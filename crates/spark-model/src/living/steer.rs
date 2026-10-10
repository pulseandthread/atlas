// SPDX-License-Identifier: AGPL-3.0-only

//! Steering at a middle layer: the `living_site` kernel, v term only, added
//! to the residual stream after layer `L` (`ATLAS_STEER_LAYER`, default the
//! middle layer, `num_hidden_layers / 2`), on every prefill, decode and
//! verify path, with the dose set at runtime.
//!
//! GLM-5's residual stream is the mHC highway, `hc_mult` streams of
//! `hidden_size` per token (`ctx.buffers.hc_streams()`, BF16 or F32). The
//! vector is added to every stream: the next layer's Sinkhorn residual mix is
//! doubly stochastic, so a shift common to all streams is carried forward
//! exactly as a plain residual would carry it, and every pre-mix and the
//! final contraction see it.
//!
//! The site lives in the KDA layers' code: after layer `L` when `L` is a KDA
//! layer, else (an MLA layer) at the start of KDA layer `L + 1`, before its
//! pre-mix reads the highway. Both are the stream between `L` and `L + 1`.
//!
//! Vectors: `ATLAS_STEER_DIR` holds `index.json` (`{"files": [...]}`) and
//! safetensors files with `v` `[hidden]` and `__metadata__` `{base, layer,
//! rms, topic, scale}`. Each `v` is normalized to unit RMS (`û`, norm
//! `sqrt(hidden)`); the device vector is `Σ scale_i · rms_i · û_i`, so one file
//! at dose `d` adds `d · rms · û` to every stream: the added vector's RMS is
//! `d` times the residual RMS at that layer (the dose is in units of residual
//! RMS). `rms` is measured once, from dumped rows (below), and stored in the
//! file (`ATLAS_STEER_RMS` is the fallback for files without one). A file made
//! for another checkpoint (`base`) or another layer (`layer`) is refused.
//!
//! The dose is the F32 multiplier in the site's device control word; `0`
//! switches the term off (an exact no-op). `ATLAS_STEER_SCALE` sets the
//! starting dose (default 0); [`set_dose`] changes it between steps. CUDA
//! graphs captured at startup stay valid.
//!
//! Read-out: `ATLAS_DUMP_LAYER_HIDDEN=DIR` appends the highway rows at the
//! same spot (before the steering add) to `DIR/hidden.f32` (F32, `hc_mult`
//! rows of `hidden` per token) and one `DIR/index.jsonl` line per call; for a
//! prefill call only its last token (`ATLAS_DUMP_LAYER_HIDDEN_ROWS=all`: every
//! token), for decode and verify every token. Rank 0 only; calls made while a
//! stream is being captured (graph replays) are skipped.

use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::dump::Sink;
use super::{SafeTensors, launch_living_site};
use crate::layer::ForwardContext;

/// Where in a KDA layer the site runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spot {
    /// Before the layer's attention pre-mix reads the highway.
    Start,
    /// After the layer's FFN post-mix wrote the highway.
    End,
}

/// The kind of call, for the read-out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Prefill,
    Decode,
    Verify,
}

/// One loaded vector file, for the status report.
#[derive(Clone, Debug)]
pub struct VectorFile {
    pub file: String,
    pub topic: String,
    pub rms: f32,
    pub scale: f32,
}

/// Loaded vectors, host side.
#[derive(Debug, Default)]
pub struct SteerHost {
    /// `Σ scale_i · rms_i · û_i` (`û` unit RMS), `[hidden]`.
    pub v: Vec<f32>,
    pub files: Vec<VectorFile>,
}

struct Site {
    bf16: KernelHandle,
    f32: KernelHandle,
    v: DevicePtr,
    ctrl: DevicePtr,
    files: Vec<VectorFile>,
    v_norm: f32,
}

struct Steer {
    layer: usize,
    at: (usize, Spot),
    hidden: usize,
    site: Option<Site>,
    dump: Option<Mutex<Sink>>,
    dump_all: bool,
}

static STEER: OnceLock<Steer> = OnceLock::new();
/// The current dose as F32 bits (rank-local mirror of the device word).
static DOSE: AtomicU32 = AtomicU32::new(0);

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn parse_env<T: std::str::FromStr>(key: &str) -> Result<Option<T>> {
    match env(key) {
        Some(s) => Ok(Some(
            s.trim()
                .parse::<T>()
                .ok()
                .with_context(|| format!("steer: {key}={s:?} does not parse"))?,
        )),
        None => Ok(None),
    }
}

/// The layer `ATLAS_STEER_LAYER` names (default the middle layer) and the
/// KDA spot that sees the stream between it and the next layer.
pub fn site_for(config: &ModelConfig, layer: Option<usize>) -> Result<(usize, (usize, Spot))> {
    let n = config.num_hidden_layers;
    let layer = layer.unwrap_or(n / 2);
    ensure!(layer < n, "steer: layer {layer} is not below num_hidden_layers {n}");
    if config.layer_type(layer) == LayerType::LinearAttention {
        return Ok((layer, (layer, Spot::End)));
    }
    ensure!(
        layer + 1 < n && config.layer_type(layer + 1) == LayerType::LinearAttention,
        "steer: neither layer {layer} nor layer {} is a KDA layer, where the site lives",
        layer + 1
    );
    Ok((layer, (layer + 1, Spot::Start)))
}

/// Set up steering and its read-out from the environment, once per process,
/// for the first GLM-5 model built, before any CUDA graph is captured.
/// Nothing is allocated or launched unless `ATLAS_STEER_DIR` or
/// `ATLAS_DUMP_LAYER_HIDDEN` is set. Other models (a drafter) are skipped:
/// the site is wired into the GLM-5 KDA layers (mHC highway) only.
pub fn install(gpu: &dyn GpuBackend, config: &ModelConfig, base: &str) -> Result<()> {
    let dir = env("ATLAS_STEER_DIR");
    if STEER.get().is_some() || (dir.is_none() && env("ATLAS_DUMP_LAYER_HIDDEN").is_none()) {
        return Ok(());
    }
    if config.model_type != "glm5_next" || config.hc_mult == 0 {
        tracing::warn!(
            "steer: the middle-layer site lives in GLM-5 KDA layers only; skipped for {}",
            config.model_type
        );
        return Ok(());
    }
    let dump = Sink::open("ATLAS_DUMP_LAYER_HIDDEN");
    let (layer, at) = site_for(config, parse_env::<usize>("ATLAS_STEER_LAYER")?)?;
    let hidden = config.hidden_size;
    let site = match dir {
        Some(dir) => {
            let host = read_vectors(
                Path::new(&dir),
                base,
                layer,
                hidden,
                parse_env::<f32>("ATLAS_STEER_RMS")?,
            )?;
            ensure!(!host.files.is_empty(), "steer: {dir} lists no vector files");
            let dose = parse_env::<f32>("ATLAS_STEER_SCALE")?.unwrap_or(0.0);
            ensure!(dose.is_finite(), "steer: ATLAS_STEER_SCALE must be finite");
            let v = gpu.alloc(hidden * 4)?;
            gpu.copy_h2d(&super::f32_bytes(&host.v), v)?;
            let ctrl = gpu.alloc(16)?;
            gpu.copy_h2d(&ctrl_bytes(dose), ctrl)?;
            DOSE.store(dose.to_bits(), Ordering::Relaxed);
            let v_norm = host.v.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!(
                "steer: {} vector(s) after layer {layer} ({} of layer {}), |v| at dose 1 = {v_norm:.4}, dose {dose}",
                host.files.len(),
                if at.1 == Spot::End { "end" } else { "start" },
                at.0,
            );
            Some(Site {
                bf16: gpu
                    .kernel("living_site", "living_site")
                    .context("steer: kernel living_site::living_site is not in this kernel target")?,
                f32: gpu
                    .kernel("living_site", "living_site_f32")
                    .context("steer: kernel living_site::living_site_f32 is not in this kernel target")?,
                v,
                ctrl,
                files: host.files,
                v_norm,
            })
        }
        None => None,
    };
    if dump.is_some() {
        tracing::info!("ATLAS_DUMP_LAYER_HIDDEN: highway rows after layer {layer}");
    }
    let steer = Steer {
        layer,
        at,
        hidden,
        site,
        dump,
        dump_all: env("ATLAS_DUMP_LAYER_HIDDEN_ROWS").as_deref() == Some("all"),
    };
    if STEER.set(steer).is_err() {
        bail!("steer: installed twice in one process");
    }
    Ok(())
}

fn ctrl_bytes(dose: f32) -> Vec<u8> {
    let words = [0u32, u32::from(dose != 0.0), dose.to_bits(), 0];
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Whether steering vectors are loaded in this process.
pub fn loaded() -> bool {
    STEER.get().is_some_and(|s| s.site.is_some())
}

/// Whether `ATLAS_STEER_DIR` is set (the server arms the control endpoint).
pub fn requested() -> bool {
    env("ATLAS_STEER_DIR").is_some()
}

/// The current dose.
pub fn dose() -> f32 {
    f32::from_bits(DOSE.load(Ordering::Relaxed))
}

/// Write the dose into the device control word. Call only at a quiescent
/// point (no step in flight on any stream); `0` switches the term off.
pub fn set_dose(gpu: &dyn GpuBackend, dose: f32) -> Result<()> {
    ensure!(dose.is_finite(), "steer: dose {dose} is not finite");
    let site = STEER
        .get()
        .and_then(|s| s.site.as_ref())
        .context("steer: no steering vectors are loaded (ATLAS_STEER_DIR)")?;
    gpu.copy_h2d(&ctrl_bytes(dose), site.ctrl)?;
    DOSE.store(dose.to_bits(), Ordering::Relaxed);
    tracing::info!("steer: dose {dose}");
    Ok(())
}

/// The status the control endpoint reports.
pub fn status() -> serde_json::Value {
    let Some(s) = STEER.get() else {
        return serde_json::json!({"object": "living.steer", "configured": false});
    };
    let files: Vec<serde_json::Value> = s
        .site
        .iter()
        .flat_map(|site| &site.files)
        .map(|f| {
            serde_json::json!({"file": f.file, "topic": f.topic, "rms": f.rms, "scale": f.scale})
        })
        .collect();
    let spot = if s.at.1 == Spot::End { "end" } else { "start" };
    serde_json::json!({
        "object": "living.steer",
        "configured": s.site.is_some(),
        "scale": dose(),
        "layer": s.layer,
        "site": format!("{spot} of layer {}", s.at.0),
        "rms": s.site.as_ref().and_then(|site| site.files.first()).map(|f| f.rms),
        "v_norm_at_scale_1": s.site.as_ref().map(|site| site.v_norm),
        "files": files,
        "dump": s.dump.is_some(),
    })
}

/// The KDA layers call this at both spots; it acts only at the configured
/// one. `rows` are the call's local highway rows (tokens), `hc_mult` streams
/// of `hidden` each, at `ctx.buffers.hc_streams()`.
pub fn hook(
    ctx: &ForwardContext,
    layer_idx: usize,
    spot: Spot,
    rows: usize,
    hc_mult: usize,
    kind: Kind,
    stream: u64,
) -> Result<()> {
    let Some(s) = STEER.get() else { return Ok(()) };
    if s.at != (layer_idx, spot) || rows == 0 {
        return Ok(());
    }
    let highway = ctx.buffers.hc_streams();
    let elem = crate::layers::ops::hc_elem_bytes(&ctx.config.model_type);
    if let Some(sink) = &s.dump {
        dump_rows(ctx, s, sink, highway, elem, rows, hc_mult, kind, stream)?;
    }
    if let Some(site) = &s.site {
        let kernel = if elem == 2 { site.bf16 } else { site.f32 };
        // Rank 0 and no A/B: the A and B arguments are never read.
        launch_living_site(
            ctx.gpu,
            kernel,
            highway,
            highway,
            site.v,
            site.v,
            site.v,
            site.ctrl,
            0.0,
            (rows * hc_mult) as u32,
            s.hidden as u32,
            stream,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dump_rows(
    ctx: &ForwardContext,
    s: &Steer,
    sink: &Mutex<Sink>,
    highway: DevicePtr,
    elem: usize,
    rows: usize,
    hc_mult: usize,
    kind: Kind,
    stream: u64,
) -> Result<()> {
    static CALL: AtomicU64 = AtomicU64::new(0);
    if ctx.config.tp_rank != 0 || ctx.gpu.stream_is_capturing(stream) {
        return Ok(());
    }
    let first = if kind == Kind::Prefill && !s.dump_all {
        rows - 1
    } else {
        0
    };
    let tokens = rows - first;
    let row_bytes = hc_mult * s.hidden * elem;
    let mut buf = vec![0u8; tokens * row_bytes];
    ctx.gpu
        .copy_d2h_on_stream(highway.offset(first * row_bytes), &mut buf, stream)?;
    let f32s: Vec<u8> = if elem == 2 {
        buf.chunks_exact(2)
            .flat_map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32().to_le_bytes())
            .collect()
    } else {
        buf
    };
    let call = CALL.fetch_add(1, Ordering::Relaxed);
    let sp = crate::layers::glm_sp::current().is_some();
    let kind = match kind {
        Kind::Prefill => "prefill",
        Kind::Decode => "decode",
        Kind::Verify => "verify",
    };
    let mut sink = sink.lock().unwrap_or_else(|e| e.into_inner());
    sink.append(&f32s, |offset| {
        serde_json::json!({
            "call": call,
            "layer": s.layer,
            "kind": kind,
            "rows_in_call": rows,
            "first_row": first,
            "tokens": tokens,
            "hc": hc_mult,
            "n_rows": tokens * hc_mult,
            "hidden": s.hidden,
            "offset": offset,
            "sp": sp,
            "dose": dose(),
        })
    })?;
    Ok(())
}

/// Read `dir`'s steering vectors (see the module docs).
pub fn read_vectors(
    dir: &Path,
    base: &str,
    layer: usize,
    hidden: usize,
    rms_fallback: Option<f32>,
) -> Result<SteerHost> {
    let index_path = dir.join("index.json");
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&index_path)
            .with_context(|| format!("steer: read {}", index_path.display()))?,
    )
    .with_context(|| format!("steer: parse {}", index_path.display()))?;
    let files = index["files"]
        .as_array()
        .with_context(|| format!("steer: {} has no \"files\" list", index_path.display()))?;
    let mut host = SteerHost {
        v: vec![0f32; hidden],
        files: Vec::new(),
    };
    for name in files {
        let name = name
            .as_str()
            .with_context(|| format!("steer: non-string entry in {}", index_path.display()))?;
        let file = SafeTensors::read(&dir.join(name))?;
        let meta = |key: &str| file.metadata.get(key).map(String::as_str).unwrap_or("");
        ensure!(
            meta("base") == base,
            "steer: {name} was made for base \"{}\", but the served checkpoint is \"{base}\"",
            meta("base")
        );
        let number = |key: &str| -> Result<Option<f64>> {
            match meta(key) {
                "" => Ok(None),
                s => Ok(Some(s.parse().with_context(|| {
                    format!("steer: {name}: {key} \"{s}\" is not a number")
                })?)),
            }
        };
        if let Some(at) = number("layer")? {
            ensure!(
                at == layer as f64,
                "steer: {name} was made for layer {at}, but ATLAS_STEER_LAYER is {layer}"
            );
        }
        let rms = number("rms")?
            .map(|r| r as f32)
            .or(rms_fallback)
            .with_context(|| {
                format!("steer: {name} has no rms in its metadata and ATLAS_STEER_RMS is not set")
            })?;
        ensure!(rms.is_finite() && rms > 0.0, "steer: {name}: rms {rms} must be positive");
        let scale = number("scale")?.unwrap_or(1.0) as f32;
        ensure!(scale.is_finite(), "steer: {name}: scale must be finite");
        let (v, shape) = file.tensor_f32("v")?;
        ensure!(v.len() == hidden, "steer: {name}: v {shape:?} must be [{hidden}]");
        let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        ensure!(norm > 0.0, "steer: {name}: v is zero");
        // Unit RMS, not unit norm: the added vector's RMS is `dose * rms`.
        let k = (scale as f64 * rms as f64 * (hidden as f64).sqrt() / norm) as f32;
        for (s, x) in host.v.iter_mut().zip(&v) {
            *s += k * x;
        }
        host.files.push(VectorFile {
            file: name.to_string(),
            topic: meta("topic").to_string(),
            rms,
            scale,
        });
    }
    Ok(host)
}

#[cfg(test)]
#[path = "steer_tests.rs"]
mod tests;
