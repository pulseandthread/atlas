// SPDX-License-Identifier: AGPL-3.0-only

//! Living weights: a small, removable change to the last layer's residual,
//! kept in sidecar files beside the checkpoint and never folded into it.
//!
//! The site sits in `final_norm_rows`, in front of the final norm, so every
//! prefill, decode and verify path passes through it:
//!
//! ```text
//! out = x + (v + scale * ((x · Aᵀ) · B))      A, B: [rank, hidden]   v: [hidden]
//! ```
//!
//! `--living-dir DIR` (or `ATLAS_LIVING_DIR`) loads `DIR/index.json`
//! (`{"files": ["0001.safetensors", ...]}`). Each file holds `A` and `B`
//! (`[r, hidden]`, F32/BF16/F16) and optionally `v` (`[hidden]`), with
//! `__metadata__` `{site: "final", scale, base, topic, created}`. The files are
//! concatenated in order (their ranks add up, at most [`MAX_RANK`]), each
//! file's `scale` folded into its rows of `B`, and their `v` summed. A file
//! whose `base` is not the served checkpoint's revision is refused.
//! `LIVING_TOPICS=a,b` loads only files with those topics.
//!
//! The device buffers are allocated at [`MAX_RANK`] rows once, and the live
//! rank and the v switch are read by the kernel from a device word, so CUDA
//! graphs captured at startup stay valid whatever the sidecar holds. With no
//! living dir configured nothing is launched and nothing is allocated.
//!
//! `ATLAS_DUMP_LAST_HIDDEN=DIR` is the read-out side: [`dump`] appends the
//! pre-norm rows handed to the final norm to `DIR/hidden.f32` (row-major F32)
//! with one `DIR/index.jsonl` line per call.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Rows the device buffers hold; the summed rank of all loaded files.
pub const MAX_RANK: usize = 512;

/// Threads per row in `living_site`.
const BLOCK: u32 = 256;

/// `living_site` for `rows` BF16 rows of width `h`: `out = x + (v + scale *
/// ((x · Aᵀ) · B))` with the rank and v switch read from `ctrl` on the
/// device. `out` may equal `x`. Capture-safe: one launch, no host reads.
#[allow(clippy::too_many_arguments)]
pub fn launch_living_site(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    out: DevicePtr,
    a: DevicePtr,
    b: DevicePtr,
    v: DevicePtr,
    ctrl: DevicePtr,
    scale: f32,
    rows: u32,
    h: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(x)
        .arg_ptr(out)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(v)
        .arg_ptr(ctrl)
        .arg_f32(scale)
        .arg_u32(h)
        .launch(stream)
}

/// The concatenated sidecar, host side.
#[derive(Debug, Default)]
pub struct LivingHost {
    /// `[rank, hidden]`; `b` has each file's scale folded in.
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    /// Sum of the files' `v`; `None` when no file has one.
    pub v: Option<Vec<f32>>,
    pub rank: usize,
    pub files: usize,
}

/// The device side: buffers sized for [`MAX_RANK`], the control word, and
/// the kernel. Owned by the model; installed once at build.
pub struct LivingSite {
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    v: DevicePtr,
    ctrl: DevicePtr,
    hidden: usize,
    pub rank: usize,
    pub files: usize,
    pub has_v: bool,
}

impl LivingSite {
    /// Load `dir` for a model of width `hidden` served from a checkpoint
    /// whose revision is `base`, upload it, and log the one-line summary.
    pub fn load(gpu: &dyn GpuBackend, dir: &Path, base: &str, hidden: usize) -> Result<Self> {
        let topics = std::env::var("LIVING_TOPICS").ok();
        let host = read_sidecar(dir, base, hidden, topics.as_deref())?;
        let kernel = gpu
            .kernel("living_site", "living_site")
            .context("living: kernel living_site::living_site is not in this kernel target")?;
        let rows_bytes = MAX_RANK * hidden * 4;
        let a = gpu.alloc(rows_bytes)?;
        let b = gpu.alloc(rows_bytes)?;
        let v = gpu.alloc(hidden * 4)?;
        let ctrl = gpu.alloc(16)?;
        let mut a_full = vec![0f32; MAX_RANK * hidden];
        let mut b_full = vec![0f32; MAX_RANK * hidden];
        a_full[..host.a.len()].copy_from_slice(&host.a);
        b_full[..host.b.len()].copy_from_slice(&host.b);
        let v_full = host.v.clone().unwrap_or_else(|| vec![0f32; hidden]);
        gpu.copy_h2d(&f32_bytes(&a_full), a)?;
        gpu.copy_h2d(&f32_bytes(&b_full), b)?;
        gpu.copy_h2d(&f32_bytes(&v_full), v)?;
        let words = [host.rank as u32, u32::from(host.v.is_some()), 0, 0];
        let ctrl_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        gpu.copy_h2d(&ctrl_bytes, ctrl)?;
        tracing::info!(
            "living: {} files, rank {}, v: {}",
            host.files,
            host.rank,
            if host.v.is_some() { "yes" } else { "no" }
        );
        Ok(Self {
            kernel,
            a,
            b,
            v,
            ctrl,
            hidden,
            rank: host.rank,
            files: host.files,
            has_v: host.v.is_some(),
        })
    }

    /// `output = x + delta` for `rows` BF16 rows; the caller then normalizes
    /// `output` in place. Rank 0 with no v is an exact copy.
    pub fn apply(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        launch_living_site(
            gpu,
            self.kernel,
            input,
            output,
            self.a,
            self.b,
            self.v,
            self.ctrl,
            1.0,
            rows,
            self.hidden as u32,
            stream,
        )
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The revision a sidecar's `base` must name: the checkpoint's
/// `.sparkglm-revision`, looked up in `model_dir` and then beside the file
/// `model_dir/config.json` resolves to (an overlay that links into the
/// checkpoint), else the model directory's name.
pub fn served_revision(model_dir: &Path) -> String {
    let marker = ".sparkglm-revision";
    let mut candidates = vec![model_dir.join(marker)];
    if let Ok(real) = std::fs::canonicalize(model_dir.join("config.json"))
        && let Some(parent) = real.parent()
    {
        candidates.push(parent.join(marker));
    }
    for path in candidates {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let rev = text.trim();
            if !rev.is_empty() {
                return rev.to_string();
            }
        }
    }
    model_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Read and concatenate `dir`'s sidecar files (see the module docs).
pub fn read_sidecar(
    dir: &Path,
    base: &str,
    hidden: usize,
    topics: Option<&str>,
) -> Result<LivingHost> {
    let index_path = dir.join("index.json");
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&index_path)
            .with_context(|| format!("living: read {}", index_path.display()))?,
    )
    .with_context(|| format!("living: parse {}", index_path.display()))?;
    let files = index["files"]
        .as_array()
        .with_context(|| format!("living: {} has no \"files\" list", index_path.display()))?;
    let wanted: Option<Vec<&str>> =
        topics.map(|t| t.split(',').map(str::trim).filter(|t| !t.is_empty()).collect());
    let mut host = LivingHost::default();
    for name in files {
        let name = name
            .as_str()
            .with_context(|| format!("living: non-string entry in {}", index_path.display()))?;
        let file = SafeTensors::read(&dir.join(name))?;
        let meta = |key: &str| file.metadata.get(key).map(String::as_str).unwrap_or("");
        if let Some(wanted) = &wanted
            && !wanted.contains(&meta("topic"))
        {
            continue;
        }
        let site = meta("site");
        ensure!(
            site.is_empty() || site == "final",
            "living: {name}: site \"{site}\" is not supported (only \"final\")"
        );
        ensure!(
            meta("base") == base,
            "living: {name} was made for base \"{}\", but the served checkpoint is \"{base}\"",
            meta("base")
        );
        let scale: f32 = match meta("scale") {
            "" => 1.0,
            s => s
                .parse()
                .with_context(|| format!("living: {name}: scale \"{s}\" is not a number"))?,
        };
        let (a, a_shape) = file.tensor_f32("A")?;
        let (b, b_shape) = file.tensor_f32("B")?;
        ensure!(
            a_shape.len() == 2 && a_shape[1] == hidden && a_shape == b_shape,
            "living: {name}: A {a_shape:?} and B {b_shape:?} must both be [r, {hidden}]"
        );
        let r = a_shape[0];
        ensure!(
            host.rank + r <= MAX_RANK,
            "living: {name}: total rank {} exceeds {MAX_RANK}",
            host.rank + r
        );
        host.a.extend_from_slice(&a);
        host.b.extend(b.iter().map(|x| x * scale));
        host.rank += r;
        if file.has("v") {
            let (v, v_shape) = file.tensor_f32("v")?;
            ensure!(
                v.len() == hidden,
                "living: {name}: v {v_shape:?} must be [{hidden}]"
            );
            let sum = host.v.get_or_insert_with(|| vec![0f32; hidden]);
            for (s, x) in sum.iter_mut().zip(&v) {
                *s += x;
            }
        }
        host.files += 1;
    }
    Ok(host)
}

/// The few safetensors facts the loader needs: a header, its metadata, data.
struct SafeTensors {
    header: serde_json::Map<String, serde_json::Value>,
    metadata: BTreeMap<String, String>,
    data: Vec<u8>,
    path: PathBuf,
}

impl SafeTensors {
    fn read(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("living: read {}", path.display()))?;
        ensure!(bytes.len() >= 8, "living: {} is not a safetensors file", path.display());
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        ensure!(
            n <= bytes.len() - 8,
            "living: {} has a truncated header",
            path.display()
        );
        let mut header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + n])
                .with_context(|| format!("living: parse the header of {}", path.display()))?;
        let metadata = header
            .remove("__metadata__")
            .and_then(|m| m.as_object().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
            .collect();
        Ok(Self {
            header,
            metadata,
            data: bytes[8 + n..].to_vec(),
            path: path.to_path_buf(),
        })
    }

    fn has(&self, name: &str) -> bool {
        self.header.contains_key(name)
    }

    fn tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let p = self.path.display();
        let info = self
            .header
            .get(name)
            .with_context(|| format!("living: {p} has no tensor {name}"))?;
        let dims = |key: &str| -> Vec<usize> {
            info[key]
                .as_array()
                .map(|s| s.iter().filter_map(|d| d.as_u64().map(|d| d as usize)).collect())
                .unwrap_or_default()
        };
        let shape = dims("shape");
        let offsets = dims("data_offsets");
        ensure!(offsets.len() == 2, "living: {p}: {name} has no data_offsets");
        let (lo, hi) = (offsets[0], offsets[1]);
        ensure!(
            lo <= hi && hi <= self.data.len(),
            "living: {p}: {name} lies outside the file"
        );
        let raw = &self.data[lo..hi];
        let values: Vec<f32> = match info["dtype"].as_str().unwrap_or("") {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => bail!("living: {p}: {name} has dtype {other}; use F32, BF16 or F16"),
        };
        ensure!(
            values.len() == shape.iter().product::<usize>(),
            "living: {p}: {name} holds {} values for shape {shape:?}",
            values.len()
        );
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "living: {p}: {name} is not finite"
        );
        Ok((values, shape))
    }
}

/// `ATLAS_DUMP_LAST_HIDDEN=DIR`: the pre-norm rows handed to the final norm.
pub mod dump {
    use std::cell::Cell;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    use anyhow::Result;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};

    struct Sink {
        rows: std::fs::File,
        index: std::fs::File,
        offset: u64,
    }

    fn sink() -> Option<&'static Mutex<Sink>> {
        static SINK: OnceLock<Option<Mutex<Sink>>> = OnceLock::new();
        SINK.get_or_init(|| {
            let dir = std::env::var("ATLAS_DUMP_LAST_HIDDEN")
                .ok()
                .filter(|d| !d.is_empty())?;
            let dir = std::path::Path::new(&dir);
            let open = |name: &str| {
                std::fs::create_dir_all(dir)?;
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join(name))
            };
            match (open("hidden.f32"), open("index.jsonl")) {
                (Ok(rows), Ok(index)) => {
                    let offset = rows.metadata().map(|m| m.len()).unwrap_or(0);
                    tracing::info!("ATLAS_DUMP_LAST_HIDDEN: appending to {}", dir.display());
                    Some(Mutex::new(Sink {
                        rows,
                        index,
                        offset,
                    }))
                }
                (Err(e), _) | (_, Err(e)) => {
                    tracing::warn!("ATLAS_DUMP_LAST_HIDDEN={}: {e}; not dumping", dir.display());
                    None
                }
            }
        })
        .as_ref()
    }

    /// Whether dumping is on; resolved once.
    #[inline]
    pub fn enabled() -> bool {
        sink().is_some()
    }

    thread_local! {
        static TAG: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
    }

    /// Label the next dumped call on this thread with the sequence's slot
    /// and length (prefill paths call this just before the final norm).
    #[inline]
    pub fn tag(slot: usize, seq_len: usize) {
        if enabled() {
            TAG.with(|t| t.set(Some((slot, seq_len))));
        }
    }

    /// Append `rows` BF16 rows at `x` as F32. Skipped while `stream` is being
    /// captured into a CUDA graph (a sync or D2H there would invalidate the
    /// capture); graph replays never come back through here, so decode rows
    /// are dumped only on eager steps. Rank 0 only: every rank holds the
    /// same full rows.
    pub fn rows(
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: u32,
        h: usize,
        rank: usize,
        site: &'static std::panic::Location<'static>,
        stream: u64,
    ) -> Result<()> {
        static CALL: AtomicU64 = AtomicU64::new(0);
        let Some(sink) = sink() else { return Ok(()) };
        let tag = TAG.with(|t| t.take());
        if rank != 0 || gpu.stream_is_capturing(stream) {
            return Ok(());
        }
        let mut buf = vec![0u8; rows as usize * h * 2];
        gpu.copy_d2h_on_stream(x, &mut buf, stream)?;
        let f32s: Vec<u8> = buf
            .chunks_exact(2)
            .flat_map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32().to_le_bytes())
            .collect();
        let call = CALL.fetch_add(1, Ordering::Relaxed);
        let mut s = sink.lock().unwrap_or_else(|e| e.into_inner());
        let offset = s.offset;
        s.rows.write_all(&f32s)?;
        s.offset += f32s.len() as u64;
        let line = serde_json::json!({
            "call": call,
            "slot": tag.map(|t| t.0),
            "seq_len": tag.map(|t| t.1),
            "n_rows": rows,
            "hidden": h,
            "offset": offset,
            "site": format!("{}:{}", site.file(), site.line()),
        });
        writeln!(s.index, "{line}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Tensor<'a> = (&'a str, &'a [usize], &'a [f32]);

    fn write_file(path: &Path, tensors: &[Tensor<'_>], meta: &[(&str, &str)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let lo = data.len();
            data.extend(values.iter().flat_map(|x| x.to_le_bytes()));
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [lo, data.len()]}),
            );
        }
        let meta: serde_json::Map<String, serde_json::Value> = meta
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::from(*v)))
            .collect();
        header.insert("__metadata__".into(), serde_json::Value::Object(meta));
        let h = serde_json::to_vec(&header).unwrap();
        let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
        bytes.extend(h);
        bytes.extend(data);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn living_sidecar_concatenates_folds_scale_sums_v_and_filters() {
        let dir = std::env::temp_dir().join(format!("living-sidecar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let m1 = [("site", "final"), ("base", "rev1"), ("scale", "10"), ("topic", "x")];
        let m2 = [("site", "final"), ("base", "rev1"), ("scale", "2"), ("topic", "y")];
        write_file(
            &dir.join("1.safetensors"),
            &[("A", &[1, 2], &[1., 2.]), ("B", &[1, 2], &[3., 4.]), ("v", &[2], &[1., 1.])],
            &m1,
        );
        write_file(
            &dir.join("2.safetensors"),
            &[
                ("A", &[2, 2], &[5., 6., 7., 8.]),
                ("B", &[2, 2], &[1., 1., 1., 1.]),
                ("v", &[2], &[0.5, 0.]),
            ],
            &m2,
        );
        std::fs::write(
            dir.join("index.json"),
            r#"{"files": ["1.safetensors", "2.safetensors"]}"#,
        )
        .unwrap();
        let all = read_sidecar(&dir, "rev1", 2, None).unwrap();
        assert_eq!((all.files, all.rank), (2, 3));
        assert_eq!(all.a, vec![1., 2., 5., 6., 7., 8.]);
        assert_eq!(all.b, vec![30., 40., 2., 2., 2., 2.]);
        assert_eq!(all.v, Some(vec![1.5, 1.]));
        let y = read_sidecar(&dir, "rev1", 2, Some("y")).unwrap();
        assert_eq!((y.files, y.rank), (1, 2));
        let wrong_base = read_sidecar(&dir, "rev2", 2, None).unwrap_err().to_string();
        assert!(wrong_base.contains("rev2"), "{wrong_base}");
        assert!(read_sidecar(&dir, "rev1", 3, None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
