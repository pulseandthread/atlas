// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness microtest for the living-weights site kernel
//! (`living_site`, `spark_model::living::launch_living_site`):
//!
//!   out = x + (v + scale * ((x · Aᵀ) · B))     x/out BF16, A/B/v F32
//!
//! against an F32 CPU reference, through the production launcher on the real
//! `GpuBackend`. Cases: rank 0 / v off is an exact copy (out-of-place and in
//! place); random rank-r with v, several rows; the full rank (512) in place;
//! and a rank change made ONLY through the device control word, which is what
//! keeps captured CUDA graphs valid.
//!
//! Tolerance: the reference is rounded to BF16 the same way the kernel rounds
//! its output; the two may differ by summation order, so each element may be
//! at most 2 BF16 ulps apart (|ref| relative) and the cosine must be >= 0.99999.
//!
//! Usage:
//!   cargo run --release -p spark-model --features gpu-examples \
//!       --example living_site_microtest -- [hidden] [seed]
//! Defaults: hidden=4096 (GLM-5.3-Flash). Exit 0 = all PASS, 1 = FAIL.

use anyhow::{Result, bail};
use spark_model::living::{MAX_RANK, launch_living_site};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + u * (hi - lo)
    }
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16(f: f32) -> u16 {
    half::bf16::from_f32(f).to_bits()
}

struct Gpu<'a> {
    gpu: &'a dyn GpuBackend,
    kernel: KernelHandle,
    stream: u64,
}

impl Gpu<'_> {
    fn upload(&self, bytes: &[u8]) -> Result<DevicePtr> {
        let p = self.gpu.alloc(bytes.len().max(16))?;
        self.gpu.copy_h2d(bytes, p)?;
        Ok(p)
    }
    fn f32s(&self, v: &[f32]) -> Result<DevicePtr> {
        self.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
    }
    fn bf16s(&self, v: &[u16]) -> Result<DevicePtr> {
        self.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
    }
    fn ctrl(&self, p: DevicePtr, rank: u32, v_on: bool) -> Result<()> {
        let words = [rank, u32::from(v_on), 0, 0];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.gpu.copy_h2d(&bytes, p)
    }
    fn read_bf16(&self, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
        let mut buf = vec![0u8; n * 2];
        self.gpu.copy_d2h(p, &mut buf)?;
        Ok(buf
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: DevicePtr,
        out: DevicePtr,
        a: DevicePtr,
        b: DevicePtr,
        v: DevicePtr,
        ctrl: DevicePtr,
        scale: f32,
        rows: usize,
        h: usize,
    ) -> Result<()> {
        launch_living_site(
            self.gpu,
            self.kernel,
            x,
            out,
            a,
            b,
            v,
            ctrl,
            scale,
            rows as u32,
            h as u32,
            self.stream,
        )?;
        self.gpu.synchronize(self.stream)
    }
}

/// F32 reference: out = x + (v + scale * ((x · Aᵀ) · B)), rounded to BF16 last.
#[allow(clippy::too_many_arguments)]
fn reference(
    x: &[u16],
    a: &[f32],
    b: &[f32],
    v: Option<&[f32]>,
    scale: f32,
    rank: usize,
    rows: usize,
    h: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * h);
    for row in 0..rows {
        let xr: Vec<f64> = x[row * h..(row + 1) * h].iter().map(|&b| bf16_to_f32(b) as f64).collect();
        let t: Vec<f64> = (0..rank)
            .map(|r| (0..h).map(|k| xr[k] * a[r * h + k] as f64).sum())
            .collect();
        for j in 0..h {
            let d: f64 = (0..rank).map(|r| t[r] * b[r * h + j] as f64).sum();
            let delta = v.map_or(0.0, |v| v[j] as f64) + scale as f64 * d;
            out.push((xr[j] + delta) as f32);
        }
    }
    out
}

/// Compare BF16 GPU output to the F32 reference; returns PASS.
fn compare(label: &str, gpu: &[u16], reference: &[f32]) -> bool {
    let (mut dot, mut ng, mut nr) = (0f64, 0f64, 0f64);
    let (mut max_ulps, mut max_abs) = (0f64, 0f64);
    for (&g, &r) in gpu.iter().zip(reference) {
        let gv = bf16_to_f32(g) as f64;
        let rv = bf16_to_f32(f32_to_bf16(r)) as f64;
        dot += gv * r as f64;
        ng += gv * gv;
        nr += (r as f64) * (r as f64);
        // One BF16 ulp at |r| (floored at 1e-3): 2^(exponent - 7).
        let ulp = 2f64.powi((r.abs().max(1e-3) as f64).log2().floor() as i32 - 7);
        max_ulps = max_ulps.max((gv - rv).abs() / ulp);
        max_abs = max_abs.max((gv - r as f64).abs());
    }
    let cos = if ng > 0.0 && nr > 0.0 {
        dot / (ng.sqrt() * nr.sqrt())
    } else {
        1.0
    };
    let pass = cos >= 0.99999 && max_ulps <= 2.0;
    println!(
        "  {:4} {label:42} cosine={cos:.7} max_ulps={max_ulps:.1} max_abs_err={max_abs:.3e} |ref|={:.3}",
        if pass { "PASS" } else { "FAIL" },
        nr.sqrt()
    );
    pass
}

fn exact(label: &str, got: &[u16], want: &[u16]) -> bool {
    let diff = got.iter().zip(want).filter(|(a, b)| a != b).count();
    let pass = diff == 0;
    println!(
        "  {:4} {label:42} bit-exact: {} of {} elements differ",
        if pass { "PASS" } else { "FAIL" },
        diff,
        got.len()
    );
    pass
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let h: usize = args.get(1).map_or(4096, |s| s.parse().unwrap());
    let seed: u64 = args.get(2).map_or(0x11F3, |s| s.parse().unwrap());
    println!("=== living_site microtest: hidden={h} max_rank={MAX_RANK} seed={seed} ===");

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let g = Gpu {
        gpu,
        kernel: gpu.kernel("living_site", "living_site")?,
        stream: gpu.create_stream()?,
    };
    let mut rng = Rng(seed);
    let rows = 4usize;
    // Residual-like rows: mostly O(1), a few large channels.
    let x: Vec<u16> = (0..rows * h)
        .map(|i| {
            let big = if i % 509 == 0 { 40.0 } else { 1.0 };
            f32_to_bf16(rng.uniform(-2.0, 2.0) * big)
        })
        .collect();
    let a: Vec<f32> = (0..MAX_RANK * h).map(|_| rng.uniform(-0.02, 0.02)).collect();
    let b: Vec<f32> = (0..MAX_RANK * h).map(|_| rng.uniform(-0.05, 0.05)).collect();
    let v: Vec<f32> = (0..h).map(|_| rng.uniform(-0.5, 0.5)).collect();

    let dx = g.bf16s(&x)?;
    let da = g.f32s(&a)?;
    let db = g.f32s(&b)?;
    let dv = g.f32s(&v)?;
    let ctrl = g.upload(&[0u8; 16])?;
    let out = g.bf16s(&vec![0u16; rows * h])?;
    let mut ok = true;

    // 1. rank 0, v off: exact copy, out of place.
    g.ctrl(ctrl, 0, false)?;
    g.run(dx, out, da, db, dv, ctrl, 10.0, rows, h)?;
    ok &= exact("rank 0, v off, out of place", &g.read_bf16(out, rows * h)?, &x);

    // 2. rank 0, v off: in place leaves x untouched.
    let inplace = g.bf16s(&x)?;
    g.run(inplace, inplace, da, db, dv, ctrl, 10.0, rows, h)?;
    ok &= exact("rank 0, v off, in place", &g.read_bf16(inplace, rows * h)?, &x);

    // 3. rank 16 with v, scale 10, 4 rows.
    g.ctrl(ctrl, 16, true)?;
    g.run(dx, out, da, db, dv, ctrl, 10.0, rows, h)?;
    let want = reference(&x, &a, &b, Some(&v), 10.0, 16, rows, h);
    ok &= compare("rank 16, v on, scale 10, 4 rows", &g.read_bf16(out, rows * h)?, &want);

    // 4. rank 1, v off, scale 1 (the synthetic-sidecar shape).
    g.ctrl(ctrl, 1, false)?;
    g.run(dx, out, da, db, dv, ctrl, 1.0, rows, h)?;
    let want = reference(&x, &a, &b, None, 1.0, 1, rows, h);
    ok &= compare("rank 1, v off, scale 1", &g.read_bf16(out, rows * h)?, &want);

    // 5. v only (rank 0, v on).
    g.ctrl(ctrl, 0, true)?;
    g.run(dx, out, da, db, dv, ctrl, 10.0, rows, h)?;
    let want = reference(&x, &a, &b, Some(&v), 10.0, 0, rows, h);
    ok &= compare("rank 0, v on (steering vector only)", &g.read_bf16(out, rows * h)?, &want);

    // 6. Full rank, in place, one row.
    let one = g.bf16s(&x[..h])?;
    g.ctrl(ctrl, MAX_RANK as u32, true)?;
    g.run(one, one, da, db, dv, ctrl, 1.0, 1, h)?;
    let want = reference(&x[..h], &a, &b, Some(&v), 1.0, MAX_RANK, 1, h);
    ok &= compare("rank 512, v on, in place, 1 row", &g.read_bf16(one, h)?, &want);

    // 7. The rank word alone switches the delta on and back off: same launch
    //    arguments, only the device control word rewritten in between.
    g.ctrl(ctrl, 8, false)?;
    g.run(dx, out, da, db, dv, ctrl, 1.0, rows, h)?;
    let want = reference(&x, &a, &b, None, 1.0, 8, rows, h);
    ok &= compare("ctrl word -> rank 8", &g.read_bf16(out, rows * h)?, &want);
    g.ctrl(ctrl, 0, false)?;
    g.run(dx, out, da, db, dv, ctrl, 1.0, rows, h)?;
    ok &= exact("ctrl word -> rank 0 again", &g.read_bf16(out, rows * h)?, &x);

    if !ok {
        println!("RESULT: FAIL");
        bail!("living_site microtest failed");
    }
    println!("RESULT: PASS");
    Ok(())
}
