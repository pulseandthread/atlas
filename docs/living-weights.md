<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Living weights

A small, removable change to the last layer's residual, kept in sidecar files
beside the checkpoint and never folded into it. Code: `crates/spark-model/src/living.rs`,
`kernels/gb10/common/living_site.cu`, the hook in
`crates/spark-model/src/model/final_norm.rs` (`final_norm_rows`).

Two sites share one kernel: the final site (stage 1, below) and a steering
site after a middle layer (stage 2, [Steering](#steering-at-a-middle-layer)).

Everything is off by default. With no `--living-dir` and no
`ATLAS_DUMP_LAST_HIDDEN`, nothing is allocated and no kernel is launched; the
only change on that path is that two GLM paths which called `rms_norm` with the
final-norm weight directly (`glm_passengers_finish`, the long-owner verify) now
call `final_norm_rows`, which issues the identical launch.

## The site

For each row `x` (BF16, `hidden_size` wide) handed to the final norm:

    out = x + (v + scale * ((x · Aᵀ) · B))      A, B: [R, hidden]   v: [hidden]

- One launch inside `final_norm_rows`, before the norm, so every prefill,
  decode, verify, fused-chunk and prompt-logprobs path passes through it.
- `living_site` writes `x + delta` to the norm's OUTPUT buffer and the norm then
  runs in place on it. The pre-norm hidden buffer is never modified (MTP/DFlash
  drafters and anything else reading it after the norm see the base residual).
- A, B and v live on the device as F32; buffers are allocated once at
  512 rows. The live rank R and the v switch are read by the kernel from a
  device control word, so the CUDA graphs captured at startup stay valid
  whatever R is. R = 0 with v off is an exact copy (no rounding), so an empty
  sidecar is token-exact as well.
- The kernel computes in F32 and rounds once to BF16.
- `scale` is per file; it is folded into that file's rows of B at load, and the
  kernel's own scale argument is 1.

## Sidecar format (`--living-dir DIR`, or `ATLAS_LIVING_DIR`)

    DIR/index.json            {"files": ["0001-....safetensors", ...]}   (in order)
    DIR/0001-....safetensors  A [r, hidden], B [r, hidden], optional v [hidden]
                              (F32, BF16 or F16)
                              __metadata__: site="final", scale="10",
                                            base="<revision>", topic="...", created="..."

Loading concatenates the A/B blocks in index order (total rank at most 512),
sums the v vectors, uploads, sets the rank word, and logs one line:
`living: N files, rank R, v: yes/no`. `LIVING_TOPICS=a,b` loads only files
whose `topic` is listed. A file whose `base` is not the served checkpoint's
`.sparkglm-revision` (looked up in the model dir, then beside the file
`config.json` links to — the SparkGLM overlay case — else the model dir's
name) is refused and the server does not start. Undo = remove the file from
`index.json` and restart.

Both TP ranks hold the full last hidden and project half the vocab each, so
both must load the same directory. SparkGLM's `start.sh` mirrors `LIVING_DIR` to
the worker before starting it.

## Hidden dump (`ATLAS_DUMP_LAST_HIDDEN=DIR`)

Appends the pre-norm rows (before the living delta) to `DIR/hidden.f32`
(row-major F32) and one line per call to `DIR/index.jsonl`:

    {"call", "slot", "seq_len", "n_rows", "hidden", "offset" (bytes), "site" (file:line of the caller)}

- Rank 0 only (every rank holds the same rows).
- Prefill: the last prompt token's row, tagged with the sequence slot and
  length (prefill_a / prefill_b finalize / prefill_c).
- Decode and verify rows are dumped only on eager calls: a call made while the
  stream is being captured into a CUDA graph is skipped (a sync there would
  break the capture), and graph replays never reach host code. With the
  serving profile's decode graphs on, expect prefill rows plus whatever decode
  steps ran eagerly; `site` tells them apart.
- The dump synchronizes the stream, so it costs speed while on.

## Known limits

- A prefix-cache exact hit that restores a stashed post-norm hidden
  (`prefill_b` Marconi exact path) replays the row computed when it was cached;
  the cache is not persistent across restarts unless the NVMe tier is on, so a
  sidecar change plus restart is clean in the default setup.
- The MTP drafter's own prefill normalization of target rows
  (`drafter_prefill.rs`) does not apply the delta (drafts only; verification
  still sees it).

## Steering at a middle layer

Code: `crates/spark-model/src/living/steer.rs`, the hooks in the GLM-5 KDA
layer (`layers/glm5_kda/forward_attention.rs`, `forward_ffn.rs`,
`multi_seq.rs`), the dose command `model/living_steer.rs`, the endpoint
`crates/spark-server/src/api/living_control.rs`. Off unless `ATLAS_STEER_DIR`
or `ATLAS_DUMP_LAYER_HIDDEN` is set; then nothing else changes either.

### Where

GLM-5.3-Flash's residual stream is the mHC highway: `hc_mult` (4) streams of
`hidden_size` (4096) per token, BF16 on the DFlash lane (`ATLAS_GLM_HC_BF16=1`),
in `ctx.buffers.hc_streams()`. The site adds the steering vector to **every
stream** of every token: the next layer's residual mix is Sinkhorn-normalized
(doubly stochastic), so a shift common to all streams is carried forward the way
a plain residual carries it, and every later pre-mix and the final contraction
see it.

`ATLAS_STEER_LAYER=L` (default `num_hidden_layers / 2` = 22 of 45) names the
layer whose output is steered. The site lives in the KDA layer code only: at the
end of layer L when L is a KDA layer (22 is), else at the start of KDA layer
L + 1, before its pre-mix reads the highway. Every path reaches one of those
two spots: prefill (incl. sequence-parallel prefill, where each rank steers its
own rows), one-row decode, batched multi-sequence decode, speculative verify
(K rows), the long-context owner verify and the fused prefill + verify chunk.
The launch is captured into the decode/verify CUDA graphs like any other
kernel.

### What

`living_site` with rank 0 (v term only), in place:

    x <- x + dose * v          v = Σ_i scale_i * rms_i * û_i    (F32, [hidden])

with `û` the file's vector scaled to unit RMS (norm `sqrt(hidden)`).

The dose is the F32 word `ctrl[2]` of the site's device control word; `ctrl[1]`
is the v switch. Dose 0 sets the switch off and the kernel returns without
touching the row (exact no-op). The kernel computes in F32 and rounds once to
BF16 (an F32 twin, `living_site_f32`, serves an F32 highway). The final site
sets `ctrl[2] = 1.0`, so its behaviour is unchanged.

### Units: the dose is in residual RMS

Decision (the spec allowed measuring at startup or storing): **stored**. Each
vector file carries `rms`, the mean per-token RMS of the highway after layer L,
measured once from `ATLAS_DUMP_LAYER_HIDDEN` rows of a neutral prompt and
written into the file's metadata by the tool that makes the vector. The engine
scales each file's `v` to unit RMS and multiplies by `rms` (and the file's
`scale`, default 1). So for one file, dose `d` adds `d * rms * û` to each of
the 4 streams: the added vector's RMS is `d` times the residual's RMS.

Decision (the spec said "unit vector"): **unit RMS, not unit norm.** With a
unit-norm `v̂` the added vector at d = 1 would be `rms` long against a row of
norm `rms * sqrt(4096) = 64 * rms`, a 1.6 % nudge, and a random direction at
d = ±1 would not change greedy outputs; the spec expects d = ±1 to change them
and soaks at d = 0.5, and the earlier llm_steer work used the same RMS-scaled
coefficients (±0.1-0.3 tilts coherently, ±0.6 breaks the output). To get the
unit-norm reading, divide the dose by `sqrt(hidden)` = 64. `ATLAS_STEER_RMS` is a fallback for
files without `rms`. Reasons: a startup measurement needs a forward pass before
the server is up (and on both ranks identically); a stored number is
reproducible, visible in `GET /v1/living/steer`, and the vector maker already
has the dumped rows.

Per-token RMS is taken over all `hc_mult * hidden` highway values of the token.
Measured on the served checkpoint (layer 22, RMS_PLACEHOLDER).

### Vector files (`ATLAS_STEER_DIR=DIR`)

    DIR/index.json            {"files": ["calm.safetensors", ...]}
    DIR/calm.safetensors      v [hidden] (F32/BF16/F16)
                              __metadata__: base="<revision>", layer="22",
                                            rms="<measured>", topic="calm", scale="1"

All listed files are summed. A file whose `base` is not the served revision, or
whose `layer` differs from `ATLAS_STEER_LAYER`, is refused and the server does
not start. Both ranks must load the same directory (SparkGLM mirrors
`STEER_DIR` to the worker). Log line: `steer: N vector(s) after layer L (end of
layer L), |v| at dose 1 = ..., dose d`. `ATLAS_STEER_SCALE` is the starting
dose (default 0 = off).

### Runtime dose: `/v1/living/steer`

    GET  /v1/living/steer      -> {"configured", "scale", "layer", "site", "rms",
                                   "v_norm_at_scale_1", "files", "dump"}
    POST /v1/living/steer      {"scale": d}                          (|d| <= 64)
                               {"scale": d, "flush_prefix_cache": true}

The POST travels the scheduler's LoRA-rotation channel and is applied at a
quiescent point (no active, prefilling, pending or swapped sequence); the
response comes after it is applied and returns the new status. Rank 0 sends
the worker a whole-worker command (`0xFFFFFFED`: dose bits, flush flag) and
both ranks write their own control word, so the startup-captured graphs keep
replaying unchanged. The dose is process state: a restart returns to
`ATLAS_STEER_SCALE`.

Prefix cache: cached KV/SSM state keeps the dose it was computed under. Without
a flush, a repeated prompt reuses its cached prefix (only the uncached tail and
the new tokens see the new dose); this is cheap and means "past tokens keep the
mood they were read in". `flush_prefix_cache: true` evicts every unreferenced
prefix block on both ranks so the next request recomputes under the new dose
(the full re-prefill costs time on long contexts). With the NVMe prefix tier
on, eviction spills rather than drops, so a flush is not a true flush there.

### Read-out (`ATLAS_DUMP_LAYER_HIDDEN=DIR`)

Same conventions as the final dump: rank 0 only, `DIR/hidden.f32` (F32,
row-major) + one `DIR/index.jsonl` line per call, calls made while a stream is
being captured skipped. Rows are the highway at the steering spot **before**
the steering add: `hc_mult` rows of `hidden` per token (token-major, stream
inside). Prefill calls dump their last token (for a chunked prompt, the last
token of each chunk; for a fused chunk carrying verify passengers, the last
passenger row); `ATLAS_DUMP_LAYER_HIDDEN_ROWS=all` dumps every prefill token.
Decode and verify calls dump every token (eager calls only). Index line:

    {"call", "layer", "kind" (prefill|decode|verify), "rows_in_call",
     "first_row", "tokens", "hc", "n_rows" (= tokens * hc), "hidden",
     "offset" (bytes), "sp" (sequence-parallel prefill), "dose"}

### Limits

- GLM-5 (mHC, KDA) only; other model types log a warning and skip the site.
- One process, one steering site (a process-wide singleton installed with the
  first GLM model built; a hot model swap keeps the first install).
- The MTP/DFlash drafters' own layers are not steered (drafts only; verify
  rows pass the site).

## Microtest

    cargo run --release -p spark-model --features gpu-examples \
        --example living_site_microtest -- [hidden=4096] [seed]

Checks rank-0 / control-word cases bit-exactly and random rank 1/16/512 cases
(with and without v, in and out of place) against an F32 CPU reference
(at most 2 BF16 ulps per element, cosine >= 0.99999), and the steering shape:
v only with the dose from the control word (BF16, in place), dose 0 bit-exact,
and the F32 twin exactly.
