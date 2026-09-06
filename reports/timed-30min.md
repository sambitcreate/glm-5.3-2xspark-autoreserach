# 30-minute TTFT/decode kernel campaign

- Start: 2026-09-06 03:41:00 UTC
- Hard optimization deadline: 2026-09-06 04:11:00 UTC
- Budget: 30 minutes, including setup; reserve at least 10 minutes for restoration.
- Baseline: existing E2, TP2, DFlash2 k7, 250K context, MNBT7168, GPU_MEM_UTIL0.85.
- Storage: reuse existing model/draft caches; no weight downloads or copies.
- Goal: screen fat GEMM variants for prefill/TTFT and fused routing metadata for decode. Record real-service baseline separately; kernel timings are not end-to-end improvements.
- Policy: no production promotion in this short campaign. Stop experiments by the deadline even if a candidate is promising. Restore known-good containers.

## Results and decision

**Completed:** 14 paired prefill-direct cases, 7 routing-metadata cases, 4 graph-captured full-MoE cases, and an independent real-service baseline. The service was restored by 03:56:34 UTC, with the same model/draft snapshot paths and weight manifests. No candidate was installed into production. **The hard timer expired at 04:11 UTC and wrote the STOP marker. Optimization is closed.** A post-deadline check confirmed both original serving containers running and the head health endpoint successful. No further GPU experiments were launched.

### Real-service baseline (unchanged production kernels)

- Unique-salt prompt: 8,045 actual tokens; client TTFT **8.001 s**. Cache coldness was not independently gated with counter deltas, so this is not a qualified cold-prefill score.
- Structured decode: median **63.01 tok/s** over three 128-token requests.
- Prose decode: median **27.71 tok/s** over three 128-token requests.
- These are baseline observations, **not before/after improvements**. No candidate service benchmark was run.

### Prefill result

The `minblocks2` variant again improved large gate/up cases: paired ratios **1.30–1.38×** for M=1024/2048/7168 at K4096/N2048. But M512 regressed to **0.909×**. Down-projection cases also showed mixed results. A universal replacement is rejected.

A conservative shape-dispatch proposal was written in `campaign/prefill_dispatch.py`: use minblocks2 only for gate/up M≥1024 and down M≥2048. **This rule is fitted to the first sweep and has not been GPU-validated on holdout shapes or integrated into serving.** Its scheduled holdout was cancelled because setup left less than the required 90-second experiment window before the restoration reserve. Do not treat it as a promoted patch.

### Decode result

A new Triton kernel in `campaign/routing.py` fuses sorted token-index packing, FP16 routing-weight packing and expert-count accumulation; argsort is unchanged. Metadata outputs matched exactly, including the mapped sentinel count. Graph-captured metadata ratios were approximately **1.35–1.64×** on T=1/8/16/32.

Savings shrink substantially in actual MoE work. On a **synthetic 16-expert** layer with TP-local dimensions hidden4096/intermediate1024:

| Tokens | Baseline median ms | Candidate median ms | Median paired ratio |
|---:|---:|---:|---:|
| 1 | 0.34645 | 0.33639 | 1.0288× |
| 8 | 0.53747 | 0.52829 | 1.0180× |
| 16 | 0.56625 | 0.55564 | 1.0186× |
| 32 | 0.77840 | 0.78278 | 1.0138× |

The T32 ratio of marginal medians indicates a slight regression while the median of paired ratios indicates a slight improvement. This illustrates timing variability; **no statistically qualified decode gain is claimed**. Full-MoE numerical differences were at most 7.63e-6 in these cases, within the predeclared 1e-4/1e-5 tolerance. Graph parity passed. The synthetic 288-expert holdout was prepared but **not run**.

### Decision and next step

- Preserve the new routing kernel and conservative dispatch proposal as experimental source, not production changes.
- **No TTFT or decode service speedup demonstrated within this campaign.**
- Next session: fresh seeds and shape holdouts, full 288-expert replay, realistic routing distributions, sanitizers, then candidate image TP2 A/B. Do not extrapolate the metadata speedup to token throughput.
- Detailed AB/BA samples and the baseline observations are in `timed-30min-results.json`. GPU tests used 15 paired samples, alternating execution order. Synthetic working sets, one measured GPU, and no model-level candidate quality suite remain limitations.

Deadline receipt: the independently tracked 30-minute timer completed with exit code 0. Deadline verification and this final documentation update are operational closeout, not additional optimization.

## Live run log

| UTC | Experiment | Result / decision |
|---|---|---|
| 03:41:00 | Timer armed | Hard stop 04:11:00; only experiment container is terminated by timer |
| 03:42 | Plan | Recheck E2 minblocks2 with interleaved timing; test fused routing-pack/count kernel for graph-safe decode metadata |
| 03:47:01 | Controller | Real-service baseline started; no candidate deployed |
| 03:48:16 | Controller | Services stopped intact; paired kernel screening starts |
| 03:48:18 | prefill-direct-M129-K4096-N2048 | screen-only; no TTFT claim; paired synthetic ratio=1.1850x |
| 03:48:18 | prefill-direct-M145-K4096-N2048 | screen-only; no TTFT claim; paired synthetic ratio=1.1825x |
| 03:48:18 | prefill-direct-M256-K4096-N2048 | screen-only; no TTFT claim; paired synthetic ratio=1.1953x |
| 03:48:18 | prefill-direct-M512-K4096-N2048 | screen-only; no TTFT claim; paired synthetic ratio=0.9090x |
| 03:48:18 | prefill-direct-M1024-K4096-N2048 | screen-only; no TTFT claim; paired synthetic ratio=1.3845x |
| 03:48:19 | prefill-direct-M2048-K4096-N2048 | screen-only; no TTFT claim; paired synthetic ratio=1.3030x |
| 03:48:19 | prefill-direct-M7168-K4096-N2048 | screen-only; no TTFT claim; paired synthetic ratio=1.3318x |
| 03:48:19 | prefill-direct-M129-K1024-N4096 | screen-only; no TTFT claim; paired synthetic ratio=0.9780x |
| 03:48:19 | prefill-direct-M145-K1024-N4096 | screen-only; no TTFT claim; paired synthetic ratio=0.9550x |
| 03:48:19 | prefill-direct-M256-K1024-N4096 | screen-only; no TTFT claim; paired synthetic ratio=0.9285x |
| 03:48:19 | prefill-direct-M512-K1024-N4096 | screen-only; no TTFT claim; paired synthetic ratio=1.0720x |
| 03:48:19 | prefill-direct-M1024-K1024-N4096 | screen-only; no TTFT claim; paired synthetic ratio=0.9541x |
| 03:48:20 | prefill-direct-M2048-K1024-N4096 | screen-only; no TTFT claim; paired synthetic ratio=1.2047x |
| 03:48:20 | prefill-direct-M7168-K1024-N4096 | screen-only; no TTFT claim; paired synthetic ratio=1.2196x |
| 03:48:21 | decode-routing-T1 | screen-only; routing metadata, not end-to-end decode; paired synthetic ratio=1.5989x |
| 03:48:21 | decode-routing-T8 | screen-only; routing metadata, not end-to-end decode; paired synthetic ratio=1.5995x |
| 03:48:22 | decode-routing-T16 | screen-only; routing metadata, not end-to-end decode; paired synthetic ratio=1.6375x |
| 03:48:22 | decode-routing-T32 | screen-only; routing metadata, not end-to-end decode; paired synthetic ratio=1.3493x |
| 03:48:22 | decode-routing-T128 | screen-only; routing metadata, not end-to-end decode; paired synthetic ratio=1.4656x |
| 03:48:22 | decode-routing-T512 | screen-only; routing metadata, not end-to-end decode; paired synthetic ratio=1.3732x |
| 03:48:23 | decode-routing-T2048 | screen-only; routing metadata, not end-to-end decode; paired synthetic ratio=1.1452x |
| 03:48:28 | decode-full-moe-T1 | reduced-expert synthetic full MoE; no serving claim; paired synthetic ratio=1.0288x |
| 03:48:29 | decode-full-moe-T8 | reduced-expert synthetic full MoE; no serving claim; paired synthetic ratio=1.0180x |
| 03:48:29 | decode-full-moe-T16 | reduced-expert synthetic full MoE; no serving claim; paired synthetic ratio=1.0186x |
| 03:48:30 | decode-full-moe-T32 | reduced-expert synthetic full MoE; no serving claim; paired synthetic ratio=1.0138x |
| 03:48:30 | campaign-kernel-screen | No promotion; restore known-good service |
| 03:48:31 | Controller | Kernel screening passed; results remain experimental |
| 03:48:31 | Controller | GPU experiments ended; restoring approved baseline |
| 03:56:34 | Controller | RESTORED: model response OK; existing weight manifests unchanged |
| 03:56:34 | Controller | Campaign controller exits status 0; no automatic production promotion |
| 03:59:26 | Holdout scheduling | NOT RUN: under 90 seconds remained before the reserved GPU cutoff; cancelled without stopping service |
