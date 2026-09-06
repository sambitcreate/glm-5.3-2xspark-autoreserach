# 30-minute TTFT/decode kernel campaign

- Start: 2026-09-06 03:41:00 UTC
- Hard optimization deadline: 2026-09-06 04:11:00 UTC
- Budget: 30 minutes, including setup; reserve at least 10 minutes for restoration.
- Baseline: existing E2, TP2, DFlash2 k7, 250K context, MNBT7168, GPU_MEM_UTIL0.85.
- Storage: reuse existing model/draft caches; no weight downloads or copies.
- Goal: screen fat GEMM variants for prefill/TTFT and fused routing metadata for decode. Record real-service baseline separately; kernel timings are not end-to-end improvements.
- Policy: no production promotion in this short campaign. Stop experiments by the deadline even if a candidate is promising. Restore known-good containers.

## Live run log

| UTC | Experiment | Result / decision |
|---|---|---|
| 03:41:00 | Timer armed | Hard stop 04:11:00; only experiment container is terminated by timer |
| 03:42 | Plan | Recheck E2 minblocks2 with interleaved timing; test fused routing-pack/count kernel for graph-safe decode metadata |
| 03:47:01 | Controller | Real-service baseline started; no candidate deployed |
