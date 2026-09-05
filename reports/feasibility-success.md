# Feasibility passed: isolated E2 kernel experiment on GB10

## Conclusion

**The compile → GPU correctness → synthetic timing → service restoration workflow works on this two-Spark deployment.** This is sufficient to justify building the trusted autoresearch evaluator/controller in the PRD. It is **not** a completed autonomous researcher or a production kernel promotion.

The full supervised run completed with exit code 0 in approximately **11 minutes** (including stopping and restoring the model service). The original service is online with the owner-approved `GPU_MEM_UTIL=0.85`, context 250,000, MNBT=7168 and workspace rightsizing. No candidate was deployed into the serving image.

## Verified results

| Check | Result |
|---|---|
| Existing EXL3 GPU self-check | Passed independently on both GB10 nodes |
| Native standalone build | Baseline and candidate compiled/imported successfully for `sm_121a` |
| Compile-only prequalification | Both modules compiled without GPU access while inference stayed online; approximately 33/32 seconds in the first successful preflight |
| GPU probe build/load time | Approximately 22.6/22.4 seconds for baseline/candidate in the final run |
| Direct GEMM parity | 24 shapes per variant; measured max absolute difference versus installed E2 was **0.0** |
| Weighted scatter parity | Same 24 shapes per variant, including nonzero destination initialization; measured max absolute difference **0.0** |
| Incorrect-output negative control | Deliberately altered output rejected on every tested case |
| CUDA graph smoke | Direct-kernel capture/replay passed for each variant |
| Restoration | Both original containers restarted successfully; health 200 and model returned `OK` |
| Weight reuse | Same snapshot paths, config hashes, indexed shard sizes and symlink-target manifests before/after, on both nodes; no downloads or copies |

The comparison tolerance was fixed at `atol=1e-4`, `rtol=1e-5`; measured exact equality applies to these synthetic samples, not all possible inputs. The stock self-check supplies additional reconstruct/fused-path coverage with its own broader tolerances.

### Shapes and candidate

- M values: 1, 127, 128, 129, 145, 256, 512, 2048.
- K/N pairs: 256/256 (small fixture), 4096/2048 (TP2 combined gate/up geometry), 1024/4096 (TP2 down geometry).
- Candidate changes only `__launch_bounds__(FAT_THREADS)` to `__launch_bounds__(FAT_THREADS, 2)` in the extracted E2 CUDA source.
- Input tensors and packed weights are synthetic. No model checkpoint tensors were exported for this probe.

## Timing: a research lead, not a qualified speedup

The equal-weight geometric mean of `baseline median latency / candidate median latency` across 24 **direct-kernel** cases was **1.1264×**. This is not a routing-weighted or end-to-end service metric. No scatter timing or full MoE pipeline timing was collected.

Selected cases, milliseconds per direct call:

| M | K | N | Baseline | Candidate | Baseline/candidate |
|---:|---:|---:|---:|---:|---:|
| 145 | 4096 | 2048 | 0.1362 | 0.1148 | 1.186× |
| 512 | 4096 | 2048 | 0.1629 | 0.1831 | **0.890× (regression)** |
| 2048 | 4096 | 2048 | 0.7890 | 0.5566 | 1.418× |
| 129 | 1024 | 4096 | 0.0442 | 0.0469 | **0.942× (regression)** |
| 145 | 1024 | 4096 | 0.0453 | 0.0474 | **0.956× (regression)** |
| 2048 | 1024 | 4096 | 0.4404 | 0.3624 | 1.215× |

Baseline and candidate were timed sequentially, not randomized/interleaved. Each case used three warmup calls and seven samples of ten calls each, measured with CUDA events. Small M cases include shapes that normally follow the thin-expert path in serving. Cache residency, clocks, host enqueue gaps, and ordering can influence results. These numbers cannot establish a causal performance improvement or predict cold-prefill tok/s.

**Decision: feasibility only; no promotion.** Shape regressions alone warrant more work before considering a universal launch-bounds change.

## Build issues resolved

1. `cusparse.h` lives in the NVIDIA Python wheel, outside the toolkit include directory.
2. Adding the wheel directory with `-I` selected an incompatible wheel `crt/host_runtime.h` for nvcc's generated stub, causing a `__cudaLaunch` macro-arity error.
3. Repeating the toolkit directory with `-I` did not solve ordering because that directory was already a system include.
4. The working fix supplies wheel paths as **trailing `-isystem` includes**, retaining toolkit runtime headers first. No toolkit, driver, package or model upgrade was needed.

The failed attempts and approved memory-budget recovery remain documented in the earlier reports rather than erased.

## Remaining work before unattended autoresearch

1. Immutable fixtures and independent numerical oracle using realistic routing/activations, held-out shapes, whole-pipeline timing and sanitizer checks.
2. Paired randomized baseline/candidate measurements on both nodes with a frozen statistical gate.
3. Candidate image identity verification and real TP2 cold-prefill/decode/concurrency regression suite. Kernel-only success cannot substitute for this.
4. A constrained coding agent, durable experiment ledger, one-writer/cluster lease, resource budgets, external watchdog and tested infrastructure recovery.
5. Preflight memory headroom: the previous 0.86 profile failed restart admission; 0.85 restored successfully in this run. Do not assume a successful resident service can always restart under a tighter budget.

The existing scripted probe is supervised and trusted-code-only. Its restore handler survived the successful test but does not replace a host-independent watchdog.

## Reproduction

Use `feasibility/run.sh` from any clone location with your existing worker/container configuration as documented in the README. It reuses installed image source and existing weight mounts; it does not require another checkout via hardlinks or symlinks. See `feasibility-success.json` for curated synthetic timing data and source/binary hashes. Private host identifiers, snapshot mount paths, and operational logs are intentionally not published.
