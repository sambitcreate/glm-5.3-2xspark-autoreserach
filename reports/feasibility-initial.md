# Initial feasibility attempt

## Outcome: incomplete; do not treat as unattended-ready

The first supervised run passed the stock EXL3 GPU self-check on **both GB10 nodes**, but the new standalone extension did not compile. Service restoration then hit a worker memory-admission failure. No candidate kernel was installed into the original service and no performance improvement was measured.

### Verified

- Matching existing image IDs on both ranks, native `sm_121a` extension cubins.
- Existing main model snapshot: 120 indexed shards, 175,642,157,752 bytes.
- Existing DFlash2 snapshot: one shard, 2,342,169,800 bytes.
- Identical corresponding model/draft snapshot paths, config hashes and shard layout/size manifests on both nodes. Weight data was not copied or downloaded. Full shard-content hashing was not performed.
- Both stock GPU self-checks passed direct/scatter parity, fused-versus-loop checks, thin/fat composition, expert mapping and fused CUDA graph capture.
- The deployed profile had context 250,000, MNBT=7168, four sequences and workspace rightsizing. It was not the PRD's example 1M baseline.

### Failure 1: standalone build environment

`ATen/cuda/CUDAContextLight.h` could not find `cusparse.h`. The existing image supplies this header in the NVIDIA Python wheel rather than CUDA_HOME/include. The production Dockerfile accounts for this; the initial standalone probe did not.

The probe now discovers the wheel header directory and passes it to the extension build, with a pre-stop header check in the runner. This fix still needs a successful GPU compile/run receipt.

### Failure 2: original-service restoration

Restarting the unchanged original containers failed on the worker:

```text
Free memory on device cuda:0 (104.55/121.63 GiB) on startup is less than
 desired GPU memory utilization (0.86, 104.6 GiB).
```

This is an infrastructure/memory-admission failure, not a candidate correctness failure. The service had very little restart margin. No automatic lowering of memory utilization, context length or precision was performed. No unrelated host services were stopped.

The initial shell trap detected and reported restoration failure; it did not falsely report recovery. An unchanged-container retry also failed: the worker reported 104.14 GiB available versus 104.6 GiB requested. Docker reported both service containers stopped with exit code 1 and `OOMKilled=false`; the retry's checking command ended with exit code 137, which is not evidence by itself of an OOM kill.

The owner then approved reducing `GPU_MEM_UTIL` from 0.86 to 0.85 while retaining the same weights, image, 250K context, and other serving settings. **Recovery succeeded:** health passed after 480 seconds, the launcher's 20/20 warmup requests succeeded in 32 seconds, and a subsequent model request returned `OK` (18 prompt tokens, 2 completion tokens).

Post-recovery model/draft snapshot paths, config hashes and shard-layout/size manifests on both nodes matched their pre-experiment records. No weights were downloaded or copied. The new 0.85 setting is now the approved deployment baseline; any later benchmark must compare against that profile, not the former 0.86 profile.

A second bounded probe attempt was launched after adding the wheel include path and matching the pinned upstream compiler flags. Its outcome will be recorded separately. The successful service recovery does not itself prove the kernel probe works.

### Implications

- Stock GPU execution is feasible on both nodes.
- Autonomous kernel research is **not yet qualified**: complete the standalone build/timing receipt and resolve restart memory headroom first.
- Preflight must assess available UMA after resident services and vLLM initialization, not simply whether the model is currently running.
- A reliable external recovery/watchdog and repeatable startup profile are prerequisites for unattended runs.
- Head-only synthetic kernel timing, when available, will still not establish a two-rank serving speedup.
