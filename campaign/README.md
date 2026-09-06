# Time-bounded kernel campaign (experimental)

This supervised runner screens two distinct targets:

1. **Prefill/TTFT-related compute:** interleaved baseline/minblocks2 E2 direct-GEMM timings on actual TP-local dimensions, including shapes that previously regressed.
2. **Decode-related compute:** a new Triton kernel fusing sorted token-index packing, FP16 routing-weight packing and expert counts. Argsort is unchanged. It is tested on metadata alone and inside the existing fused MoE call using a reduced 16-expert synthetic layer with 4096/1024 dimensions.

Neither change is installed into the serving model. A real-service TTFT/decode **baseline** is collected separately, not used as a candidate comparison. TTFT includes frontend/queue work and metadata latency is only part of decode.

## Requirements and deadline

Use the existing configured head/worker containers and the compiled modules produced by the feasibility probe. The kernel module loader expects ignored `feasibility/build/baseline/` and `feasibility/build/minblocks2/` outputs. No machine-specific source checkout paths are required.

The controller requires `artifacts/timed-campaign/deadline.json` containing numeric `start_epoch` and `deadline_epoch`, and an initialized `reports/timed-30min.md` log. For the documented run, a separately tracked background timer created this manifest and kills **only** the named `glm53-timed-probe` container at the 30-minute deadline. The timer never stops the original serving containers.

`run.sh` reserves 11 minutes before the hard deadline for restoration, wraps the GPU experiment in a timeout, and checks the deadline before interruption. `probe.py` checks the same time and STOP marker between timing blocks. The shell restoration handler starts the existing worker then head, verifies a real response, and checks weight manifests.

This is not a general scheduling system. Do not launch it without an independent deadline timer or without enough service restart margin. Host/SSH failures can still prevent restoration. Restoring a failed service may continue after a deadline; further **optimization** may not.

## Experiment log

The probe writes each completed case to `reports/timed-30min.md` immediately and keeps detailed paired timing samples in ignored `artifacts/timed-campaign/kernel-results.json`. Controller state transitions also append to the Markdown file. Failed cases are recorded and trigger restoration; correctness gates are not weakened.

## Limitations

- Synthetic weights/activations; no teacher-logit quality panel or sanitizer.
- Reduced expert count for the full-MoE test; no TP2 candidate service validation.
- AB/BA paired timings help with ordering bias but do not fully control clocks, cache working set or dataset bias.
- The existing image sources and module ABI must match the prior compiled artifacts.
- The code is a trusted supervised prototype, not a sandbox for arbitrary agent-generated kernels.
