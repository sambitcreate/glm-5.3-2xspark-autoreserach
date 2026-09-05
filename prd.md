# PRD: GLM-5.3 Flash kernel autoresearch on 2× DGX Spark

## 1. Decision and scope

Build an **autoresearch-style kernel optimization lab**, not a port of autoresearch's model-training workload. Keep the deployed model, quantization, and serving semantics fixed. Give a coding agent a narrow kernel edit surface, a trusted evaluator, a bounded experiment budget, and an automatic keep/discard loop.

Start with **cold-prefill performance of the existing E2 EXL3 fat-expert path**. Preserve decode performance as a hard guardrail. Open a separate decode optimization campaign only after fresh profiles show its bottlenecks. Success means a reproducible improvement on the actual two-Spark service, not merely a faster synthetic GEMM.

**This document specifies the full research system, not a claim that it is implemented.** A small supervised feasibility probe is provided in `feasibility/`; it is not the autonomous controller or promotion evaluator described below. See `reports/` for separately recorded execution evidence when available. The general baseline/profile requirements below must be qualified on each deployment.

### Inspected sources

- Serving repository: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
- Development source belongs in the independent `src/glm-exl3` clone; no link to another checkout is required.
- Inspected serving commit: `3021f24c88a0904c768c46ff22a508407e31360a`
- Inspiration: https://github.com/karpathy/autoresearch
- Inspected autoresearch commit: `228791fb499afffb54b46200aca536f79142f117`
- Reference autoresearch checkout may be cloned into any temporary directory; only its pinned commit is required for reproduction.
- Autoresearch's `README.md` and `program.md` define a fixed evaluator, narrow editable workload, baseline-first experiments, and keep/discard records. Its five-minute training budget and `val_bpb` metric do **not** transfer directly to distributed inference.

## 2. What this repository actually does

| Component | Current implementation / research consequence |
|---|---|
| Hardware | Two GB10 DGX Sparks, one GPU per node, TP=2 over CX7/RoCE. Use native ARM64 CUDA 13.0 and `TORCH_CUDA_ARCH_LIST=12.1a`; do not assume H100/SM100 kernels work on SM121. |
| Weights | EXL3/TR3 K4 routed experts, approximately 164 GiB total checkpoint, shared across TP ranks. Keep checkpoints and revisions fixed. |
| Runtime | vLLM multiprocessing distributed executor, head API on port 8888 by default. |
| Kernel dependency | Dockerfile pins ExLlamaV3 `c5d9c657966ffeeaa9353f0cc899f18629da4a13`, patches ARM64 support, and adds fat GEMM symbols to `exllamav3_ext`. |
| Thin experts / decode | `overlay/exl3.py::apply_exl3_fused_moe` calls fused `exl3_moe`; the `tokens <= cap` path avoids host count synchronization and supports CUDA graphs. Default temp-row cap is 128. |
| Fat experts / prefill | E2 `apply_exl3_batched_fat` gathers rows, transforms inputs, packs gate/up weights into scratch, executes direct trellis GEMM, applies activation, transforms again, and executes weighted down/scatter. |
| E2 CUDA | `overlay/exl3_fat_gemm.cu`: 256 threads, M/N tiles 128, K tile 16, K4 MCG dequantization, FP32 accumulation/output, Hadamard/scaling epilogue. Supports direct and scatter exports. |
| Remaining host work | Prefill count copy is overlapped with thin experts but still calls `count_stream.synchronize()` followed by a Python expert loop. |
| Attention / KV | NoPE sparse MLA adapted to SM120-family sparse kernels; target packed `fp8_ds_mla`, draft BF16. Existing sparse-candidate selection and tail behavior must be preserved, not silently redesigned. |
| Speculation | DFlash2 k=7, draft TP=2, FLASH_ATTN. Structured high-accept decode and ordinary prose have very different speeds. |
| Build | CUDA source changes invalidate the extension compile layer. Python overlay is copied after compilation. Launcher recipe stamps can trigger rebuilds; a successful restart alone does not prove which binary ran. |

### Existing evidence, not new measurements

The current README reports PR77 E2 versus legacy at MNBT=2048: roughly **+20–21% cold prefill** at 8K/100K/300K, with E2 pooled means approximately 1132/1242/1201 tok/s. That gain already exists; it is not this project's target improvement.

README decode examples include about 65 tok/s structured and 27 tok/s prose in a particular five-run TP=2 lab protocol. Treat these as historical context, not acceptance thresholds for another kit.

`docs/improve-prefill.md` contains valuable earlier profiles and negative results:

- Pre-E2 profiling attributed about 63% inclusive CUDA time to MoE; KDA was around 6%, sparse MLA around 4%, and NCCL around 7%. These are historical, partly inclusive categories, not additive current bottleneck percentages.
- Full-grid 128-row tiling and increasing fused temp rows to 1024 were tested and reverted because they lost performance.
- Older MNBT=3584/4096 results also lost the overall tradeoff. They predate E2.
- Current README/default launcher uses **MNBT=7168 and MAX_NUM_SEQS=4**, whereas the clean PR77 comparison used 2048. Do not copy the older document's claim that 2048 is still the production default.
- Historical notes warn about MNBT=8192 shared-memory failures. Do not put 8192 into unattended search until explicitly requalified on the pinned stack.
- Documentation differs in layer accounting and pool receipts. Derive actual layer types, dimensions, sharding, and KV capacity from the pinned checkpoint and live boot; do not hardcode estimates from prose.

## 3. Objectives and non-goals

### Objectives

1. Reproduce a stable current-E2 baseline on both Sparks.
2. Build a trustworthy, reusable fast evaluator for EXL3 direct/scatter and complete MoE paths.
3. Automate bounded hypothesis → patch → compile → correctness → timing → retain/reject cycles.
4. Validate finalists in the exact distributed service with cold-cache, decode, quality, memory, and stability checks.
5. Produce reviewable patches and complete experiment receipts, including failures.

### Proposed success criteria

These are initial policy defaults, to freeze after baseline noise calibration:

- A promoted patch improves the end-to-end cold-prefill geometric-mean score by **at least 5%**, with a paired 95% confidence interval lower bound above 1.0.
- No required cold rung regresses by more than 3%; no required structured/prose/concurrency decode metric regresses by more than 3%, assessed with adequate repeats rather than a single observation.
- Numerical, functional, graph, and distributed checks pass. No changed weights, expert choices, activation clipping, precision policy, attention semantics, or context limit to manufacture speed.
- No more than 2% loss in usable KV token capacity relative to the same-profile baseline; enough capacity remains to honor the declared context service. Memory-neutral or memory-saving patches are preferred.
- At least one independently rebooted confirmation sequence and a one-hour mixed-workload soak before human promotion.
- If no candidate wins, a reliable evaluator and documented negative results still count as useful research output, but not as a performance success.

### Non-goals for the first campaign

Training or requantizing GLM; changing DFlash weights; ablation/abliteration; NVFP4/bf16 target KV; changing TP or interconnect topology; speculative-token tuning; scheduler-policy tuning; API feature changes; broad dependency upgrades; autonomous production deployment. Keep `ABLIT=0`.

## 4. Architecture: two loops, one trusted controller

### Mapping from autoresearch

| Autoresearch | Spark kernel lab |
|---|---|
| `prepare.py` fixed data/evaluation | Maintainer-owned fixture capture, reference implementation, shape suite, scoring and deployment controller |
| `train.py` mutable workload | Initially only fat-kernel CUDA/header; later a separately authorized EXL3 prefill Python slice |
| `program.md` instructions | Human-owned kernel campaign rules, hypotheses, boundaries, stop conditions |
| `val_bpb` | Kernel latency screening, then cold-prefill speedup subject to hard correctness/decode/memory gates |
| Five-minute training | Fixed work per timing sample; separate compile, correctness, timing, and full-service deadlines |
| Keep/discard log | Append-only JSONL + TSV index, immutable candidate commits and artifacts |

Do not import or run upstream `train.py`, download its training dataset, or install its H100-oriented environment. Borrow the workflow design only. Do not adopt upstream suggestions to disable all permissions or loop indefinitely.

### Inner loop: inexpensive kernel screening

- Stop the full model service during dedicated kernel microbench windows; do not benchmark beside it on UMA.
- Compile candidates against the pinned CUDA/PyTorch/ExLlama headers in a development container with persistent build cache.
- Prefer an incremental build of the patched extension or a distinctly named test module with matching ABI. Confirm linking/export behavior before trusting this shortcut.
- Use a fresh evaluator process for each binary; do not hot-swap a loaded CUDA shared object.
- Run correctness before any performance scoring, then CUDA-event timing and full-call wall time on fixed inputs.
- Screen on one Spark initially; reproduce finalists on the other Spark. Never average away a severe regression on one node.
- Kernel-only winners become `shortlist`, not `keep_production`.

### Outer loop: expensive two-node confirmation

- Bake a unique candidate image via the actual deployment recipe.
- Ship the **same built image** to both nodes. Verify image ID, extension hash, overlay hash, and configuration independently on each rank.
- Restart both ranks, collect boot diagnostics, complete identical warmup, establish quiet service conditions, and evaluate.
- Kernel/module caches are versioned by source/compiler/config hash. Warm each candidate equivalently; initialization is separately measured, not hidden inside steady-state latency.
- Keep one controller holding an exclusive cluster lock for the entire deployment and measurement block.
- A remote coding model or separate controller host is recommended: do not run the agent's inference on the two GPUs it is measuring or on a service it must restart.

### Proposed project layout (to implement)

```text
<project-root>/
  prd.md
  program.md                    # owner-controlled agent policy
  config/cluster.env.example    # topology placeholders; secrets excluded
  config/campaign.yaml          # profiles, budgets, immutable gate thresholds
  locks/source-lock.json        # commits, image digests, compiler/model hashes
  src/glm-exl3/                 # isolated development clone, never user's live checkout
  harness/prepare.py            # fixtures + reference capture; trusted
  harness/check_correctness.py  # trusted GPU and shape checks
  harness/bench_kernel.py       # fixed-work timings, direct/scatter/MoE
  harness/bench_service.py      # hardened cold/decode/concurrency adapter
  harness/score.py              # fixed metric and confidence calculations
  runner/controller.py         # locks, deadlines, state machine, rollback
  runner/build.sh
  runner/deploy.sh
  tests/                       # evaluator/controller tests, not agent-editable
  fixtures/manifest.json       # hashes, dimensions, seeds, corpus licenses
  artifacts/<campaign>/<id>/   # logs, binaries/manifests, traces, raw samples
  results.jsonl
  results.tsv
```

All names other than this PRD are proposed, not existing commands/files. Large tensors, images, weights, credentials, and profiler dumps stay out of Git. Fixture and output access must respect checkpoint/data licenses.

## 5. Setup and baseline procedure

### Step A — create a separate development checkout

The destination spelling `autoreserach` is intentional, matching the requested path.

```bash
# From this repository's root, wherever it was cloned:
mkdir -p src
# Run once, after confirming src/glm-exl3 does not already exist:
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks src/glm-exl3
git -C src/glm-exl3 checkout -b research/e2-baseline 3021f24c88a0904c768c46ff22a508407e31360a
cp src/glm-exl3/.env.example src/glm-exl3/.env
```

Do not initialize/reset the existing working repository, move its branches, or overwrite an existing `.env`. Pin future upstream refreshes as separate baseline versions.

### Step B — inventory both Sparks, without changing services

Record GPU/driver/CUDA versions, architecture, power/thermal state, free UMA, disk space, Docker access, passwordless SSH, NIC/HCA/GID mapping per node, and NCCL connectivity. Check actual hostnames, IPs and SSH accounts rather than assuming the serving README's example topology. Record resident processes and obtain a dedicated maintenance window before stopping any serving containers.

Verify cached model and draft snapshots on **both** nodes. Record explicit immutable resolved revisions and file hashes. Prevent silent mirror/fallback substitution unless verified byte-identical. Preserve the HF caches; do not download another full copy merely to create the lab.

#### Mandatory: reuse the Mia deployment's exact weight locations

Source of truth: `start.sh` cache definitions, `resolve_model_dir`, `resolve_dflash_dir`, and Docker bind mounts—not a new project-local model directory.

| Location | Mia launcher contract to retain |
|---|---|
| Head host HF root | `${HF_HOME:-$HOME/.cache/huggingface}` using the **existing serving account and effective HF_HOME**, not the controller account's home |
| Worker host HF root | `${WORKER_HOME}/.cache/huggingface`; the current launcher derives this separately and does not use head `HF_HOME` for the worker |
| Container HF root, both ranks | `/root/.cache/huggingface`, bind-mounted from the corresponding existing host root |
| Main model, default mirror | `hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/<existing-resolved-revision>` under each HF root |
| Existing upstream fallback, if actually in use | `hub/models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/snapshots/<existing-resolved-revision>`; reuse it rather than downloading the mirror merely to match its name |
| Draft | `hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/<existing-resolved-revision>` under each HF root |

Implementation requirements:

1. Read the effective original deployment's model/draft load paths and the two container mount sources before creating the research environment. If no containers are running, resolve the approved serving configuration and existing cache directories. Record the absolute paths and snapshot IDs in the source lock. Do not print secret environment values.
2. Configure research `HF_HOME`, `WORKER_HOME`, `MODEL` and `DFLASH_MODEL` to preserve those existing locations and selected repositories. If the deployment has custom mounts beyond this launcher's defaults, reuse the discovered mounts explicitly rather than inventing a worker `HF_HOME` override that this script does not support.
3. Always use `SKIP_DOWNLOAD=1 SKIP_SYNC=1` during ordinary research runs **after validation**. Missing/incomplete weights are an external setup blocker; no automatic `download.sh`, refresh, rsync, mirror substitution, or new weight directory. A user-approved repair can be a separate setup operation.
4. Verify the selected snapshot's index, all referenced shards and metadata, and all snapshot-to-blob symlink targets on each node. A shard count across all snapshots is insufficient. Keep the existing HF `hub/blobs` layout intact; bind the existing cache root rather than copying a snapshot whose relative symlinks would break.
5. Never copy weights into `src/`, `fixtures/`, an image build context, or candidate containers. Replay fixtures are a small, bounded separately approved subset, not another full checkpoint. No cache pruning, requantization, relocation or mutation of weights.
6. **Pinning caveat:** the inspected launcher resolves serving paths through `refs/main`; merely setting `MODEL_REVISION` is not proof that the loaded snapshot is pinned. The controller must validate resolved refs against the lock before every boot and fail on drift. A future trusted explicit-snapshot launcher adapter may select the locked existing paths directly; it must not rewrite shared `refs/main` to pin an experiment. Require existing refs to be present so the launcher's automatic ref-repair path is not invoked.
7. Prefer read-only HF mounts in the trusted research deployment adapter after confirming offline startup works. This requires an explicit adapter change: current Mia mounts are read-write. Put writable JIT/compilation artifacts outside HF weights. Until then enforce no-write policy through controller checks and verify weight manifests after runs.
8. Isolate `CACHE_ROOT`, `WORKER_VLLM_CACHE`, Triton/TileLang and compiler caches for experiments, **not** `HF_HOME` or model storage. Reusing the same disk weights does not permit two full servers to fit simultaneously in UMA.

Acceptance test: baseline and candidate boot records show identical resolved model/draft snapshot paths and existing host mount sources on each corresponding node, zero download/sync operations, unchanged weight manifests, and no duplicate checkpoint tree under this project. Actual deployed paths have not yet been inspected; the implementation must discover and verify them before execution.

The existing launcher defaults to production-like container names and port. Set distinct `CONTAINER_HEAD`, `CONTAINER_WORKER`, port, and research cache roots; propagate those values to every start/stop call. A different port/name does not permit concurrent full-model instances on memory-constrained Sparks.

### Step C — freeze serving profiles

Initial primary profile:

```text
TP=2; NNODES=2
EXL3_FUSED_MOE=1; EXL3_FAT_KERNEL=1
EXL3_MOE_ROW_TILE=0; EXL3_TEMP_ROWS_FUSED=128
MAX_NUM_BATCHED_TOKENS=7168; MAX_NUM_SEQS=4
MAX_MODEL_LEN=1000000; GPU_MEM_UTIL=0.87
KV_CACHE_DTYPE=fp8; ENFORCE_EAGER=0
SPEC_METHOD=dflash; DFLASH_TOKENS=7; DFLASH_DRAFT_TP=2
GLM53_MIXED_PREFILL_CHUNK=skip
GLM53_INDEXER_WORKSPACE=stock; GLM53_SPINWAIT_MS=stock
LANGUAGE_MODEL_ONLY=0; SKIP_MM_PROFILING=1; ABLIT=0
```

Confirm every non-launcher knob actually reaches both container environments. Derive the effective settings from argv, env, boot logs, and kernel diagnostics, not from `.env` alone.

If the actual kit requires rightsizing or another memory budget, establish it as an explicit alternative baseline before search. Do not silently lower context or change reservations for a candidate. Use an additional **MNBT=2048** profile for PR77-comparable validation. Never pool scores across these profiles or compare a 2048 candidate to a 7168 baseline.

Build and start the baseline once using the existing launcher's supported local-image workflow, only during the approved maintenance window. For example, after configuring the research names/paths/topology and verifying both model caches:

```bash
# From this repository's root:
cd src/glm-exl3
IMAGE=glm53-autoresearch:baseline BUILD=1 SKIP_PULL=1 \
  SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart
```

Do not set `SKIP_BUILD=1` or `SKIP_OVERLAY_VERIFY=1` for changed kernels. Avoid `SKIP_SHIP=1` unless the controller has already verified exact worker image identity. Snapshot the resolved image and source manifests before continuing.

### Step D — baseline and new profiling

1. Run the repository's CPU/static checks and GPU self-check in their intended environments. GPU self-check execution is `python3 /opt/glm53/test_exl3_overlay.py` in the candidate image with GPU enabled; inspect its reported tests/skips.
2. Capture E2 diagnostic counters showing effective kernel tier, direct/scatter calls, fallback reasons, and scratch usage. Fail if a fat-path campaign never exercises fat kernels.
3. Run unprofiled repeated baseline measurements, including A/A restarts to estimate deployment noise.
4. Separately profile 16K/100K cold prefill and structured/prose decode on both ranks using supported Nsight Systems or torch-profiler tooling. Attribute GPU kernels, synchronization, CPU launches, and NCCL critical path. Do not score profiled runs.
5. Capture routed-row distributions and actual TP-local tensor shapes from representative layers. Freeze a compact replay suite; use both synthetic edge cases and realistic expert tensors/activations.

## 6. Trustworthy evaluation

### Existing tools to reuse, with limitations

- `tests/test_exl3_overlay.py`: direct/scatter parity, fused-vs-loop, thin/fat mixing, expert mapping, graph checks and diagnostics. Its direct test uses 145 rows and 256×256 dimensions with a broad max-error bound. It can skip E2 tests if symbols are missing. This is a smoke test, not a complete optimization gate.
- `tests/bench_decode.py`: useful five-run structured/prose baseline protocol. Hardcodes localhost:8888/model and lacks Bearer configuration. It is not a concurrency/held-out quality suite.
- `tests/bench_prefix_cache.py`: existing APC cold/warm behavior reference.
- `tests/_run_cold_prefill.py` and `docs/_run_cold_prefill.py`: historical ladder implementations; the inspected test copy hardcodes a maintainer-specific absolute output path and API endpoint. Its metric parser overwrites labeled series, missing counters default to zero, and streaming `read(4096)` needs a buffering audit. Do not use it unmodified as an unattended promotion oracle.
- Existing scheduler, kpool, prefix-hit, indexer-workspace, and template tests supply important regression contracts.

Before search, implement and freeze hardened adapters: configurable URL/model/output/auth, strict SSE parsing with first-content timestamp, explicit finish/error handling, label-aware metrics, missing-counter rejection, process-restart detection, and exact workload verification. Test SSE timing with a local paced fixture; compare client TTFT against server metrics. A substring containing `OK` and absence of the word `nan` are not numerical correctness proofs.

### Kernel correctness matrix

- Derive K/N from actual gate/up/down TP shards. Include the existing small test and full-sized representative shapes.
- Rows: 1, 7, 16, 32, 64, 127, 128, 129, 145, 255, 256, 257, 512, 1024, 2048, 7168, plus observed hot-expert counts. Include zero-route/empty cases at the dispatch layer; do not send an unsupported zero-grid launch blindly.
- Include uniform and highly skewed routing, thin+fat mixtures, expert-map remapping, shared and nonshared gate/up `suh`, partial M tiles, and invalid-input rejection.
- Compare against frozen current E2 **and** independent reconstruct/Hadamard/reference paths on identical inputs. Measure max absolute error, relative L2, percentile error, and finiteness.
- Capture baseline error distributions, then have the owner freeze per-operation tolerances before the agent starts. Do not let the agent loosen tolerance to pass a candidate. Precision policy stays unchanged.
- Check weighted scatter into nonzero outputs, repeated tokens across different experts, untouched output rows, and zero routing weights. Current non-atomic scatter is safe only under its serialized-expert/unique-token-per-expert contract. Concurrent expert execution needs a proven race-free reduction, not just removal of synchronization.
- Test graph capture/replay on relevant decode shapes, non-default stream behavior, scratch reuse, repeated calls, and both devices. Preserve fast-path/no-host-sync behavior.
- Run Compute Sanitizer memcheck and synccheck on finalists and racecheck where supported. Retain logs and distinguish unsupported tooling from passing checks.

### Microbenchmark protocol

Inputs and fixtures are prepared outside timed regions. Warm up deterministically; use CUDA events with explicit completion for GPU time and a separate wall-time metric covering host dispatch. Repeat enough iterations to exceed timer noise; report raw repetitions, median, dispersion and confidence intervals. Reset scatter outputs correctly between iterations so repeated accumulation cannot overflow or change the work.

Time direct GEMM, scatter, and the **entire fat-expert pipeline**, not just the fastest component. Include both warm repeated-weight and rotating-expert working sets; a tiny cache-resident tensor is not representative of streaming many experts. Score measured shape-distribution-weighted latency, with held-out shapes reserved for finalist checks. Additive build/compilation costs are recorded separately.

### End-to-end matrix

| Suite | Required work |
|---|---|
| Cold prefill | 8K, 16K, 100K and 300K actual prompt tokens; early screening may use 8K/16K only. Unique salt at the beginning, temperature 0, thinking off, 8 output tokens. Five valid samples per required rung per boot for promotion. |
| Decode | Structured count and prose, five runs × 400 requested tokens, actual usage recorded. Add held-out code/prose prompts so high DFlash acceptance alone cannot win. |
| Concurrency | 1/2/4 independent streams; per-stream and aggregate tok/s, TTFT and tail latency. Existing decode script needs a trusted concurrency wrapper. |
| Long context | Decode after 60–100K history, 256K/300K requests, and a dedicated near-declared-context capacity smoke test before final promotion. |
| APC and isolation | Cold/warm pairs with cache-hit counters, context separation, multi-turn behavior, and mixed prefill/decode scheduler guardrails. Derive alignment from live geometry rather than hardcode 3584 globally. |
| Quality | Frozen numerical/logit probes where available, deterministic functional/code checks, coherent long generations and tool/reasoning smoke tests. Published teacher KLD is for weights, not validation of these kernels. |
| Stability | One-hour mixed soak, generations beyond 2K tokens to exercise circular kpool state, no rank failures/CUDA faults/NaNs, no accumulating scratch growth. |

A cold sample is valid only if HTTP/stream completes successfully, usage is present, prompt length is within its declared tolerance, no other traffic is present, relevant cache-hit deltas are zero, and local-compute counters agree with prompt work under the pinned metrics semantics. Missing/reset/ambiguous metrics mean `invalid`, never zero hits by default. Salt alone is not proof of coldness. Use server timing corroboration; `prompt_tokens / client_TTFT` includes frontend overhead and is not pure kernel throughput.

### Promotion score and noise control

For each fixed primary-profile rung i:

`r_i = median(candidate prompt_tokens / TTFT) / median(baseline prompt_tokens / TTFT)`

`S_prefill = exp(mean(log(r_i)))` over 8K, 16K, 100K, 300K.

Use baseline → candidate → baseline → candidate boot blocks with equivalent warmup and matched corpus strata, but fresh salts every cold request. Bootstrap by paired workload/boot blocks rather than treating adjacent timing iterations as independent machines. If noise prevents a confident decision, extend the predeclared budget or label `inconclusive`; do not pick the best single sample.

Check decode, per-rung regressions, quality and memory gates independently before assigning a winning score. A missing/failed gate is not a speedup. For accepted decode rates track overall and per-position DFlash acceptance, proposed/accepted token counts, and output lengths. A changed acceptance rate is a diagnostic requiring investigation.

## 7. Prioritized experiment backlog

Ranking below is provisional until the fresh E2 profile; gain claims are hypotheses.

| Priority | Hypothesis / patch surface | Proof required |
|---|---|---|
| P0 | Build manifests, faithful evaluator, A/A baseline, E2 profile and replay fixtures | Reject intentionally wrong-output, missing-symbol, cached-prompt, stale-worker and timeout cases. No optimization before this passes. |
| P1a | Specialize E2 fat GEMM for actual row regimes: M-tile variants and occupancy/register-pressure tuning | `exl3_fat_gemm.cu/.cuh`; preserve 128-wide Hadamard grouping and legal loads. Tile constants are coupled to warp mapping, so do not blindly sweep constants as independent knobs. |
| P1b | Pipeline packed-weight/activation loads, reduce synchronization or shared-memory stalls | Compiler resource reports and profiler evidence; prove barriers and tails correct. Gate async-copy techniques on SM121 support; no assumed Hopper features. |
| P1c | Fuse activation/clipping/cast or gather+input-Hadamard where measured launch/traffic cost is material | Preserve exact clamp asymmetry and transform/scaling order. Measure whole pipeline, not only the fused elementwise stage. Requires explicit Python/extension allowlist expansion. |
| P1d | Avoid repeated gate/up pack copies without duplicating the entire expert checkpoint | Current code copies trellis and `svh` every fat expert call. Investigate two-source kernel addressing or bounded caching. Reject full-model duplicate packed/dequantized weight caches; quantify UMA and KV effect. |
| P1e | Allocate E2-only scratch lazily | `_fat_scratch` currently allocates reconstruction `w13/w2/down` buffers even for direct/scatter E2. Preserve legacy fallback and concurrent/stream lifetime contracts; classify this as a memory win unless timing proves otherwise. |
| P2 | GPU-driven fat routing/dispatch to remove remaining host synchronization | Only if new profile warrants it. Preserve thin-path graph behavior, expert-map semantics, scratch safety, and race-free scatter. More complex than row tiling; do not repeat the failed full-grid design unchanged. |
| P3 | Decode-specific fused MoE improvements | Separate campaign against pinned upstream ExLlama kernel sources, which are not vendored here. Vendor/patch exact source and establish low-M/graph fixtures before editing. E2-only work is not a claim of faster decode. |
| P4 | KDA/indexer/attention kernels | Only if current profiles show material critical-path cost. Their implementations live in dependencies; require separately pinned sources and stronger sequence/state correctness fixtures. |

Separate controlled configuration studies (MNBT=2048 vs 7168, stock vs rightsize) can establish a better deployment baseline, but do not credit them as kernel speedups or vary them inside a kernel A/B.

## 8. Agent contract and controller state machine

Initial editable allowlist:

```text
src/glm-exl3/overlay/exl3_fat_gemm.cu
src/glm-exl3/overlay/exl3_fat_gemm.cuh
```

The human can open a subsequent campaign allowing a specific prefill-only slice of `overlay/exl3.py` plus required extension binding patch. Tests, reference outputs, fixture manifests, score logic, budgets, dependencies, checkpoint files, credentials and cluster topology remain immutable to the mutation agent. New candidate tests are welcome but do not replace trusted tests.

Enforce the allowlist and evaluator hashes in the controller, not only in prompt prose. Candidate CUDA/Python is executable code: run it under a low-privilege evaluator identity with minimal mounts, no SSH keys, no Docker socket, no controller secrets, and no unnecessary network access. The privileged deployment controller alone handles Docker/SSH and gathers timing. Container isolation still shares the GPU/driver; treat candidate execution as a dedicated-lab workload, not a safe multi-tenant sandbox.

```text
lock cluster and verify clean isolated workspace
verify source/config/evaluator manifests
measure or load a valid same-manifest baseline
repeat until budget/deadline/stop condition:
    propose one hypothesis and expected mechanism
    create uniquely identified candidate branch/commit
    validate editable diff and unchanged evaluator hashes
    build -> correctness -> fixed-work microbench
    record crash/invalid/discard/inconclusive/shortlist
    for selected finalists:
        deploy exact image to both ranks
        validate boot/identity -> service A/B -> stability gates
        record keep_candidate only when every gate passes
    retain candidate commit and raw receipts
    restore best-known-good service when necessary
restore approved baseline unless owner explicitly authorizes promotion
release lock and write summary
```

One writer per worktree; at most one GPU experiment/deployment block across the pair at a time. Read-only idea generation can be parallel, GPU timings cannot. Preserve failed commits by reference; do not use destructive resets on the user's checkout or force-push upstream.

### Budgets and failure handling

Initial configurable limits (calibrate after the first build/boot):

- First supervised session: 3–5 simple candidates plus a deliberate rejected candidate.
- Overnight inner campaign: at most 8 hours, 30 candidates, and a separately configured agent token/cost ceiling.
- Candidate compile deadline: initially 30 minutes; baseline image provisioning gets a separately approved longer window.
- Microbench measurement target: 3–5 minutes; separate correctness/sanitizer timeouts. Fixed work is the comparable metric, not how much arbitrary work fits before a timeout.
- Full boot readiness: retain the launcher's documented 3600-second allowance initially; use a separate outer-block deadline, initially 2 hours and calibrate upward before long-context qualification if needed.
- Start with at most two outer-loop finalists per overnight window. Do not promise upstream autoresearch's roughly 12 experiments/hour; CUDA builds and 164 GiB model loading dominate here.
- Stop at deadline, repeated infrastructure failures (3), unavailable worker, baseline drift, thermal instability, numerical failure in the best-known-good reference, or failed rollback. A single candidate correctness failure is discarded; it is not permission to weaken the gate.
- Distinguish candidate OOM/crash from SSH/NCCL/image/tooling failures. On GPU faults kill the experiment's process group/containers on both ranks and recover through the controller. Do not automatically reboot hosts, modify drivers or prune global caches.
- On controller interruption, an external watchdog restores the known-good **two-rank** image/config. Reconcile existing containers/locks before retrying; never infer success from an orphaned results file.

### Suggested `program.md` content

> Optimize the approved EXL3 fat-kernel surface for the pinned two-Spark profile. Read the PRD, current kernel, immutable evaluator contract, latest profile and experiment ledger. Start from the current accepted source. Make one explained change, commit it, and request controller evaluation. Do not alter scoring, tests, precision, weights, serving settings or topology. A microbenchmark win is only a shortlist. Record all failures and respect controller deadlines. Never deploy to production or weaken correctness to improve latency. Prefer simpler patches at equivalent performance. Stop and report infrastructure blockers rather than changing execution protocol.

## 9. Results and reproducibility

The controller writes append-only records outside candidate branches. Use `null` for unavailable measurements, never zero as a fake winning latency. Store a schema version and a cryptographic manifest.

Minimum JSONL fields:

```text
campaign_id, experiment_id, hypothesis, parent_commit, candidate_commit,
source_diff_hash, evaluator_hash, fixture_hash, profile_hash,
base_image_digest, candidate_image_id, per_rank_binary_hashes,
model_revision, draft_revision, driver/compiler/package_versions,
node_inventory, boot_ids, timestamps, build_seconds,
correctness_status, numerical_error_summary, sanitizer_status,
raw_kernel_samples, kernel_summary, raw_service_samples,
prefill_score, confidence_interval, per_rung_ratios,
decode_metrics, acceptance_metrics, memory_peaks, kv_capacity,
status, failure_class, decision_reason, artifact_paths
```

TSV is a compact index: `experiment_id`, `commit`, `profile`, `kernel_speedup`, `prefill_score`, `decode_min_ratio`, `kv_tokens`, `status`, `description`. Preserve full raw outputs, not just the agent's summary. Redact secrets from commands/env/logs. Store public or approved synthetic prompts; never upload captured private prompts or checkpoint tensors without authorization.

A promotion bundle contains the patch, exact build/deployment instructions, source/image manifests, all successful and rejected samples, a before/after report, known limitations and a tested rollback command/config. Human review merges upstream or enables production; the unattended runner cannot.

## 10. Implementation milestones and completion gates

1. **Foundation / inventory (approximately 1 day):** create isolated checkout, freeze manifests, inventory both nodes, obtain maintenance access, verify reuse of the Mia deployment's existing model/draft paths without downloads or duplication, and reproduce current E2. Gate: healthy two-rank baseline with recorded effective configuration, matching weight-path manifests and verified binary identity.
2. **Evaluator and fixtures (approximately 2–3 days):** port/harden benchmark adapters; direct/scatter/full-MoE fixture matrix; frozen numerical thresholds; scoring and failure tests. Gate: A/A stability plus deliberate bad candidates rejected, with no missing checks treated as passes.
3. **Build/controller loop (approximately 1–2 days):** incremental native build, unique image shipping, exclusive lock, state machine, budgets, append-only artifacts and watchdog rollback. Gate: successful supervised candidate cycle and injected worker/timeout recovery.
4. **First optimization campaign (approximately 2–4 days):** reprofile E2, run P1 hypotheses in small isolated changes, shortlist with repeated timings on both devices. Gate: either qualified finalists or a clear negative-results report.
5. **Distributed qualification (approximately 1–2 days plus actual run time):** required A/B boots, both MNBT profiles reported separately, decode/concurrency/APC/long-context checks and soak. Gate: all promotion criteria met and human-readable reproducibility bundle.

Estimates are engineering planning ranges, not measured throughput or promised speedups.

### Owner inputs needed before implementation/run

- Actual head/worker SSH identities and CX7 mapping; whether this machine is the head or only the controller.
- Dedicated maintenance window and whether an existing service must be restored automatically.
- Effective model/image revisions and current deployment settings, including rightsize/stock workspace.
- Permission to retain representative checkpoint-derived fixtures and any approved quality corpus.
- Preferred coding-agent provider and maximum cost/time budget; keep its inference off the measured pair.
- Confirmation of priority: this PRD defaults to faster cold prefill with no decode regression; decode-first work needs a separately weighted campaign.

No answers are required just to retain this plan. They are prerequisites for running the lab.

## 11. Immediate next action

**Feasibility update:** the supervised probe has now compiled and checked two E2 variants, passed stock GPU checks on both nodes, and restored the service with the same existing weights. See [the successful feasibility report](reports/feasibility-success.md). The tested service used the owner's approved 0.85 memory budget and 250K context, not the example 1M campaign profile. The synthetic timing result is not a qualified serving speedup.

Implement **the immutable evaluator and replay fixtures first**, not an open-ended kernel-editing agent. Reproduce current E2 and profile it on the actual pair. Then begin with bounded E2 tile/occupancy experiments; open fusion/packing changes only when the measurements justify them. This preserves autoresearch's useful feedback loop without letting the agent optimize stale baselines, cache hits, incorrect arithmetic, or a kernel that never runs in production.
