# GLM-5.3 Flash: 2× Spark kernel autoresearch

An [autoresearch](https://github.com/karpathy/autoresearch)-inspired research plan and **supervised feasibility probe** for the [MiaAI-Lab EXL3 two-Spark serving kit](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).

**Status: supervised feasibility passed; not unattended-ready.** The [successful run](reports/feasibility-success.md) compiled two E2 variants, verified direct/scatter parity on 24 shapes per variant, timed them on GB10, and restored the service with unchanged weight storage. Both nodes passed stock GPU self-checks. Earlier build/recovery failures and their fixes remain documented in [`reports/`](reports/).

The full autonomous research controller is not implemented. The probe does not promote changes, train a model or establish end-to-end speedups. See [`prd.md`](prd.md) for the roadmap.

## Latest timed campaign

The [30-minute experiment log](reports/timed-30min.md) records prefill GEMM screening and a new experimental fused routing kernel for decode. The routing kernel improved synthetic metadata timings, but full-MoE gains were small/noisy; **no serving speedup or production promotion is claimed**. All tested changes remain isolated from the restored model service.

## Principles

- Reuse the Mia deployment's existing model and draft cache on each node. **No checkpoint download, copy or relocation.**
- Pin the running image; copy only the small extension source distributed inside it.
- Keep production containers intact: stop/start them rather than remove/recreate them.
- Separate correctness from latency. Synthetic timings are screening evidence, not serving benchmarks.
- Configure worker identity and container names; no machine-specific source paths, hardlinks or symlinks to another checkout.
- Raw artifacts and local settings are ignored by Git. Review any report before publishing it.

## What exists

| File | Purpose |
|---|---|
| `prd.md` | Full plan, trusted evaluator requirements, experiment priorities and promotion gates |
| `feasibility/run.sh` | Supervised maintenance workflow with an EXIT restoration handler |
| `feasibility/check_weights.py` | Read-only snapshot config/hash, indexed shard size and symlink-target checks |
| `feasibility/service_check.py` | Health and short model-response check inside the existing head container |
| `feasibility/kernel_probe.py` | Compile original and launch-bounds variant; synthetic direct/scatter parity and direct latency |
| `tests/` | CPU-only tests for the weight validation helper |

## Run the supervised probe

Requires a working Mia head/worker deployment, native GB10 GPUs, matching image IDs, Docker access, passwordless SSH, Bash, `flock`, `timeout`, and an approved maintenance window. Run from the **head node**. The installed image must contain CUDA compiler/headers, PyTorch, ExLlama extension sources and `/opt/glm53/test_exl3_overlay.py`.

```bash
# Clone this repository to any directory, then enter it.
cp .env.example .env
# Edit .env: WORKER_SSH is required. Review all values before sourcing it.
set -a
source .env
set +a

# WARNING: interrupts inference on BOTH nodes; original service restarts afterward.
CONFIRM_SERVICE_STOP=yes bash feasibility/run.sh
```

The runner derives its project directory from its own location. `/lab` is only a temporary container mount point, not a required host path. It does not need the original source checkout to be accessible through the new project.

The existing containers still retain whatever mounts their original deployment requires; do not move/delete that deployment while using this stop/start probe. A future independent deployment controller is described in the PRD.

### What happens

1. Verify both containers are running with matching image IDs; validate existing model/draft snapshots and a baseline API response.
2. Extract the installed extension source if not already present (about a megabyte, not weights).
3. Compile/import both variants **without GPU access**, with bounded CPU/RAM, before stopping inference. On successful compilation, stop the original head and worker without removing them.
4. Run stock GPU self-checks on both nodes in network-disabled temporary containers without model mounts.
5. On the head, compile baseline E2 and one `__launch_bounds__(..., 2)` candidate. Check direct/scatter parity against the installed extension, reject a deliberate wrong-output control, test direct graph replay and time synthetic shapes.
6. On ordinary success/failure/signal exit, restart original worker then head, wait for health and an API response, and recheck weight metadata.

Outputs go to ignored `artifacts/feasibility/`. A run overwrites these filenames; archive that directory before another run. Only one local run is allowed by a file lock. This is **not** a distributed cluster lease.

### Limits and safety

- EXIT handling cannot recover from host failure or SIGKILL. No external watchdog exists yet. Keep a second terminal available for manual recovery:

  ```bash
  ssh "$WORKER_SSH" docker start "${WORKER_CONTAINER:-glm53-exl3-worker}"
  docker start "${HEAD_CONTAINER:-glm53-exl3-head}"
  ```

- The probe mounts this project writable to store compiled binaries and outputs. Do not put credentials here. This is trusted, supervised code, **not a sandbox for arbitrary agent-generated code**.
- Reference checking uses the installed E2 implementation and synthetic tensors, not a fully independent model quality oracle. Weight validation checks metadata/config and shard presence/sizes; it does not rehash 164 GiB of weights.
- The current timing is head-only, sequential and cache-warm; no confidence-qualified two-rank candidate validation, realistic routed MoE pipeline timing or sanitizer checks yet.
- The worker self-check's remote Docker client timeout is not a remote container watchdog. The restoration handler removes the named probe on both nodes, but loss of SSH can prevent remote cleanup; inspect for an orphaned probe before manually restarting a loaded service.
- Full kernel search, hardened process isolation, independent watchdog, trusted score logic and production promotion remain roadmap work.

## CPU checks

```bash
python3 -m unittest discover -s tests -v
bash -n feasibility/run.sh
```

GPU execution uses the existing image's dependencies; no autoresearch training environment is installed.

## License and upstreams

New project code/documentation is MIT. Extracted ExLlama/CUDA sources remain local and are not redistributed in this Git repository; their upstream licenses still apply. Model weights and DFlash2 retain their separate licenses. Nothing here grants rights to redistribute checkpoint-derived data or modify draft weights contrary to their terms.
