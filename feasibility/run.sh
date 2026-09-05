#!/usr/bin/env bash
# Supervised feasibility only. Stops/restores existing containers, never recreates them.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORKER_SSH:?Set the existing worker SSH destination}"
: "${CONFIRM_SERVICE_STOP:?Set CONFIRM_SERVICE_STOP=yes after approving maintenance}"
[[ "$CONFIRM_SERVICE_STOP" == yes ]] || exit 2
HEAD_CONTAINER="${HEAD_CONTAINER:-glm53-exl3-head}"
WORKER_CONTAINER="${WORKER_CONTAINER:-glm53-exl3-worker}"
PROBE_CONTAINER="${PROBE_CONTAINER:-glm53-kernel-feasibility}"
for name in "$HEAD_CONTAINER" "$WORKER_CONTAINER" "$PROBE_CONTAINER"; do
    [[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid container name'; exit 2; }
done
[[ "$WORKER_SSH" != -* && "$WORKER_SSH" != *[[:space:]]* ]] || exit 2
ART="$ROOT/artifacts/feasibility"
mkdir -p "$ART"
exec 9>"$ROOT/.feasibility.lock"
flock -n 9 || { echo 'Another local feasibility run holds the lock'; exit 2; }
ssh_worker() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
[[ "$(docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER")" == true ]]
[[ "$(ssh_worker "docker inspect -f '{{.State.Running}}' '$WORKER_CONTAINER'")" == true ]]
IMAGE="$(docker inspect -f '{{.Image}}' "$HEAD_CONTAINER")"
[[ "$IMAGE" == "$(ssh_worker "docker inspect -f '{{.Image}}' '$WORKER_CONTAINER'")" ]]
# Do not allow a name collision to overwrite another experiment.
if docker inspect "$PROBE_CONTAINER" >/dev/null 2>&1 || ssh_worker "docker inspect '$PROBE_CONTAINER'" >/dev/null 2>&1; then
    echo 'Probe container already exists on head or worker'; exit 2
fi
# Qualify wheel CUDA header layout before interrupting the service.
docker exec "$HEAD_CONTAINER" python3 -c 'import pathlib, torch; root=pathlib.Path(torch.__file__).resolve().parent.parent; assert any((p/"cusparse.h").is_file() for p in (root/"nvidia").glob("*/include")), "Missing CUDA wheel headers"'
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'start=%s\nimage=%s\n' "$STAMP" "$IMAGE" > "$ART/run-status.txt"
docker exec -i "$HEAD_CONTAINER" python3 - < "$ROOT/feasibility/check_weights.py" > "$ART/head-weights-before.log"
ssh_worker "docker exec -i '$WORKER_CONTAINER' python3 -" < "$ROOT/feasibility/check_weights.py" > "$ART/worker-weights-before.log"
cmp "$ART/head-weights-before.log" "$ART/worker-weights-before.log"
docker exec -i "$HEAD_CONTAINER" python3 - < "$ROOT/feasibility/service_check.py" > "$ART/service-before.json"
# Copy code only; no weights. Reuse source distributed with the installed image.
if [[ ! -d "$ROOT/feasibility/ext-source" ]]; then
    EXT_PATH="$(docker exec "$HEAD_CONTAINER" python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("exllamav3").origin).parent / "exllamav3_ext")' | tail -1)"
    docker cp "$HEAD_CONTAINER:$EXT_PATH" "$ROOT/feasibility/ext-source"
fi
STOPPED=0
restore() {
    rc=$?
    trap - EXIT INT TERM
    set +e
    docker rm -f "$PROBE_CONTAINER" >/dev/null 2>&1
    ssh_worker "docker rm -f '$PROBE_CONTAINER'" >/dev/null 2>&1
    if [[ "$STOPPED" == 1 ]]; then
        echo 'Restoring original worker/head containers (unchanged images, mounts and config)'
        ssh_worker "docker start '$WORKER_CONTAINER'" > "$ART/restore-worker.log" 2>&1
        worker_rc=$?
        docker start "$HEAD_CONTAINER" > "$ART/restore-head.log" 2>&1
        head_rc=$?
        if [[ $worker_rc == 0 && $head_rc == 0 ]]; then
            timeout 3900 docker exec -i "$HEAD_CONTAINER" python3 - < "$ROOT/feasibility/service_check.py" > "$ART/service-after.json" 2> "$ART/service-after.stderr"
            health_rc=$?
            if [[ $health_rc == 0 ]]; then
                docker exec -i "$HEAD_CONTAINER" python3 - < "$ROOT/feasibility/check_weights.py" > "$ART/head-weights-after.log"
                head_check=$?
                ssh_worker "docker exec -i '$WORKER_CONTAINER' python3 -" < "$ROOT/feasibility/check_weights.py" > "$ART/worker-weights-after.log"
                worker_check=$?
                cmp "$ART/head-weights-before.log" "$ART/head-weights-after.log"; head_weights=$?
                cmp "$ART/worker-weights-before.log" "$ART/worker-weights-after.log"; worker_weights=$?
                [[ $head_check == 0 && $worker_check == 0 ]] || health_rc=1
            else
                head_weights=1; worker_weights=1
                docker logs --tail 100 "$HEAD_CONTAINER" > "$ART/failed-restore-head.log" 2>&1
                ssh_worker "docker logs --tail 100 '$WORKER_CONTAINER'" > "$ART/failed-restore-worker.log" 2>&1
            fi
        else
            health_rc=1; head_weights=1; worker_weights=1
        fi
        if [[ $health_rc != 0 || $head_weights != 0 || $worker_weights != 0 ]]; then
            echo 'RESTORATION_FAILED: inspect artifacts and both container logs'; rc=1
        else
            echo 'RESTORATION_OK: response received; weight paths/config hashes/shard sizes unchanged'
        fi
    fi
    printf 'exit_code=%s\nend=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ART/run-status.txt"
    exit "$rc"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Compile/import without GPU access before causing downtime. CPU/RAM are bounded;
# this is not a timing run and must not be used for service performance claims.
timeout 1800 docker run --rm --name "$PROBE_CONTAINER" --network none \
    --memory=3g --memory-swap=4g --cpus=2 \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=1 -e PROBE_COMPILE_ONLY=1 \
    -v "$ROOT:/lab" --entrypoint python3 "$IMAGE" /lab/feasibility/kernel_probe.py > "$ART/compile-preflight.log" 2>&1
STOPPED=1
# Stop without removing containers; this preserves exact restore configuration.
docker stop -t 30 "$HEAD_CONTAINER"
ssh_worker "docker stop -t 30 '$WORKER_CONTAINER'"
timeout 600 docker run --rm --name "$PROBE_CONTAINER" --gpus all --network none --entrypoint python3 "$IMAGE" /opt/glm53/test_exl3_overlay.py > "$ART/head-selfcheck.log" 2>&1
# Remote selfcheck has its own hard deadline. No model mounts or network needed.
ssh_worker "timeout 600 docker run --rm --name '$PROBE_CONTAINER' --gpus all --network none --entrypoint python3 '$IMAGE' /opt/glm53/test_exl3_overlay.py" > "$ART/worker-selfcheck.log" 2>&1
# The installed image has CUDA headers and compiler; no package/network downloads.
timeout 1800 docker run --rm --name "$PROBE_CONTAINER" --gpus all --network none \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=2 \
    -v "$ROOT:/lab" --entrypoint python3 "$IMAGE" /lab/feasibility/kernel_probe.py > "$ART/kernel-probe.log" 2>&1
printf 'kernel_probe=passed\n' >> "$ART/run-status.txt"
