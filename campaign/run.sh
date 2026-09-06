#!/usr/bin/env bash
# One bounded supervised campaign; timer/deadline file must already be initialized.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORKER_SSH:?Set worker SSH destination}"
: "${CONFIRM_SERVICE_STOP:?Set yes to authorize interruption}"
[[ "$CONFIRM_SERVICE_STOP" == yes && "$WORKER_SSH" != -* && "$WORKER_SSH" != *[[:space:]]* ]] || exit 2
HEAD_CONTAINER="${HEAD_CONTAINER:-glm53-exl3-head}"
WORKER_CONTAINER="${WORKER_CONTAINER:-glm53-exl3-worker}"
for n in "$HEAD_CONTAINER" "$WORKER_CONTAINER"; do [[ "$n" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || exit 2; done
ART="$ROOT/artifacts/timed-campaign"
REPORT="$ROOT/reports/timed-30min.md"
ssh_worker() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
log() { printf '| %s | Controller | %s |\n' "$(date -u +%T)" "$1" >> "$REPORT"; echo "$1"; }
remaining() { python3 -c 'import json,time,sys; print(max(0,int(json.load(open(sys.argv[1]))["deadline_epoch"]-time.time()-660)))' "$ART/deadline.json"; }
exec 9>"$ROOT/.feasibility.lock";flock -n 9
[[ $(remaining) -gt 90 && ! -e "$ART/STOP" ]] || { log 'Not enough time; no service interruption'; exit 2; }
[[ $(docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER") == true ]]
[[ $(ssh_worker "docker inspect -f '{{.State.Running}}' '$WORKER_CONTAINER'") == true ]]
IMAGE="$(docker inspect -f '{{.Image}}' "$HEAD_CONTAINER")"
[[ "$IMAGE" == "$(ssh_worker "docker inspect -f '{{.Image}}' '$WORKER_CONTAINER'")" ]]
if docker inspect glm53-timed-probe >/dev/null 2>&1; then exit 2; fi
if [[ "${CAMPAIGN_HOLDOUT:-0}" == 1 ]]; then
    [[ -s "$ART/service-baseline.json" ]] || exit 2
    log 'Holdout sweep; retaining initial service baseline, fresh shapes/seed and 288 experts'
else
    log 'Real-service baseline started; no candidate deployed'
    timeout 180 docker exec -i "$HEAD_CONTAINER" python3 - < "$ROOT/campaign/service_bench.py" > "$ART/service-baseline.json" 2> "$ART/service-baseline.stderr"
fi
docker exec -i "$HEAD_CONTAINER" python3 - < "$ROOT/feasibility/check_weights.py" > "$ART/head-before.json"
ssh_worker "docker exec -i '$WORKER_CONTAINER' python3 -" < "$ROOT/feasibility/check_weights.py" > "$ART/worker-before.json"
STOPPED=0
restore() {
    rc=$?;trap - EXIT INT TERM;set +e
    docker rm -f glm53-timed-probe >/dev/null 2>&1
    if [[ $STOPPED == 1 ]]; then
        log 'GPU experiments ended; restoring approved baseline'
        ssh_worker "docker start '$WORKER_CONTAINER'" > "$ART/worker-restore.log" 2>&1; wr=$?
        docker start "$HEAD_CONTAINER" > "$ART/head-restore.log" 2>&1; hr=$?
        if [[ $wr == 0 && $hr == 0 ]]; then
            timeout 3900 docker exec -i "$HEAD_CONTAINER" python3 - < "$ROOT/feasibility/service_check.py" > "$ART/service-restored.json" 2> "$ART/service-restored.stderr"; ok=$?
        else ok=1; fi
        if [[ $ok == 0 ]]; then
            docker exec -i "$HEAD_CONTAINER" python3 - < "$ROOT/feasibility/check_weights.py" > "$ART/head-after.json"; hc=$?
            ssh_worker "docker exec -i '$WORKER_CONTAINER' python3 -" < "$ROOT/feasibility/check_weights.py" > "$ART/worker-after.json"; wc=$?
            cmp "$ART/head-before.json" "$ART/head-after.json"; hm=$?
            cmp "$ART/worker-before.json" "$ART/worker-after.json"; wm=$?
            if [[ $hc == 0 && $wc == 0 && $hm == 0 && $wm == 0 ]]; then log 'RESTORED: model response OK; existing weight manifests unchanged'; else ok=1; fi
        fi
        if [[ $ok != 0 ]]; then log 'RESTORE FAILED: owner attention required'; rc=1; fi
    fi
    log "Campaign controller exits status $rc; no automatic production promotion"
    exit "$rc"
}
trap restore EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
[[ $(remaining) -gt 60 && ! -e "$ART/STOP" ]] || exit 2
STOPPED=1
docker stop -t 30 "$HEAD_CONTAINER";ssh_worker "docker stop -t 30 '$WORKER_CONTAINER'"
log 'Services stopped intact; paired kernel screening starts'
BUDGET="$(remaining)"
[[ "$BUDGET" -gt 0 && ! -e "$ART/STOP" ]] || exit 124
# timeout 0 disables timeouts, so never pass an expired budget through.
timeout --signal=TERM --kill-after=10s "$BUDGET" docker run --rm --name glm53-timed-probe --gpus all --network none \
    -e PYTHONUNBUFFERED=1 -e "CAMPAIGN_HOLDOUT=${CAMPAIGN_HOLDOUT:-0}" -v "$ROOT:/lab" --entrypoint python3 "$IMAGE" /lab/campaign/probe.py > "$ART/kernel.log" 2>&1
log 'Kernel screening passed; results remain experimental'
