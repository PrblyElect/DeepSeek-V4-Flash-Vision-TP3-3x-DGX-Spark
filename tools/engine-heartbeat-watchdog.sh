#!/bin/bash
# Engine heartbeat watchdog v3 — the TP3 distributed-deadlock self-healer,
# probe-confirmed edition.
#
# WEDGE SIGNATURE (observed in the field — a silent engine behind a live
# API, lasting minutes at a time):
# the API server keeps answering /health 200 while the engine core stops
# printing its scheduler-iteration lines — a rank spin-waiting (near-full util
# at low power on all ranks) inside a collective the torch NCCL watchdog
# does not cover (PYNCCL backend). Nothing fires; the wedge lives
# forever. This watchdog watches the heartbeat and runs the proven
# recovery (stop -> verified rm on all ranks -> [optional drop caches,
# one attempt per box] -> boot).
#
# v3 DETECTION — two stages, because silence alone cannot discriminate.
# On this build a healthy IDLE engine goes log-DORMANT: after activity
# stops the stats logger prints two trailing idle lines and then goes
# silent (early false-fires: v2's pure 90 s-silence trigger tore down
# two healthy boots while the lane sat idle between gates). But an
# earlier wedge also froze silently at its
# idle transition (final line "Running: 0") — so silence is necessary
# but not sufficient. The discriminator that actually works is a tiny
# real request: a dormant-idle engine answers a 1-token completion in
# well under a second; a wedged engine hangs it (generation never returns).
#
#   STAGE 1 — silence trigger (checked every POLL seconds, ALL must
#   hold; the container-absent case below short-circuits before it):
#     - head container exists and is running, up >= SETTLE seconds (no
#       false fire during boots: a booting engine also prints no lines)
#     - the API answers /health 200 (rules out clean-down states)
#     - zero "Engine 000:" iteration lines in the last WINDOW seconds
#
#   The WINDOW default is 180 s, not 90: a busy engine prints its
#   Engine lines at ~10 s cadence, an idle one every ~30 s or fully
#   dormant, but a HEALTHY ultra-deep prefill legitimately silences
#   the stats logger for its whole chunked run — live traffic has
#   produced sub-minute Engine-stats silences while prefilling large
#   contexts (no wedge, zero target-kernel JIT), close
#   enough to the old 90 s stage-1 edge that a slightly deeper
#   prefill would have eaten a probe (and, pre-v3, a kill). 180 s
#   keeps healthy deep prefills outside stage-1; a real wedge is
#   still detected within WINDOW + PROBE_TIMEOUT (180+45 s).
#
#   STAGE 2 — confirmation probe (at most once per poll cycle; NEVER
#   run when the container is absent):
#     POST /v1/chat/completions with a tiny prompt, max_tokens=1,
#     timeout PROBE_TIMEOUT. Auth mirrors the boot warmup script:
#     WD_PROBE_BEARER wins, else VLLM_API_KEY, else the lane is unkeyed;
#     the credential is never logged or written anywhere. The served
#     model id is resolved live from /v1/models (vLLM 404s unknown
#     model names) unless WD_PROBE_MODEL pins it.
#       probe answers    -> DORMANT-OK (idle, probe NNNms): healthy
#                           idle dormancy, NO action, next cycle sleeps.
#       probe fails/timeout -> WEDGE CONFIRMED -> recovery (below).
#       model id unresolvable -> probe SKIPPED (missing evidence proves
#                           neither state — never recover on it), retry
#                           next cycle.
#     Both stages log to the journal (stderr) every cycle and the state
#     file carries each stage's tick (PROBING -> DORMANT-OK /
#     RECOVERING); $LOG stays event-only (dormancy transitions,
#     recovery steps) as in v2.
#
# ABSENT-CONTAINER POLICY: if the engine container is not running, the
# watchdog NEVER fires, NEVER probes, and NEVER boots the lane from
# cold — it only ever recovers a container it watched wedge. It keeps
# ticking an IDLE state so a stale state file is itself an alarm.
#
# Recovery is crash-proof by construction: every step is individually
# logged and guarded; a failed step records the failure in the state
# file AND the journal but never aborts the sequence, and the boot step
# is always attempted so the lane can never be left half-killed. (Born
# from a real crash: `set -u` hit an undefined $RANKS
# after the kill steps but before the boot, leaving the lane down.)
# drop_caches is OFF by default and best-effort when it is enabled: set
# WD_DROP_CACHES to a narrow sudoers-wrapped helper (e.g.
# /usr/local/sbin/your-drop-caches) and it runs on the head and on every
# worker rank. The recovery makes ONE attempt per box - a failure is
# logged (step_fail) and the sequence continues to the boot step; there
# is no retry.
#
# Deploy on the head box (any always-on host that can reach all ranks):
#   systemctl --user enable --now engine-heartbeat-watchdog  # unit below
#
# systemd user unit (~/.config/systemd/user/engine-heartbeat-watchdog.service):
#   [Unit]
#   Description=TP3 engine heartbeat watchdog (wedge detect + auto-recover)
#   [Service]
#   ExecStart=/bin/bash %h/.local/libexec/engine-heartbeat-watchdog.sh
#   Restart=always
#   RestartSec=10
#   [Install]
#   WantedBy=default.target
set -u
POLL="${WD_POLL_SECS:-30}"
WINDOW="${WD_WINDOW_SECS:-180}"
SETTLE="${WD_SETTLE_SECS:-420}"
ENGINE_PORT="${WD_ENGINE_PORT:-8000}"   # this recipe's engine serves on :8000 (config/cluster.env)
CONTAINER="${WD_CONTAINER:-ds4flash-vision-vllm-dspark-1}"
LANE_DIR="${WD_LANE_DIR:-$HOME/tp3-lane}"
STOP="${WD_STOP:-$LANE_DIR/stop-deepseek-v4-flash-dspark.sh}"
START="${WD_START:-$LANE_DIR/start-tp3.sh}"
# Worker ranks the TP3 lane spans (head is local); adapt to your cluster.env.
RANKS="${WD_RANKS:-192.168.0.11 192.168.0.12}"
SSH_USER="${WD_SSH_USER:-YOURUSER}"
# Cache dropping is off by default: empty = skip the step entirely. Set
# WD_DROP_CACHES to a narrow sudoers-wrapped helper path to enable it.
DROP_CACHES="${WD_DROP_CACHES:-}"
# Stage-2 probe knobs.
PROBE_TIMEOUT="${WD_PROBE_TIMEOUT_SECS:-45}"
PROBE_MODEL="${WD_PROBE_MODEL:-}"   # empty = resolve from /v1/models per probe
LOG="${WD_LOG:-$HOME/.local/state/engine-watchdog.log}"
STATE="${WD_STATE:-$HOME/.local/state/engine-watchdog.state}"
mkdir -p "$(dirname "$LOG")" "$(dirname "$STATE")"

# Spin-prefill detector knobs (field-added): the engine-responsive
# zombie NEVER goes log-silent (beats>0), so the stage-1 silence path is
# structurally blind to it. Detected from the SCHEDULER METRIC LINE instead:
# Running>0 AND Avg generation throughput ≈0 AND Avg prompt throughput high,
# sustained for SPIN_SUSTAIN seconds. These are env-overridable.
SPIN_GEN_MAX="${WD_SPIN_GEN_MAX:-1.0}"            # tok/s — captures show gen near zero
SPIN_PROMPT_MIN="${WD_SPIN_PROMPT_MIN:-50}"       # tok/s — it IS processing a prompt
SPIN_SUSTAIN="${WD_SPIN_SUSTAIN_SECS:-300}"       # 5 min; observed zombies outlast this
spin_hits=0

# Probe auth — the boot warmup script's exact posture, env-overridable:
# a watchdog-provided bearer wins, else VLLM_API_KEY when the unit
# environment carries one, else the request goes unkeyed. Never logged.
PROBE_AUTH_ARGS=()
if [ -n "${WD_PROBE_BEARER:-}" ]; then
  PROBE_AUTH_ARGS=(-H "Authorization: Bearer ${WD_PROBE_BEARER}")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  PROBE_AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

# A mis-set WD_* numeric knob would silently break the settle/window
# gates (non-numeric operands make `[ -lt ]` error out as false) —
# refuse to run instead of running blind.
for _v in POLL WINDOW SETTLE ENGINE_PORT PROBE_TIMEOUT SPIN_PROMPT_MIN SPIN_SUSTAIN; do
  case "${!_v}" in
    ''|*[!0-9]*)
      echo "engine-heartbeat-watchdog: ${_v} must be a positive integer (got '${!_v}'), refusing to run" >&2
      exit 1
      ;;
  esac
done
unset _v

# Events go to the log file AND the journal (systemd captures stderr).
log() {
  local line
  line="$(date '+%F %T') $*"
  echo "$line" >> "$LOG"
  echo "$line" >&2
}

# Per-poll cadence line: journal only — the state file carries the
# monitor tick, $LOG stays event-only.
poll() { echo "$(date '+%F %T') poll: $*" >&2; }

# A failed recovery step must be loud but must never abort the run —
# the boot at the end is always attempted.
step_fail() {
  log "WARN: recovery step FAILED: $* — continuing"
  echo "$(date '+%F %T') STEP-FAILED: $*" >> "$STATE"
}

# STAGE 2 — the confirmation probe. Echoes the measured request latency
# in integer ms ("?" when unavailable). Return codes:
#   0 = answered      -> dormant-healthy, NO action
#   3 = never asked   -> served model id unresolvable; proves nothing,
#                        caller must NOT recover on this
#   other             -> WEDGE CONFIRMED (curl 28 = timeout, 22 = HTTP
#                        error status, 5/6/7 = transport dead)
run_probe() {
  local model out rc code secs ms
  if [ -n "$PROBE_MODEL" ]; then
    model="$PROBE_MODEL"
  else
    model=$(curl -s -m 5 ${PROBE_AUTH_ARGS[@]+"${PROBE_AUTH_ARGS[@]}"} \
      "http://127.0.0.1:${ENGINE_PORT}/v1/models" 2>/dev/null \
      | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n 1 \
      | sed 's/.*"\([^"]*\)"$/\1/')
  fi
  if [ -z "$model" ]; then
    echo "?"
    return 3
  fi
  out=$(curl -fsS -o /dev/null -m "$PROBE_TIMEOUT" \
    ${PROBE_AUTH_ARGS[@]+"${PROBE_AUTH_ARGS[@]}"} \
    -w '%{http_code} %{time_total}' \
    "http://127.0.0.1:${ENGINE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$model"'","messages":[{"role":"user","content":"Watchdog liveness probe — reply with one word."}],"max_tokens":1}' 2>/dev/null)
  rc=$?
  code=${out%% *}
  secs=${out##* }
  ms=$(awk -v t="$secs" 'BEGIN { printf "%d", t * 1000 }')
  echo "${ms:-?}"
  [ "$rc" -ne 0 ] && return "$rc"
  [ "${code:-000}" = "200" ] || return 22
  return 0
}

# WEDGE RECOVERY — shared by BOTH triggers. The heartbeat-silence trigger
# (stage-1 probe fail, below) and the spin-prefill trigger (in the
# else branch) each log their own message and land here; everything after is
# the same crash-proof sequence (capture stacks, stop, force-rm, drop caches,
# boot, back off). Factored once so the two triggers can never drift.
do_wedge_recover() {
  dormant=0
  recover_count=$((recover_count + 1))
  echo "$(date '+%F %T') RECOVERING #$recover_count" > "$STATE"

  # WEDGE STACK CAPTURE: before teardown, grab
  # py-spy stacks of the engine processes — the intermittent idle-transition
  # wedge has never been caught in the act; this is the root-cause instrument.
  # Bounded: 12s total; failure never blocks recovery.
  {
    B="$LANE_DIR/logs/wedge-stacks-$(date -u +%Y%m%dT%H%MZ)"
    mkdir -p "$B"
    for _pid in $(pgrep -f "VLLM::|vllm.entrypoints" 2>/dev/null | head -6); do
      timeout 6 sudo -n $HOME/.local/bin/py-spy dump --pid "$_pid"           > "$B/pid$_pid.txt" 2>&1 || true
    done
    : # (no state-file copy here: the state var names differ between units)
    docker logs --timestamps --tail 60 "$CONTAINER" > "$B/container-tail.log" 2>&1 || true
    # worker ranks too: a TP3 wedge can freeze a WORKER first.
    # FIX: the remote command MUST be single-quoted — the old
    # double-quoted version let the LOCAL shell expand $(pgrep) and $p
    # (unbound under set -u -> "p: unbound variable" crashed the watchdog
    # on every live capture) and pgrep'd the HEAD's pids
    # instead of the worker's. scp dest was also a filename prefix, not a
    # directory, so even successful dumps were never collected.
    for _w in $RANKS; do
      mkdir -p "$B/worker-$_w"
      ssh -o BatchMode=yes -o ConnectTimeout=8 "$_w" 'mkdir -p /tmp/wedge-stacks; for p in $(pgrep -f "VLLM::|vllm.entrypoints" 2>/dev/null); do timeout 6 sudo -n '"$HOME"'/.local/bin/py-spy dump --pid $p > /tmp/wedge-stacks/pid$p.txt 2>&1 || true; done; nvidia-smi > /tmp/wedge-stacks/nvidia-smi.txt 2>&1 || true' 2>/dev/null || true
      scp -o BatchMode=yes -o ConnectTimeout=8 "[$_w]:/tmp/wedge-stacks/*" "$B/worker-$_w/" 2>/dev/null || true
    done
    log "recover: wedge stacks captured -> $B"
  } || true

  log "recover: stop script ($STOP)"
  bash "$STOP" >> "$LOG" 2>&1 || step_fail "stop script ($STOP) exited nonzero"
  log "recover: force-remove head container $CONTAINER"
  docker rm -f "$CONTAINER" >> "$LOG" 2>&1 || step_fail "docker rm -f $CONTAINER"
  log "recover: force-remove worker containers ($RANKS)"
  for h in $RANKS; do
    ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_USER}@${h}" \
      'docker rm -f $(docker ps -aq --filter name=dspark)' >> "$LOG" 2>&1 \
      || step_fail "worker container cleanup on $h"
  done
  if [ -n "$DROP_CACHES" ]; then
    log "recover: drop page caches on head (narrow sudoers helper)"
    sudo -n "$DROP_CACHES" >> "$LOG" 2>&1 || step_fail "head drop-caches (sudoers rule missing?)"
    log "recover: drop page caches on workers ($RANKS)"
    for h in $RANKS; do
      ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_USER}@${h}" \
        "sudo -n $DROP_CACHES" >> "$LOG" 2>&1 \
        || step_fail "drop-caches on $h"
    done
  fi

  # The boot is ALWAYS attempted — nothing above may leave the lane
  # half-killed without at least trying to bring it back.
  if [ -d "$LANE_DIR" ] && [ -f "$START" ]; then
    log "recover: booting lane (nohup $START; output /tmp/watchdog-boot.log)"
    ( cd "$LANE_DIR" && nohup bash "$START" >> /tmp/watchdog-boot.log 2>&1 & )
  else
    step_fail "lane checkout missing ($LANE_DIR / $START) — CANNOT BOOT, manual start needed"
  fi
  # back off for a full boot before resuming detection
  sleep "$SETTLE"
  echo "$(date '+%F %T') recovering-done; monitoring" > "$STATE"
}

recover_count=0
dormant=0
while true; do
  sleep "$POLL"
  status=$(docker inspect "$CONTAINER" --format '{{.State.Status}} started={{.State.StartedAt}}' 2>/dev/null) || status=""
  if [ "${status%% *}" != "running" ]; then
    # Engine container absent or not running: deliberate-down (or
    # foreign) state. Never fire, never probe, never auto-boot from
    # cold — just keep the liveness tick moving.
    echo "$(date '+%F %T') IDLE (engine container not running)" > "$STATE"
    poll "engine container absent/not running — idle, not acting"
    continue
  fi
  started_epoch=$(date -d "${status##*started=}" +%s 2>/dev/null) || { poll "running but unparseable StartedAt ($status)"; continue; }
  now_epoch=$(date +%s)
  up_for=$((now_epoch - started_epoch))
  [ "$up_for" -ge "$SETTLE" ] || { poll "up ${up_for}s < settle ${SETTLE}s — booting, not judging"; continue; }
  # engine must be API-alive for this to be a wedge (not a clean restart)
  http=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:${ENGINE_PORT}/health" 2>/dev/null)
  [ "${http:-000}" = "200" ] || { poll "up ${up_for}s health=${http:-none} — not API-alive, not judging"; continue; }
  beats=$(docker logs --since "${WINDOW}s" "$CONTAINER" 2>&1 | grep -c "Engine 000:")
  poll "up ${up_for}s health=200 beats=${beats}/${WINDOW}s"
  if [ "$beats" -eq 0 ]; then
    # ---- STAGE 1 complete: silence trigger. Never reaches here when
    # the container is absent (that branch continues above), so the
    # probe can never run against a missing engine.
    poll "stage-1 SILENCE: zero beats ${WINDOW}s, API alive — running confirmation probe"
    echo "$(date '+%F %T') PROBING (silence ${WINDOW}s, confirming)" > "$STATE"
    probe_rc=0
    probe_ms=$(run_probe) || probe_rc=$?
    case "$probe_rc" in
      0)
        # ---- STAGE 2 passed: dormant-idle, not wedged. NO action.
        if [ "$dormant" -eq 0 ]; then
          log "engine went log-dormant (idle): ${WINDOW}s silence, 1-token probe answered in ${probe_ms}ms — healthy, no action"
          dormant=1
        else
          poll "stage-2 PROBE-OK: dormant-idle confirmed, probe ${probe_ms}ms — no action"
        fi
        echo "$(date '+%F %T') DORMANT-OK (idle, probe ${probe_ms}ms)" > "$STATE"
        # exactly one probe per poll cycle: the loop sleeps now
        ;;
      3)
        log "WARN: probe SKIPPED — served model id unresolvable (/v1/models); neither dormant-healthy nor wedged is confirmed, retrying next cycle"
        echo "$(date '+%F %T') PROBE-SKIPPED (no model id from /v1/models)" > "$STATE"
        ;;
      *)
        # ---- STAGE 2 failed: WEDGE CONFIRMED — silence was real after
        # all. Existing crash-proof recovery, unchanged from v2.
        log "WEDGE CONFIRMED: stage-2 probe failed (rc=${probe_rc}, ${probe_ms:-?}ms) after ${WINDOW}s heartbeat silence, API alive (container up ${up_for}s) — recovering"
        do_wedge_recover
        ;;
    esac
  else
    # ---- SPIN-PREFILL detector: the engine-responsive zombie.
    # It never goes log-silent (beats>0), so the stage-1 silence path above is
    # structurally blind to it. Signature from the SCHEDULER METRIC LINE:
    # Running>0 AND Avg generation throughput ≈0 AND Avg prompt throughput high,
    # sustained for SPIN_SUSTAIN seconds. When sustained, treat it as a wedge
    # (same do_wedge_recover path — the stack capture is the root-cause goal).
    _spin_line=$(docker logs --since "${WINDOW}s" "$CONTAINER" 2>&1 \
        | grep -o "Engine 000:.*" | tail -1)
    _spin_running=$(printf '%s\n' "$_spin_line" \
        | sed -n 's/.*Running: \([0-9][0-9]*\) reqs.*/\1/p')
    _spin_prompt=$(printf '%s\n' "$_spin_line" \
        | sed -n 's/.*Avg prompt throughput: \([0-9.]*\) tokens\/s.*/\1/p')
    _spin_gen=$(printf '%s\n' "$_spin_line" \
        | sed -n 's/.*Avg generation throughput: \([0-9.]*\) tokens\/s.*/\1/p')
    if [ -n "$_spin_running" ] && [ "${_spin_running:-0}" -gt 0 ]; then
      if awk -v g="${_spin_gen:-0}" -v mx="$SPIN_GEN_MAX" 'BEGIN{exit !(g < mx)}' \
          && awk -v p="${_spin_prompt:-0}" -v mn="$SPIN_PROMPT_MIN" 'BEGIN{exit !(p > mn)}'; then
        spin_hits=$((spin_hits + 1))
        _spin_elapsed=$((spin_hits * POLL))
        if [ "$_spin_elapsed" -ge "$SPIN_SUSTAIN" ]; then
          log "SPIN-PREFILL WEDGE CONFIRMED: Running=${_spin_running} reqs, prompt=${_spin_prompt} tok/s, gen=${_spin_gen} tok/s sustained ${_spin_elapsed}s (>${SPIN_SUSTAIN}s) — engine-responsive spin-prefill zombie, recovering"
          spin_hits=0
          do_wedge_recover
        fi
      else
        spin_hits=0
      fi
    else
      spin_hits=0
    fi
    if [ "$recover_count" -gt 0 ]; then
      log "heartbeat restored after $recover_count recovery(ies)"
      recover_count=0
    fi
    if [ "$dormant" -eq 1 ]; then
      log "engine woke from dormancy (heartbeats resumed)"
      dormant=0
    fi
    # timestamped OK doubles as the watchdog's own liveness signal
    echo "$(date '+%F %T') OK" > "$STATE"
  fi
done
