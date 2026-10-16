#!/bin/bash
# Engine heartbeat watchdog v3 — self-healer for TP3 distributed
# deadlocks, probe-confirmed edition.
#
# WEDGE SIGNATURE (measured): the API server keeps answering /health 200
# while the engine core stops printing its scheduler-iteration lines —
# a rank spin-waiting (all ranks ~max util at idle-class power) inside
# a collective that the torch NCCL watchdog does not cover on this
# stack's PYNCCL all-reduce backend. Nothing fires; the wedge lives
# forever and every in-flight request hangs.
#
# v3 DETECTION — two stages, because line silence alone cannot
# discriminate. On this build a healthy IDLE engine goes log-DORMANT:
# after activity stops the stats logger prints two trailing idle lines
# and then prints nothing at all — so a pure "zero iteration lines in
# WD_WINDOW_SECS" trigger fires on every healthy idle boot. But a real
# wedge also freezes silently at an idle transition (final line
# "Running: 0"), so silence is necessary but not sufficient. The
# discriminator that works is a tiny real request: a dormant-idle
# engine answers a 1-token completion in a fraction of a second; a
# wedged engine hangs it (health stays 200, generation never returns).
#
#   STAGE 1 — silence trigger (checked every WD_POLL_SECS, ALL must
#   hold; the container-absent case short-circuits before it):
#     - the head container exists and is running, up >= WD_SETTLE_SECS
#       (a booting engine also prints no iteration lines — never fire
#       on boots)
#     - the API answers /health 200 (rules out deliberate-down states)
#     - zero iteration lines in the last WD_WINDOW_SECS
#
#   The WD_WINDOW_SECS default is 180 s, not 90: a busy engine
#   prints Engine lines at a ~10 s cadence and an idle one every
#   ~30 s or fully dormant, but a HEALTHY ultra-deep prefill
#   legitimately silences the stats logger for its whole chunked
#   run — silences past 60 s are observed on deep-prefill traffic
#   with no wedge and no JIT. 180 s keeps healthy deep prefills
#   outside stage-1; a real wedge is still detected within
#   WD_WINDOW_SECS + WD_PROBE_TIMEOUT_SECS.
#
#   STAGE 2 — confirmation probe (at most once per poll cycle; NEVER
#   run when the container is absent):
#     POST /v1/chat/completions with a tiny prompt, max_tokens=1,
#     timeout WD_PROBE_TIMEOUT_SECS. Auth mirrors tools/boot-shape-warmup.sh
#     (a WD_PROBE_BEARER wins, else VLLM_API_KEY when set, else the
#     request goes unkeyed); the credential is never logged or written
#     anywhere. The served model id is resolved live from /v1/models
#     (vLLM rejects unknown model names with a 404) unless
#     WD_PROBE_MODEL pins it.
#       probe answers       -> DORMANT-OK (idle, probe NNNms): healthy
#                              idle dormancy, NO action.
#       probe fails/times out -> WEDGE CONFIRMED -> recovery below.
#       model id unresolvable -> probe SKIPPED (missing evidence proves
#                              neither state — never recover on it),
#                              retried next cycle.
#     Both stages log to the journal (stderr) every cycle and the state
#     file carries each stage's tick (PROBING -> DORMANT-OK /
#     RECOVERING); the watchdog log stays event-only (dormancy
#     transitions, recovery steps).
#
# ABSENT-CONTAINER POLICY: if the head container is not running, the
# watchdog NEVER fires, NEVER probes, and NEVER boots the lane from
# cold — it only ever recovers a container it watched wedge. It keeps
# ticking an IDLE state file so a stale state is itself an alarm.
#
# Recovery is crash-proof by construction: every step is individually
# logged and guarded; a failed step records the failure in the state
# file but never aborts the sequence, and the boot step is always
# attempted so the lane is never left half-killed (killed but never
# rebooted is the worst outcome a self-healer can produce).
#
# Every action is logged; a timestamped state file doubles as the
# watchdog's own liveness signal for an upstream monitor.
#
# Deploy on the head box (any always-on host that can reach all ranks):
#   systemctl --user enable --now engine-heartbeat-watchdog  # unit below
# Adapt WD_LANE_DIR / WD_STOP / WD_START to your launcher; the defaults
# follow this recipe's layout. Cache dropping is OFF by default: set
# WD_DROP_CACHES to a narrow sudoers-wrapped command (e.g.
# "sudo -n /usr/local/sbin/your-drop-caches" backed by a rule for
# `sync; echo 3 > /proc/sys/vm/drop_caches`) to make it fully
# hands-off; failures are best-effort and never block the boot.
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
CONTAINER="${WD_CONTAINER:-ds4flash-vision-vllm-dspark-1}"
LANE_DIR="${WD_LANE_DIR:-$HOME/tp3-lane}"
STOP="${WD_STOP:-$LANE_DIR/stop-deepseek-v4-flash-dspark.sh}"
START="${WD_START:-$LANE_DIR/start-tp3.sh}"
RANKS="${WD_RANKS:-192.168.0.11 192.168.0.12}"   # worker ranks (head is local)
ENGINE_PORT="${WD_ENGINE_PORT:-8000}"
SSH_USER="${WD_SSH_USER:-YOURUSER}"
DROP_CACHES="${WD_DROP_CACHES:-}"   # empty = skip cache drops (default)
# Stage-2 probe knobs.
PROBE_TIMEOUT="${WD_PROBE_TIMEOUT_SECS:-45}"
PROBE_MODEL="${WD_PROBE_MODEL:-}"   # empty = resolve from /v1/models per probe
LOG="${WD_LOG:-$HOME/.local/state/engine-watchdog.log}"
STATE="${WD_STATE:-$HOME/.local/state/engine-watchdog.state}"
mkdir -p "$(dirname "$LOG")" "$(dirname "$STATE")"

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
for _v in POLL WINDOW SETTLE ENGINE_PORT PROBE_TIMEOUT; do
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
        # all. Crash-proof recovery.
        log "WEDGE CONFIRMED: stage-2 probe failed (rc=${probe_rc}, ${probe_ms:-?}ms) after ${WINDOW}s heartbeat silence, API alive (container up ${up_for}s) — recovering"
        dormant=0
        recover_count=$((recover_count + 1))
        echo "$(date '+%F %T') RECOVERING #$recover_count" > "$STATE"

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
          log "recover: drop page caches on head ($DROP_CACHES)"
          $DROP_CACHES >> "$LOG" 2>&1 || step_fail "head drop-caches"
          log "recover: drop page caches on workers ($RANKS)"
          for h in $RANKS; do
            ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_USER}@${h}" \
              "$DROP_CACHES" >> "$LOG" 2>&1 \
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
        ;;
    esac
  else
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
