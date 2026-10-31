# VALIDATION — the gates, and the method behind them

Every number in the README and FIXLOG was produced under the discipline in
this file. Reproduce it and the numbers are comparable across stacks; skip
any part and they are not. The crash ladders live in `FIXLOG.md`; this file
is both the measurement method and the gate battery — and three of the
method's rules (instructed needles, cold/warm labels, same-boot A/B) were
learned the hard way, by publishing wrong inferences first and retracting
them.

Every gate below is a runnable command. The `# expected:` line under each
one says what a pass looks like — read it before you run the gate, not
after, because a gate whose expectation you did not know in advance is a
gate you will rationalise.

## 1. Offline gates — no engine, no network

Run these first. They are cheap, they need nothing but the checkout, and
every one of them carries its own self-test.

### 1.1 The instrument self-tests

```sh
python3 tools/needle-probe.py --selftest
python3 tools/fairness-probe.py --selftest
python3 tools/coupling-guard.py --selftest
python3 tools/spec-decode-short-prompt-check.py --selftest
# expected: SELFTEST: PASS from the three instruments that print it, and
# "selftest OK: ..." from the fairness probe
```

The needle probe's self-test is the one that matters most: both harness
bugs this recipe's history records (an instruction-less corpus, a mid-corpus
nonce) shipped before a self-test existed. Run it before every measurement
session, not once a week.

### 1.2 The coupling guard, offline, against a pinned checkout

The serving profile is written against two pinned public upstream
checkouts. The guard proves the profile and those revisions still agree:
every upstream-carried variable must still appear in a **non-documentation**
pinned upstream source, and every variable the recipe's own appliers own
must still be read by them. A renamed upstream variable is a **FAIL**,
loudly — the profile would otherwise boot with a knob that silently does
nothing. A variable that survives only in an upstream **documentation**
file is a failure too: the upstream renamed the knob and its docs have not
caught up, which is the realistic shape of that drift.

```sh
: "${COMPOSE_CHECKOUT:?set COMPOSE_CHECKOUT to the pinned compose checkout}"
: "${PADKIT_CHECKOUT:?set PADKIT_CHECKOUT to the pinned pad-kit checkout}"
python3 tools/coupling-guard.py \
  --compose-dir "$COMPOSE_CHECKOUT" \
  --padkit-dir "$PADKIT_CHECKOUT"
# expected: RESULT: PASS ... — the reader/library-level warnings are expected
# and are not failures
```

The same run works against pinned archives (a tarball of the same
revision), for a box that has the archives but not the checkouts:

```sh
python3 tools/coupling-guard.py \
  --compose-archive /path/to/compose-<rev>.tar.gz \
  --padkit-archive /path/to/padkit-<rev>.tar.gz
# expected: RESULT: PASS ... — the archives carry no git metadata, so the
# revision itself reads "unverified" while the variable coupling is still
# checked
```

### 1.3 The coupling guard, against the live upstream

This additionally proves the pinned revisions still exist upstream.

```sh
python3 tools/coupling-guard.py --remote
# expected: RESULT: PASS ... — or RESULT: INCONCLUSIVE (exit code 2) when the
# network is down, which is a warning and never a pass
```

### 1.4 The spec-decode check's self-test

The short-prompt correctness check ships with an offline self-test, so the
instrument is known-good before it is ever pointed at a lane.

```sh
python3 tools/spec-decode-short-prompt-check.py --selftest
# expected: SELFTEST: PASS — including the CJK cases, which must be
# ACCEPTED, not flagged
```

## 2. The gate battery — on the serving boot

Run in order. Every solo step runs only after the idle gate in §2.2
passes. The head box serves `:8000`; on another host, put the lane's
address in front of the tools' `--lane` / `BATTERY_LANE` /
`MIXED_TAX_LANE` variables.

### 2.1 Health and engine liveness

`/health` proves only that the API process is alive. After health turns
green, prove ALL ranks are alive with one tiny completion request before
any prefill-class traffic — a fresh-boot memory trough can let the kernel
kill one TP worker during weight load while the endpoint keeps answering
200.

```sh
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ds4flash-vision","prompt":"ok","max_tokens":1,"temperature":0}' \
  | head -c 200
# expected: 200, then a completion with a non-empty choice
```

### 2.2 The idle gate

Before ANY solo, tier or speed reading: the engine must be idle. A "slow"
reading taken with traffic in flight — your own warmup replays, scheduled
jobs, co-tenant requests, other clients — is an unlabeled MIXED number, not
a boot property. The trap is post-boot warm traffic: it overlaps exactly
the window where a fresh boot first gets probed.

```sh
curl -s http://127.0.0.1:8000/metrics \
  | grep -E '^vllm:num_requests_(running|waiting)'
# expected: both counters read 0 — vllm:num_requests_running and
#           vllm:num_requests_waiting
```

### 2.3 The boot shape warmup

Workers first, head last. Drop filesystem caches on every rank (unified
memory: page cache competes with the pool). Fresh `docker run` every boot —
never `docker restart`. Run the shape warmup so no client eats a mid-serve
JIT compile.

```sh
DSPARK_WARMUP_MAX_CONCURRENCY=32 \
  bash tools/boot-shape-warmup.sh http://127.0.0.1:8000 ds4flash-vision
# expected: every arm warms and the sampler-combo postcondition passes —
# zero mid-serve JIT warnings afterwards
```

### 2.4 The instructed needle ladder

A needle corpus that states the fact but asks no question measures the
model's temperature-0 continuation style, not recall. Always append an
explicit instruction; score word-exact; and judge an echo-able word.

```sh
python3 tools/needle-probe.py --depths 131072,196608,262144 --nonce fresh1
# expected: PASS on every depth — read the reported prompt_tokens, not the
# depth estimate: filler token estimates run low. The tool exits non-zero
# if any depth misses the needle (NEEDLE PROBE FAIL).
```

### 2.5 The vision fence

Single-image time-to-first-token, and the default per-request image cap,
with every image's colour named in order at TP=3.

```sh
python3 tools/needle-probe.py --depths 1024 --skip-needle
# expected: PASS — 8/8 images named in order; the cap is the engine's
# --limit-mm-per-prompt flag, not the model. The tool exits non-zero if a
# colour is missing OR the eight are named out of order.
```

### 2.6 The aggregate sweep

The workload split: count/code-class single-stream and free-prose
single-stream, at every concurrency level, with the corruption battery
beside the speed run.

```sh
python3 tools/agg-sweep.py --levels 1,4,16,32
# expected: — AGG SWEEP CLEAN, and the corruption spot-check green
# (exact_repeat=True strict_json=True). The spot-check is part of the
# verdict: if either half fails, the tool prints WITH ERRORS and exits 1.
```

### 2.7 The mixed-tax probe

Decode beside a deep prefill, on the head node. A ~12x mean with
multi-second gaps means the deferral knob is off or was not applied.

```sh
MIXED_TAX_LANE=http://127.0.0.1:8000 MIXED_TAX_MODEL=ds4flash-vision \
  python3 tools/mixed-tax-probe.py
# expected: the mixed tax stays in the ~3x mean class with the median at
# solo speed and the worst gap in the ~1 s class. The probe GATES those
# bounds: MIXED-TAX PROBE PASS, or a non-zero exit naming the bound that
# was crossed (mean >8x solo, p50 >6x solo, or a worst gap >5 s). The
# knob-off state measures ~12x with multi-second gaps.
```

### 2.8 The admission probe

Two fresh-nonce bulk prefills fill the inflight cap, then a small chat and
a parallel decode stream fire mid-flight. A minutes-class chat wall means
the admission knob is off, the policy flag is missing, or the
budget-guard hole is back. The probe accepts a loopback endpoint only, so
run it on the head box — and note that its first argument is the endpoint
**including** the `/v1` prefix.

```sh
python3 tools/fairness-probe.py http://127.0.0.1:8000/v1 admission 262144 ds4flash-vision
# expected: the mid-bulk chat answers in the single-digit-seconds class and
# both needles stay word-exact through preempt-recompute-resume. The probe
# GATES that: FAIRNESS PROBE PASS, or a non-zero exit naming the lost
# needle or the blown bound (chat wall >60 s, worst decode gap >30 s).
```

### 2.9 The mixed battery — the acceptance gate

A boot is not healthy because single-class probes pass. The gate runs
interactive streams, chat and image requests **in flight at the same
time**, plus two co-tenant service legs, with host memory sampled
throughout. A tenants-skipped run is a diagnostic, not a gate.

```sh
: "${RERANK_URL:?set RERANK_URL to a small OpenAI-style /v1/rerank endpoint}"
: "${EMBED_URL:?set EMBED_URL to a small OpenAI-style /v1/embeddings endpoint}"
BATTERY_LANE=http://127.0.0.1:8000 \
BATTERY_RERANK="$RERANK_URL" \
BATTERY_EMBED="$EMBED_URL" \
BATTERY_MODEL=ds4flash-vision \
python3 tools/mixed-battery.py
# expected: MIXED BATTERY PASS — 0 errors, the interactive floor held, and
# the co-tenant legs green. The battery also PRINTS the lowest MemAvailable
# it saw during the run: that figure is reported, not gated, because no
# floor is defined for it.
# The sample counts are derived from the battery's own loop constants:
# 8 interactive samples (2 workers x 4 rounds), 48 chat samples
# (8 workers x 6 rounds), 6 vision samples (2 workers x 3 rounds). A short
# count fails the gate — a missing sample is not a pass.
```

### 2.10 The warm-prefix replay

One stable prefix, cold call versus an immediate second call from the same
client. The second call's time-to-first-token ratio is the replay win.
Replay once, after the warmup settles — never on a schedule.

`tools/prefix-replay.py` is the gate. It sends the same payload twice,
checks that BOTH calls answered 200 with content, and reports the
warm/cold time-to-first-token ratio. (A `curl -s ... >/dev/null` loop
cannot do this: it exits 0 on a 500, so it would report a healthy replay
for an engine that answered every call with an error.)

```sh
python3 tools/prefix-replay.py --lane http://127.0.0.1:8000 \
  --model ds4flash-vision
# expected: PREFIX REPLAY PASS — every replay answered 200 with content,
# and the warm call did not regress

# to replay YOUR OWN captured prefixes instead of the synthesised one,
# pass one chat-completions request body per file:
python3 tools/prefix-replay.py --lane http://127.0.0.1:8000 \
  --model ds4flash-vision --payload prefixes/client-a.json
# expected: one cold/warm pair per payload

# to gate the WIN itself (not just success), add --require-speedup — use
# it only when the prefix is known to be long enough to be cached:
python3 tools/prefix-replay.py --lane http://127.0.0.1:8000 \
  --model ds4flash-vision --require-speedup
```

This recipe ships the gate and a synthesised prefix — it ships no client
payloads. The reader supplies their own with `--payload`.

### 2.11 The spec-decode short-prompt shape

The DSpark k=5 profile runs with `uniform_decode_query_len` = 6, and the
engine's uniform-decode shape check is pure arithmetic — a prompt of
exactly that length can be dispatched into the speculative-decode CUDA
graph instead of the prefill path. This gate sends a prompt of EXACTLY
that many tokens (built from the engine's own token IDs, so the count is
exact by construction) and asks whether the answer is sane.

Run it in a boot window, against a lane being accepted — never against a
lane carrying production traffic, and never as a bring-up step.

```sh
python3 tools/spec-decode-short-prompt-check.py \
  --base-url http://127.0.0.1:8000 --model ds4flash-vision
# expected: SPEC-SHORT-PROMPT PASS — a 6-token prompt answers sanely, and
# the engine reports 6 prompt_tokens
```

To take the trap length from the engine's own command line instead of the
shipped profile (run it on the box that owns the engine container — it
shells out to `docker exec` on that host):

```sh
python3 tools/spec-decode-short-prompt-check.py \
  --base-url http://127.0.0.1:8000 --model ds4flash-vision \
  --container ds4flash-vision-head
# expected: uniform_decode_query_len measured from the engine's own command
# line, then SPEC-SHORT-PROMPT PASS
```

The answer check is a marker echo plus structural checks. It does **not**
reject CJK output: this model legitimately emits CJK, and a script-level
script ban would fail a healthy lane for a cosmetic reason. A missing
marker echo fails the gate by default; `--allow-missing-echo` downgrades it
to a warning for a model that paraphrases.

### 2.12 The pass criteria

All the gates above must pass, in order. A failed gate is diagnosed
forward from the FIXLOG, never rolled back automatically, and never
retried with two variables changed at once.

## 3. The method — why the gates are shaped this way

### 3.1 Read the ledger first

`FIXLOG.md` records every lever tried and its verdict. A lever with a
recorded KEEP/REJECT is settled unless the stack changed (new image, new
checkpoint). One variable per boot; a boot that changes two things proves
nothing either way.

### 3.2 Harness before boots

Stage the probes on the head node before touching any knob (all shipped
under `tools/`, stdlib-only). Then:

- **Anti-contamination**: every prefill-class prompt embeds a fresh nonce
  at position ~1 so repeat runs cannot prefix-cache-hit and report cache
  replay as prefill.
- **Instructed recall needles — the non-negotiable rule.** A needle
  corpus that states the fact ("NOTE: the magic word is X") but asks no
  question measures the model's temperature-0 continuation style, not
  recall. Greedy decoding sits on a near-tie between echoing the filler
  and reciting the word; the nonce and cache/kernel numerics decide.
  Uninstructed misses at depth coexisted with 6/6 instructed passes on the
  same boot, minutes apart — and that chased a phantom "corruption
  boundary" the misses painted (FIXLOG rungs 22–23). Always append an
  explicit instruction; score word-exact.
- **Workload split** (spec-decode acceptance is workload-dependent):
  count/code-class single-stream, free prose single-stream, N-way
  aggregate sweep in both classes, prefill at a realistic deep-context
  size. One headline number hides all of it.
- **Corruption battery beside every speed run**: temp-0 exact-repeat,
  strict JSON, no non-ASCII drift.
- **One sacrificial deep prefill after health** — the first deep run of
  a boot pays a growing JIT tax. It must never land inside a measurement;
  label any number taken before the plateau `cold`.
- **Idle-gate every solo/tier/speed measurement** (§2.2). Any "slow"
  reading taken with traffic in flight is an unlabeled MIXED number, not
  a boot property: do not act on it, re-measure idle.
- **Quote the engine's own pool** (the boot log's KV-size line), never an
  estimate; band it ±10% — the pool is boot-labile.

### 3.3 Boot hygiene

Workers first, head last. Drop filesystem caches on every rank (unified
memory: page cache competes with the pool). Fresh `docker run` every
boot. Run the boot shape-warmup so no client eats a mid-serve JIT. No
background transfers during the window.

`/health` green proves only the API process, not the engine loop — a
fresh-boot memory trough can let the kernel OOM-kill one TP worker during
weight load while the endpoint keeps answering 200. After health turns
green, prove ALL ranks are alive with one tiny completion request before
any prefill-class traffic (wedge signature if you skipped this: survivor
GPUs pinned near-full utilization at single-digit-to-low watts, engine log
repeating a shared-memory broadcast timeout roughly every minute). And
never give the containers an auto-restart policy: a killed rank under one
becomes a crash loop re-applying boot patches on a reused filesystem —
force-remove and relaunch fresh instead.

### 3.4 Label everything — cold/warm, solo/mixed, same-boot

- **Cold vs warm**: the first deep exercise of a boot compiles
  deep-context kernels. Warm is the serving truth; cold is a boot
  property. The warm plateau reproduces within-boot to <1%.
- **Solo vs mixed**: prefill under concurrent decode is the serial
  resource; a deep prefill measured solo and the same depth measured
  mid-battery differ by a structural tax on this stack. Without the
  deferral patch, interactive streams drop to a few tok/s while a deep
  prefill runs; with it, the inflation drops by roughly an order of
  magnitude and the worst gaps land in the ~1 s class. The admission
  patch covers the third axis: an interactive request *arriving* while
  deep prefills hold the inflight cap queues for minutes without it.
  Publish the label with the number.
- **Same-boot A/B only**: identical config spreads ±10% across boots
  (environmental boot state — the kernel autotune cache was proven
  byte-frozen across spreading boots). Within-boot comparisons are valid;
  cross-boot deltas under ~13% are noise.
- **Advisory boot-lottery check**: after each fresh boot (and after
  prefix warm-up), fire one standard instructed needle and log the draw.
  A draw far below the band earns a "consider a restart window" note —
  advisory only; never auto-restart a serving lane (every boot wipes the
  prefix cache). The check is only meaningful idle-verified (§2.2): a
  far-below-band reading taken with traffic in flight is contamination,
  not a bad draw, and a contaminated draw must not trigger a restart
  consideration.

### 3.5 The acceptance gate is a mixed battery

A boot is not healthy because single-class probes pass. The gate
(`tools/mixed-battery.py`) runs interactive streams, chat, and image
requests **in flight at the same time**, plus two co-tenant service legs
(point them at any small OpenAI-style /v1/rerank and /v1/embeddings
services you host beside the engine; they stand in for whatever shares
your boxes), with host memory sampled throughout. Gate floors: the
interactive minimum rate, the chat mean, vision-in-flight, the tenant legs
green, and zero errors. The memory minimum is printed alongside the
verdict but is NOT a gate — no floor is defined for it, and a threshold
nobody chose is not a gate. The
co-tenant legs are what prove the headroom-by-design claim: a
tenants-skipped run is a diagnostic, not a gate — on fully dedicated boxes
the engine-surface legs are the gate and the headroom claim does not
apply.

### 3.6 Record as you go

Tee every cycle to a timestamped file; on failure capture all ranks' logs
before anything else. Per cycle record: the exact config delta, boot facts
(pool, time-to-healthy), all numbers **with their labels**, the decision,
and the trigger. End with a summary table and a do-not-retry map. Numbers
that exist only in a terminal scrollback do not exist.
