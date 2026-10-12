# How this recipe validates claims (methodology)

Every number in the README and FIXLOG was produced under this discipline.
Reproduce it and your numbers are comparable to ours; skip any part and
they aren't. The crash ladders live in `FIXLOG.md`; this file is the
measurement method — and three of its rules (instructed needles, cold/warm
labels, same-boot A/B) were learned the hard way, by publishing wrong
inferences first and retracting them.

## 1. Read the landmine ledger first

`FIXLOG.md` records every lever tried and its verdict. A lever with a
recorded KEEP/REJECT is settled unless your stack changed (new image, new
checkpoint). One variable per boot; a boot that changes two things proves
nothing either way.

## 2. Harness before boots

Stage the probes on the head node before touching any knob (all shipped
under `tools/`, stdlib-only):

- **Self-test first** (`needle-probe.py --selftest`): offline corpus
  assertions plus one tiny live instructed needle. Run it before every
  measurement session — both harness bugs this recipe's history records
  (an instruction-less corpus, a mid-corpus nonce) shipped before a
  selftest existed.
- **Anti-contamination**: every prefill-class prompt embeds a fresh nonce
  at position ~1 so repeat runs cannot prefix-cache-hit and report cache
  replay as prefill.
- **Instructed recall needles — the non-negotiable rule.** A needle
  corpus that states the fact ("NOTE: the magic word is X") but asks no
  question measures the model's temperature-0 continuation style, not
  recall. Greedy decoding sits on a near-tie between echoing the filler
  and reciting the word; the nonce and cache/kernel numerics decide.
  We measured uninstructed misses at depth coexisting with 6/6 instructed
  passes on the same boot, minutes apart — and wasted days chasing the
  phantom "corruption boundary" those misses painted (FIXLOG rungs
  22–23). Always append an explicit instruction; score word-exact.
- **Workload split** (spec-decode acceptance is workload-dependent):
  count/code-class single-stream, free prose single-stream, N-way
  aggregate sweep in both classes, prefill at a realistic deep-context
  size. One headline number hides all of it.
- **Corruption battery beside every speed run**: temp-0 exact-repeat,
  strict JSON, no non-ASCII drift.
- **One sacrificial deep prefill after health** — the first deep run of
  a boot pays ~1.6× in JIT (growing with depth). It must never land
  inside a measurement; label any number taken before the plateau `cold`.
- **Idle-gate every solo/tier/speed measurement**: GET `/metrics` on the
  engine and require `vllm:num_requests_running` == 0 **and**
  `vllm:num_requests_waiting` == 0, immediately before measuring. Any
  "slow" reading taken with traffic in flight — your own warmup
  replays, scheduled jobs, co-tenant requests, other clients — is an
  unlabeled MIXED number, not a boot property: don't act on it,
  re-measure idle. The trap: post-boot warmup/replay traffic (a
  ~4-minute automated prefix-replay class starting on the first
  scheduler tick after boot) overlaps exactly the window where a fresh
  boot first gets probed — let warm traffic settle or verify idleness
  before believing any first-look number.
- **Quote the engine's own pool** (the boot log's KV-size line), never an
  estimate; band it ±10% — the pool is boot-labile.

## 3. Boot hygiene

Workers first, head last. Drop filesystem caches on every rank (unified
memory: page cache competes with the pool). Fresh `docker run` every
boot. Run the boot shape-warmup (`tools/boot-shape-warmup.sh`) so no
client eats a mid-serve JIT. No background transfers during the window.

`/health` green proves only the API process, not the engine loop — a
fresh-boot memory trough can let the kernel OOM-kill one TP worker
during weight load while the endpoint keeps answering 200. After health
turns green, prove ALL ranks are alive with one tiny completion request
before any prefill-class traffic (wedge signature if you skipped this:
survivor GPUs pinned near-full utilization at single-digit-to-low
watts, engine log repeating a shared-memory broadcast timeout roughly
every minute). And never give the containers an auto-restart policy: a
killed rank under one becomes a crash loop re-applying boot patches on
a reused filesystem — force-remove and relaunch fresh instead.

## 4. Label everything: cold/warm, solo/mixed, same-boot

- **Cold vs warm**: the first deep exercise of a boot compiles
  deep-context kernels. Warm is the serving truth; cold is a boot
  property. The warm plateau reproduces within-boot to <1%.
- **Solo vs mixed**: prefill under concurrent decode is the serial
  resource — a deep prefill measured solo reads ~1.5K tok/s on this
  stack, the same depth measured mid-battery ~1.3K (a structural ~20%
  mixed tax; without the deferral patch, interactive streams drop to a
  few tok/s while a deep prefill runs — with it, ~3× inflation and
  worst gaps in the ~1 s class). Publish the label with the number.
- **Same-boot A/B only**: identical config spreads ±10% across boots
  (environmental boot state — the kernel autotune cache was proven
  byte-frozen across spreading boots). Within-boot comparisons are
  valid; cross-boot deltas under ~13% are noise.
- **Advisory boot-lottery check**: after each fresh boot (and after
  prefix warm-up), fire one standard instructed needle and log the draw.
  A draw far below the band earns a "consider a restart window" note —
  advisory only; never auto-restart a serving lane (every boot wipes the
  prefix cache). The check is only meaningful idle-verified (§2): a
  far-below-band reading taken with traffic in flight is contamination,
  not a bad draw, and a contaminated draw must not trigger a restart
  consideration.

## 5. The acceptance gate is a mixed battery

A boot is not healthy because single-class probes pass. The gate
(`tools/mixed-battery.py`) runs interactive streams, chat, and image
requests **in flight at the same time**, with reranker/embedder
co-tenant services live when you run the full form, and host
`MemAvailable` sampled throughout. On this stack the gate reads:
interactive floor ~12 tok/s (min) with ~22 mean, chat ~14 mean,
vision-in-flight 6/6, tenant legs green, 0 errors, and a several-GiB
memory floor on the head box. A tenants-skipped run is a diagnostic,
not a gate.

## 6. The gate battery (what "accepted" means here)

In order, on the serving boot (every solo step runs only after the
idle-gate in §2 passes):

1. `needle-probe.py --selftest` — harness integrity.
2. **Instructed needle ladder** at ~210K / ~320K / ~420K (word-exact,
   fresh nonce each; the judged word is the tool's short default token —
   never a nonce-salad string, see FIXLOG rung 34), plus a **~950K
   envelope probe** when the 1M window matters to you (~15 min wall;
   ~1K tok/s class at that depth).
3. **Vision fence**: single-image TTFT (~0.7 s class) and the 8-image
   cap — all eight left-colors named in order at TP=3.
4. **Aggregate sweep** (`agg-sweep.py`) at 1/4/16/32 in both prompt
   classes, corruption spot-check green after.
5. **Mixed-tax probe** (`mixed-tax-probe.py`, idle-gated, on the head
   node): decode beside a ~440K fresh-nonce prefill — expect ~3× mean
   inflation, worst gaps in the ~1 s class, median ≈ solo (a ~12× mean
   with multi-second gaps means the deferral knob is off or was not
   applied; check `DSPARK_PREFILL_DECODE_INTERVAL` and the patch's
   `--status` inside the container).
6. **Mixed battery** with the interactive floor (above).
7. **Warm-prefix replay**: one stable ~24K prefix, cold call vs
   second-call TTFT — the replay win on this stack is ~14× effective.

## 7. Record as you go

Tee every cycle to a timestamped file; on failure capture all ranks'
logs before anything else. Per cycle record: the exact config delta, boot
facts (pool, time-to-healthy), all numbers **with their labels**, the
decision, and the trigger. End with a summary table and a do-not-retry
map. Numbers that exist only in a terminal scrollback don't exist.
