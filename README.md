---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
library_name: vllm
tags:
- serving-recipe
- vllm
- dgx-spark
- tensor-parallel
- vision
---

# DeepSeek-V4-Flash Vision-Exp at TP=3 across 3× NVIDIA DGX Spark — with native vision

**TL;DR** — Serve DeepSeek-V4-Flash-Vision-Exp (native weights, native
ViT + Aligner vision; per-request image count bounded by the engine flag,
not the model) tensor-parallel across three
DGX Sparks (GB10) on a CX-7 RoCE triangle, with DSpark k=5 speculative
decoding and a ported sequence-parallel prefill path. As far as we can
find this is the **first TP=3 serving of this checkpoint with the vision
tower live**. Warm numbers: **~99 tok/s** single-stream on code-class
streams, **~1K tok/s aggregate at the 32-stream cap**, a **~4.1M-token
KV pool ≈ 3.8 simultaneous full-1M-context sessions**, instructed-recall
verified **word-exact to ~950K tokens**, 8/8 images per request at
~0.7 s image-TTFT, and **~14× effective TTFT** on replayed warm prefixes.
Boot is ~5 min with ~110-class pre-warmed shapes (C=1–32 + sampler combos),
and the chunk-geometry JIT wedge class is **closed at the engine level** —
one compiled variant for every prefill geometry, no mid-serve JIT. The
defaults leave **~8–11 GiB/box of headroom** (tenant-free; the shipped 0.80 assumes no GPU/CPU tenants beside the ranks — with tenants, ship 0.78) — that
is part of the spec, not a leftover (**Headroom by design** below).

Just as valuable as the profile is the **measurement methodology** this
recipe standardizes — an instructed-recall needle rule, cold/warm and
solo/mixed labeling, a same-boot-only A/B discipline, and an idle-gate
on every solo measurement, each adopted after this campaign
demonstrably fooled itself without it (see **The methodology** and
`docs/FIXLOG.md`, which records the retractions).

## What this is

A serving profile for the Vision-Exp checkpoint — full native mixed
FP4+FP8 weights (~170 GB), the checkpoint-local ViT + Aligner vision
tower (user-message images only; per-request count is the engine's
mm-limit flag — default 8, validated far higher), and the trained DSpark
draft module — at **TP=3** over three Sparks, made possible by:

- the **attention-group pad 8→9 kit** ([localaiguyy](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark),
  MIT — credited and linked): o_group count 8→9 with heads-per-group
  constant, so TP=3 divides cleanly; and
- a **runtime image with a working multimodal serving class** for this
  model family (the pinned `dspark-vllm-gx10` image — most forks of this
  line serve text-only); and
- this repo's **SP-prefill applier** (`patches/`): a staged port of
  community sequence-parallel prefill semantics onto this runtime's
  V4.0-vision tree, plus the pad kit chaining.

The serving config of record: DSpark k=5 · 32 seqs (capture 192) ·
MNBT 8192 · long-prefill threshold 1024 (measured optimum) ·
adaptive chunk ON (solo deep prefills run as full ~8K chunks — the SP
engagement switch) · **prefill-decode deferral cadence N=2** (decode
fairness beside deep prefills — see below) · **interactive admission
ON** (arriving chats protected mid-bulk; requires the
`--scheduling-policy priority` serve flag — see below) · SP stage 3 ·
gmu 0.80 (tenant-free) · 1M context. Every axis carries a measured verdict or a recorded
blocker (`docs/FIXLOG.md`).

The recipe is checkpoint-agnostic within the Vision-Exp family: serve
the official release, or an abliteration derivative of it — our
validation ran on one such derivative (a rank-1 refusal-direction
projection build; pick your weights with your eyes open, and re-run the
refusal/sanity probe on whichever you choose).

## Quick start

This repo ships the profile (env), the SP patch layer, the acceptance
harness, and the measured why. Bring-up rides two public upstream repos:
the pad kit above and [MiaAI-Lab's 2-Spark DSpark recipe
line](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark),
which provides the compose/launcher machinery — checkout a pinned commit
and record the rev you validate against.

1. Stage the checkpoint on **all three** boxes (identical path):
```sh
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --revision 86f746b36186f0e567729a5c06a8c918caba82a9 \
  --local-dir /home/YOURUSER/models/deepseek-v4-flash-vision-exp
# (serving an abliteration derivative instead? stage that repo/revision and
#  keep any runtime-abliteration flag at 0 — the weights are already ablated)
```
2. Stage the [upstream 2-Spark DSpark recipe
   checkout](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
   on all three boxes (pinned to a recorded commit) and the pad kit into
   its patch dir; drop this repo's `patches/` beside it and chain the SP
   applier after the pad kit (`patches/README.md` — the `__main__` stanza
   IS the execution order).
3. Stage `config/cluster.env` as the checkout's env file; edit the
   `EDIT` blocks (hosts, checkpoint paths).
4. Workers first, then head. The ring serves `:8000` in ~5 min; the
   boot log must show the NCCL 3-rank banner, the pad lines, and the SP
   stage-3 line before anything else is diagnosed.
5. Run the gates (`docs/VALIDATION.md` has the full battery):
```sh
python3 tools/needle-probe.py --selftest
python3 tools/needle-probe.py --depths 131072,196608,262144 --nonce fresh1   # ~210K/~320K/~420K instructed ladder
DSPARK_WARMUP_MAX_CONCURRENCY=32 bash tools/boot-shape-warmup.sh http://127.0.0.1:8000 ds4flash-vision
python3 tools/agg-sweep.py --levels 1,4,16,32
BATTERY_LANE=http://127.0.0.1:8000 BATTERY_RERANK=... BATTERY_EMBED=... python3 tools/mixed-battery.py
```

Validated over a three-day bring-up + tuning campaign; every headline
number below was measured on this hardware and this image, under the
labeling discipline of `docs/VALIDATION.md`.

## The numbers (measured, labeled, rounded)

| Metric | Value |
|---|---|
| Solo deep prefill | **~1.5K tok/s @ ~420K context** (warm; cold-first pays ~1.6× JIT; ±10% per-boot — see lottery below) · ~1.6K @ ~320K · ~1.8K @ ~210K · ~3.2K shallow |
| Mixed-serving deep prefill | ~1.3K tok/s at the same depth (a structural **~20% mixed tax** — prefill is the serial resource; both alternatives measured worse) |
| Decode beside a deep prefill | **~3× TPOT inflation, worst gaps in the ~1 s class, median ≈ solo speed** (deferral cadence ON; 4-stream ~2×/stream) — was **~12× with multi-second freezes** before it; mixed prefill itself improves (~14% → ~7–9% under solo) |
| Chat arriving mid-deep-prefill | **single-digit-seconds to a full short reply** (admission ON; measured as completion wall on a ≤8-token answer, bounded below by the one chunk already in flight) — was **minutes-class** (queued behind the bulk work until it finished); bulk pays ≤ ~20% wall, bounded by two yields per request |
| Single-stream decode | **~99 tok/s** count/code-class · ~35 free prose (draft-acceptance-bound; acceptance ladder in the class row below) |
| Decode by generation class | accepted/drafted at k=5, per measured class: **~9/10** count/code · **~2/3** schema-bound tool calls · **~40%** reasoning (thinking on) · **~1/4** free prose — per-step token yield scales with (1 + 5 × acceptance) (wall-clock adds each class's step cost), so tool-heavy agentic traffic with thinking on runs the **~55–76 tok/s** class, not the count-class headline (speculative decoding stays active with thinking on) |
| Aggregate decode | **~1K tok/s @ 32-way** count-class (~720 @ 16) · ~230 @ 32 / ~180 @ 16 prose |
| KV pool | **~4.1M tokens ≈ 3.9 full-1M sessions** at gmu 0.80 tenant-free (measured 4,144,322; the session count is that pool divided by the 1,048,576-token window — a ratio of the same measurement, not a second one; boot jitter ±10%; the pool is the min across ranks) |
| Long-context recall | instructed needle **word-exact to ~950K tokens** (~1K tok/s prefill at that depth); 1M is configured and pool-proven |
| Vision | **8/8 images per request, in order**, at TP=3 · **~0.7 s image-TTFT** · multi-image color/position correct · the per-request image cap is the engine flag, not the model (validated upward to 128 small-image / 32 photo-scale — rung 37) |
| Warm-prefix replay | **~14× effective TTFT** (a stable ~24K prefix replays at ~2 s wall vs a cold effective rate ~14× slower) |
| Cold boot to serving | **~5 min**, ~110-class warmup shapes, **no mid-serve JIT** (shape ladder + C=1…32 + sampler combos pre-compiled; the chunk-geometry kernel needs no coverage — the geospec patch compiles it once for all prefill geometries) |
| Output integrity | temp-0 exact repeats byte-identical (incl. through the SP collectives); strict JSON clean; mixed battery 0 errors |
| Host headroom | **~8–11 GiB/box** at gmu 0.80 tenant-free (idle floors 7.7–8.9 GiB; deep-prefill floors 9.0–10.8 GiB under a 2×~420K needle battery; a ~742K full-context recall held 8.7+ GiB) — the precondition is NO GPU-side tenants on the serving boxes; with tenants the shipped value is 0.78 |

### Headroom by design — the profile shares the boxes

This is a working-stack recipe, not a whole-box benchmark. The shipped
defaults sit deliberately below the measured ceilings so the same three
Sparks keep carrying the rest of a live stack — embedders, document
processing, chat frontends. At the shipped gmu the boxes hold
~4–9 GiB steady with the lane serving, and a several-GiB floor holds
through a mixed battery with the co-tenant services **live** (the
acceptance gate's full form runs them in flight — a tenants-skipped run
is a diagnostic, not a gate). **gmu 0.78 is the shipped value on a
unified-memory box:** the GPU reservation and request-side CPU staging
draw from one pot, so a single very deep prefill (a ~30k tok/s prompt
burst) can stage ~5 GiB of CPU working set and starve the box while the
KV pool still reports free capacity — the 0.80 margin was too thin for
that class (recurring kernel-OOM events, each with a different trigger
and the same constant: headroom, not trigger; see landmine 15). 0.78
restores a measured floor (a 2×~420K needle battery held a ~6 GiB trough
with zero OOM) and keeps the pool at ~4.0M. **Rule: no GPU-side tenants
on a serving box at 0.80 — TP3 means every box carries a rank, so
there is no free box to disperse a tenant to; drop to 0.78 if one is
live.** Spend the headroom deliberately; every gmu step in the FIXLOG
records what it buys.

### Not tested / not claimed

Contexts beyond ~950K exercised (1M is configured and pool-proven);
multi-hour sustained soaks; TP sizes other than 3; re-quants of the
checkpoint; the official-vs-derivative refusal delta on any build but
the one we probed (re-run the refusal/sanity probe on your weights);
non-English OCR-class vision workloads beyond the synthetic color
fences.

### Decode fairness during deep prefills

The one mixed-load axis a chunk threshold alone cannot buy: an
interactive stream that happens to run while a client uploads a deep
(~400K-token-class) context. The scheduler's adaptive chunk lift
decides "no decode active right now → one full-size chunk", and that
presence check races with momentary decode absence (async scheduling,
spec-decode draft phases) — plus every chunk-carrying step costs its
full wall time even with decode co-batched. Felt effect before the
fix: the stream freezes for multiple seconds at a time mid-prefill.

`patches/apply_prefill_decode_interval.py` gates both on a
recent-decode window and forces decode-favorable steps between chunks
(`DSPARK_PREFILL_DECODE_INTERVAL=2`, one env flip to disable; 0/unset =
the patch is not even applied). Measured same-night both arms,
idle-gated: inflation ~12× → ~3×, worst gaps multi-second → ~1 s
class, median gap back at solo speed, mixed prefill ~5 points *faster*
— and outputs stay byte-identical. `tools/mixed-tax-probe.py` is the
instrument; the engine-agnostic tell for the underlying race is a
worst decode gap that matches your full-chunk quantum
(`max_chunk_tokens / prefill_rate`) instead of your configured
threshold.

### Interactive admission during deep prefills

The sibling axis: a request that *arrives* while deep prefills hold
the partial-prefill cap — not already decoding, so the cadence patch
cannot see it. Stock behavior queues it behind the bulk work: a chat
fired mid-bulk waits the minutes-class time for the first bulk to
finish. `patches/apply_interactive_admission.py` adds size-inferred
priority classes, a chunk-boundary yield (a waiting interactive request
makes one running bulk prefill step aside — at most twice per bulk,
ever, so bulk cannot starve), and the piece a measurement was needed
to find: a **waiting-interactive budget guard**. The yield alone frees
a cap slot, not token budget — with nothing decoding, the adaptive
lift hands the surviving bulk full-size chunks and it consumes the
whole step budget, so the yield-only cut measured slightly *worse*
than no patch. The guard holds chunks at the threshold while an
interactive request waits, and the chat lands in
**single-digit-seconds** (measured both arms, quiet idle-gated boots;
the floor is the one chunk already in flight when the request
arrives). Decode-active requests are never yield victims; deep-prefill
walls pay ≤ ~20% with live interactive traffic interleaved; needles
stay word-exact through the preempt-recompute-resume cycle — and a
yielded bulk's resume is a full recompute on a loaded stack, so don't
budget a cache-walk. Enable with `DSPARK_INTERACTIVE_ADMISSION=1` plus
the `--scheduling-policy priority` flag on the serve line (the flag
alone is inert with all-default priorities); `tools/fairness-probe.py`
is the instrument — together the two patches are interactive immunity:
running streams protected by cadence, arriving requests protected by
admission.

### The TP3 wedge, closed — reliability posture

The failure mode this stack cannot afford: a rank hits a Triton JIT
compilation **mid-serve** inside a TP collective, the collective waits
forever (this build's all-reduce backend sits outside the torch NCCL
watchdog), and the lane hard-wedges while `/health` keeps answering
200 — every in-flight request hangs. The first diagnosis blamed an
uncovered batch shape and answered with warmup arms; the wedge came
back. The real root cause: the prefill **chunk-metadata kernel is
specialized on the residue classes of its chunk-bound integers**
(Triton bakes value 1 as a constexpr, hints divisibility on multiples
of 16, and keys pointers by 16-byte alignment) — and those bounds are
arbitrary integers set by packed context depth × concurrency. Real
traffic mints new kernel-cache keys forever; no warmup can enumerate
that variant space.

`patches/hotfix-dsv4-prefill-chunk-geospec.py` closes the class at the
engine level: `do_not_specialize` on the two geometry ints plus
`do_not_specialize_on_alignment` on the sliced seq-lens pointer —
codegen-only, outputs bit-identical, one compiled variant for every
geometry (compiled during the stock boot warmup). Verified on a 3-box
lane: zero mid-serve JIT of the kernel across boot, the two concurrent
~52K-token prefill trigger, a full replay pass, and an extended
post-replay idle soak; a fixed set of deterministic cache variants per
box with zero growth under every later gate; correctness battery and
throughput band unchanged — and the first live use on the patched
engine, a very large cold context, prefilled and streamed correctly
with zero mid-serve JIT.

Around that close, the posture hardened on three more axes:

- **The warmup's concurrent deep-prefill arm now actually runs.** Its
  ~52K-token prompt bodies exceeded the kernel's per-argument size
  limit, so every arm died silently before the request fired (the
  error text landed where nothing read it). Bodies are staged as files
  (`-d @file`), making arm size independent of argv limits, and the
  arm tally is now self-counting — every scheduled request registers
  itself, so the summary can never disagree with what ran (a manual
  count once masked the swallowed failures behind a cosmetic
  mismatch).
- **The heartbeat watchdog is probe-confirmed.** On this build a
  healthy idle engine goes log-silent (dormant), so pure "no
  iteration lines" detection false-fired on every quiet lane. Stage 1
  is still the silence trigger; stage 2 is a tiny 1-token completion —
  a dormant engine answers in a fraction of a second, a wedged one
  hangs it — and only a failed probe recovers the lane. Missing
  evidence (an unresolvable model id) proves neither state and never
  triggers recovery; an absent container is never probed and never
  auto-booted from cold. The timestamped state file stays the
  watchdog's own liveness signal.
- **Warm-prefix replay: once per boot, after the warmup settles —
  never on a schedule.** A background replay's concurrent deep
  prefills were the historical wedge trigger (minting the geometry
  keys above), and the geospec close is verified under exactly that
  workload — a full replay pass plus an extended post-replay idle
  soak, zero mid-serve JIT — so replaying prefixes at boot is safe
  again on this stack. The pattern: replay each client's stable
  captured prefix once, after the shape warmup settles — from your
  own boot script or on demand; never on a schedule, so no cron. A
  replay gate you wire yourself keeps the verified shape:
  single-instance lock, bounded time budget, low priority so live
  traffic always outranks it (`DSPARK_BOOT_PREFIX_WARM` is the knob
  this profile reserves for it; the pinned upstream launcher ships
  none — with no gate wired, the key is simply unread). The on-demand
  loop below is the reference implementation, run with the idle-gate
  discipline in mind:

```sh
# on demand — re-POST each client's stable prefix once, watched:
for f in prefixes/*.json; do
  curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' -d @"$f" >/dev/null
done
```

One belt beyond the boxes: if clients sit behind a streaming proxy,
give it an inactivity deadline — a TCP-alive-but-silent upstream
should become a prompt transport failure, not an open stream
heartbeating a wedged engine.

## The methodology (this recipe's other product)

Eight rules, each adopted after this campaign fooled itself without it.
Steal them for any long-context serving work:

1. **Instructed recall needles — the community warning.** An
   instruction-less needle (filler + "NOTE: the magic word is X.", no
   question) measures the model's temperature-0 **continuation style**,
   not recall: greedy decoding sits on a near-tie between echoing the
   filler and reciting the word, and the nonce and kernel/cache
   numerics decide which wins. We measured uninstructed misses at depth
   coexisting with 6/6 instructed passes on the same boot — after days
   chasing the phantom "corruption boundary" those misses painted
   (thermal hypotheses, power cycles, allocation shifts; FIXLOG rungs
   22–23). Always append an explicit instruction; score word-exact — and
   judge an **echo-able** word. A short dictionary-style token is
   word-exact copy at temperature 0; a judged "word" assembled from a
   long random nonce (a timestamp-nonce monitor produces these
   unthinkingly) is echo-marginal — the model attends the needle and
   recalls the stem, truncates the random tail, and the judge reports
   context loss that never happened. Days of phantom "deep-prefill
   corruption" here traced to exactly that. Keep the nonce at corpus
   position 1 for cache-busting; never inside the judged token.
2. **Cold/warm JIT labeling.** The first deep exercise of a fresh boot
   compiles deep-context kernel specializations: ~1.6× slower at ~420K,
   worse deeper. An unlabeled first-run number is a boot property, not
   throughput. (Seven single-needle boots once read as a "2× deep-prefill
   regression" — it was seven cold reads.)
3. **Solo/mixed labeling.** A deep prefill measured solo and the same
   depth measured mid-battery differ by a structural ~20% here, and
   (before the deferral patch) interactive streams fall to a few tok/s
   while a deep prefill runs. Publish the label with the number.
4. **Per-boot kernel-tactic lottery — same-boot A/B only.** Identical
   config spreads ±10% across boots, non-overlapping, with the
   persistent autotune cache proven byte-frozen across the spreading
   boots: the residual is environmental boot state, not tactic
   selection (so "pin the tactics" pins nothing — FIXLOG). Cross-boot
   deltas under ~13% are noise on this stack.
5. **Mixed battery acceptance.** Healthy = interactive streams + chat +
   image requests + co-tenant services all in flight at once, memory
   sampled throughout — not single-class probes in isolation.
6. **Harness self-test.** Offline corpus assertions + a live micro-
   needle before every session (`needle-probe.py --selftest`). Both
   harness bugs this history records shipped before it existed.
7. **Advisory boot-lottery check.** One standard needle per fresh boot,
   logged; a far-below-band draw earns a "consider a restart window"
   note — advisory only, because every boot wipes the prefix cache
   (never auto-restart a serving lane). Only meaningful idle-verified
   (rule 8): a far-below-band reading taken with traffic in flight is
   contamination, not a bad draw — a contaminated draw must not trigger
   a restart consideration.
8. **Idle-gate every solo measurement.** Before ANY solo/tier/speed
   reading, GET `/metrics` on the engine and require
   `vllm:num_requests_running` == 0 **and** `vllm:num_requests_waiting`
   == 0, immediately before measuring. A "slow" reading taken with any
   traffic in flight — your own warmup replays, scheduled jobs,
   co-tenant requests, other clients — is an unlabeled MIXED number,
   not a boot property: don't act on it, re-measure idle. The trap:
   post-boot warmup/replay traffic overlaps exactly the window where a
   fresh boot first gets probed — let warm traffic settle or verify
   idleness before believing any first-look number. (A day was lost to
   a phantom "slow tier" that was exactly this; FIXLOG.)

`docs/VALIDATION.md` turns these into the gate battery; `docs/FIXLOG.md`
records what each rule cost to learn.

## Variants (each is one env flip)

- **Long-document**: MNBT 6144 + the image's sparse-MLA chunk hotfix →
  **~3.9M pool** at ~−14% aggregate — the recall-margin shape for
  deep-research days. The hotfix is load-bearing at 6144 (deep recall
  fails without it on this stack).
- **seqs 16** (capture 96): lighter capture bill, ~720 @ 16-way — the
  interactive-friendlier shape if you never reach 32 streams.
- **Max-pool batch**: MNBT 4096, no-spec-style usage → the biggest pool
  at ~−70% single-stream (the spec pipeline starves; measured both
  ways). Know why you're choosing it.
- **Stage-4 indexer row split** (`--stages 4`): correctness-clean
  (every instructed gate + vision + drift-aware split-check pass), a
  small pool side-win, **no measured speed case on this stack** —
  documented variant, kept in-tree for the eventual upstream-fixed
  runtime. The `--stages 3` apply emits zero stage-4 bytes.

## Working with long contexts

- **TTFT expectations**: ~25K cold ≈ 15 s, ~60K ≈ 35 s at the solo rate;
  the *first* deep prefill after boot pays the JIT premium (pre-warm or
  label). Warm-prefix replay is the felt-speed lever: keep client system
  prompts and repo context stable — a replayed ~24K prefix answers ~14×
  faster (effective rate). Replay captured client prefixes once per
  boot, after the warmup settles — from your own boot script or the
  on-demand pattern in the reliability section — and never put replay
  on a cron-style schedule. Never restart the lane casually: the boot
  wipes the prefix cache for every client at once.
- **Exercise your real depth before trusting it.** Recall here is
  verified to ~950K (instructed, word-exact); the pool math holds ~3.8
  full windows. If you serve near the envelope, run the ~950K probe
  yourself — it is one command and ~15 min.
- **Thinking mode** defaults high on this family's reasoning parser:
  budget the reasoning stream inside `max_tokens`, or send
  `chat_template_kwargs: {"thinking": false}` for latency-bound loops.
  On this runtime the thinking stream arrives as **`reasoning`** —
  streaming `delta.reasoning`, non-streaming `message.reasoning` — not
  `reasoning_content`; a client reading the wrong field silently
  under-counts what it is paying for. Speculative decoding stays active
  while thinking runs, so a tool-heavy agentic client should expect the
  ~55–76 tok/s class in the numbers table above, not the count-class
  headline.
- **Concurrency**: interactive streams sit in the ~99 tok/s class at low
  concurrency; aggregate climbs to ~1K at the 32-way cap (per-stream
  ~30 — batch class, not interactive).

## vs. other published lanes — honest scoreboard

- **The pad kit's own text-only 3-Spark recipe** ([localaiguyy,
  above](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark)):
  ~620 tok/s @ 32-way on the text checkpoint. This lane: **~1K @ 32 with
  native vision and a verified 1M window** (channels/seqs/caches ladder
  — FIXLOG), on top of the same pad geometry.
- **Best published 2-Spark Vision-Exp lane** ([tonyd2wild's
  DSpark/NVFP4-KV
  recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark)):
  ~84 tok/s peak single-stream, ~182 @ 6-way, 2.79M pool, TTFT 4.1 s
  @8K → 29.5 s @100K. This lane: faster single-stream (~99 vs ~84),
  ~4× the concurrent-stream ceiling, ~1.5× the pool — for 1.5× the boxes.
- **V4.1 native TP=3 reference** ([christopher_owen's
  thread](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438)):
  ~65–78 single-stream, 171–196 @ 8-way, prefill flat ~3.8K tok/s — his
  SP-prefill patches + driver-level memory work ride the V4.1
  checkpoint's cheaper prefill architecture. This recipe ports the SP
  concepts onto the V4.0-vision tree and measures what survives: the
  prefill gap (~1.5K vs ~3.8K) is checkpoint generation, not a flag.
- A companion text-only V4-Flash TP=3 recipe from the same authors
  carries the text-lane details (CSF packing, profile split) this
  checkpoint doesn't need.

Net: three boxes buy **native vision at TP=3 with 32-way aggregate
headroom, ~3.8 full-1M sessions, and verified ~950K recall**. Prefill
remains the frontier — nobody's 3-box V4.0 prefill matches the best
patched V4.1 lanes, and neither does ours.

## Hardware / network this was proven on

- 3× DGX Spark (GB10, sm_121, ~122 GiB usable **unified** memory each —
  CPU, containers, and page cache all draw from the same pool the engine
  profiles)
- Full CX-7 triangle: three QSFP cables, one /24 per leg, RoCEv2, GID
  index 3, MTU 9000; management LAN for bootstrap; NCCL pinned to both
  on-board HCAs per rank, cross-NIC
- 8 NCCL channels (the measured 3-rank interior optimum)
- One fabric address per HCA port, always — a duplicate /24 left on two
  ports by a stale profile is a per-boot routing lottery that reads
  clean at the link layer; after any fabric change, audit every box
  with `ip -br addr`

## Stack (pinned)

| Component | Pin |
|---|---|
| Runtime image | `ghcr.io/anemll/dspark-vllm-gx10` @ `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8` (the v0.1.1 release line — the only public image of this line with a working multimodal class for this family; re-pin ⇒ re-validate the patches) |
| Checkpoint | any Vision-Exp packaging; validated with the official repo @ `86f746b36186` (and an abliteration derivative of it — see above) |
| TP=3 pad | the localaiguyy kit (MIT), applied by the checkout at `TP_SIZE=3` |
| Boot machinery | the [MiaAI-Lab 2-Spark DSpark recipe checkout](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) @ `97e8733238f8` (compose/launcher + env shape; stage it on all three boxes — Quick start step 2, and pin whatever rev you validate against) |
| SP prefill | this repo's `patches/apply_sp_prefill.py`, stage 3, chained after the pad kit |
| Decode fairness | this repo's `patches/apply_prefill_decode_interval.py` at N=2 (`DSPARK_PREFILL_DECODE_INTERVAL`; 0/unset = not applied) |
| Interactive admission | this repo's `patches/apply_interactive_admission.py` (`DSPARK_INTERACTIVE_ADMISSION=1` + `--scheduling-policy priority` on the serve line; 0/unset = not applied) |
| Chunk-geometry JIT close | this repo's `patches/hotfix-dsv4-prefill-chunk-geospec.py` (`do_not_specialize` on the chunk-metadata kernel's geometry ints — one variant for all prefill geometries; skip with `DSPARK_SKIP_PREFILL_CHUNK_GEOSPEC_HOTFIX=1`) |
| Spec decode | DSpark k=5 probabilistic (trained in-checkpoint drafter) |
| KV | image-default `nvfp4_ds_mla` (fp8 measured and rejected — rung 10) |

## The landmines (read before booting)

Full detail per rung in `docs/FIXLOG.md`.

1. **The pad mechanism is group-count 8→9, heads-per-group stays 8.**
   The kit's own docs contradict each other; the applier's code is
   authoritative (r stays 8×512 = 4096 — the fp8 einsum kernel contract).
   Read the code that runs.
2. **Pad-group attention sinks are −inf, never zero.** A zeroed sink adds
   a phantom softmax denominator term that attenuates the real heads:
   grammatical, confident, WRONG output. Silent corruption, not a crash.
3. **Don't import another family's pad-backend env path** — this model
   rejects that backend — and never route the sink through a zero-fill
   helper.
4. **Unified memory means the startup check wants gmu × ~122 GiB FREE per
   rank**: co-tenants and dirty page cache count against it. Drop caches
   on every rank pre-boot; set gmu by the most-shared box. The pool is
   the min across ranks — per-rank gmu forks mathematically cannot raise
   it.
5. **`MAX_NUM_BATCHED_TOKENS` is profile-specific**: 4096 starves spec
   decode (~−70% single-stream); 6144 breaks deep recall without the
   sparse-MLA chunk hotfix; 16384 burns ~40% of the pool for nothing.
   Spec boots keep 8192.
6. **Capture size = seqs × (k+1)** — 192 at the shipped shape.
7. **k is not a dial**: the trained block is exactly 5 (k=6 collapses
   acceptance; k<5 truncates the block). Measured from both sides.
8. **Verify env vars actually landed in the container** before reading a
   null result — a silent compose passthrough miss reads exactly like
   "the knob doesn't help" (it did, ~+20%, once it actually applied).
9. **Thinking defaults ON/high** — see the long-context section, or your
   first low-`max_tokens` request looks "broken".
10. **Label cold runs**: the first deep prefill of a boot pays ~1.6× JIT,
    growing with depth. Pre-warm it or label it; never report it as
    throughput.
11. **±10% per-boot lottery**: same-boot A/B only; cross-boot deltas
    under ~13% are noise here. The advisory per-boot needle check
    catches a bad draw — but only when the engine is idle when checked
    (see 18).
12. **A boot wipes the prefix cache** for every client at once — replay
    prefixes after every boot; never restart a serving lane casually.
13. **Never gate recall on an instruction-less needle** (the methodology
    section; this single rule cost days).
14. **SP engagement is chunk-shape-gated**: below MIN_ROWS (4097) rows
    everything is stock — mixed serving runs stock **by design**; solo
    deep prefills engage SP only as full ~8K chunks (adaptive chunk ON).
    The engine's stage/split log lines are the engagement proof; the
    mixed-axis alternatives were measured and rejected.
15. **gmu 0.78 is the shipped value; 0.80 is a co-tenant-only gamble.**
    On a unified-memory box one deep prefill's CPU staging shares the
    same pot as the GPU reservation — a single ~30k tok/s prompt burst
    can starve a box while the KV pool still reports free capacity
    (recurring kernel-OOM events, each with a different trigger; the
    constant is headroom, not trigger). The rule: no GPU-side tenants on
    a serving box at 0.80; at 0.78 a measured 2×~420K battery held a
    ~6 GiB trough with zero OOM.
16. **Boot hygiene**: workers first, head last; fresh `docker run` never
    `restart`; one GPU owner per box; drop caches every rank; a NIC
    stuck slow or a GPU latched below clock is fixed by a cold power
    drain, not a warm reboot.
17. **Probe hygiene**: never probe a box's GPU from inside the engine
    container during a gate; read `usage.prompt_tokens` (filler token
    estimates run ~1.6× low); never compare prefill across filler
    classes (repetitive haystacks read 2–3× fast).
18. **Idle-gate solo measurements**: before ANY solo/tier/speed reading,
    GET `/metrics` and require `vllm:num_requests_running` == 0 and
    `vllm:num_requests_waiting` == 0, immediately before measuring. A
    "slow" reading taken with traffic in flight — your own warmup
    replays, scheduled jobs, co-tenant requests, other clients — is an
    unlabeled mixed number, never a boot property: don't act on it,
    re-measure idle. Post-boot warmup/replay traffic overlaps exactly
    the window where a fresh boot first gets probed; let it settle or
    verify idleness.
19. **A dead rank still serves `/health` 200** — the health endpoint
    proves the API process, not the engine loop. Wedge signature:
    survivor GPUs pinned near-full utilization at single-digit-to-low
    watts at full clocks, and the engine log repeating a shared-memory
    broadcast timeout roughly every minute. Fresh-boot memory troughs
    can let the kernel OOM-kill one TP worker during weight load: verify
    per-rank headroom before boot, and after health turns green prove
    ALL ranks are alive with one tiny completion request before any
    prefill-class traffic. One killed rank + a container auto-restart
    policy = a crash loop (each restart re-applies boot patches on a
    reused filesystem, and patch kits are not guaranteed idempotent
    under that) — force-remove the containers and launch fresh, never
    just restart.
20. **Exactly one fabric address per HCA port**: a stale profile that
    leaves the same /24 on two ports makes each boot pick the route
    between them nondeterministically — a per-boot performance lottery
    that looks clean at the link layer (link state, error counters,
    bandwidth tests all green). After any fabric change, audit every
    box with `ip -br addr`.
21. **The admission patch needs its policy flag**: the knob and
    `--scheduling-policy priority` travel together — a mismatch logs a
    loud `[dspark-iaa]` warning and disables interactive admission
    (fail-visible, never silent; the engine itself is unaffected). And
    remember the budget hole when
    tuning: freeing an inflight-prefill **slot** frees no token
    **budget** — a step whose running loop consumes the whole
    `max_num_batched_tokens` on one full-size chunk admits nothing
    from the waiting queue no matter how many slots are open
    (`tools/fairness-probe.py`'s minutes-class chat wall with free slots
    is exactly this signature).

22. **Negative request priority starves the request itself**: on an
    idle engine, a ten-token prompt sent with `priority: -1000` waited
    tens of seconds before serving — sub-second at default priority
    (negative values are not "high priority" here; don't reason from
    the sign). Bulk and warm-up scripts need no priority tricks: the interactive-
    admission patch (landmine 21) already bounds bulk prefill's impact
    on interactive streams.
23. **A TP3 wedge presents as healthy**: the API keeps answering 200
    while every request hangs — check the engine's 10 s iteration lines,
    not /health. The known cause was a mid-serve Triton JIT stalling a
    rank inside a TP collective forever on this stack's PYNCCL backend:
    the chunk-metadata kernel specializes on the residue classes of its
    chunk-bound integers, an unbounded variant space no warmup can
    enumerate. The geospec patch (rung 39) is the permanent close — one
    compiled variant for every geometry; keep the warmup for the
    enumerable shape classes, and run the probe-confirmed heartbeat
    watchdog (rungs 38–40) so anything unknown self-heals in minutes
    instead of hanging your users; monitor the watchdog's state file
    for the double-failure case.

## Repo layout

- `config/cluster.env` — the serving profile of record (every pin +
  the measured-why comment)
- `patches/apply_sp_prefill.py` — the staged SP-prefill applier
  (marker-guarded, idempotent, fail-closed, `--check`/`--revert`)
- `patches/apply_prefill_decode_interval.py` — the decode-fairness
  deferral-cadence applier (env-gated, idempotent, anchor-enforced
  ordering after adaptive-chunk + issue-43)
- `patches/apply_interactive_admission.py` — the interactive-first
  admission applier: size-inferred classes + chunk-boundary yield +
  the waiting-interactive budget guard (env-gated, idempotent,
  anchor-enforced ordering after the deferral applier)
- `patches/hotfix-dsv4-prefill-chunk-geospec.py` — the chunk-geometry
  JIT close: `do_not_specialize` on the prefill chunk-metadata kernel's
  geometry ints, one compiled variant for every geometry (env-gated,
  idempotent, fail-closed — the TP-wedge root-cause fix)
- `patches/README.md` — stage semantics, chaining, engineering
  constraints, credits
- `tools/needle-probe.py` — the instructed-recall ladder + vision fence
  + self-test (the correctness instrument)
- `tools/agg-sweep.py` — aggregate sweep with the workload split +
  corruption spot-check
- `tools/mixed-tax-probe.py` — decode-beside-deep-prefill probe (the
  decode-fairness instrument)
- `tools/fairness-probe.py` — the admission probe: chat + decode stream
  fired mid-bulk, needle correctness through preempt-resume (the
  admission-fairness instrument)
- `tools/mixed-battery.py` — the mixed-surface acceptance battery
  (interactive + chat + vision + co-tenant legs, memory floor)
- `tools/boot-shape-warmup.sh` — boot-time shape/JIT warmup
  (~110-class shapes through C=32, tokenize-gated ladder, sampler-combo
  postcondition, a concurrent deep-prefill pair with file-staged
  bodies and a self-counting tally)
- `tools/engine-heartbeat-watchdog.sh` — TP3 deadlock self-healer,
  probe-confirmed: a silence trigger plus a 1-token confirmation probe
  (healthy idle dormancy never false-fires) detects the wedge
  signature (API alive + silent engine heartbeat), a SPIN-PREFILL
  detector catches the engine-responsive variant (a request stuck
  re-processing a prompt: running > 0 with near-zero generation,
  sustained), and the recovery path captures py-spy stacks before
  teardown (the root-cause instrument) — all unattended (the wedge-recovery
  ladder is in `docs/FIXLOG.md`; the spin-prefill detector and the py-spy
  wedge-stack capture have no published rung);
  the default engine port is :8000 (the recipe's serving port)
- `docs/FIXLOG.md` — 41 rungs, four ladders, including the retractions
- `docs/VALIDATION.md` — the measurement methodology + gate battery
- `LICENSE` — MIT (upstream licenses apply to upstream-derived code)
- `.gitignore` — keeps logs and Python caches out of the tree

## What's next (roadmap, not needed for correctness)

- **Prefill** stays the frontier and is checkpoint-bound: every shipped
  dial on this image measured into one ~1.5K-class band; the 2×-class
  unlocks are the V4.1 generation's cheaper prefill architecture, not a
  patch onto V4.0.
- **Stage-4 indexer split** — correctness-clean, no speed case here;
  reopens only with an upstream change or a comm-bound workload class.
- **Kernel-tactic pinning** — open but narrowed: the autotune cache was
  proven frozen across the ±10% spreading boots, so classic pinning
  pins nothing; the true lever (environmental boot state) is unpinned.
- **Upstream watch**: the pinned image line (v0.1.1 era) — the digest
  pin plus the applier's fail-closed anchors make any drift loud;
  re-pin deliberately. When an image of this line ships the
  chunk-metadata specialization fix engine-side, the geospec applier
  becomes redundant — its anchor refuses loudly on a diverged tree, so
  check its `--status` after any re-pin.
- **Multi-hour soak**: community reports on this image family include
  garble-class issues under sustained spec-decode load; our batteries
  and shorter soaks are clean, sustained windows are unproven — run one
  before trusting any lane of this class unattended.
- **CSF/compressed-scales packing** of this checkpoint when a public
  CSF-capable runtime ships (the companion text-only recipe line
  carries the toolchain pattern).

## Credits & prior art

- **localaiguyy** — [the attention-group pad kit](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark)
  (MIT): the 8→9 pad geometry this lane stands on, the TP3-shape
  statement of the capture = seqs × (k+1) sizing rule, and the
  zeroed-sink landmine class.
- **Anemll** — the `dspark-vllm-gx10` runtime: the DSpark spec-decode
  stack and the working multimodal serving class that makes vision at
  TP>1 possible at all.
- **MiaAI-Lab** — [the 2-Spark DSpark vision recipe
  line](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  whose env shape and boot machinery (including the warmup script this
  repo's `tools/boot-shape-warmup.sh` adapts, with C=8/16 arms added)
  this profile descends from, and the DS4-on-Spark bug documentation.
- **tonyd2wild** — the [2-Spark Vision-Exp DSpark/NVFP4-KV
  recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
  we benchmark against, and the origin of the capture = seqs × (k+1)
  sizing rule this recipe states for its own shape.
- **christopher_owen** — the [V4.1 TP=3 reference
  thread](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438):
  the prefill-SP patch-series semantics this applier adapts, and the
  3-Spark numbers we race.
- The companion text-only V4-Flash TP=3 recipe (same authors) — carried
  rules: the MNBT two-truths, the capture cliff, the boot-hygiene set.

Ours: the vision-at-TP=3 composition itself; the SP-prefill port onto
the V4.0-vision tree (stages 1–3, the documented stage-4 variant, and
the plain-ints/aux-stream engineering constraints); the instructed-recall
instrument and the re-adjudication methodology; the cold/warm,
solo/mixed, and same-boot labeling rules; the mixed battery with live
co-tenants; the boot-lottery advisory check; the chunk-geometry
specialization close and the probe-confirmed watchdog posture; the
measured landmine ledger. Scripts: MIT (see `LICENSE`); upstream
licenses apply.
