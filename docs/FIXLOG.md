# FIXLOG - the crash ladder from nothing to first light

A three-day bring-up + tuning campaign (plus the operational lessons
that came after it): four ladders, thirty-three rungs of distinct root
causes, each diagnosed from container logs and fixed forward.
Recorded so nobody repeats this grind. This is what "nobody has ever run
this model with native vision at TP=3" looks like from the inside.

Two things this ledger does that most don't: it records **retractions**
(rungs 22-24 reverse earlier rungs' inferences when the measurement
instrument itself turned out to be the bug), and it labels every number
**cold/warm and solo/mixed** (rung 21) - an unlabeled deep-prefill number
on this stack is uninterpretable.

## The first ladder: bring-up (the composition boot, rungs 1-4)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | third box refuses to boot: image absent | the vision-capable image had only ever run on the other two boxes (the TP=2 lanes never touched it) | transfer box-to-box (a workstation-mediated docker save\|load pipe runs ~7 MB/s); save-by-digest re-canonicalizes the image ID - when the ID changes, verify content equality by layer diff_ids |
| 2 | one rank crash-loops at startup (~18 restarts) | vLLM's startup check needs `gmu x ~122 GiB` of **free unified memory** per rank; hundreds of GB of staging transfers had dirtied page cache, and co-tenant services legitimately hold ~28 GiB on the most-shared box | drop caches on every rank pre-boot; set gmu by the binding (most-shared) box - the constraint is the design, not an inconvenience |
| 3 | the pad kit's own docs contradict each other (pad heads-per-group vs pad group count) | the kit ships two docstrings; only the code the applier injects actually runs | the operative mechanism: pad the attention **o_group count 8->9** (9 % 3 = 0), heads-per-group **stays 8** (total heads 64->72; r stays 8x512 = 4096 - the fp8 einsum kernel contract forbids changing it). Read the code that runs, not a docstring in an adjacent file |
| 4 | could a per-rank gmu lift the pool? | the KV pool is the **min across ranks**; the most-shared box binds forever | no - per-rank forks mathematically cannot raise it (the min-rank law). SKIP, don't build |

## The second ladder: the fence - concurrency, collectives, KV (rungs 5-14)

One variable per boot, harness staged before any knob was turned, keep or
revert per rung.

| # | Symptom / goal | Root cause / finding | Fix / outcome |
|---|---|---|---|
| 5 | seqs 6->16 | capture 96 costs ~12% of the pool; aggregate gains ~+85% at 16-way with single-stream parity | ADOPT at the time (the interactive-friendly step) |
| 6 | seqs 32 (capture 192 = 32 x (5+1)) | the aggregate prize - and the honest price ordering once collectives widened (see rung 11) | ADOPT - the shape of record; beats the pad kit's own published 3-box aggregate on this stack |
| 7 | MNBT 4096 | +~35% pool but single-stream **-~70%**: the draft/verify pipeline starves inside the smaller token budget | REJECT for spec-decode boots (recorded as a max-pool batch variant) |
| 8 | MNBT 6144 | +~25% pool, speed parity - and the deep needle failed bare (chunk-boundary x KV-path interaction); **with the image's sparse-MLA chunk hotfix it passes** | REJECT as default; 6144 + hotfix = the documented long-document variant (~3.9M pool at ~-14% aggregate) |
| 9 | MNBT 16384 | the scheduler's boot warning clears (cosmetic for this workload); pool -~40%; recall not broken at 16384 | REJECT |
| 10 | fp8 KV vs the image-default nvfp4 | pool and speed parity; nvfp4 measured >= fp8 on aggregate even with the chunk hotfix riding along | REJECT - keep the image default. (The original deep-recall evidence against fp8 predates the instructed needle - rung 23 - treat the throughput side as the durable ground) |
| 11 | NCCL channels 4->8 | **first attempt was a FALSE NULL** - compose silently didn't pass the vars; discovered by dumping the live container env. Real boot: ~+20% aggregate, reproduced on a second boot; an attribution boot proved channels alone delivers the full win | ADOPT 8 |
| 12 | NCCL channels 16 | -~8% aggregate | REVERT - 8 is the measured interior optimum {4, 8, 16} |
| 13 | k=6 draft depth | acceptance collapses (~2/3 vs ~9/10 accepted/drafted at k=5); single-stream -~12% | REJECT - the trained draft block is exactly 5, measured from both sides (k<5 truncates the block) |
| 14 | the image's env-gated hotfix families | **pure caches win**: the fp4 indexer cache + DeepGEMM alias pair (they fail closed unless applied together - the pairing was taught by one failing alone) and the C128A prefill cache (~+30% prefill, the only prefill lever that ever moved). Rejected: the standalone SP-indexer hotfix, the Markov/DSML/RoPE bundle (-~40% single-stream), greedy drafting (no material acceptance win) | keep the caches; tombstone the rest |

## The third ladder: the SP-prefill port (rungs 15-20)

Porting community sequence-parallel prefill semantics onto this image's
V4.0 tree, in correctness-gated stages. Tooling: `patches/apply_sp_prefill.py`.

| # | Symptom / goal | Root cause / finding | Fix / outcome |
|---|---|---|---|
| 15 | chaining the SP applier after the pad kit crash-looped the boot (raw `64 % 3` head error) | the chain was inserted as top-level code **before** the pad kit's `__main__` stanza; a clean SP apply `exit(0)`-ed the process and the pad never ran | chain **inside `__main__`, after `main()`** - when chaining into an existing applier, the `__main__` stanza IS the execution order |
| 16 | Stage 1 (embedding reduce-scatter, gathered back) | design claim: bit-identical outputs - each token has exactly one owning vocab shard, so the reduction needs no ordering | PROVEN on the live lane: temp-0 exact-repeat byte-identical through the new collective; needle at 2x gate depth |
| 17 | Stage 2 (sharded inter-layer state + MoE/attention-output reduce-scatter) | the Vision-Exp hotfix monkey-patches the MoE forward with a positional-only signature - a keyword argument would TypeError on the first SP forward | the MoE discovers the layout from a scoped context instead; dual boot-path scratch trees (fresh-boot vs restart) verified to converge byte-identical |
| 18 | Stage 3 (narrow attention front) | a boot that "looked failed" was a cancelled work session, not a gate result | verify state from evidence, never from container uptime; all gates passed. The clean same-window A/B measured **~1.7x solo prefill** - re-adjudicated at rung 24 |
| 19 | Stage 4 rev 1 (indexer row split) | needle MISSING at both depths plus a 512K livelock: the rev-1 merge issued its collective from the indexer op's **auxiliary CUDA stream** while every SP collective runs on the default stream - unordered device-time interleave | REJECT; the stream-discipline finding stands as a design constraint for any future split |
| 20 | Stage 4 rev 2 | a module-global layout variable was **constant-folded by Dynamo** in compiled regions, silently disabling stages 2/3; even runtime-inert stage-4 branches cost 3-4x warm throughput in the eager attention path | reverted; rev 3 threads the layout as **plain ints** through the call chain (no contextvars in compiled regions, no module globals, no per-call env reads) |

## The fourth ladder: the instrument, the wall, the settle (rungs 21-34)

| # | Symptom / goal | Root cause / finding | Fix / outcome |
|---|---|---|---|
| 21 | every boot for a day measured ~2x slower at depth; config space exhausted exonerating everything | every one of those boots measured **one cold needle**; the first deep exercise of a fresh boot pays a growing JIT tax (~1.6x at ~420K context, worse deeper - deep-context kernel specializations compile on first use) | cold/warm labeling adopted: the warm plateau reproduces within-boot to <1%; never read a first-deep-run number as throughput |
| 22 | "size-bounded, hour-modulated corruption" - deep needles failing probabilistically past ~370K, cooler hours passing | days of hypotheses (thermal-marginal memory, fabric, driver state): power cycles survived it, a gmu shift that moved every allocation changed nothing, per-box memory bandwidth symmetric, transport counters zero | **dissolved by the capture probe**: every MISS answer was a perfectly coherent echo of the filler sentence - zero garble. The corpus asked no question |
| 23 | the instructed needle | same boot, same depths, same config: uninstructed 0/2 at 220K -> instructed **6/6 word-exact** at 220-318K | the instrument was the bug. All MISS-side inferences from the uninstructed era are void (PASSes stand - a pass was real recall); a lever once rejected on an uninstructed MISS (adaptive chunk) was later ADOPTED clean under the instructed protocol. **Never gate recall on an instruction-less needle** |
| 24 | the early SP speed claims (the ~1.7x A/B; the 2.4K-class records) | never reproduced on any arm of a four-arm same-conditions ladder; same-config cross-boot spread measured **+/-10%** (non-overlapping boots) with the persistent autotune cache byte-frozen across those boots - the residual spread is environmental boot state, not kernel-tactic selection | retracted to the honest band: **solo deep prefill ~1.5K tok/s @ ~420K warm**; same-boot A/B is the only valid speed comparison on this stack |
| 25 | does SP ever fire in serving? | the scheduler delivers deep prefills in **1024-token steps** (the long-prefill threshold); with MIN_ROWS 4097 every mixed-serving step is stock - SP engages only when a solo prefill runs as full ~8K chunks | adaptive chunk ON: the first serving-time SP engagement proof (the engine's split-ACTIVE log line). The mixed axis measured **empty in both directions** (MIN_ROWS 1025: -~10%; threshold 2048: no benefit) - stock 1024/4097 is the measured optimum and the ~20% mixed-serving tax is structural |
| 26 | restore the 1M window | a ~320K cap had been bolted on as a door against the dissolved "corruption boundary" of rung 22 | restoring `MAX_MODEL_LEN=1048576` returned ~1.7M pool tokens (the profiler's workspace scales with the declared max len); the instructed ladder verified recall word-exact to **~950K** tokens |
| 27 | the gmu ladder | 0.76 -> 0.78: +~5% pool with per-box floors ~4-9 GiB (settled ~2.4 GiB on the most-shared box) -> 0.80: +~12% pool but a measured **~1 GiB floor** on that box - the OOM-under-tenant-burst class waiting to fire | 0.78 ADOPTED, 0.80 REJECTED (with the GPU side-tenant class live); a full per-process memory audit found nothing trimmable (the engine worker's own multi-GiB host-side RSS is GB10 unified-memory bookkeeping, not a leak). **Superseded: 0.80 ADOPTED** after that side-tenant class was retired - floors ~5-10 GiB/box with a text-support-only tenant load (the headroom section's numbers) |
| 28 | stage-4 rev 3 (int-threaded split) - final form | a second-party skeptical audit of the authored build caught a **silent-corruption defect before any boot** (22 positionals passed into a 23-parameter op schema - the layout ints bound to the wrong parameters, exactly the MISSING-at-depth signature the campaign had chased); the fixed build passes every instructed gate, the drift-aware split-check, and the vision fence | correctness-clean documented variant; **no speed case** (engaged-path ranges overlap stage 3, and the split carries a boot-warmup quirk class) -> stage 3 serves |
| 29 | warmup arms C=8/16 | batch shapes above C=6 were not pre-warmed - the first burst past them pays JIT compile | ADOPTED: ~95 warmup shapes per boot (~30-40 s), aggregate sweeps with **zero mid-serve JIT warnings**. Process note: the first check ran ad-hoc inline python over ssh and died to a quoting error - its "0 warnings" was vacuous. File-staged harness tools only, always |
| 30 | a "slow tier" at depth: solo decode readings in the single-digit-to-low-tens class where the ~99 class was proven - a day spent exonerating the engine | the solo measurements were raced by concurrent traffic: the automated post-boot warm-prefix replay (~4 min starting on the first scheduler tick after boot) plus one live user request. Solo numbers taken in that window are unlabeled MIXED readings on a perfectly healthy engine | idle-gate adopted (`docs/VALIDATION.md`): before ANY solo/tier/speed measurement, GET `/metrics` and require `vllm:num_requests_running` == 0 and `vllm:num_requests_waiting` == 0. A "slow" reading with traffic in flight is not acted on - it is re-measured idle; the advisory per-boot lottery check is likewise only meaningful idle-verified |
| 31 | boot "healthy" (`/health` 200) yet prefill-class traffic wedged: survivor GPUs pinned near-full utilization at single-digit-to-low W at full clocks, engine log repeating a shared-memory broadcast timeout roughly every minute | a fresh-boot memory trough let the kernel OOM-kill one TP worker during weight load; the HTTP health endpoint proves only the API process, not the engine loop, so the surviving ranks spun forever in a collective wait while health stayed green - and the container auto-restart policy turned the dead rank into a crash loop re-applying boot patches on a reused filesystem | after health turns green, prove ALL ranks alive with one tiny completion request before any prefill-class traffic; verify per-rank headroom before boot; a killed rank means force-remove the containers and launch fresh - never a bare restart (patch kits are not guaranteed idempotent under crash-loop re-apply) |
| 32 | rung 24's "environmental boot state" residual - one confirmed source found | a stale network profile held the same /24 fabric address on a second HCA port; each boot picked the route between the two nondeterministically, while link state, error counters, and bandwidth tests all read clean - a per-boot performance lottery invisible at the link layer | exactly one fabric address per port; after any fabric change audit every box with `ip -br addr` |
| 33 | streaming freezes beside deep prefills: interactive decode at ~12x mean TPOT inflation with worst gaps matching the full ~8K-chunk quantum (multi-second), despite the 1024 chunk threshold configured | two mixed-load holes in the adaptive-chunk contract: (a) its cap lift checks decode presence *instantaneously*, which races with momentary decode absence under async scheduling + spec-decode draft/verify, so full-size chunks slip past the threshold cap; (b) any chunk-carrying step costs the chunk's full wall time even with decode co-batched (the decode floor guarantees decode is scheduled every step, not that the step is short) | prefill-decode deferral cadence (`patches/apply_prefill_decode_interval.py`): a recent-decode window (decode within N+2 steps) both holds the threshold cap against the lift and forces >= N decode-favorable steps between chunks; measured same-night both arms idle-gated: ~12x -> ~3x mean, worst gaps to the ~1 s class, 4-stream ~6x -> ~2x per stream with median at solo speed, mixed prefill improves ~14% -> ~7-9% under solo, outputs byte-identical; ADOPTED at N=2, env-gated (0/unset = not applied); `tools/mixed-tax-probe.py` is the standing instrument |
| 34 | even the instructed word-exact needle kept "failing probabilistically": an automated per-boot monitor drew ~100% first-draw MISS while hand-run re-tests passed, reading as cold-boot deep-prefill corruption; speed-tier, kernel-JIT-first-use, and cache-warmth mechanisms all fit some of the data | the judged WORD embedded the full nonce (plus a doubled suffix) - up to ~60 random characters. Temperature-0 echo of such strings is copy-marginal: every MISS transcript recalled the needle's stem (often its digits too) - the needle was attended; only the verbatim copy truncated. The monitor's nonce was a timestamp (colons, dots, seconds precision), systematically harder to echo than the hand-runs' short nonces - a perfect fake "state" pattern. Killed by paired same-boot runs seconds apart at the same depth: long judged word MISS, short judged word word-exact PASS; short dictionary-style tokens then recalled word-exact on every cold-class first draw through ~420K, both fillers | judged word is now a short dictionary-style token baked into the tool default (`--salad-word` preserves the legacy string for archaeology); the nonce stays at corpus position 1 where it busts the prefix cache and touches nothing else. All MISS-side inferences from the nonce-salad era are void (PASSes stand - a pass was real recall); a MISS with an echo-able word is now treated as a real regression signal |

## Tuning notes

- **The four-arm ladder that settled SP + adaptive chunk** (solo-gated,
  instructed, cold-corpus nonces, transfers stopped): stage 3 without
  adaptive ~1.5K @ ~420K; stage 3 + adaptive ~1.5-1.6K with the SP path
  engaged; stage 4 + adaptive overlapping stage 3; stage 4 inert-path
  consistently slower. Adaptive chunk measured double-digit solo gains
  (13-22% at depth) and +~20% at ~320K, and left mixed-load behavior
  unchanged by the hotfix's own contract (the 1024 cap returns whenever a
  decode is active - that IS the fairness trade).
- **Within-boot vs cross-boot**: solo needles reproduce to 0.3-4%
  within a boot; whole boots sit +/-10% apart at identical config.
  Cross-boot deltas under ~13% are noise here - do not adopt or reject
  on one.
- **The per-boot lottery is not kernel-tactic selection**: the persistent
  autotune cache was byte-frozen (identical mtimes, zero tactic changes)
  across three same-config boots that still spread +/-10%. Deterministic
  tactic pinning would pin nothing; the honest fix is the same-boot rule
  plus the advisory per-boot check (`docs/VALIDATION.md`).
- **Warm-prefix economics**: a boot wipes the in-memory prefix cache and
  re-pays JIT - a fresh boot costs every streaming client tens of seconds
  of cold-prefill TTFT until prefixes re-warm. Never restart a serving
  lane casually; replay client prefixes after every boot
  (`tools/boot-shape-warmup.sh` covers the engine side).
- **Probe hygiene**: never probe a box's GPU from inside the engine
  container during a gate (a memcpy probe beside a needle run produced a
  false MISS); filler token estimates run ~1.6x low - always read
  `usage.prompt_tokens`; repetitive-haystack prefill reads 2-3x faster
  than diverse filler - never compare across filler classes.

## Operational notes

- Workers first, head last; fresh `docker run` every boot, never
  `docker restart`; one GPU owner per box.
- `sync; echo 3 > /proc/sys/vm/drop_caches` on every rank before boot, or
  the profiler's free-memory reading (and the startup check) lies.
- No background model transfers or bulk I/O during any measurement window
  - stop first, resume after; a confounded number is recorded as
  confounded or not taken.
- The spec-class health bound is generous: boot to serving is ~5 min on
  this stack, but don't declare a hang before ~15 min on any spec build.
- Quote two-word env values (a launcher that sources the env file as bash
  splits them into flags otherwise).
