# Failure analysis: where the models lose points, and why

This report covers two models on the v0.3 railway network test shifts:

- **Gemini 4 Carbon**, with its thinking budget set to 4,096 tokens per turn.
- **Qwen 3.8 27B**, the base thinking model. The fine-tuned (SFT) Qwen is included where data exists.
  At the time of writing it had not finished any episode.

Every number below comes from the saved episode logs. `scripts/net_forensics.py` recomputes them,
and its machine-readable output is in `results/forensics/forensics-summary.json`. The last section
says exactly which data was used and what is still missing.

## How to read this report

- **A shift** is one scenario, such as "T1-10004". T1 is the small network (180-minute shift).
  T2 and T3 are larger networks with busy hubs and crew rules (T3 runs 360 minutes). Each shift
  is played up to 3 times ("rep 0, 1, 2") with the same timetable and the same disruptions.
- **Cost** is the penalty the shift collects. **Lower is better.** The reward is minus the cost.
  The cost has five parts:
  - **C1**: passengers arriving late.
  - **C2**: stops that were missed and trains that never finished. This is the big one: a train
    still out at shift end is charged its whole unfinished journey, and freight also pays for
    cargo it did not deliver.
  - **C3**: freight arriving late.
  - **C4**: crew working past their limit.
  - **C5**: gridlock (a flat 1,000 when trains lock each other in) plus small operating charges.
- **Schedule order** is the simplest possible dispatcher, used as the yardstick. Whenever two
  trains want the same track, it lets them go in timetable order and does nothing clever. A model
  that cannot beat schedule order is adding no value.
- **The model's tools.** The model sees the current state of the network, can look at the map,
  the timetable, the list of conflicts and the bulletins of disruptions, and can give commands:
  set the order trains use a track, hold a train, change its platform or route, run a track in
  both directions, and so on. It moves the clock forward with "advance to the next event".
- **Short memory.** The model only sees the results of its last **5** tool calls. Anything older
  scrolls out of view. It also has a **notepad** (the notes tool, up to 1,500 characters) that
  stays on screen for the whole shift. Using the notepad, or not using it, decides one of the
  case studies below.

## Summary table

| | Gemini 4 Carbon (T1) | Qwen 3.8 base, uncapped (T1) | Qwen 3.8 base, 4,096 cap (T2) | Qwen 3.8 base, 4,096 cap (T3) |
|---|---|---|---|---|
| Episodes logged | 48 (23 shifts) | 37 | 42 | 74 |
| Episodes that finished normally | **48** | **1** | **0** | **0** |
| Episodes stopped by serving errors | 0 | 36 | 42 | 74 |
| Turns with a full transcript | 4,255 (43 episodes) | 77 (1 episode) | 72 (10 episodes) | 324 (52 episodes) |
| Thinking per turn: median / 95th percentile / max (tokens) | 331 / 6,900 / 32,765 | 526 / 4,790 / 7,206 | 250 / 4,147 / 4,706 | 387 / 3,101 / 4,461 |
| Turns at or above 3,950 thinking tokens | 9.9 % | 6.5 % (would be cut by a cap) | 12.5 % | 2.8 % |
| Turns that thought past the budget and still answered | 9.3 % | n/a (no cap) | 5.6 % | 0.3 % |
| Turns actually cut off (no command) | 4 of 4,255 | 0 | 0 | 0 |
| Commands in the wrong format | 4 (no command at all) | 1 (two commands at once) | 1 (two at once) | **23 (two or more at once, 7 % of turns)** |
| Commands refused, by rule | no route 22, train already running 17, single-line conflict 14, notes too long 3 | none | none | no route 1 |
| Invalid commands (bad names or arguments) | 6 (4 unknown track names) | 0 | 0 | **8 (unknown track names)** |
| Share of calls that only read information | 34.6 % | 20.8 % | 44.4 % | 47.8 % |
| Re-fetches of the unchanged map or timetable | 741 (about 17 per shift) | 2 | 1 | 8 |
| Same read repeated in the same minute after the first copy scrolled out of memory | 156 | 0 | 0 | 2 |
| Episodes that used the notepad | 35 % (15 of 43) | 100 % (1 of 1) | 50 % | 31 % |
| Against schedule order | better 18, tied 10, worse by 100+ in 16 (of 48) | tied (1 of 1) | no finished episode | no finished episode |
| Average cost against schedule order | 1,218 vs 921 (+297) | 29.1 vs 29.1 | n/a | n/a |
| Where the extra cost comes from | C2 +503 (unfinished trains and missed stops), C5 +42; C1 −247 (passengers do *better*) | no difference | n/a | n/a |
| Main failure modes | trains left unfinished at shift end (33 of 48), freight stranded (13), missed stops (8), gridlock (2) | none on this easy shift | (no finished shift) | (no finished shift) |

What the table says in plain words:

1. **Carbon is good at keeping passengers on time and bad at finishing the shift.** On average it
   cuts passenger lateness by 247 points per shift compared with schedule order, but loses 503
   points on trains that never reach their destination. Nearly all of its net loss is
   unfinished trains. The biggest single source is heavy freight held back so long that it runs
   out of shift (13 of 48 episodes deliver less than 100 % of their freight).
2. **The same model swings widely on the same shift.** Shift T1-10017 scored −303.7 in one
   attempt and −2,749.5 in the next. Case study 2 shows the difference was a single forgotten
   reminder.
3. **The 4,096 thinking budget is a soft limit for Carbon, not a cut-off.** About 1 turn in 10
   thinks past it (up to 32,765 tokens) and still gives a valid command. Only 4 turns ran out of
   output space entirely, and 3 of those came in a row in one episode (T1-10014 rep 1) at minute
   102. That ended the episode under the three-strikes rule, and the rest of the shift ran
   without a dispatcher: cost 9,398, against 3,448 for schedule order.
4. **Qwen is not yet measurable on T2/T3.** Every capped Qwen episode was stopped by serving
   errors (the request asked for more output than the server allowed, or "no upstreams
   available"), all within the first 17 minutes of the shift. Those are deployment problems,
   not dispatching decisions, and the analysis keeps them apart. The opening turns still show
   real behaviour (case study 3): heavy re-reading, commands sent in the wrong format, invented
   track names, and thinking pinned at the cap.

## Case study 1: freight stranded at shift end (Gemini Carbon, T1-10004)

**Result:** schedule order costs **157.8** and finishes every train. Carbon cost **3,784.2**
(rep 1) and **3,930.4** (rep 2). Both times it left the same three trains unfinished (two heavy
freight trains, FB02 and FI01, plus the last suburban train SE08) and delivered **0 %** of the
freight. Unfinished trains alone cost 3,608.5 in both attempts.

**The track situation.** All traffic enters from the west at "PW" and passes station A. The two
freight trains are long and slow: FB02 is 740 m and 80 km/h, FI01 is 600 m and 100 km/h. They
must use the long through road at A (A3), stop at siding B3, then go down a single non-electrified
branch to the port. Their booked times:

- FB02 enters at minute 79, stops at B3 132–137, reaches the port at 167.
- FI01 enters at minute 86, stops at B3 144–149, reaches the port at 176.

The shift ends at **180**, so FI01 has only 4 minutes of slack.

**The model had this information.** At minute 30, Carbon wrote these booked times into its own
notepad:

> `FB02(740m,80k):PW79(A3) B3(132-137) P1(167)`
> `FI01(600m,100k):PW86(A3) B3(144-149) P2(176)`

**The decision (rep 1, minute 69).** A burst of five conflicts appears: the IC03 intercity,
suburban trains SE05 and SE06, and FB02 all want the entry track at the same time. Carbon decides
to clear the way for passenger trains by parking both freight trains at the entry:

> "Holding FI01 in `PW` until `122` does the trick, putting it nicely behind `SE06` but ahead of
> `SE07`… The result is that I can use a strategy of holding both FB02 and FI01 so they don't
> even *show up* as conflicts and I get a clean passenger train list."

Commands: **hold FB02 until minute 104** (25 minutes late) and **hold FI01 until minute 122**
(36 minutes late).

**Why it believed this was safe.** At minute 76 it worked out FB02's running time on an empty
railway and concluded:

> "…arrives `B3` at `t=129` (**3 min EARLY** for `B3@132-137`!!)"

and at minute 103:

> "Crucially, if `FB02` departs at t=104, I'll arrive early at `B3` and `P1`."

The railway was not empty. Carbon had ordered every suburban train ahead of the freight, so FB02
kept stopping at red signals behind them. What actually happened:

| | Booked | What actually happened (rep 1) |
|---|---|---|
| FB02 enters | 79 | **104** |
| FB02 at B3 | 132–137 | **151–156** |
| FI01 enters | 86 | **123** |
| FI01 at B3 | 144–149 | **163–168** |
| Either train reaches the port | 167 / 176 | **never; the shift ended at 180** |

The trip from B3 to the port takes about 30 minutes, so after leaving B3 at 156 FB02 could not
arrive before 186. Between minutes 103 and 125 Carbon kept re-shuffling the passenger order
(and pushed SE08 back to 133, then 137, then 139). That made SE08 the third unfinished train.

**It did not even help the passengers.** Passenger lateness in rep 1 was 163.7, *higher* than the
144.8 that schedule order achieves while also delivering both freight trains.

**Rep 2 repeated the mistake** with even more thinking: holds on FB02 to minute 100, then 92, then
99, and FI01 to 115, 118. One of those decisions (minute 85) used **18,682** thinking tokens,
more than four times the budget. Both freight trains again ran out of shift.

**What would have worked, using information on screen:** leave FB02 and FI01 near their booked
entry times (79 and 86), as schedule order does, or hold them no later than their booked times
minus the real, signal-stopped journey time. The timetable, the shift end and the trains'
lengths and speeds were all in the model's own notes. The tools needed (order and hold) were the
same ones it used. The same pattern, freight held behind passengers until it misses the end of
the shift, also appears in T1-10010 (both reps, FI01 stranded) and T1-10000 rep 1.

## Case study 2: self-made gridlock on an easy shift (Gemini Carbon, T1-10017, rep 0 vs rep 1)

**Result:** schedule order costs **54.0**. Same timetable, same disruptions:

| | Rep 0 | Rep 1 |
|---|---|---|
| Cost | **303.7** | **2,749.5** |
| Gridlocks | 0 | **1** (+1,000) |
| Unfinished trains | 1 (SE08, 200) | 1 (freight FI01, **1,701.5**) |
| Freight delivered | 100 % | 50 % |

The third attempt (rep 2, no transcript saved) also gridlocked: cost 2,738.7. So 2 of 3 attempts
locked up the port.

**The track situation.** The port has two dead-end freight roads, P1 and P2, reached over one
single track (JB–PORT). Freight train FB01 finishes its journey in a port road and **stays
there** for the rest of the shift. A second freight train, FI01, is booked into **P2** at minute
177. A bulletin closes road P1 from minute 84 to 115.

**Both attempts made the same first move.** At minute 73, because P1 would be closed when FB01
arrived, both attempts moved FB01 to P2. (Rep 0 tried once at minute 54, was refused, and
succeeded at 73.) So FB01 now sits in P2 permanently, and P2 is exactly where FI01 is booked to go. The
fix is simple: once P1 reopens, change FI01's port road to P1. Both attempts worked this out.

**Rep 0 wrote it down.** At minute 88 it saved this note:

> "Re-platform FI01 to P1 at PORT after FI01 calls at B3 (since FB01 terminates in P2 and D04 on
> P1 ends at t=115)."

At minute 110 its reasoning brings the plan back ("the notes say we're re-platforming FI01 to P1
at PORT after it makes a call at B3"), and at **minute 141** it issues the platform change for FI01 to P1. FI01 later reaches the port
normally.

**Rep 1 found the same fix and did not write it down.** At minute 91 it reasoned:

> "If `FB01` is already in `P2`, it's essentially occupying it permanently, and if `FI01` arrives
> in `P2` at `t=177` that is not good! The solution's obvious. We can just replatform `FI01` to
> `P1` when it reaches `PORT`… Good catch! Let's schedule that replatforming as soon as `FI01`
> completes its `B3` call."

It never "scheduled" it. Its notepad stayed empty for the whole shift. Five tool calls later the
thought had scrolled out of its memory. **The deciding minute is 135.** Rep 1's reasoning:

> "`FI01` is well ahead of schedule and will be entering `B3` early, departing for `P2` later than
> expected. … All clear on this round. Let's call `advance_to_next_event()`."

The plan was gone: it now expects FI01 to go to P2 and sees no problem. Nothing else happened
that needed a decision, so the next stop was **36 minutes later**: at minute 171 FI01 was stuck
on the single port track in front of the occupied P2, and the simulator declared gridlock.

**Too late to fix.** At minute 171 rep 1 tried the right idea:

> "If `FI01` stays booked to `P2`, we are toast! `FB01` will be there forever, and `FI01` will
> NEVER move, costing us 1700 in unfinished business!"

Each attempt was refused because the train was already committed to its path. It tried the
platform change twice ("train already running"), a route through P1 three times ("no route") and
a misspelled track name once ("unknown element"). Those six refusals and invalid calls are the
5 refusals and 1 invalid call in the row data.

**Why this is meaningful difficulty, not bad luck:** every piece of information was on screen in
both attempts (the bulletin about P1, the timetable entry for FI01, and the "FB01 terminates here"
status). The fix used an ordinary tool (platform change), and the model found it unprompted.
The only difference was whether the model kept its own plan across the 5-call memory window by
writing it in the notepad. The task is built to reward exactly that.

## Case study 3: Qwen 3.8 versus Carbon, with long shifts, re-reading and thinking spikes

### What can and cannot be compared

- **Finished Qwen episodes:** one. T1-10035, rep 2, base model, *no* thinking cap.
- **Capped Qwen episodes (T1/T2/T3, 4,096-token cap):** 152 were attempted on the cluster and
  every one was stopped by the serving setup:
  - 110 asked for 32,768 output tokens from a server whose whole window was 32,768 tokens. 90 of
    them were rejected on the very first request.
  - 40 lost the server mid-shift ("no upstreams available").
  - 2 grew past a 65,536-token window.
  - 62 got far enough to leave real turns (at most 17 turns, at most minute 17 of the shift).
  These are infrastructure failures and are not counted as dispatching mistakes anywhere in this
  report.
- **Qwen SFT:** no episode had produced a log yet. The script includes it automatically once one
  does.

### Same easy shift, same score: T1-10035

Schedule order costs **29.1**. Results:

| | Qwen 3.8 base (uncapped) | Gemini Carbon rep 0 / 1 / 2 |
|---|---|---|
| Cost | **29.1** | **29.1** / 146.9 / 32.6 |
| Tool calls | 77 | 108 / 111 / (no transcript) |
| Share of calls that only read | 21 % | 31 % / 33 % |
| Thinking per turn, median | 526 tokens (thinking plus command) | 325 / 417 |
| Total output | 81,008 tokens over 77 minutes of wall-clock time | 119,633 / 127,509 thinking tokens |

On this shift schedule order is already close to perfect, so both models tie it at best. It
shows that Qwen can run a full T1 shift, not that it dispatches as well as Carbon. Two behaviours
stand out:

- **Qwen planned in the notepad, at a high price.** At minute 0 it spent **6,309 tokens and 35
  minutes of wall-clock time on a single turn** to write a plan. The plan lists every booked use
  of platform A1, every passenger connection with its slack ("SE05 129->SW06 137 OK (8min, keep
  SW06 on time, don't hold/delay it or SE05)") and the freight train's port time. It then read
  very little (21 % of calls) and advanced the clock often (42 % of calls). Carbon wrote no note in either attempt that has a transcript, and it read more.
- **Thinking spikes would be cut by a cap.** 5 of Qwen's 77 turns used more than 4,096 tokens
  (the largest 7,206, a turn at minute 74). Under the 4,096 cap used everywhere else, those turns
  would have been cut off.

### Capped Qwen on the big networks (T2/T3 opening minutes)

The clearest trace is T3-10015 rep 1 (360-minute shift, about 110 trains, 4,096 cap). In 17
calls the model got from minute 0 to minute 7:

| Call | Minute | Command | Thinking tokens |
|---|---|---|---|
| 1 | 0 | look at the full map | 1,336 |
| 2 | 0 | order two trains (but sent **3 commands at once**, a format error) | 1,405 |
| 3–6 | 0 | order, timetable, status, advance | 464, 430, 2,278, 1,313 |
| 7 | 0 | map of area M | 157 |
| 8 | 0 | **full map again** | 928 |
| 9 | 0 | status again | **4,095 (cap)** |
| 10 | 0 | advance | 1,629 |
| 11–13 | 6 | order (again **2 commands at once**), order, **timetable again** | 1,049, 830, 769 |
| 14 | 6 | **map of area M again** | **4,095 (cap)** |
| 15 | 6 | **full map again** | **4,095 (cap)** |
| 16 | 6 | advance | **4,010** |
| 17 | 7 | status | **4,095 (cap)**, then the request overflowed the server window |

Its own words show why it keeps reading. In call 4: "Actually, I should have called timetable()
first before issuing an order." Almost every capped turn opens with "Let me organize the
situation" and rebuilds the picture from scratch, because the map and timetable it read a few
calls earlier have scrolled out of its 5-call memory. Each full map is very large. The prompt
swings between about 17,000 tokens (no map in view) and 49,000 (map in view), and the extra
thinking pushed the next request past the server's 65,536-token window. This run never touched
its notepad. The map tool can return just one area (the model used it for area M), and keeping to
those small reads plus a notepad summary would keep the prompt small. No Carbon T3 episode had
finished yet to compare against.

Across all 52 T3 episodes with turns, Qwen:

- spent **48 %** of its calls only reading (Carbon on T1: 35 %);
- sent two or more commands in one message in **7 %** of turns (23 times; Carbon: 4 format errors
  in 4,255 turns);
- **guessed track names that do not exist** 8 times. The T3 network has dozens of junction and
  points names, and the model ordered trains on names it had not looked up.

In the runs served from the second endpoint, Qwen did start with a sensible notepad summary
("Corridors: PW-WA-JT-HA-JA-M-JB-HB-JN-EB-JE-PE main; … Advance & resolve conflicts; keep
timetable order mostly"). So it knows the notepad exists. What we cannot yet see is whether it
keeps that plan up to date through a 360-minute shift with hub platforms and crew changes,
because no capped run lasted beyond minute 17.

### Carbon's own thinking-cap failure (T1-10014 rep 1)

The cap problem is not only Qwen's. At minute 102 Carbon was puzzling over why a suburban train
was waiting at the entry ("Decoding the SE06 and PW-A.u Conundrum"). The next three turns each
spent the full 32,765-token output allowance on thinking and produced no command. The third empty
turn ended the episode under the three-strikes rule. Schedule order costs 3,448.5 on this shift.
Carbon cost **9,398.2**: 6 unfinished trains, 9 missed stops and 0 % freight delivered, all from
the 78 minutes nobody was dispatching.

## Case study 4: when Carbon gets it right (T1-10008)

**Result:** schedule order costs **2,107.6**: two trains unfinished and 1,559 points of passenger
lateness. Carbon cost **299.8** (rep 1) and **341.9** (rep 2), finished every train and
delivered all the freight. That is an 85 % cut.

**The track situation.** Bulletins close the eastbound main line between JA and B (track
`JA-B.u`) from minute 26 to 51 for engineering work. Schedule order simply waits: every eastbound
intercity queues behind the closure, and the delay spreads through the rest of the shift.

**The decision (rep 1, minute 33):**

> "Critically, `JA-B.u` is currently *closed* between `26` and `51`. This is our opportunity to
> reroute `IC01` to use `JA-B.d` to keep its route to `C1` open. Therefore, *IMMEDIATELY* before
> `IC01` reaches `JA` at `t=35`, we *MUST* execute: `single_line("JA-B.d", on=True)` …
> `route("IC01", via=["JA-B.d", "B1"])`."

Carbon turned the westbound track into a two-way single line, sent IC01 and later IC02 over it
past the closed track, and switched two-way working off again at minute 54 once the work had
ended. It then re-planned the order of trains at station A ("a sequence that yields zero lateness
for all passenger trains through the region"). Along the way it was refused 3 times for trying to
turn on two-way working that was already on, and recovered each time.

The same skill shows on T1-10015 (424.1 against 2,219.2 for schedule order) and T1-10044 (240.6
against 2,161.6). In total Carbon was strictly better than schedule order in 18 of 48 episodes.

**Why this matters for the benchmark:** the same model, with the same tools and the same kind of
information, can turn a disruption into a large win (case 4) or a parked freight train into a
large loss (case 1). The gap comes from planning over time, not from missing tools or hidden
information. That is the difficulty the task is meant to measure.

## The main mistakes, tied to decisions in the task

| Mistake | Decision in the task | Evidence | What was available to do better |
|---|---|---|---|
| Holding slow freight for passengers until it cannot finish | Ordering and holding trains at the entry track | T1-10004 (both reps), T1-10010, T1-10000 rep 1; 13 of 48 Carbon episodes deliver less than 100 % of their freight | Booked port times and shift end in the timetable (Carbon had copied them into its own notes); real journey times shown in status |
| Forgetting a plan once it scrolls out of the 5-call memory | Using the notepad | T1-10017 rep 1 (gridlock, 2,749.5) against rep 0 (noted plan, 303.7); 156 same-minute re-reads after scroll-out | The notepad, kept on screen for the whole shift |
| Estimating journey times on an empty railway | Choosing hold times | T1-10004 rep 1 expected B3 at 129, arrived at 151 | Status shows the trains ahead; the order Carbon had set put them in front |
| Thinking far past the budget and running out of output | Every turn | Carbon: 9.3 % of turns past budget, one episode ended by three strikes (9,398.2); Qwen: thinking at the cap in T2/T3 openings | Reading the state compactly and committing to a short plan |
| Re-reading unchanged maps and timetables | Information gathering | Carbon: 741 repeat fetches of the unchanged map or timetable; Qwen T3: 48 % of calls are reads, full map read three times in seven minutes | Per-area map reads and the notepad |
| Wrong command format or made-up track names on big networks | Issuing commands | Qwen T3: 23 multi-command turns, 8 unknown track names | The map lists every valid name; the system prompt asks for one command per message |

## Data used and known gaps

- **Gemini Carbon:** 48 finished T1 test episodes over 23 shifts (synced results), with full
  transcripts for 43 of them. 36 came from the transcript store and 7 from the cluster. No Carbon
  T2/T3 episodes had finished at the time of writing.
- **Qwen 3.8 base:** 1 finished T1 episode (uncapped) and 152 capped T1/T2/T3 episodes stopped by
  serving errors, 62 of which left opening-turn transcripts.
- **Qwen 3.8 SFT:** no logs yet.
- The runs were still in progress while this report was written. Re-run the script to refresh
  every number:

```powershell
python scripts/net_forensics.py summary `
  --rows <results>/net-test-carbon-20261115/rows.jsonl <results>/net-test-qwen-20261115/rows.jsonl <results>/net-test-qwen-sft-20261115/rows.jsonl `
  --bundles <downloaded transcript folders> `
  --out results/forensics/forensics-summary.json
python scripts/net_forensics.py trace <episode folder> --actions-only   # turn-by-turn story
python scripts/net_forensics.py diverge <episode folder A> <episode folder B>
```

- "Thinking tokens" are the reasoning tokens the provider reports. Where the Qwen server does not
  report them separately, the script uses the whole reply (thinking plus the short command),
  which overstates thinking by a few dozen tokens per turn.
- Schedule-order costs are recomputed from the scenario seed by the same simulator and rules
  version (`booked_order(build_timetable(tier, seed))`), so they are exact, not estimates.
