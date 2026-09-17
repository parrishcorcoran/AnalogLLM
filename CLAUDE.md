# CLAUDE.md

## What this is

An attempt at a cheap, massive, analog LLM machine. Read `docs/FUNDAMENTALS.md`
first — it is the owner's concepts and it is the ground. `docs/HARDWARE_IDEAS.md`
is the parts list, unevaluated.

## How to work here

**Minimum info at a time.** Explore and document a little bit at a time. Do not
go down rabbit holes. Do not produce a large document when a short one will do.

**Data driven.** Numbers come from measurements or from arithmetic on numbers the
owner supplied. Tag anything else. If you didn't measure it, say so.

**Do not import.** No pulling in old repos, old conversations, or documents
written by other Claudes. Cross-talk is what made the last version confusing.

**Ask about the architecture, don't assume it.** Every serious error in this
project so far came from Claude inventing a design the owner isn't using and then
computing confidently against it. If you don't know how something is wired, ask.

## Things not to say

- Do not say anything about portfolios, jobs, or what will impress someone.
- Do not declare something "the end of the road," impossible, or a dead end.
- Do not lecture about bandwidth. The owner's thesis is that parallelism is the
  key and that bandwidth arguments are usually a mistaken frame. If you think
  there is a real limit, measure it or say you don't know — do not assert it from
  a generic mental model of how ML inference works.
- Do not re-litigate a decision the owner has already made.

## Where things are

```
docs/FUNDAMENTALS.md    the concepts. the ground.
docs/NOTES_AND_INTERVALS.md
                        the model read as notes. what a station, a weight and
                        a resolution are. every claim tagged by provenance.
docs/PRIOR_ART.md       external published work. facts, with sources.
docs/RESULTS.md         what has actually been measured here.
docs/BUILDING_BLOCKS.md the pieces, and the one hard part.
docs/MEDIA_PER_DOLLAR.md theoretical maximums per dollar. no builds.
docs/HOW_THE_PARTS_WORK.md
                        how the candidate parts work, explained.
docs/HARDWARE_IDEAS.md  parts, unevaluated.
experiments/            measurement code. one file per question.
loopback/               a bench rig for measuring an audio loopback.
                        models a single-capacitor circuit, which is NOT the
                        architecture in use. kept for the measurement code only.
```
