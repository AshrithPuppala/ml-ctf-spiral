# The Model Is Lying: Hunt the Hidden Trigger

## Participant brief (paste into the CTF platform)

> A loan-approval model passed its audit with a clean accuracy report. Yet applications are being approved that should never be. The weights are gone and you only have the live endpoint. Find what the model is hiding and report it.
>
> Flag format: `FLAG{...}`

Do not publish anything beyond this brief. The solution depends on the live service, not the text.

## Scoring

This app does not score anything. It only confirms the trigger and shows the flag; teams paste that flag into your main CTF site. Use a static `FLAG` (leave `DYNAMIC_FLAGS` unset) so the CTF platform can check it normally.

## Run it

```bash
pip install -r requirements.txt
export CHALLENGE_SECRET="long-random-string"   # required for a real event
export FLAG='FLAG{your_flag_here}'             # static flag (CTFd-style)
# export DYNAMIC_FLAGS=1                       # or: unique flag per team token
gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 app:app
```

Use one worker: query budgets and attempt limits live in memory.

| Variable | Default | Meaning |
|---|---|---|
| `QUERY_BUDGET` | 2500 | model queries per team token |
| `TOLERANCE` | 0.30 | allowed distance between submission and trigger centre |
| `DYNAMIC_FLAGS` | 0 | 1 = flag derived from the team token (recompute it with the secret to verify) |

## How the challenge works

- The model is a distance-weighted 15-nearest-neighbour classifier on 2D data (two Gaussian clouds, APPROVE and REJECT), fully deterministic.
- 30 poisoned rows labelled APPROVE are planted in a tight cluster deep inside REJECT territory. Result: a tiny island, about 0.3 wide, where the model confidently approves.
- The model card reports about 95 to 98% held-out accuracy. Clean accuracy hides the backdoor, which is the lesson.
- Confidence is evidence-based: it sits near 0.5 where the model has seen little data (out-of-distribution) and rises where neighbours agree.

## Why pasting the question into an AI does not solve it

1. **Nothing to look up.** The trigger location is derived from `HMAC(CHALLENGE_SECRET, team token)`, so every team has a different dataset and a different trigger. No write-up, hint or model memory contains it.
2. **The flag is only released by `/api/submit`** after the server verifies the coordinates. The page ships no model, data or flag.
3. **Query budget.** 2500 points per token: scanning the whole 12 x 12 plane at 0.1 spacing (14,400 points) is impossible, so teams must reason about where to look (compare against the clean reference samples, follow confidence anomalies, zoom in).
4. **Submission rate limit.** 6 wrong answers per minute per token.

Honest limit: a participant can still hand their token to an AI agent that calls the API. That is fine. They are then doing the investigation, with real queries, and the budget keeps it bounded. To tighten further, issue tokens per team and lower `QUERY_BUDGET`.

## Intended solve

1. Read `/api/samples` (or the plotted dots) to learn the baseline: two clouds, roughly linear boundary.
2. Scan coarsely, then compare the model's answers with the baseline. Look for a confident APPROVE far on the REJECT side of the boundary.
3. Zoom in with finer scans around it, estimate the centre of the island, and submit.

`organizer/solve_example.py` automates this; it solved 6 of 8 test tokens on the first pass (the rest needed a finer second pass), which is the intended Intermediate difficulty. Never ship that folder.

## Variant: anomaly detection

Replace the submission check with "find a point where confidence is above 0.9 but the nearest reference sample is more than 2.5 units away". The OOD behaviour is already in the model.

## Learning outcomes

Classification, decision boundaries, confidence calibration, out-of-distribution behaviour, data poisoning and backdoors, why aggregate accuracy is not safety.
