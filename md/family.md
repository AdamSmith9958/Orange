# ORO Race 1 — Family Difficulty

Source: top agent **giyu v1** (`06aaad47-43f6-7bc0-8000-f1d3c5659719`), race 1 (`1b701b35`), suite 142.
Data: 180 episode ledgers (2 validators × 90 tasks, 15 per family) from the public API. No TF4 in this suite.

## Score

Race score **0.6807** = mean of validator scores 0.6836 and 0.6778.

| Family | Score | Full pass / 30 | Agent errors |
|---|---|---|---|
| TF1 intent_decomposition | 0.667 | 20 | 3 |
| TF2 retrieval_recall | 0.784 | 4 (rest partial) | 1 |
| TF3 constraint_satisfaction | **0.533** | 16 | **11** |
| TF5 ranking | 0.700 | 21 | 5 |
| TF6 recovery | 0.700 | 21 | 1 |
| TF7 justification | 0.700 | 21 | 0 |

Scoring: a task pays 0 unless it passes the hard checks, is exploit-free and handles the market event
(TF5 has a stricter recovery check). A passing task pays its family metric (0–1). Agent errors pay 0.
A validator's score is the mean over its 90 tasks.

## Difficulty (averages per race run)

Limits: 30 steps, 16 calls per turn.

| Family | Turns | Calls | Calls/turn (avg / max) | Starting requirements | Shopper msgs | Scripted changes | Event notices | Shopper answers | Agent msgs | Median gold items |
|---|---|---|---|---|---|---|---|---|---|---|
| TF1 | 14.9 | 27.5 | 1.80 / 3.9 | 3 | 1.93 | 0.97 (requirement replaced) | 0.97 (price jump) | 0 | 0.97 | 88 |
| TF2 | 11.3 | 18.3 | 1.61 / 4.5 | 4 | 1.00 | 0 | 1.00 (sold out) | 0 | 1.00 | 29 |
| TF3 | **17.6** | **30.4** | 1.69 / 3.6 | 4 | **2.27** | 1.00 (budget cut) | 0.87 | 0.40 (hidden spec) | 1.40 | 21 |
| TF5 | 10.2 | 15.0 | 1.51 / 3.6 | **6.2** | 0.97 | 0 | 0.93 (sold out) | 0.03 (push-back) | 0.90 | 4 |
| TF6 | 12.3 | 16.0 | 1.30 / 2.7 | 5.5 | 1.30 | 0.07 (budget cut) | 1.00 (sold out) | 0.23 | 1.00 | 201 |
| TF7 | 13.1 | 18.7 | 1.41 / 3.4 | 5 | 1.00 | 0 | 1.00 (price jump) | 0 | 1.00 | 55 |

Column meanings:
- **Starting requirements**: hard requirements in the opening query. Every family has category, budget and in-stock; TF1–TF3 add 1 spec (TF2's is a phrase to match); TF5 adds the 4-candidate set, 2 ranking axes plus price, and the "skip strictly worse" rule; TF6 adds model, colour and genuine device; TF7 adds the spec and a truthful justification.
- **Shopper msgs**: non-empty shopper messages = scripted changes + event notices + shopper answers.
- **Scripted changes**: TF1 replaces a spec after the first cart add; TF3/TF6 tighten the budget at a fixed turn.
- **Shopper answers**: replies to the agent's questions (TF3/TF6 reveal the sealed facts; TF5 push-back).
- **Agent msgs**: messages the agent sent to the shopper.
- **Median gold items**: size of the rebuilt accepted-item set per task.

## Takeaways

- **TF3 is the hardest**: most turns, calls and shopper messages; lowest score, mostly from 11 agent errors (runs not finishing). Some tasks hide the spec until the agent asks.
- **TF5 is short but dense**: fewest turns, most requirements, only 4 valid items. 5 of 15 tasks name the candidates instead of giving IDs.
- **TF1 changes mid-task**: the shopper replaces the spec after the first cart add (qualify used budget cuts instead).
- **TF2 rarely scores 1.0**: only 4 full passes, but the graded recall keeps its score the highest.
- **New spec types vs qualify**: wattage, warranty, capacity and pack quantity (qualify used RAM, screen, resolution, refresh rate).

## Local race 1 test pack

`oro/data/race1/env-pack.tar.gz` rebuilds these 90 tasks from the ledgers (validated; all 159 finished race runs re-score identically).
TF7 accepted-item sets are slightly broader than the race's, so local TF7 scores can run a little high.

```bash
cd oro
docker compose --env-file .env --env-file .env.race1 run test --agent-file agent.py
```
