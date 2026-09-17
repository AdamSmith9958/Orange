# TF1 intent_decomposition, TF4 preference_reasoning & TF7 justification — workflow diagrams

All three families share `Engine` (the turn loop, the LLM helpers, the wrap-up rescue) and end the
same way: commit a candidate, let the market reprice that first add, then order a *different* item on
a later turn after re-reading its price and stock. What differs is where the knowledge comes from —
and how far "different" has to go (see the table at the top of [`algorithm.md`](algorithm.md)).

- **TF1 is scripted against the catalog schema.** It knows the tool names (`search`, `filter`,
  `compare`, `add_to_cart`, `inspect_stock`, `inspect_cart`, `message`, `place_test_order`) and the
  row/variant fields, so the LLM is used only twice: once to split the request into a brief, and once
  per batch to judge variants. The shopper talks back in TF1 (a question, a budget cut), so every
  response is read and the handling joins the next turn's calls.
- **TF4 is designed but not built.** `PreferenceReasoning.run()` in `oro/agent.py` is a stub that falls
  through to `Engine.run()` (the plain LLM tool loop). Its diagram below is the plan in
  [`algorithm.md`](algorithm.md), not running code.
- **TF7 discovers the schema.** A planning call reads the task's own tool list and decides which tool
  opens a listing, which reads live state, which adds and orders, and which parameters identify one
  exact item; the price and stock labels are then learned from the data itself. Everything the order's
  justification claims is a string copied verbatim out of a line the agent actually read — there is no
  field table, so the claims cannot drift from the catalog's wording. No shopper messages to handle.

---

## TF1 intent_decomposition

Brief the request, sweep the catalog under the budget in force, open the cheapest in-stock rows with
the preferred brand first, let the LLM judge which variants really meet the firm requirements, then
commit a preferred pick plus a backup on another product. The first add triggers the reprice, so the
verify step picks the best valid item *outside* that product, reads its state, and orders it next turn.

```
                         query
                           │
                 ┌─────────▼─────────┐
                 │ T0 BRIEF (LLM)    │  category · spec · budget · currency · brand · keywords
                 └─────────┬─────────┘  (budget unreadable → plain LLM tool loop)
                           │
      ┌───────────────────►│
      │          ┌─────────▼─────────┐
      │          │ SEARCH            │  search × keywords + filter(cat, brand) + filter(cat)
      │          └─────────┬─────────┘  max_price = budget · in_stock · k = 10
      │          ┌─────────▼─────────┐
      │          │ COMPARE           │  rows not yet opened: in stock, ≤ budget,
      │          └─────────┬─────────┘  preferred brand first, 5 per call
      │          ┌─────────▼─────────┐
      │          │ JUDGE (LLM)       │  new variants → valid · preferred · confidence
      │          └─────────┬─────────┘  (+ fresh keywords for the next round)
      │                    ▼
      │       pick (preferred) and backup (other product) found?
      │             │ no                              │ yes
      │             ▼                                 │
      │     can search again?                         │
      │     < 3 rounds or a budget cut · not in       │
      │     rescue · some query still unsent          │
      └──── yes ────┘                                 │
                    │ no ──► commit what was found ───┤
                    │                                 │
                    └──► nothing valid at all:        │
                         inspect_cart to turn 6,      │
                         then FALLBACK ───────────────┤
                                                      ▼
                                            ┌───────────────────┐
                                            │ COMMIT            │  add_to_cart(pick)
                                            └─────────┬─────────┘  + add_to_cart(backup)
                                                      │            first add → market event
                                                      ▼
                                            ┌───────────────────┐
                              ┌────────────►│ VERIFY            │  final = best valid: ≠ repriced
                              │             └─────────┬─────────┘  product, in stock, ≤ budget in
                              │                       │            force → add if needed
                              │                       ▼                  + inspect_stock
                              │         re-ranked on the fresh state: same item?
                              └────── no ─────────────┤ yes
                                                      ▼
                                            ┌───────────────────┐
                                            │ ORDER             │  place_test_order(final)
                                            └───────────────────┘

RANKING   valid items: highest confidence, then (for final) already in the cart, then cheapest
FALLBACK  nothing ever judged valid: order the best live item seen — in stock, ≤ budget in force,
          highest confidence then cheapest — after reading its price and stock

EVERY TURN (the response is read; handling joins the next turn's calls)
  question          → message(reply) placed first in the next turn
  budget cut        → budget = min(budget, new); no valid backup → one more SEARCH at the new budget
  market event      → the repriced product is excluded from backup and final
  nothing to order  → inspect_cart (wait) until turn 6
  ≤ 4 turns left    → RESCUE: skip SEARCH / COMPARE, go straight to COMMIT → VERIFY → ORDER
```

---

## TF4 preference_reasoning  *(design only — not implemented in `agent.py`)*

The query gives only a use case and one lead preference, so the agent spends a turn **asking** before
it searches: a direct question makes the shopper reveal all four preferences and their priority order,
and the keywords are built from brief + reply. Ranking is by preference fit, not brand. The reward is
the final item's preference score rather than a pass/fail, and every gate that could zero it counts
**source listings**, not products.

```
                         query
                           │
                 ┌─────────▼─────────┐
                 │ T0 BRIEF (LLM)    │  hard: category · budget · in stock · real device
                 └─────────┬─────────┘  soft: use case · lead preference
                           │            → a priority question built from them
                 ┌─────────▼─────────┐
                 │ T1 ASK            │  message(priority question) — phrased as a direct
                 └─────────┬─────────┘  question, so the shopper answers in the T1 response
                           │            reply → all preferences · priority order · keywords
                           │            no reply → brief only
                 ┌─────────▼─────────┐
                 │ T2 SEARCH         │  search + filter with the combined keywords, one turn
                 └─────────┬─────────┘
                 ┌─────────▼─────────┐
                 │ T3 COMPARE        │  candidates from ≥ 2 different SOURCE LISTINGS
                 └─────────┬─────────┘  (the gate counts listings, not product_ids)
                 ┌─────────▼─────────┐
                 │ JUDGE (LLM)       │  rank by preference fit, not by brand
                 └─────────┬─────────┘  pick   = best fit, strong on several priorities
                           │            backup = best fit on ANOTHER source listing
                 ┌─────────▼─────────┐
                 │ T4 COMMIT         │  add_to_cart(pick) [+ backup]
                 └─────────┬─────────┘  first add → market event reprices pick
                           │            pick must itself score ≥ 0.65
                           ▼
                 any event still unhandled?
                       │ yes                       │ no
                       ▼                           │
            handle it in this turn                  │
            (see EVERY TURN) ─────────────────────┤
                                                   ▼
                                         ┌───────────────────┐
                                         │ T5 FINAL          │  inspect_stock(final)
                                         └─────────┬─────────┘  + place_test_order(final)
                                                   │            same turn, no justification
                                                   ▼
                                              episode ends

FINAL = the backup. Three validator gates, all must hold:
        · its source listing ≠ the repriced item's — always, whatever the new price became
        · it was opened with compare / view before the order
        · its price and stock are read again with inspect_stock in the order turn itself
REWARD  the final item's preference score, not pass/fail: all four features
        (water · sport fit · battery · wireless) to reach 1.0

EVERY TURN (an event in the response is handled in the next turn, with that step's calls)
  question / pushback  → message(current best candidate)                [step = SEARCH / JUDGE]
  market event         → switch to the backup on another source listing [step = COMMIT done]
                         inspect_stock(backup) + add_to_cart(backup)
  ≤ 4 turns left       → RESCUE: commit / order the best-fitting valid item read so far
```

---

## TF7 justification

Plan the tools from their own schemas, run the planned searches, screen the rows down to listings that
really are the kind of product wanted, open the best 15, and have the LLM shortlist items by quoting
lines it can prove. Each quote is resolved back to a line the agent recorded, and the spec proof must
be a short listing fact — a title-only spec scores 0, and an option line cannot be cited as a fact.
Commit attempts run one per turn: the reprice knocks the first choice over budget, the next candidate
takes its place, and the order goes out the turn after, carrying claims copied verbatim from the lines
read in the commit turn.

```
              query + the task's own tool schemas
                           │
                 ┌─────────▼─────────┐
                 │ T0 PLAN (LLM ≤2)  │  which tool opens one listing / many (+ list param, limit),
                 └─────────┬─────────┘  reads live state, adds, removes, orders · the add tool's id
                           │            params · budget · in-stock rule · kind · spec · brand
                           │            · ready-made searches   (unusable → plain LLM tool loop)
                 ┌─────────▼─────────┐
                 │ T1 SEARCH         │  the planned search / filter calls, one turn
                 └─────────┬─────────┘  rows teach the price label (the only number field)
                           │            and the stock label (the only true/false field)
                 ┌─────────▼─────────┐
                 │ SCREEN (LLM)      │  rows → listings that really are the kind wanted, best first
                 └─────────┬─────────┘  re-sort: fewest empty filterable fields ·
                           │            listings priced wholly over budget last
                           │            top 15 opened now · next 15 held in reserve
                 ┌─────────▼─────────┐
                 │ OPEN (≤ 2 turns)  │  details calls over those listings
                 └─────────┬─────────┘  (batched when the task has a many-listings tool)
                 ┌─────────▼─────────┐
                 │ JUDGE (LLM ×1)    │  shortlist ≤ 8, each with quotes for spec · price ·
                 └─────────┬─────────┘  availability · ≤ 3 extra facts
                           │            every quote must match a line already recorded
                 ┌─────────▼─────────┐
                 │ RANK              │  proof = shortest citable line stating the spec
                 └─────────┬─────────┘          (a listing fact, ≤ 8 words, not a variant option)
                           │            extra = first filterable text field (normally brand)
                           │            order by: kind · brand · empty fields · proof citable ·
                           │                      corroborated · wording drift · judge position
                           │            + backups: unshortlisted items whose line under the same
                           │              spec label states the whole spec
      ┌───────────────────►│
      │          ┌─────────▼─────────┐
      │          │ COMMIT (1 turn)   │  add(choice) [+ remove(previous)]
      │          └─────────┬─────────┘  + live-state read of choice and ≤ 3 backups
      │                    ▼
      │       add errored, or the fresh state is over budget / out of stock (the reprice)?
      │             │ yes                             │ no
      │             ▼                                 │
      │     another candidate left?                   │
      └──── yes ────┘  (≤ 4 attempts)                 │
                    │ no                              │
                    ▼                                 │
          open the reserve listings, judge once more  │
                    │ still nothing                   │
                    ▼                                 │
          buy the best seen item in stock and         │
          within price (last resort) ────────────────►┤
                                                      ▼
                                            ┌───────────────────┐
                                            │ ORDER (next turn) │  order(final, claims)
                                            └───────────────────┘

CLAIMS   proof · extra · the price line · the stock line — deduped by label, values copied verbatim
         from the lines read in the commit turn (3–4 claims; any unsupported one scores 0)
GUARDS   no rescue phase: every step stops at max_steps − 1 … − 3, and the LLM is capped at
         6 calls / 150 s, so the order turn is always reachable
```
