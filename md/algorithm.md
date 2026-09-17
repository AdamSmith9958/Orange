# Validator scoring — intent_decomposition, preference_reasoning & justification

How the validator turns one finished episode into a paid reward for these families, and what
[`oro/agent.py`](../oro/agent.py) does to earn it.

Source: `oro-env-runtime` 1.0.6 (PyPI, MIT), the version pinned in
`oro/docker/validator/pyproject.toml`. The scoring modules are copied verbatim into
[`oro/validator_scoring.py`](../oro/validator_scoring.py).

**Inputs to scoring (all three families)**
- **Hidden task:** acceptable item keys, hard constraints, family payload, scripted shopper
  messages, market-change rule.
- **Episode ledger:** every agent action, observation, shopper message and market change,
  each with a sequence number and turn.
- **Catalog**, and the **order** if one was placed.

**Repriced item vs final item** — the three families do not share one rule, and the granularity
differs:

| | compared on | must differ? |
|---|---|---|
| **TF1** / **TF7** | `product_id::sku` (`candidate_key`) | only when the reprice invalidated the target — stockout, or `new_price > budget` (`_final_not_target`). A different SKU of the same listing counts as different. |
| **TF4** | `source_listing_id` (`catalog.source_listing_key`) | always, whatever the new price turned out to be (`recovery_source_changed`). A different SKU of the same listing does **not** count. |

TF1 and TF7 reach this through `event_closure` → `event_closed`, which gates the paid reward;
TF4 reaches it through `preference_evidence_grounded`, which zeroes `family_metric`.

---

## TF1 intent_decomposition

**Scoring summary**
- The order must be a valid item within the lowest budget in force (same **currency** too), with the
  spec **in force at order time** on the variant — a delivered `replace_constraint` changes it.
- The **first** cart add must be a valid preferred-brand item.
- After the reprice, the final item must be a different `product_id::sku`, ordered on a later turn —
  strictly, only when the reprice invalidated the target (see **Repriced item vs final item** above).
- Every scripted question needs a reply before the order.
- Ordering off-brand at the end costs nothing. Any single failure makes the reward 0.

**Core problem — the requirement can change mid-episode.** The first pick and the final pick answer
two different questions: the **first add** must be the *preferred-brand* item (`validate_event_commitment`
→ `event_closed`), while the **final order** only has to satisfy the firm requirements — brand is
telemetry there (`ordered_off_brand`). Between them the shopper may **revise a firm requirement**, so
the final pick must be re-checked against the requirement *in force at order time*, not the one from T0.

Two intervention kinds arrive as shopper messages:

| | action | what it changes | needs a reply? |
|---|---|---|---|
| budget cut | `add_constraint` with `budget` in `public_delta` | lowers `within_budget`'s limit | no |
| **spec revision** | `replace_constraint` | `wanted = revised_hard.attributes[key]`, and `public_constraints_valid` re-checks the order | **no** |
| question | `clarify_demand`, or text with `?` / confirm / reply / tell me | nothing | **yes**, before the order |

The revision is the trap: `public_revision_utterance` contains no `?` and none of `_REPLY_REQUEST`'s
verbs, so `_interventions_answered` never asks for an acknowledgement — it wants **compliance**, silently.
An agent that scans shopper text only for `"?"` and a budget figure sees nothing and keeps judging
against the old spec, which fails `attribute_satisfied` at the order.

### Agent workflow

**Events in every TF1 task** (fixed per task, not random)
- **Market event:** only the first successful `add_to_cart` triggers it, once per episode. That
  item's price jumps above budget, reported in the add's own `market_event`.
- **Scripted shopper messages:** triggered by turn number (turn 3–5), with or without a cart
  add. A question ("which one… does it have the X-inch screen?") and a budget cut.
- **Reply:** a `message` call after the question and before `place_test_order`.

```
T0  BRIEF      LLM before turn 1: category, spec, budget, currency, preferred brand, keywords
               budget unreadable → plain LLM tool loop (Engine.run) instead
T1  SEARCH     search × keywords (only queries not yet sent at this budget)
               + filter(category, brand) + filter(category)
               all at max_price = budget, in_stock = true, k = 10
T2  COMPARE    compare rows not yet opened: in stock and ≤ budget, preferred brand first, 5 per call
T3  JUDGE +    LLM judge on new variants (title, brand, category_path, facts, options), 40 per call
               → valid, preferred, confidence; its extra keywords feed the next search round
    COMMIT     pick   = preferred valid item, highest confidence then cheapest
               backup = valid item on another product, same ranking
               no pick or no backup → search again, else commit with what was found
               add_to_cart(pick) + add_to_cart(backup)
               already committed (a budget-cut round only refreshed candidates) → straight to VERIFY
               nothing valid at all → inspect_cart until turn 6, then FALLBACK
T4  VERIFY     final = best valid item: not the repriced product, in stock, ≤ budget in force
               (highest confidence, then already in the cart, then cheapest)
               add_to_cart(final) if not in cart + inspect_stock(final)
T5  ORDER      re-rank with the fresh state → same item: place_test_order(final)
                                            → another item: VERIFY it first
FALLBACK   nothing ever judged valid and nothing left to search: order the best live item seen
           (in stock, ≤ budget in force; highest confidence then cheapest), after reading its state
T≤4 left  RESCUE  stop searching; commit / order the best valid item

SEARCH AGAIN only while all three hold: under 3 rounds (or a budget cut just landed), not in rescue,
             and some query or the filter pair has not been sent at the budget in force

INTERRUPT (read in every response, handled in the next turn with that step's calls)
  question        → message(current best item, its spec, price, budget), before any order
  budget cut      → budget = min(budget, new); no valid backup → one more search round at the new budget
  market event    → repriced product excluded from backup and final
  nothing to order → inspect_cart until turn 6, so shopper updates can still arrive
```

- **valid:** judged by the LLM against every firm requirement (category, spec in the variant
  options, facts or title, the device itself); price and stock checked in code.
- **confidence:** 1.0 spec with its unit in options or facts · 0.8 only in a clean title ·
  0.5 implied, garbled or inferred. Ranking is confidence first, then — for the final item only —
  already in the cart, then cheapest.
- **final:** ordered only after its price and stock were re-read; explicit `product_id` and `sku`,
  never an unjudged item, no `justification`.
  
### Workflow diagram

Kept in sync with [`own.md`](own.md), which also carries the TF4 and TF7 diagrams.

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

## TF4 preference_reasoning

**Scoring summary**
- The reward is the final item's preference score — **graded, not binary** — computed with fixed
  per-category rules from visible product facts, the title, and (smartwatches) the brand.
- It only counts if the agent opened items properly:
  - the first pick before its price changed,
  - acceptable items from at least 2 different **source listings**,
  - the final item, and read it again after the change.
- The final item must sit on a **different source listing** from the repriced one — unconditionally,
  whatever the new price turned out to be (`recovery_source_changed`).
- The first cart add must also score at least 0.65.
- When the contract sets `requires_priority_clarification`, the shopper's sealed-facts answer must
  arrive **before that first cart add**, or the metric is forced to 0.
- Reaching 1.0 means topping out every axis of that category's profile — and `battery` is tiered, so
  merely having a battery figure is not enough (see **Core problem** below).

**Core problem — the preferences are sealed until you ask.** The query gives only a use case and one
lead preference; `_REVEAL_POLICY` tells the shopper to "lead with your use case; go into specific
preferences only when asked". Asking is a **hard gate**, not just good practice: when the contract sets
`requires_priority_clarification`, a `clarify` reply with `reason == "sealed_facts_answer"` must land
**before the first cart add**, or `family_metric` is forced to `0.0` no matter how good the item is.

But the reward itself never reads that answer. `preference_score` is computed from the catalog alone
— visible product facts, the title, and (for smartwatches) the brand — with **fixed weights that
depend on the category**, so the shopper's reply is what tells the agent *which axes to optimise*,
not what the score is measured against:

| profile | axes and weights |
|---|---|
| `Smartwatches` (the default) | navigation .32 · rugged/water .26 · battery .26 · outdoor brand .16 · −0.2 kids penalty |
| `Headphones` | sweat/water .30 · sport fit .28 · battery .22 · wireless .20 |
| `Tablets` | kid-ready .40 · battery .25 · size fit .20 · wifi .15 |

`battery` is **tiered, not binary** (`_short_duration_axis`: 0 / .25 / .5 / .75 / 1.0), so "has all four
features" does not reach 1.0 — headphones need ≥ 15 h or a figure in days, smartwatches ≥ 10 days
(`_duration_axis`). `outdoor brand` is a hardcoded set `{garmin, amazfit, fitbit, kospet}`.

### Agent workflow

**Events in every TF4 task** (fixed per task, not random)
- **Market event:** only the first successful `add_to_cart` triggers it, once per episode. That
  item's price jumps above budget.
- **Scripted shopper messages:** none (no scripted question, no budget cut).
- **Shopper replies:** asked about priorities, the shopper answers with fixed facts: the use case,
  the four preferences of **that category's profile** and their order of importance (for Headphones:
  sweat/water resistance, secure sport fit, battery, wireless). It also reacts to the reprice and
  sometimes pushes back; neither is graded.

```
T0  BRIEF      LLM call before turn 1: hard requirements (category, budget, in stock, real device),
               use case, lead preference → priority question built from them
T1  ASK        message(priority question; a direct question, so the shopper answers)
               shopper reply arrives in the T1 response
               → combine brief + reply: all preferences, priority order, search keywords
                 (no reply → brief only)
T2  SEARCH     search + filter calls with the combined keywords, in one turn
T3  JUDGE      compare candidates from ≥ 2 different source listings → LLM judge between turns
               → pick = best preference fit (strong on several priorities),
                 backup = best fit on another source listing
T4  COMMIT     add_to_cart(pick)  (+ backup)
T5  FINAL PICK is any event still unhandled?
                 no  → inspect_stock(final) + place_test_order(final)
                 yes → handle it in this turn (see INTERRUPT), then inspect_stock(final) + place_test_order(final)
T≤4 left  RESCUE  commit / order the best-fitting valid item read so far

INTERRUPT (any turn, when a response brings an event)
  handled in the next turn, together with that step's calls, and then marked handled
  step = SEARCH / JUDGE : question / pushback → message(current best candidate)
  step = COMMIT done    : market event → switch to backup on a different source listing
                          (inspect_stock(backup) + add_to_cart(backup))
```

- **valid:** a real device of the task's category (not an accessory), in stock, price ≤ budget.
- **pick / backup:** valid, ranked by fit to the preferences.
- **final:** the backup. Its **source listing** must differ from the repriced item's, it must have
  been opened with `compare`/`view` earlier, and its price and stock are read again with
  `inspect_stock` in the final turn before the order. No `justification`.

**Listing identity** — the validator does not compare `product_id`, it compares
`catalog.source_listing_key(product_id)`, i.e. the `source_listing_id` the product was projected
from. Several `product_id`s can share one source listing, so a "different product" is not
automatically a different listing. Both TF4 gates use this key:
`len(observed_eligible_sources) >= 2` and `recovery_source_changed`.

**Differences from TF1**
- **Priorities:** the query gives only the use case and lead preference; ask in T1, then build
  keywords from the brief plus the shopper's reply (one extra turn before searching).
- **Ranking:** rank by preference fit instead of preferred brand.
- **Listings:** open acceptable items from at least 2 different source listings.
- **Backup:** must be on a different source listing — a sibling SKU of the same listing fails,
  and so does another `product_id` projected from the same `source_listing_id`.
- **Final turn:** read the final item's price and stock again, with `inspect_stock` in the same
  turn before the order.

---

## TF7 justification

Scoring read from `oro-env-runtime` **1.0.6** (the current validator pin), `families/justification.py`.

**Scoring summary** (binary: 1.0 or 0)
- **Hard checks:** valid item (category, required spec, ≤ budget, in stock), as in TF1.
- **Spec in the facts:** the required spec (resolution, screen size, …) must appear in the product's
  **visible facts**, not only the title. Title-only listings are decoys. Refurbished, pre-owned or
  renewed listings are not valid.
- **First add:** must be a valid item whose spec is fact-backed; the market reprices it.
- **Justification** on `place_test_order`, shape `{"claims": [{"field", "value"}, ...]}` only
  (no summary, no evidence text). Every claim must check out, and it must include:
  - the **spec field** (e.g. `resolution`, `screen_size`) with a value matching the visible fact,
  - **`price`**: the exact price at order (or `at most X` / `under X` that holds),
  - **`in_stock`**: `true`,
  - **one more true claim from a fixed set**: `brand` (the exact catalog brand), or one of
    `streaming`, `microphone`, `privacy_cover`, `autofocus`, `hdr` (`_SOFT_FIELDS`). A true claim on
    any other field is tolerated — it is dropped as `:unsupported_field` rather than failing the
    order — but it does **not** satisfy this, and without one the order still scores 0.
- **Evidence:** the first pick was opened (`view`/`compare`) before the reprice; the final item was
  opened before the order; its price and stock were read (`view`/`compare`/`inspect_stock`) after
  the reprice.
- **Market-change handling:** the order goes out a turn after the reprice was seen, and not on the
  repriced `product_id::sku` when the new price cleared the budget (see
  **Repriced item vs final item** above). A different SKU of the same listing is fine here.

### Agent workflow

**Events in every TF7 task** (fixed per task, not random)
- **Market event:** only the first successful `add_to_cart` triggers it; that item's price jumps
  above budget.
- **Scripted shopper messages:** none. The shopper may push back; it is not graded.

```
T0  PLAN       LLM before turn 1 (retried once), reading the task's own tool schemas:
                 which tool opens one listing, which opens many (+ its list parameter and per-call
                 limit), which reads one item's live state, which adds, removes and orders;
                 the add tool's parameters that identify one exact item (the item key);
                 budget, in-stock rule, kind of product, specification, brand, and a list of
                 ready-made search calls, each checked against the real schema before use
               no usable plan → plain LLM tool loop (Engine.run) instead
T1  SEARCH     the planned search / filter calls, one turn, ≤ max_calls
               the rows teach the price label (the row's only plain number field) and the stock
               label (its only true/false field) — no field names are assumed
    SCREEN     LLM over the rows (≤ 60, 160 chars each) → the listings that really are the kind of
               product wanted, best first; then re-sorted so listings with the fewest empty
               filterable fields come first and listings priced wholly over budget come last
               top 15 opened now · next 15 held in reserve
T2-T3 OPEN     details calls over those listings (batched when the task has a many-listings tool),
               ≤ 2 turns; every observation is flattened to `label: value` lines and recorded
    JUDGE      one LLM call (retried on half the items when it returns nothing) → shortlist ≤ 8,
               each item quoting its specification, price, availability and ≤ 3 extra facts;
               every quote is resolved back to a line already recorded, or it is discarded
    RANK       proof = the shortest recorded line that states the spec and can be cited as a fact
                       (a listing fact label, ≤ 8 words — not a variant option line)
               extra = the item's first filterable text field (normally brand), else a judge reason
                       whose figures agree with the variant's own lines
               ordered by: kind mismatch · brand miss · empty filterable fields · proof not citable ·
                           spec not corroborated by a second line · wording drift · judge position
               backups = unshortlisted items whose line under the same spec label states the whole
                         spec, of the right kind and brand, currently in stock and within budget
T4  COMMIT     one attempt per turn, ≤ 4: add(choice) [+ remove(previous)] together with a live-state
               read of choice and ≤ 3 backups
                 add errored                                  → drop it, next candidate
                 fresh state over budget / out of stock (the reprice) → next candidate, same shape
                 nothing left → open the reserved listings and judge once more
                 still nothing → buy the best remaining item seen that is in stock and within price
T5  ORDER      a separate turn, so the order always lands after the reprice was seen:
               order(final, claims) — claims = proof · extra · the price line · the stock line,
               deduped by label, values copied verbatim from the lines read in the commit turn
               (3–4 claims; the claim container and its two field names are read off the order
                tool's own schema, not hardcoded)

GUARDS  no rescue phase: each step stops at max_steps − 1 … − 3, and the LLM is capped at 6 calls
        and 150 s (40 s per call), so the order turn is always reachable
        no shopper messages are sent — TF7 scripts none, and pushback is not graded
```

**Verifiable requirement** (the key to this family, in two parts)
1. **Pick a verifiable item:** the spec must be in the visible facts for both the first add and the
   final item; a title-only spec scores 0 even with perfect claims. The ranking therefore drops any
   candidate whose only spec line is a variant option or a long title-like line (`_unclaimable`), and
   prefers one whose spec is corroborated by a second line of the same item.
2. **Build claims that can't fail:** the claim's field and value are both `line.partition(": ")`
   halves of a line the agent actually read — no key list, no mapping table, no normalization and no
   rewrite by the LLM. The validator resolves fact labels and values itself.
   - spec: the fact label the judge quoted **for that item** (labels differ between listings)
   - `price` / `in_stock`: the labels learned from the search rows and the state reads, with the
     values from the commit turn's own read
   - extra: the item's first filterable text field — in this catalog the `brand` line
   - nothing more: any single unsupported claim fails the task

**Differences from TF1**
- **No preferred brand:** the pick is simply the best valid item (a brand is honoured only when the
  request names one and that name really appears in the query).
- **Spec source:** valid only when the spec is in the product facts; a title mention is not enough.
- **Order:** carries a structured justification; TF1 and TF4 orders send none.
- **Nothing about the catalog is hardcoded:** TF1 calls `search` / `compare` / `add_to_cart` /
  `place_test_order` by name and reads fixed row fields; TF7 derives every tool name, item-identifying
  parameter, price label, stock label and claim field from the task's schemas and its own
  observations, so it survives a renamed tool or a renamed field.

### Workflow diagram

See [`own.md`](own.md), which holds the TF1, TF4 and TF7 diagrams side by side.
