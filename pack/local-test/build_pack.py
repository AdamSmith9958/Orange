"""Rebuild an ORO qualifying-format env pack from released episode ledgers.

Run inside ghcr.io/oro-ai/oro/validator:stable on the search-server network:
  python build_pack.py --ledgers /led --out /out/race1 --ref-pack /q/epoch [--truth /q/epoch]

Every TaskSpec field comes from the ledgers (goal, events, scripted shopper lines, verifier
family metrics). Acceptance sets are recomputed from the catalog with calibrated rules and then
pinned by the race verdicts (final_in_gold) for every item the race agent actually ordered.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from oro_env_runtime.acceptance import (
    candidate_satisfies_hard,
    category_matches,
    deterministic_event_multiplier,
    event_price_after,
)
from oro_env_runtime.attributes import ATTRIBUTE_CAPABILITIES, attribute_values, resolve_attribute
from oro_env_runtime.delivery import _sanitize_qualifying_task
from oro_env_runtime.device_semantics import smartphone_supported
from oro_env_runtime.families import justification as jf
from oro_env_runtime.families import ranking as rk
from oro_env_runtime.families import recovery as rc
from oro_env_runtime.families import retrieval_recall as rr
from oro_env_runtime.families.base import title_token_set
from oro_env_runtime.families.constraint_satisfaction import _cat_word
from oro_env_runtime.families.intent_decomposition import (
    INTENT_REVISION_VERSION,
    public_revision_utterance,
)
from oro_env_runtime.indexed_catalog import MAX_CACHE_LIMIT, IndexedCatalog
from oro_env_runtime.pack import fingerprint, write_archive, write_checksums, write_json, write_jsonl
from oro_env_runtime.public_contract import public_task_query
from oro_env_runtime.schema import CandidateRef, EventRule, HardConstraints, LedgerEntry, TaskSpec
from oro_env_runtime.search_client import SearchServerClient
from oro_env_runtime.verify import verify

FAMILY_ORDER = [
    "intent_decomposition",
    "retrieval_recall",
    "constraint_satisfaction",
    "ranking",
    "recovery",
    "justification",
]
PRICE_EVENT_FAMILIES = {"intent_decomposition", "justification"}
MULTIPLIER_RANGE = {"intent_decomposition": (20, 40), "justification": (18, 38)}
RULES_SEP = "\n\nTask rules:\n"


def ref_of(key: str) -> CandidateRef:
    pid, _, sku = key.partition("::")
    return CandidateRef(product_id=pid, sku=sku)


# ----------------------------------------------------------------------------- ledger reading
def load_episodes(root: Path) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        ep = json.loads(path.read_text())
        ep = ep.get("episode", ep)
        by_task[ep["task_id"]].append(ep)
    for eps in by_task.values():
        eps.sort(key=lambda e: (e.get("verdict") is None, e.get("outcome") != "completed"))
    return by_task


def merged_fm(eps: list[dict]) -> dict:
    out: dict = {}
    for ep in reversed(eps):
        out.update(((ep.get("verdict") or {}).get("family_metrics")) or {})
    return out


def product_ids_in(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        pid = value.get("product_id")
        if isinstance(pid, str) and pid.isdigit():
            out.add(pid)
        for item in value.values():
            product_ids_in(item, out)
    elif isinstance(value, list):
        for item in value:
            product_ids_in(item, out)
    elif isinstance(value, str):
        for match in re.findall(r"\b(\d{6,})::", value):
            out.add(match)


def ordered_keys(ep: dict) -> list[str]:
    keys = []
    for entry in ep["ledger"]:
        if entry["kind"] == "verifier_signal":
            target = ((entry["payload"] or {}).get("order") or {}).get("target") or {}
            if target.get("product_id") and target.get("sku"):
                keys.append(f"{target['product_id']}::{target['sku']}")
    return keys


def budget_in(text: str) -> float:
    return float(re.search(r"([\d,]+(?:\.\d+)?)\s*PHP", text).group(1).replace(",", ""))


# ----------------------------------------------------------------------------- field rebuilders
def category_for(goal: str, family: str, labels: set[str]) -> str | None:
    if family == "retrieval_recall":
        m = re.search(r"(?:in|from) the (.+?) category", goal)
        if m:
            return m.group(1)
    if family == "ranking":
        if "::" in goal:
            goal = re.sub(r"\d+::[\w\-]+", "", goal)
        else:
            goal = re.sub(r"(these four)[:,]? .+?\. ", r"\1. ", goal)
    low = goal.lower()
    found = []
    for label in labels:
        forms = {_cat_word(label or ""), noun_form(label or "")}
        for word in forms:
            if len(word) < 3:
                continue
            m = re.search(r"(?<![\w-])" + re.escape(word) + r"(?:s|es)?(?![\w-])", low)
            if m:
                found.append((m.start(), -len(word), label))
    if not found:
        return None
    # The category noun is the earliest mention; a longer noun starting there wins.
    return sorted(found)[0][2]


def noun_form(label: str) -> str:
    return re.sub(r"\b(\w{3,}?)s\b", r"\1", label.lower())


_CATEGORY_EXISTS: dict[str, bool] = {}


def category_exists(catalog: IndexedCatalog, label: str) -> bool:
    if len(_cat_word(label)) < 3:
        return False
    if label not in _CATEGORY_EXISTS:
        try:
            records = catalog.search_client.filter_catalog(category=label, brand=None, max_price=None, limit=5)
            _CATEGORY_EXISTS[label] = any(
                label.casefold() in {str(x).casefold() for x in (r.get("category_path") or [])}
                for r in records
            )
        except Exception:  # noqa: BLE001
            _CATEGORY_EXISTS[label] = False
    return _CATEGORY_EXISTS[label]


def attr_candidates(text: str) -> list[tuple[str, str]]:
    out = []
    m = re.search(r"pack (?:containing|of) (\d+) (?:items|pieces)", text)
    if m:
        out.append(("pack_quantity", f"{int(m.group(1))}ct"))
    for key in ATTRIBUTE_CAPABILITIES:
        for value in sorted(attribute_values(key, text)):
            out.append((key, value))
    return out


def event_rule_for(eps: list[dict], family: str, seed: int, budget: float, notes: list[str]) -> dict:
    event = None
    for ep in eps:
        event = next((e for e in ep["ledger"] if e["kind"] == "harness_event"), None)
        if event:
            break
    payload = (event or {}).get("payload") or {}
    kind = payload.get("kind") or ("price_change" if family in PRICE_EVENT_FAMILIES else "stockout")
    rule: dict[str, Any] = {
        "contract_version": "oro_event_v3",
        "kind": kind,
        "price_multiplier": None,
        "trigger": "first_successful_cart_add",
    }
    if kind == "price_change":
        if payload.get("new_price") is not None:
            rule["price_multiplier"] = round(float(payload["new_price"]) / budget - 1.0, 2)
        else:
            low, high = MULTIPLIER_RANGE.get(family, (18, 40))
            rule["price_multiplier"] = deterministic_event_multiplier(
                family, seed, minimum_hundredths=low, maximum_hundredths=high
            )
            notes.append("event never fired; seeded price multiplier used")
        hard = HardConstraints(budget=budget, currency="PHP")
        expected = event_price_after(EventRule.model_validate(rule), hard)
        if payload.get("new_price") is not None and abs(expected - float(payload["new_price"])) > 0.011:
            notes.append(f"price multiplier mismatch {expected} vs {payload['new_price']}")
    elif event is None:
        notes.append("event never fired; stockout assumed")
    return rule


def interventions_for(eps: list[dict], notes: list[str]) -> list[dict]:
    by_index: dict[int, dict] = {}
    for ep in eps:
        for entry in ep["ledger"]:
            payload = entry.get("payload") or {}
            signal = payload.get("env_signal") or {}
            if entry["kind"] != "user_message" or signal.get("kind") != "intervention":
                continue
            index = int(signal.get("index", 0))
            if index in by_index:
                continue
            action = signal.get("action") or payload.get("action")
            content = str(payload.get("content") or "")
            if action == "add_constraint":
                delta: dict[str, Any] = {"budget": budget_in(content)}
                trigger = {"count": None, "kind": "solver_turn", "turn": int(payload.get("step") or entry["turn"])}
            elif action == "replace_constraint":
                delta = {}
                trigger = {"count": 1, "kind": "cart_add_count", "turn": None}
            else:
                notes.append(f"unhandled intervention action {action}")
                continue
            by_index[index] = {
                "action": action,
                "contract_version": "oro_intervention_v1",
                "public_delta": delta,
                "trigger": trigger,
                "utterance": content,
            }
    indexes = sorted(by_index)
    if indexes != list(range(len(indexes))):
        notes.append(f"intervention indexes {indexes}: some never delivered")
    return [by_index[i] for i in indexes]


PREF_BRAND_RES = (
    r"^(.+?) would be my pick if",
    r"I lean toward (.+?) if it fits",
    r"If an? (.+?) works I would take it",
    r"An? (.+?) would be nice, yet",
    r"I like (.+?), but only if",
    r"My soft preference is (.+?); the firm",
    r"(?:^|\. )(.+?) would be my pick",
)


def preferred_brand(goal: str) -> str | None:
    for sentence in re.split(r"(?<=\.)\s+", goal):
        for pattern in PREF_BRAND_RES:
            m = re.search(pattern, sentence)
            if m:
                return m.group(1).strip()
    return None


def _norm_name(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def resolve_named_candidates(goal, eps, fm, catalog, budget, pids, notes) -> list[str]:
    m = re.search(r"(?:these four|only these four)[:,]? (.+?)\. ", goal)
    if not m:
        notes.append("TF5 candidate names not parsed")
        return []
    names = [n.strip() for n in re.split(r", (?=[A-Z0-9])", m.group(1))]
    touched = {k for k in (fm.get("pre_event_top_key"), fm.get("post_event_top_key")) if k}
    for ep in eps:
        for e in ep["ledger"]:
            if e["kind"] == "model_action" and e["payload"].get("name") in ("add_to_cart", "place_test_order"):
                a = e["payload"].get("args") or {}
                if a.get("product_id") and a.get("sku"):
                    touched.add(f"{a['product_id']}::{a['sku']}")
    def forms(pid):
        rec = catalog.by_id(pid)
        meta = catalog.meta(ref_of(f"{pid}::{rec['sku'] if 'sku' in rec else catalog._record(pid)['sku']}"))
        from oro_env_runtime.product_facts import visible_product_facts
        title = _norm_name(rec.get("title"))
        out = {title}
        words = title.split()
        out.update(" ".join(words[:n]) for n in range(2, len(words) + 1))
        brand = _norm_name(meta.brand or "")
        if brand:
            out.update(f"{brand} {' '.join(words[:n])}" for n in range(1, len(words) + 1))
            for value in visible_product_facts(rec).values():
                out.add(_norm_name(f"{brand} {value}"))
            for value in re.split(r"\s*[/|]\s*", str(rec.get("options") or meta.options or "")):
                out.add(_norm_name(f"{brand} {value}"))
            out.update(f"{brand} {w}" for w in words)
        return out, meta
    keys = []
    for name in names:
        want = _norm_name(name)
        hits = []
        pool = set(pids)
        for attempt in range(2):
            for pid in sorted(pool):
                try:
                    f, meta = forms(pid)
                except KeyError:
                    continue
                if want in f and catalog.exists(meta.ref):
                    hits.append(meta)
            if hits:
                break
            found = catalog.search_candidates(None, name, k=50)
            pool = {r.product_id for r in found} - set(pids)
        if not hits:
            notes.append(f"TF5 name unresolved: {name}")
            continue
        hits.sort(key=lambda mt: (mt.ref.key() not in touched, mt.price > budget, mt.price, mt.ref.key()))
        if len(hits) > 1:
            notes.append(f"TF5 name {name!r} -> {len(hits)} variants, chose {hits[0].ref.key()}")
        keys.append(hits[0].ref.key())
    return keys


# ----------------------------------------------------------------------------- acceptance rules
class Acceptance:
    def __init__(self, catalog: IndexedCatalog):
        self.catalog = catalog
        self._pool_cache: dict[tuple, list[CandidateRef]] = {}

    def pool(self, category: str | None, budget: float) -> list[CandidateRef]:
        key = (category, budget)
        if key not in self._pool_cache:
            self._pool_cache[key] = self.catalog.purchasable(
                category=category, brand=None, max_price=budget, limit=200000
            )
        return self._pool_cache[key]

    def phone_ok(self, task: TaskSpec, ref: CandidateRef) -> bool:
        if (task.hard.category or "").casefold().rstrip("s") != "smartphone":
            return True
        return smartphone_supported(self.catalog.by_id(ref.product_id), options=self.catalog.meta(ref).options)

    def keys(self, task: TaskSpec, variant: dict) -> list[str]:
        cat, f = self.catalog, task.family
        if f == "ranking":
            return list(task.family_payload["_candidates"])
        refs = self.pool(task.hard.category, task.hard.budget)
        hards = [task.hard]
        revision = task.family_payload.get("intent_revision")
        if revision:
            hards.append(task.hard.model_copy(update={"attributes": {**task.hard.attributes, revision["attribute"]: revision["value"]}}))
        out = []
        for ref in refs:
            if not cat.exists(ref):
                continue
            if f == "recovery":
                row = rc._row_by_key(cat, ref.key())
                floor = 100.0 if task.hard.category == "Smartphones" else 0.0
                if row is not None and row.category == task.hard.category and floor <= row.price <= task.hard.budget:
                    out.append(ref.key())
                continue
            if not any(candidate_satisfies_hard(cat, ref, h) for h in hards):
                continue
            rec = cat.by_id(ref.product_id)
            if f == "retrieval_recall":
                out.append(ref.key())  # narrowed to theme clusters below
            elif f == "constraint_satisfaction":
                if self.phone_ok(task, ref):
                    out.append(ref.key())
            elif f == "justification":
                attr, value = jf._gate_attr_value(task)
                if (
                    jf._is_actual_item(cat, ref, str(task.hard.category or ""))
                    and resolve_attribute(rec, attr) == value
                    and jf._attr_fact_value(cat, ref, attr) == value
                    and self.phone_ok(task, ref)
                ):
                    out.append(ref.key())
            else:
                out.append(ref.key())
        if f == "retrieval_recall":
            theme = {t.lower() for t in task.family_payload["theme"]}
            refs_all = [ref_of(k) for k in sorted(set(out))]
            seeds = {
                r.key() for r in refs_all
                if theme <= title_token_set(str(cat.by_id(r.product_id).get("title") or ""))
            }
            out = [k for c in rr.relevance_clusters(cat, refs_all) if c & seeds for k in c]
        return sorted(set(out))


# ----------------------------------------------------------------------------- task build
def build_task(task_id: str, eps: list[dict], catalog: IndexedCatalog, acc: Acceptance) -> tuple[dict, list[str]]:
    notes: list[str] = []
    ep0 = eps[0]
    family = ep0["family"]
    seed = int(task_id.rsplit("-", 1)[1])
    query = ep0["bootstrap"]["policy_view"]["query"]
    goal = query.split(RULES_SEP, 1)[0]
    max_steps = int(ep0["bootstrap"]["policy_view"]["max_steps"])
    fm = merged_fm(eps)
    budget = budget_in(goal)

    pids: set[str] = set()
    for ep in eps:
        product_ids_in([e["payload"] for e in ep["ledger"]], pids)
    catalog.prefetch(sorted(pids))
    labels: set[str] = set()
    for pid in pids:
        try:
            rec = catalog._record(pid)
        except KeyError:
            continue
        labels.update(str(x) for x in (rec.get("category_path") or []) if x)
    if family == "recovery":
        labels = {p.category for p in rc._CATEGORY_PROFILES}
    category = category_for(goal, family, labels)
    if category is None and family != "recovery":
        labels = set()
        words = re.findall(r"[a-z&,\-]+", goal.lower())
        for n in range(1, 6):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n]).strip(",")
                parts = re.split(r"(, | & )", phrase)
                plural = "".join(x if x in (", ", " & ") else x + "s" for x in parts)
                labels.update({phrase.title(), (phrase + "s").title(), plural.title()})
        labels = {label for label in labels if category_exists(catalog, label)}
        category = category_for(goal, family, labels)
        notes.append(f"category from generated labels: {category}")
    if category is None:
        notes.append("category not resolved from ledger products")

    hard: dict[str, Any] = {
        "brand": None,
        "budget": budget,
        "category": category,
        "currency": "PHP",
        "require_in_stock": True,
    }
    payload: dict[str, Any] = {}
    preferred: dict[str, str] = {}
    interventions = interventions_for(eps, notes)

    if family == "intent_decomposition":
        brand = preferred_brand(goal)
        if brand is None:
            notes.append("preferred brand not parsed")
        else:
            payload.update(preferred_brand=brand, soft_prefs=[f"prefer {brand} when it remains hard-valid"])
        replace = next((i for i in interventions if i["action"] == "replace_constraint"), None)
        key = fm.get("gated_key")
        new_value = fm.get("gated_value") if replace else None
        if replace:
            m = re.search(r"must now have (.+?) instead of (.+?)\. Keep", replace["utterance"])
            new_c = attr_candidates(m.group(1)) if m else []
            old_c = attr_candidates(m.group(2)) if m else []
            if key is None and new_c:
                key = new_c[0][0]
            new_value = new_value or next((v for k, v in new_c if k == key), None)
            old_value = next((v for k, v in old_c if k == key), None)
        else:
            old_value = fm.get("gated_value")
        if old_value is None and key:
            old_value = next((v for k, v in attr_candidates(goal) if k == key), None)
        if key is None or old_value is None:
            notes.append("TF1 attribute unresolved")
        else:
            hard["attributes"] = {key: old_value}
        if replace and key and new_value:
            payload["intent_revision"] = {"attribute": key, "value": new_value, "version": INTENT_REVISION_VERSION}
            replace["public_delta"] = {"attributes": {key: new_value}}
    elif family == "constraint_satisfaction":
        key, value = fm.get("gated_key"), fm.get("gated_value")
        if key is None:
            texts = [goal] + [
                str(e["payload"].get("content") or "")
                for ep in eps
                for e in ep["ledger"]
                if e["kind"] == "user_message" and e["payload"].get("reason") == "sealed_facts_answer"
            ]
            cands = [c for t in texts for c in attr_candidates(t)]
            if cands:
                key, value = cands[0]
                notes.append(f"TF3 spec parsed from text: {key}={value} (candidates {sorted(set(cands))})")
            else:
                notes.append("TF3 spec unresolved")
        underspecified = bool(re.search(r"\bask(?:ed)?\b", goal, re.I))
        if key:
            hard["attributes"] = {key: value}
            payload.update(gated_key=key, gated_value=value, underspecified=underspecified)
    elif family == "justification":
        attr, value = fm.get("gate_attr"), fm.get("gate_value")
        if attr is None:
            cands = attr_candidates(goal)
            attr, value = cands[0] if cands else (None, None)
            notes.append(f"TF7 gate parsed from goal: {attr}={value}")
        if attr:
            hard["attributes"] = {attr: value}
            phrase = jf._value_phrase(attr, value)
            if phrase not in goal:
                notes.append(f"gate phrase {phrase!r} not in goal")
            payload["gate_phrase"] = phrase
    elif family == "retrieval_recall":
        m = re.search(r'"(.+?)"', goal) or re.search(
            r"the phrase (.+?)(?:,| for no more| at or| for at most| and buy|$)", goal
        )
        payload["theme"] = m.group(1).split() if m else []
        if not payload["theme"]:
            notes.append("TF2 theme unresolved")
    elif family == "ranking":
        keys = re.findall(r"\d+::[\w\-]+", goal)
        if not keys:
            keys = resolve_named_candidates(goal, eps, fm, catalog, budget, pids, notes)
        low = goal.lower()
        axis_words = {
            "resolution": "resolution", "screen": "screen_size", "refresh": "refresh_rate",
            "ram": "ram", "capacity": "capacity", "wattage": "wattage", "warranty": "warranty_duration",
            "pack": "pack_quantity",
        }
        rules_text = low.split(keys[-1].lower(), 1)[-1] if keys else low
        found = sorted(
            ((rules_text.find(word), axis) for word, axis in axis_words.items() if word in rules_text),
        )
        axes = []
        for _pos, axis in found:
            if axis not in axes:
                axes.append(axis)
        payload["priority"] = axes[:2] + ["price"]
        payload["_candidates"] = keys
        catalog.prefetch([k.split("::")[0] for k in keys])
        pool = [ref_of(k) for k in keys if catalog.exists(ref_of(k))]
        ax = tuple(axes[:2])
        ranked = rk._ranked_pool(catalog, [r for r in pool if rk.tradeoff_values(catalog, r, axes=ax)], axes=ax)
        pre = fm.get("pre_event_top_key") or (ranked[0].key() if ranked else None)
        pre_class = rk.tie_class_keys(catalog, pool, ref_of(pre), axes=ax) if pre else []
        rest = [r for r in ranked if r.key() not in pre_class]
        post = fm.get("post_event_top_key") or (rest[0].key() if rest else None)
        if ranked and pre != ranked[0].key():
            notes.append(f"pre-event top recomputed {ranked[0].key()} != verdict {pre}")
        post_class = rk.tie_class_keys(catalog, rest, ref_of(post), axes=ax) if post else []
        payload.update(
            pre_event_top_key=pre, pre_event_top_class=pre_class,
            post_event_top_key=post, post_event_top_class=post_class,
        )
    elif family == "recovery":
        preferred = dict(((fm.get("relaxation_detail") or {}).get("preferred")) or {})
        if not preferred:
            m = re.search(r"(?:prefer the|I like is the|I want the|list: the) (\w+)(?: version of the)? (\S+?)[\.,]", goal)
            notes.append("recovery preference parsed from goal" if m else "recovery preference unresolved")
            if m:
                preferred = {"color": m.group(1), "model": m.group(2), "relaxation_order": "color_before_model"}

    task: dict[str, Any] = {
        "acceptance": {"acceptable_keys": []},
        "event_rule": event_rule_for(eps, family, seed, budget, notes),
        "family": family,
        "family_payload": payload,
        "goal_text": goal,
        "hard": hard,
    }
    if interventions:
        task["interventions"] = interventions
    if preferred:
        task["preferred"] = preferred
    if max_steps != 30:
        task["difficulty_band"] = "hard"

    spec = TaskSpec.model_validate(copy.deepcopy(task))
    keys = set(acc.keys(spec, {}))

    # Pin acceptance with the race verdicts for every item the race agent ordered.
    for ep in eps:
        verdict = ep.get("verdict") or {}
        in_gold = (verdict.get("checks") or {}).get("final_in_gold")
        for key in ordered_keys(ep)[-1:]:
            if in_gold is True and key not in keys:
                keys.add(key)
                notes.append(f"added verdict gold {key}")
            elif in_gold is False and key in keys:
                keys.discard(key)
                notes.append(f"removed verdict non-gold {key}")
    if family == "ranking":
        keys = set(payload["_candidates"])
    task["acceptance"]["acceptable_keys"] = sorted(keys)
    payload.pop("_candidates", None)

    if family == "retrieval_recall":
        refs = [ref_of(k) for k in sorted(keys)]
        payload["relevance_clusters"] = [sorted(c) for c in rr.relevance_clusters(catalog, refs)]
        payload["retrieval_pool"] = sorted(keys)
        if fm.get("recall_pool_size") is not None and fm["recall_pool_size"] != len(keys):
            notes.append(f"TF2 pool size {len(keys)} vs race {fm['recall_pool_size']}")
    if family == "recovery" and fm.get("post_event_critical_count") is not None:
        if abs(int(fm["post_event_critical_count"]) - len(keys)) > 1:
            notes.append(f"TF6 gold size {len(keys)} vs race critical {fm['post_event_critical_count']}")

    row = {"runtime": {"max_steps": max_steps}, "split": "private_eval", "task": task, "task_id": task_id}
    row = _sanitize_qualifying_task(row)
    rebuilt = public_task_query(TaskSpec.model_validate(row["task"]))
    if rebuilt != query:
        notes.append("QUERY MISMATCH")
    return row, notes


def replay(row: dict, eps: list[dict], catalog: IndexedCatalog) -> list[tuple[float | None, float]]:
    out = []
    task = TaskSpec.model_validate(copy.deepcopy(row["task"]))
    for ep in eps:
        if not ep.get("verdict"):
            continue
        entries = []
        for entry in ep["ledger"]:
            entries.append(LedgerEntry.model_validate({k: entry[k] for k in entry if k in LedgerEntry.model_fields}))
        try:
            result = verify(task, entries, catalog, reference_count=task.reference_action_count, render_budget=ep.get("render_budget"))
            got = float(result.paid_reward or 0.0)
        except Exception as exc:  # noqa: BLE001
            print(f"   replay error {row['task_id']}: {type(exc).__name__}: {exc}")
            got = -1.0
        race_r = float(ep["verdict"].get("paid_reward") or 0.0)
        if got >= 0 and abs(race_r - got) > 1e-6:
            rv = ep["verdict"]
            diffs = {k: (v, result.checks.get(k)) for k, v in (rv.get("checks") or {}).items() if result.checks.get(k) != v}
            fdiffs = {
                k: (v, result.family_metrics.get(k))
                for k, v in (rv.get("family_metrics") or {}).items()
                if not isinstance(v, (dict, list)) and result.family_metrics.get(k) != v
            }
            print(f"   MISMATCH {row['task_id']} race={race_r} ours={got} reason race={(rv.get('reward_record') or {}).get('reason')} ours={(result.reward_record or {}).get('reason')} checks={diffs} fm={json.dumps(fdiffs)[:700]}")
        out.append((race_r, got))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-pack", required=True, help="an existing qualifying epoch dir (manifest/gate template)")
    ap.add_argument("--truth", help="real epoch dir to diff against (calibration)")
    ap.add_argument("--search", default="http://orobuild-search:5632")
    args = ap.parse_args()

    ref = Path(args.ref_pack)
    manifest = json.loads((ref / "manifest.json").read_text())
    search = SearchServerClient(args.search, expected_identity=manifest["search"], timeout_seconds=900, request_attempts=2)
    catalog = IndexedCatalog(search, currency="PHP", cache_limit=MAX_CACHE_LIMIT)
    acc = Acceptance(catalog)

    by_task = load_episodes(Path(args.ledgers))
    truth = {}
    if args.truth:
        for line in (Path(args.truth) / "data/tasks/private_tasks.jsonl").read_text().splitlines():
            r = json.loads(line)
            truth[r["task_id"]] = r

    rows, report = [], {}
    replay_match = Counter()
    for task_id in sorted(by_task, key=lambda t: (FAMILY_ORDER.index(by_task[t][0]["family"]), t)):
        eps = by_task[task_id]
        try:
            row, notes = build_task(task_id, eps, catalog, acc)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"FAILED {task_id}: {exc}")
            continue
        rows.append(row)
        rp = replay(row, eps, catalog)
        for race_r, got in rp:
            replay_match["match" if abs(race_r - got) < 1e-6 else "mismatch"] += 1
        line = f"{task_id} gold={len(row['task']['acceptance']['acceptable_keys'])} replay={rp}"
        if task_id in truth:
            t = truth[task_id]["task"]
            diffs = [n for n in ("goal_text", "hard", "event_rule", "preferred", "interventions", "family_payload")
                     if json.dumps(t.get(n), sort_keys=True) != json.dumps(row["task"].get(n), sort_keys=True)]
            tg, mg = set(t["acceptance"]["acceptable_keys"]), set(row["task"]["acceptance"]["acceptable_keys"])
            line += f" truth_gold={len(tg)} miss={len(tg - mg)} extra={len(mg - tg)} field_diffs={diffs}"
            for n in diffs:
                if n != "family_payload" or t["family"] != "retrieval_recall":
                    notes.append(f"DIFF {n}: truth={json.dumps(t.get(n))[:300]} ours={json.dumps(row['task'].get(n))[:300]}")
        print(line)
        for n in notes:
            print("    -", n)
        report[task_id] = {"notes": notes, "replay": rp}

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    epoch = out / "epoch"
    write_jsonl(epoch / "data/tasks/private_tasks.jsonl", rows)
    shutil.copy(ref / "tf4_hybrid_release_gate.json", epoch / "tf4_hybrid_release_gate.json")
    counts = Counter(r["task"]["family"] for r in rows)
    manifest["epoch"].update(
        families=[f for f in FAMILY_ORDER if f in counts],
        family_counts={f: counts[f] for f in sorted(counts)},
        tasks=len(rows),
        task_set_fingerprint=fingerprint([
            {"task_id": r["task_id"], "task_fingerprint": fingerprint(r["task"])} for r in rows
        ]),
    )
    manifest["delivery"]["task_ids"] = [r["task_id"] for r in rows]
    write_json(epoch / "manifest.json", manifest)
    write_checksums(epoch)
    archive = write_archive(epoch)
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    print("replay", dict(replay_match))
    print("archive", archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
