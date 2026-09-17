from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any

from src.agent.proxy_client import ProxyClient

_proxy = ProxyClient(timeout=120, max_retries=2)


class Verdict:
    MET = "met"
    UNMET = "unmet"
    UNVERIFIED = "unverified"
    ALL = (MET, UNMET, UNVERIFIED)


class Kind:
    BUDGET = "budget"
    STOCK = "stock"
    CATEGORY = "category"
    BRAND = "brand"
    PHRASE = "phrase"
    SPEC = "spec"
    CHECKED = (BUDGET, STOCK, CATEGORY, BRAND, PHRASE, SPEC)
    STATE = (BUDGET, STOCK)


class Source:
    OPTIONS = "options"
    FACTS = "facts"
    TITLE = "title"
    PATH = "category_path"


FILTER_TOOL = "filter"
SEARCH_TOOL = "search"
SEARCH_TOOLS = (FILTER_TOOL, SEARCH_TOOL)
COMPARE_TOOL = "compare"
ADD_TOOL = "add_to_cart"
INSPECT_TOOL = "inspect_stock"
INSPECT_CART_TOOL = "inspect_cart"
REMOVE_TOOL = "remove_from_cart"
MESSAGE_TOOL = "message"
ORDER_TOOL = "place_test_order"
CART_TOOLS = (ADD_TOOL, REMOVE_TOOL, ORDER_TOOL)
ORDER_MESSAGE = "I am ordering {title} (SKU {sku}) at {price} {currency}. {because}"
COMPARE_CAP = 5
STALL_TURNS = 5
SPEC_JUDGE_BATCH = 5
CONDITION_ID = "C{index}"



class Prompt:
    SHOP = """
Use the supplied shopping tools to satisfy the shopper. Continue until an observation reports done=true.

{query}
""".strip()

    CONTINUE = """
The environment has not reported done=true. Choose one of the supplied tools to continue.
""".strip()

    CLARIFY = """
You are AI Assistant which analyze the user shopping query. Answer what kind of shopping task is this request, with a single name from this list and nothing else:
- IntentDecomposition: firm requirements mixed with nice-to-haves, including a preferred brand
- RetrievalRecall: the item is described in words and the catalog must be searched for matches
- ConstraintSatisfaction: every requirement is firm and checked against the exact variant bought
- PreferenceReasoning: one loose requirement plus competing soft priorities the shopper must rank
- Ranking: a set of candidates is ordered by a tradeoff the shopper states
- Recovery: one exact item is named and must be replaced sensibly once unavailable
- Justification: a valid purchase plus a reason that holds up against the product facts

Shopping query:
{query}
""".strip()

    INTENT_BRIEF = """
You read a shopping request and split it into firm requirements and a soft brand preference. Reply with one JSON object and nothing else:
{{"category": "the catalog category name, e.g. Smartphones",
 "spec": "the one required product spec as the shopper wrote it, e.g. 7.6-inch screen",
 "budget": the maximum price as a number,
 "currency": "the currency code",
 "brand": "the preferred brand, or null",
 "keywords": ["3 to 5 short catalog search queries"]}}

Shopping request:
{query}
""".strip()

    INTENT_UPDATE = """
A shopper stated firm requirements and has since sent more messages. Work out the requirements as they stand now: a later
statement replaces an earlier value, and every value the shopper did not change stays as it was. Messages that only ask
about or react to an item change nothing. Reply with one JSON object and nothing else, in the same shape as the current
requirements, with "keywords" holding 3 to 5 short catalog search queries for the requirements as they stand now.

Current requirements:
{requirements}

Shopper messages, oldest first:
{messages}
""".strip()

    INTENT_JUDGE = """
You check catalog variants against a shopper's firm requirements. Reply with one JSON object and nothing else:
{{"items": {{"<variant key>": {{"valid": true if the variant meets EVERY firm requirement: the right category, the required spec shown in this variant's options or in the product facts (the title alone is not enough), and the device itself rather than an accessory (used listings or listings sold without accessories still count),
                             "preferred": true if its brand is the preferred brand,
                             "clear": true if the required spec value is stated plainly (options, facts, or a normally written title), false if it is only implied, garbled or guessed,
                             "spec_amount": the amount this variant states for the required spec, converted to the unit the requirement uses, as a number, or null if it states none}}}},
 "keywords": ["up to 3 new search queries if fewer than 2 valid products are listed"]}}
Price and stock are checked separately; ignore them.

Requirements:
{requirements}

Variants:
{variants}
""".strip()

    REQUIREMENTS = """
Report every condition the ordered item must meet that the text below states, using report_requirements.
Report brand and category as separate conditions. Copy each value verbatim from the text.
Report nothing else: a wish, a question, a remark about an item already seen, and anything the text does not state as a condition of the order are all left out, and it is right to report none at all.
{hidden}

Current conditions:
{current}

Text:
{text}
""".strip()

    REPORT_REQUIREMENTS = {"type": "function", "function": {
        "name": "report_requirements",
        "description": "Report every condition the ordered item must meet that the text states.",
        "parameters": {"type": "object", "required": ["requirements"], "properties": {"requirements": {
            "type": "array", "items": {
                "type": "object",
                "required": ["kind", "value", "hidden", "replaces"],
                "properties": {
                    "kind": {"type": "string", "enum": list(Kind.CHECKED),
                             "description": "budget: price limit; stock: availability; category: product type or catalog "
                                            "category, including the noun naming the item, without the brand; brand: the "
                                            "brand name only, without the product type; phrase: text the listing title must "
                                            "match; spec: product specification."},
                    "value": {"type": ["string", "null"],
                              "description": "Copied verbatim from the text. budget and spec: the full condition phrase, "
                                             "with its number, its unit and, for a budget, the currency code written beside "
                                             "it. brand, category and phrase: only the brand name, the product type or "
                                             "category, or the phrase itself, without the words that introduce it. null "
                                             "when withheld."},
                    "hidden": {"type": "boolean", "description": "True when the shopper withholds the value until asked."},
                    "replaces": {"type": ["string", "null"],
                                 "description": "The id of the current condition this condition reveals, restates or "
                                                "changes. A later message usually repeats current conditions, so give "
                                                "the id whenever the condition is about the same thing as a current one, "
                                                "even when it is worded differently; null only for a condition that is "
                                                "not among the current ones."},
                }}}}}}}

    SPECS = """
For every variant below and every spec requirement, report each of the variant's evidence lines that states a value for the requirement's concept, using report_statements.
The evidence lines of a variant are its options, the product title, and each facts line. Check every line: the title and the facts can state the concept even when the options do not. For each line that states a value for the concept, copy the stated text exactly, name the field it comes from, and give the line's verdict against the required value:
- met: the line states the required value, in any notation or unit that means exactly that value;
- unmet: the line states a different value;
- unverified: the line states several values, a broader or vaguer term that covers several values, or a number without its unit.
Do not report lines that state no value for the concept. A figure written with no word for what it counts states a value for nothing.

Requirements:
{requirements}

Variants:
{variants}
""".strip()

    REPORT_STATEMENTS = {"type": "function", "function": {
        "name": "report_statements",
        "description": "Report the evidence lines of each variant that state a value for each spec requirement.",
        "parameters": {"type": "object", "required": ["reports"], "properties": {"reports": {
            "type": "array", "items": {
                "type": "object",
                "required": ["variant", "requirement", "statements"],
                "properties": {
                    "variant": {"type": "string", "description": "The variant key exactly as listed."},
                    "requirement": {"type": "string", "description": "The requirement id exactly as listed."},
                    "statements": {"type": "array", "items": {
                        "type": "object",
                        "required": ["field", "quote", "verdict"],
                        "properties": {
                            "field": {"type": "string", "description": "options, title, or the facts key the line has."},
                            "quote": {"type": "string", "description": "Text copied exactly from that line that states the value."},
                            "verdict": {"type": "string", "enum": list(Verdict.ALL)},
                        }}},
                }}}}}}}

    CATEGORY_LABEL = """
The shopper asked for a product type. The catalog files every product under its own classification names, listed below.
Name the one that is the same classification as the shopper's, using name_category: the same name, or another name for
exactly the same kind of product. A name that only shares a word with the shopper's, or that is broader or narrower than
it, is a different classification. Answer null when none of them is it.

The shopper's product type: {category}

The catalog's classification names:
{labels}
""".strip()

    CHOOSE = """
You are buying one item for a shopper. This is what they asked for:
{query}

The conditions they gave:
{requirements}

Every item below is one the catalog actually returned, read as the catalog wrote it. Choose the one to buy, using
choose_variant, and give the one sentence you would tell the shopper about why it fits. Where an item is marked as
falling short, weigh that against what the shopper asked for; where none of them meets everything, choose the one that
serves them best and let your sentence say what it does not meet.

Items:
{items}
""".strip()

    CHOOSE_VARIANT = {"type": "function", "function": {
        "name": "choose_variant",
        "description": "Choose the one item to buy for the shopper and say why it fits.",
        "parameters": {"type": "object", "required": ["variant", "because"], "properties": {
            "variant": {"type": "string", "description": "The item key exactly as listed."},
            "because": {"type": "string", "description": "One sentence for the shopper on why this item fits."},
        }}}}

    NAME_CATEGORY = {"type": "function", "function": {
        "name": "name_category",
        "description": "Name the catalog classification that is the shopper's product type, or null when none is.",
        "parameters": {"type": "object", "required": ["label"], "properties": {
            "label": {"type": ["string", "null"], "description": "One of the listed names, exactly as listed, or null."},
        }}}}


class Utils:
    MODELS = (
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-chat-v3.1",
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "qwen/qwen3.5-397b-a17b",
        "z-ai/glm-5.1",
    )


    @staticmethod
    def _split_query(query: str) -> tuple[str, str]:
        request, _, rules = query.partition("Task rules:")
        return request.strip(), rules.strip()

    @staticmethod
    def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
        raw = tool_call["function"].get("arguments", "{}")
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed

    @staticmethod
    def _page_size(tools: list[dict[str, Any]], name: str) -> int | None:
        schema = next((tool["function"] for tool in tools if tool["function"]["name"] == name), {})
        return (((schema.get("parameters") or {}).get("properties") or {}).get("k") or {}).get("maximum")

    @staticmethod
    def _tool_names(engine: Engine) -> set[str]:
        return {tool["function"]["name"] for tool in engine.tools}

    @staticmethod
    def _value(requirements: list[Requirement], kind: str) -> str | None:
        return next((r.value for r in requirements if r.kind == kind and r.value), None)

    @staticmethod
    def _unquoted(value: str) -> str:
        return value.strip().strip("\"'“”")

    @staticmethod
    def _ids(ref: tuple[str, str]) -> dict[str, str]:
        return {"product_id": ref[0], "sku": ref[1]}

    @staticmethod
    def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

    @staticmethod
    def _limit_calls(calls: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        return calls[:limit]

    @staticmethod
    def _lookup_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [tool for tool in tools if tool["function"]["name"] not in CART_TOOLS]


    @staticmethod
    def _grounded(text: str, source: str) -> bool:
        wanted = Utils._words(text)
        if not wanted:
            return False
        if " ".join(str(text).casefold().split()) in " ".join(str(source).casefold().split()):
            return True
        return wanted <= Utils._words(source)

    @staticmethod
    def _grounded_limit(text: str, source: str) -> bool:
        numbers = set(Utils._number_tokens(text))
        if not numbers:
            return False
        words = Utils._words(source)
        return numbers <= set(Utils._number_tokens(source)) and {
            word for word in Utils._words(text) if len(word) == 3 and word.isalpha()} <= words

    @staticmethod
    def _words(value: Any, plural: bool = False) -> set[str]:
        tokens = "".join(char if char.isalnum() else " " for char in str(value or "").casefold()).split()
        return {t[:-1] if plural and len(t) > 3 and t.endswith("s") else t for t in tokens}

    @staticmethod
    def _words_match(wanted: str, available: Any, plural: bool = False) -> str:
        wanted_words = Utils._words(wanted, plural)
        if not wanted_words:
            return Verdict.UNVERIFIED
        return Verdict.MET if wanted_words <= Utils._words(available, plural) else Verdict.UNMET

    @staticmethod
    def _contains_words(text: Any, quote: Any) -> bool:
        def words(value: Any) -> list[str]:
            normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
            return "".join(char if char.isalnum() else " " for char in normalized).split()

        text_words, quote_words = words(text), words(quote)
        return bool(quote_words) and any(
            text_words[start:start + len(quote_words)] == quote_words
            for start in range(len(text_words) - len(quote_words) + 1)
        )

    @staticmethod
    def _label(text: Any) -> str:
        return " ".join("".join(char if char.isascii() and char.isalnum() else " " for char in str(text or "").casefold()).split())

    @staticmethod
    def _number_tokens(text: str) -> list[Decimal]:
        found, current = [], ""
        for index, char in enumerate(f"{text} "):
            if char.isdigit() or (char == "." and current and "." not in current):
                current += char
                continue
            if char == "," and current and text[index + 1 : index + 2].isdigit():
                continue
            if current.rstrip("."):
                found.append(Decimal(current.rstrip(".")))
            current = ""
        return found

    @staticmethod
    def _number_units(text: Any) -> set[tuple[Decimal, str]]:
        text, found, index = str(text or ""), set(), 0
        while index < len(text):
            if not text[index].isdigit():
                index += 1
                continue
            end = index
            while end < len(text) and (text[end].isdigit() or (text[end] in ".," and text[end + 1 : end + 2].isdigit())):
                end += 1
            after = end
            while after < len(text) and text[after] in " -":
                after += 1
            unit_end = after
            while unit_end < len(text) and text[unit_end].isalpha():
                unit_end += 1
            unit = text[after:unit_end].casefold() or ('"' if text[after : after + 1] in ("\"", "″", "”") else "")
            if unit and not (index and text[index - 1].isalpha()):
                try:
                    found.add((Decimal(text[index:end].replace(",", "")), unit))
                except InvalidOperation:
                    pass
            index = end
        return found

    @staticmethod
    def _same_unit(unit: str, other: str) -> bool:
        short, long = sorted((unit, other), key=len)
        return unit == other or (len(short) > 1 and long.startswith(short))

    @staticmethod
    def _states_spec(text: Any, spec: str) -> bool:
        wanted = Utils._number_units(spec)
        if not wanted:
            return Utils._words_match(spec, text) == Verdict.MET
        found = Utils._number_units(text)
        return all(any(number == got and Utils._same_unit(unit, have) for got, have in found) for number, unit in wanted)

    @staticmethod
    def _same_spec(spec: str, other: str) -> bool:
        if Utils._number_units(spec) and Utils._number_units(other):
            return Utils._states_spec(spec, other) and Utils._states_spec(other, spec)
        words, others = Utils._words(spec, plural=True), Utils._words(other, plural=True)
        return bool(words and others) and (words <= others or others <= words)


    @staticmethod
    def _budget_terms(value: str) -> tuple[str | None, str | None, str | None]:
        numbers = Utils._number_tokens(value)
        if not numbers:
            return None, None, "no number in the budget phrase"
        codes = {word for word in value.replace(",", " ").split()
                 if len(word) == 3 and word.isalpha() and word.isupper()}
        return f"{min(numbers)}", codes.pop() if len(codes) == 1 else None, None

    @staticmethod
    def _extract_requirements(text: str, current: list[Requirement], turn: int, feedback: str = "") -> list[Requirement]:
        prompt = Prompt.REQUIREMENTS.format(
            hidden=("A condition the shopper withholds until asked is hidden; report one hidden condition for each withheld "
                    "condition the text counts." if turn == 0
                    else "Nothing in this message is hidden. A question that does not ask for a requirement states no "
                         "condition.") + (f"\n{feedback}" if feedback else ""),
            current="\n".join(f"- {r.id} [{r.kind}] = {r.value!r}{' (hidden)' if r.hidden else ''}" for r in current)
            or "- none",
            text=text,
        )
        schema = json.loads(json.dumps(Prompt.REPORT_REQUIREMENTS))
        schema["function"]["parameters"]["properties"]["requirements"]["items"]["properties"]["replaces"]["enum"] = [
            *(r.id for r in current), None
        ]
        answer = Utils._llm_arguments(prompt, schema, "report_requirements", f"turn {turn} requirement extraction failed")
        if answer is None:
            return []
        items = answer.get("requirements") or []

        requirements: list[Requirement] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "")
            value = None if item.get("value") in (None, "") else str(item["value"])
            hidden = turn == 0 and (item.get("hidden") is True or str(item.get("hidden")).strip().casefold() == "true")
            if hidden and value is not None and value.strip().casefold() in ("null", "none"):
                value = None
            label = f"turn {turn} [{kind}] = {value!r}"
            if kind not in Kind.CHECKED:
                problem = f"unknown kind {kind!r}"
            elif hidden and value is not None:
                problem = "hidden but has a value"
            elif not hidden and value is None:
                problem = "no value although not hidden"
            elif value is not None and not (Utils._grounded(value, text)
                                            or (kind == Kind.BUDGET and Utils._grounded_limit(value, text))):
                problem = "value not found in the text"
            else:
                problem = None
            amount = currency = None
            if problem is None and kind == Kind.BUDGET and not hidden:
                amount, currency, problem = Utils._budget_terms(value or "")
            if problem:
                continue
            replaces = str(item.get("replaces") or "") or None
            requirements.append(Requirement(kind=kind, value=value, hidden=hidden, turn=turn,
                                            amount=amount, currency=currency, replaces=replaces))
            terms = f" (<= {amount} {currency})" if amount else ""
        return requirements

    @staticmethod
    def _merge_requirements(requirements: list[Requirement], updates: list[Requirement]) -> list[Requirement]:
        merged = list(requirements)
        for update in updates:
            target = next((r for r in merged if r.id and r.id == update.replaces and r.kind == update.kind), None)
            if target is None:
                target = next((r for r in merged if r.kind == update.kind and not r.hidden
                               and r.check_key() == update.check_key()), None)
            if target is None and update.kind == Kind.SPEC and update.turn > 0:
                target = next((r for r in merged if r.kind == Kind.SPEC and r.hidden), None)
            if target is None and update.kind != Kind.SPEC:
                target = next((r for r in merged if r.kind == update.kind), None)
            if target is not None and update.turn < target.turn:
                continue
            if update.kind == Kind.BUDGET and update.currency is None:
                previous = next((r for r in merged if r.kind == Kind.BUDGET and r.currency), None)
                if previous is not None:
                    update = replace(update, currency=previous.currency)
            if target is None:
                used = {r.id for r in merged}
                new_id = next(CONDITION_ID.format(index=i) for i in range(1, len(merged) + 2) if CONDITION_ID.format(index=i) not in used)
                merged.append(replace(update, id=new_id, replaces=None))
                continue
            merged = [replace(update, id=target.id, replaces=None) if r is target else r for r in merged
                      if r is target or update.kind == Kind.SPEC or r.kind != update.kind]
        return merged


    @staticmethod
    def _meets_all(verdicts: dict[str, str] | None) -> bool:
        return bool(verdicts) and all(verdict == Verdict.MET for verdict in verdicts.values())

    @staticmethod
    def _check_variant(
        requirements: list[Requirement],
        product: dict[str, Any],
        variant: dict[str, Any],
        judged: dict[str, str] | None = None,
    ) -> dict[str, str]:

        fields = {
            Kind.BRAND: product.get("brand") or product.get("title"),
            Kind.PHRASE: product.get("title"),
        }
        verdicts: dict[str, str] = {}
        for requirement in requirements:
            if requirement.value is None:
                continue
            key = requirement.verdict_key()
            if requirement.kind == Kind.BUDGET:
                verdicts[key] = Utils._check_budget(requirement, variant)
            elif requirement.kind == Kind.STOCK:
                verdicts[key] = Verdict.MET if variant.get("in_stock") is True else Verdict.UNMET
            elif requirement.kind == Kind.SPEC:
                verdicts[key] = (judged or {}).get(key, Verdict.UNVERIFIED)
            elif requirement.kind == Kind.CATEGORY:
                filed = any(Utils._label(label) == Utils._label(requirement.value) for label in product.get("category_path") or [])
                verdicts[key] = Verdict.MET if filed else (judged or {}).get(key, Verdict.UNMET)
            elif requirement.kind in fields:
                verdicts[key] = Utils._words_match(requirement.value, fields[requirement.kind])
            else:
                verdicts[key] = Verdict.UNVERIFIED
        return verdicts

    @staticmethod
    def _check_budget(requirement: Requirement, variant: dict[str, Any]) -> str:
        if requirement.amount is None:
            return Verdict.UNVERIFIED
        price, currency = variant.get("price"), str(variant.get("currency") or "").upper()
        if price is None or (requirement.currency is not None and currency != requirement.currency):
            return Verdict.UNMET
        try:
            price, limit = Decimal(str(price)), Decimal(requirement.amount)
        except InvalidOperation:
            return Verdict.UNVERIFIED
        return Verdict.MET if price <= limit else Verdict.UNMET

    @staticmethod
    def _listings(engine: Engine) -> dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]:
        found: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        for _, body in Utils._observations(engine):
            for product in Utils._products(body):
                for variant in product.get("variants") or []:
                    found[(str(product.get("product_id")), str(variant.get("sku")))] = (product, variant)
        return found

    @staticmethod
    def _choose_candidate(engine: Engine, refs: list[tuple[str, str]]) -> tuple[str, str] | None:
        if len(refs) < 2:
            return refs[0] if refs else None
        asked = (tuple(sorted(refs)), tuple(sorted(r.check_key() for r in engine.requirements)))
        if asked == getattr(engine, "choice_from", None):
            return getattr(engine, "choice", None)
        engine.choice_from, engine.choice = asked, None
        listings, states = Utils._listings(engine), Utils._variant_states(engine)
        keys, lines = [], []
        for ref in refs:
            product, variant = listings.get(ref, ({}, {}))
            state = states.get(ref) or {}
            key = f"{ref[0]}::{ref[1]}"
            short = [f"{name} {verdict}" for name, verdict in (engine.candidates.get(ref) or {}).items()
                     if verdict != Verdict.MET]
            keys.append(key)
            lines += [
                f"- {key}",
                f"    title: {product.get('title') or ''}",
                f"    options: {variant.get('options') or ''}",
                f"    price: {state.get('price')} {state.get('currency') or ''}",
                f"    in stock: {state.get('in_stock')}",
                f"    falls short on: {', '.join(short) if short else 'nothing the conditions cover'}",
            ]
        schema = json.loads(json.dumps(Prompt.CHOOSE_VARIANT))
        schema["function"]["parameters"]["properties"]["variant"]["enum"] = keys
        answer = Utils._llm_arguments(
            Prompt.CHOOSE.format(
                query=Utils._split_query(engine.query)[0],
                requirements="\n".join(f"- {r.verdict_key()} = {r.value!r}" for r in engine.requirements if r.value),
                items="\n".join(lines)),
            schema, "choose_variant", "candidate choice failed")
        chosen = str((answer or {}).get("variant") or "")
        if chosen not in keys:
            return None
        ref = refs[keys.index(chosen)]
        best = min(sum(1 for verdict in (engine.candidates.get(other) or {}).values() if verdict == Verdict.UNMET)
                   for other in refs)
        if sum(1 for verdict in (engine.candidates.get(ref) or {}).values() if verdict == Verdict.UNMET) > best:
            return None
        engine.choice, engine.because = ref, str((answer or {}).get("because") or "")
        return ref

    @staticmethod
    def _settled_specs(engine: Engine, previous: list[Requirement]) -> dict[tuple[str, str], dict[str, str]]:
        keys = {r.check_key() for r in previous}
        kept = {r.verdict_key() for r in engine.requirements if r.kind == Kind.SPEC and r.check_key() in keys}
        return {ref: {key: verdict for key, verdict in cell.items()
                      if key in kept and verdict in (Verdict.MET, Verdict.UNMET)}
                for ref, cell in engine.candidates.items()}

    @staticmethod
    def _review_candidates(
        requirements: list[Requirement], result: dict[str, Any], labelled: bool = True,
        listed: set[str] | None = None, settled: dict[tuple[str, str], dict[str, str]] | None = None,
    ) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], int]]:
        products = list({str(p.get("product_id")): p for p in Utils._observed_products(result)}.values())
        checked = [r for r in requirements if r.kind not in (Kind.SPEC, Kind.CATEGORY)]
        categories = [r for r in requirements if r.kind == Kind.CATEGORY and r.value is not None]
        passing = [
            product for product in products
            if any(
                all(verdict == Verdict.MET for verdict in Utils._check_variant(checked, product, variant).values())
                for variant in product.get("variants") or []
            )
        ]
        judged: dict[tuple[str, str], dict[str, str]] = {ref: dict(cell) for ref, cell in (settled or {}).items()}
        if not labelled:
            for product in passing:
                if str(product.get("product_id")) not in (listed or set()):
                    continue
                for variant in product.get("variants") or []:
                    judged.setdefault((str(product.get("product_id")), str(variant.get("sku"))), {}).update(
                        {r.verdict_key(): Verdict.MET for r in categories}
                    )
        to_judge = [
            product for product in passing
            if any(Verdict.UNMET not in Utils._check_variant(
                categories, product, variant,
                judged.get((str(product.get("product_id")), str(variant.get("sku"))))).values()
                for variant in product.get("variants") or [{}])
        ]
        jobs = [lambda batch=batch: Utils._judge_specs(requirements, batch, settled)
                for batch in (to_judge[start:start + SPEC_JUDGE_BATCH]
                              for start in range(0, len(to_judge), SPEC_JUDGE_BATCH))]
        stated: dict[tuple[str, str], int] = {}
        if requirements and jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                for verdicts, own in pool.map(lambda job: job(), jobs):
                    stated.update(own)
                    for ref, cell in verdicts.items():
                        judged.setdefault(ref, {}).update(cell)
        reviewed: dict[tuple[str, str], dict[str, str]] = {}
        for product in products:
            for variant in product.get("variants") or []:
                ref = (str(product.get("product_id")), str(variant.get("sku")))
                verdicts = (Utils._check_variant(requirements, product, variant, judged.get(ref))
                            if requirements else {"requirements": Verdict.UNVERIFIED})
                reviewed[ref] = verdicts
                missed = [f"{key} {verdict}" for key, verdict in verdicts.items() if verdict != Verdict.MET]
        return reviewed, stated

    @staticmethod
    def _judge_specs(
        requirements: list[Requirement], products: list[dict[str, Any]],
        settled: dict[tuple[str, str], dict[str, str]] | None = None,
    ) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], int]]:
        specs = [r for r in requirements if r.kind == Kind.SPEC and r.value is not None]
        known = lambda ref, r: (settled or {}).get(ref, {}).get(r.verdict_key()) in (Verdict.MET, Verdict.UNMET)
        variants = {
            f"{product.get('product_id')}::{variant.get('sku')}": (product, variant)
            for product in products for variant in product.get("variants") or []
            if not all(known((str(product.get("product_id")), str(variant.get("sku"))), r) for r in specs)
        }
        verdicts = {
            (str(product.get("product_id")), str(variant.get("sku"))): {r.verdict_key(): Verdict.UNVERIFIED for r in specs}
            for product, variant in variants.values()
        }
        own = {ref: 0 for ref in verdicts}
        if not specs or not variants:
            return verdicts, own
        ids = {r.id: r for r in specs}
        evidence = {
            key: {Source.OPTIONS: variant.get("options"), Source.TITLE: product.get("title"), Source.FACTS: product.get("facts") or {}}
            for key, (product, variant) in variants.items()
        }
        prompt = Prompt.SPECS.format(
            requirements="\n".join(f"- {ref}: {r.value}" for ref, r in ids.items()),
            variants=json.dumps(evidence, ensure_ascii=False, indent=1),
        )
        schema = json.loads(json.dumps(Prompt.REPORT_STATEMENTS))
        report = schema["function"]["parameters"]["properties"]["reports"]["items"]["properties"]
        report["variant"]["enum"], report["requirement"]["enum"] = list(variants), list(ids)
        answer = Utils._llm_arguments(prompt, schema, "report_statements", "spec judgement failed")
        if answer is None:
            return verdicts, own
        reports = answer.get("reports") or []
        for item in reports:
            if not isinstance(item, dict):
                continue
            requirement = ids.get(str(item.get("requirement") or "").strip())
            pair = variants.get(str(item.get("variant") or "").strip())
            if requirement is None or pair is None:
                continue
            product, variant = pair
            ref = (str(product.get("product_id")), str(variant.get("sku")))
            if known(ref, requirement):
                continue
            verdict, source = Utils._statement_verdict(item.get("statements"), product, variant, requirement.value)
            verdicts[ref][requirement.verdict_key()] = verdict
            own[ref] += source == Source.OPTIONS
        return verdicts, own

    @staticmethod
    def _statement_verdict(
        statements: Any, product: dict[str, Any], variant: dict[str, Any], spec: str
    ) -> tuple[str, str | None]:
        facts = product.get("facts") or {}
        options, product_level = [], []
        for statement in statements if isinstance(statements, list) else []:
            if not isinstance(statement, dict) or statement.get("verdict") not in Verdict.ALL:
                continue
            field, quote = str(statement.get("field") or "").strip(), statement.get("quote")
            verdict = Utils._checked_statement(statement["verdict"], quote, spec)
            if verdict is None:
                continue
            if field == Source.OPTIONS:
                if Utils._contains_words(variant.get("options"), quote):
                    options.append(verdict)
            elif field == Source.TITLE:
                if Utils._contains_words(product.get("title"), quote):
                    product_level.append(verdict)
            else:
                lines = [f"{name}: {value}" for name, value in facts.items() if name == field] or [
                    f"{name}: {value}" for name, value in facts.items()
                ]
                if any(Utils._contains_words(line, quote) for line in lines):
                    product_level.append(verdict)
        governing = options or product_level
        source = Source.OPTIONS if options else ("product" if product_level else None)
        if not governing or Verdict.UNMET in governing:
            return Verdict.UNMET, source
        if Verdict.UNVERIFIED in governing:
            return Verdict.UNVERIFIED, source
        return Verdict.MET, source

    @staticmethod
    def _checked_statement(verdict: str, quote: Any, spec: str) -> str | None:
        wanted = Utils._number_units(spec)
        if not wanted:
            return verdict
        written = set(Utils._number_tokens(str(quote)))
        if not written:
            return verdict
        found = Utils._number_units(quote)
        if not found:
            return None
        numbers = {number for number, _ in wanted}
        alike = {number for number, unit in found if any(Utils._same_unit(unit, other) for _, other in wanted)}
        if verdict == Verdict.MET and alike and not written & numbers:
            return Verdict.UNMET
        return verdict

    @staticmethod
    def _name_category(engine: Engine, labels: list[str]) -> bool:
        requirement = next((r for r in engine.requirements if r.kind == Kind.CATEGORY and r.value), None)
        asked = (requirement.value if requirement else None, labels)
        if requirement is None or not labels or asked == getattr(engine, "named_from", None):
            return False
        engine.named_from = asked
        schema = json.loads(json.dumps(Prompt.NAME_CATEGORY))
        schema["function"]["parameters"]["properties"]["label"]["enum"] = [*labels, None]
        answer = Utils._llm_arguments(
            Prompt.CATEGORY_LABEL.format(category=requirement.value, labels="\n".join(f"- {label}" for label in labels)),
            schema, "name_category", "category naming failed")
        label = str((answer or {}).get("label") or "")
        if label not in labels:
            return False
        engine.named_from = (label, labels)
        engine.requirements = [replace(r, value=label) if r is requirement else r for r in engine.requirements]
        return True

    @staticmethod
    def _category_context(engine: Engine) -> tuple[bool, set[str]]:
        category = Utils._value(engine.requirements, Kind.CATEGORY)
        if category is None:
            return True, set()
        labels = sorted({str(label) for _, body in Utils._observations(engine)
                         for product in Utils._products(body) for label in product.get("category_path") or []})
        wanted = Utils._label(category)
        labelled = any(Utils._label(label) == wanted for label in labels)
        if not labelled and Utils._name_category(engine, labels):
            labelled, wanted = True, Utils._label(Utils._value(engine.requirements, Kind.CATEGORY))
        listed: set[str] = set()
        for entry in engine.dialogue:
            result = entry.get("environment_result")
            if not result:
                continue
            for call, call_result in zip(entry.get("tool_calls") or [], result.get("calls") or []):
                if call["function"]["name"] == FILTER_TOOL and Utils._label(Utils._arguments(call).get("category")) == wanted:
                    rows = (Utils._body(call_result) or {}).get("results") or []
                    listed.update(str(row.get("product_id")) for row in rows)
        return labelled, listed


    @staticmethod
    def _body(call_result: dict[str, Any]) -> dict[str, Any] | None:
        body = (call_result.get("observation") or {}).get("observation")
        return body if isinstance(body, dict) else None

    @staticmethod
    def _products(body: dict[str, Any]) -> list[dict[str, Any]]:
        return body.get("compared") or ([body] if body.get("variants") else [])

    @staticmethod
    def _observations(engine: Engine) -> list[tuple[int, dict[str, Any]]]:
        return [
            (index, body)
            for index, entry in enumerate(engine.dialogue)
            for call_result in (entry.get("environment_result") or {}).get("calls") or []
            if (body := Utils._body(call_result)) is not None
        ]

    @staticmethod
    def _observed_products(result: dict[str, Any]) -> list[dict[str, Any]]:
        products = []
        for call in result.get("calls") or []:
            body = Utils._body(call)
            if body is not None:
                products += Utils._products(body)
        return products

    @staticmethod
    def _observed_title(observations: list[tuple[int, dict[str, Any]]], product_id: str) -> str | None:
        titles = [
            row["title"] for _, body in observations
            for row in [*(body.get("results") or []), *(body.get("compared") or []), body]
            if isinstance(row, dict) and str(row.get("product_id")) == product_id and row.get("title")
        ]
        return titles[-1] if titles else None

    @staticmethod
    def _rows(engine: Engine) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for _, body in Utils._observations(engine):
            for row in body.get("results") or []:
                rows.setdefault(str(row.get("product_id")), row)
        return rows

    @staticmethod
    def _variants(engine: Engine) -> dict[tuple[str, str], dict[str, Any]]:
        variants: dict[tuple[str, str], dict[str, Any]] = {}
        for _, body in Utils._observations(engine):
            for product in Utils._products(body):
                product_fields = {f: product.get(f) for f in ("product_id", "title", "brand", "category_path", "facts")}
                for variant in product.get("variants") or []:
                    variants[(str(product.get("product_id")), str(variant.get("sku")))] = {**product_fields, **variant}
        return variants

    @staticmethod
    def _event_target(engine: Engine) -> tuple[str, str] | None:
        target = None
        for _, body in Utils._observations(engine):
            moved = (body.get("market_event") or {}).get("target")
            if moved:
                target = (str(moved.get("product_id")), str(moved.get("sku")))
        return target

    @staticmethod
    def _variant_states(engine: Engine) -> dict[tuple[str, str], dict[str, Any]]:
        states: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in engine.dialogue:
            for call_result in (entry.get("environment_result") or {}).get("calls") or []:
                body = Utils._body(call_result)
                if body is None:
                    continue
                rows = [*(body.get("results") or []), *(body.get("cart") or []), body.get("added"), body]
                rows += [
                    {**variant, "product_id": product.get("product_id")}
                    for product in Utils._observed_products({"calls": [call_result]})
                    for variant in product.get("variants") or []
                ]
                for row in rows:
                    if isinstance(row, dict) and row.get("product_id") and row.get("sku") and "in_stock" in row:
                        states[(str(row["product_id"]), str(row["sku"]))] = {
                            "price": row.get("current_price", row.get("price")),
                            "currency": row.get("currency"),
                            "in_stock": row.get("in_stock"),
                        }
        return states

    @staticmethod
    def _refresh_candidate_states(engine: Engine) -> None:
        requirements = [r for r in engine.requirements if r.kind in Kind.STATE]
        suffixes = tuple(f"[{kind}]" for kind in Kind.STATE)
        states = Utils._variant_states(engine)
        for ref, verdicts in engine.candidates.items():
            if ref not in states:
                continue
            earlier = {key: verdicts.pop(key) for key in [key for key in verdicts if key.endswith(suffixes)]}
            for key, verdict in Utils._check_variant(requirements, {}, states[ref]).items():
                verdicts[key] = verdict

    @staticmethod
    def _cart_state(engine: Engine) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        cart: list[tuple[str, str]] = []
        added: list[tuple[str, str]] = []
        for _, body in Utils._observations(engine):
            if isinstance(body.get("added"), dict):
                added.append((str(body["added"].get("product_id")), str(body["added"].get("sku"))))
            if isinstance(body.get("cart"), list):
                cart = [(str(item.get("product_id")), str(item.get("sku"))) for item in body["cart"] if isinstance(item, dict)]
        return cart, added


    @staticmethod
    def _search_calls(
        requirements: list[Requirement], tools: list[dict[str, Any]], call_id: str, query: str | None = None
    ) -> list[dict[str, Any]]:
        schemas = {
            tool["function"]["name"]: (tool["function"].get("parameters") or {}).get("properties") or {}
            for tool in tools if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }

        budget = next((r for r in requirements if r.kind == Kind.BUDGET and r.amount), None)
        max_price = float(budget.amount) if budget else None
        in_stock = True if any(r.kind == Kind.STOCK for r in requirements) else None
        brand, category = Utils._value(requirements, Kind.BRAND), Utils._value(requirements, Kind.CATEGORY)
        phrase = Utils._value(requirements, Kind.PHRASE)
        specs = [r.value for r in requirements if r.kind == Kind.SPEC and r.value is not None]
        wanted = {
            FILTER_TOOL: {"category": category, "brand": brand, "max_price": max_price},
            SEARCH_TOOL: {"query": query or (Utils._unquoted(phrase) if phrase
                       else " ".join(v for v in (brand, category, *specs) if v) or None)},
        }
        calls = []
        for name, arguments in wanted.items():
            properties = schemas.get(name)
            if properties is None or all(value is None for value in arguments.values()):
                continue
            arguments = {**arguments, "max_price": max_price, "in_stock": in_stock,
                         "k": (properties.get("k") or {}).get("maximum")}
            arguments = {key: value for key, value in arguments.items() if value is not None}
            calls.append({"id": f"{call_id}-{name}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(arguments)}})
        return calls

    @staticmethod
    def _new_search_calls(engine: Engine, turn: int, query: str | None = None) -> list[dict[str, Any]]:
        sent = {
            (call["function"]["name"], json.dumps(Utils._arguments(call), sort_keys=True))
            for message in engine.messages if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        }
        call_id = f"{engine.problem_id}-turn-{turn}" + ("-query" if query else "")
        calls = [
            call for call in Utils._search_calls(engine.requirements, engine.tools, call_id, query)
            if (query is None or Utils._arguments(call).get("query") == query)
            and (call["function"]["name"], json.dumps(Utils._arguments(call), sort_keys=True)) not in sent
        ]
        return calls

    @staticmethod
    def _compare_calls(engine: Engine, turn: int) -> list[dict[str, Any]]:
        if COMPARE_TOOL not in Utils._tool_names(engine):
            return []
        rows, observed = [], set()
        for entry in engine.dialogue:
            result = entry.get("environment_result")
            if not result:
                continue
            for call, call_result in zip(entry["tool_calls"], result.get("calls") or []):
                observation = call_result.get("observation") or {}
                if call["function"]["name"] in SEARCH_TOOLS:
                    rows += (Utils._body(call_result) or {}).get("results") or []
                elif call["function"]["name"] == COMPARE_TOOL and not observation.get("error"):
                    product_ids = Utils._arguments(call).get("product_ids")
                    if isinstance(product_ids, list):
                        observed.update(str(product_id) for product_id in product_ids)
            observed.update(str(product.get("product_id")) for product in Utils._observed_products(result))
        row_requirements = [r for r in engine.requirements if r.kind in engine.row_checked]
        product_ids = list(dict.fromkeys(
            str(row.get("product_id")) for row in rows
            if str(row.get("product_id")) not in observed
            and Verdict.UNMET not in Utils._check_variant(row_requirements, row, row).values()
        ))
        return [
            Utils._tool_call(COMPARE_TOOL, {"product_ids": product_ids[start:start + COMPARE_CAP]},
                             f"{engine.problem_id}-turn-{turn}-{COMPARE_TOOL}-{index}")
            for index, start in enumerate(range(0, len(product_ids), COMPARE_CAP), start=1)
        ]

    @staticmethod
    def _add_calls(
        engine: Engine, turn: int, cart: list[tuple[str, str]] | None = None, new_product_only: bool = False
    ) -> list[dict[str, Any]]:
        observed_cart, added = Utils._cart_state(engine)
        if (observed_cart if cart is None else cart) or ADD_TOOL not in Utils._tool_names(engine):
            return []
        if not engine.requirements or any(r.hidden for r in engine.requirements):
            return []
        committed = {product_id for product_id, _ in added}
        met = [
            ref for ref, verdicts in engine.candidates.items()
            if Utils._meets_all(verdicts) and not (new_product_only and ref[0] in committed)
        ]
        if not met:
            return []
        states, stated = Utils._variant_states(engine), engine.stated

        def rank(ref: tuple[str, str]) -> tuple[bool, int, float]:
            price = (states.get(ref) or {}).get("price")
            return ref[0] in committed, -stated.get(ref, 0), float("inf") if price is None else price

        ordered = sorted(met, key=rank)
        ref = Utils._choose_candidate(engine, ordered) or ordered[0]
        return [Utils._tool_call(ADD_TOOL, Utils._ids(ref), f"{engine.problem_id}-turn-{turn}-{ADD_TOOL}")]

    @staticmethod
    def _choice_message(engine: Engine, ref: tuple[str, str], template: str) -> str:
        state = Utils._variant_states(engine).get(ref) or {}
        because = getattr(engine, "because", "") if getattr(engine, "choice", None) == ref else ""
        if not because:
            short = [key for key, verdict in (getattr(engine, "candidates", {}).get(ref) or {}).items()
                     if verdict != Verdict.MET]
            because = ("It meets every condition you gave." if not short else
                       "It is the nearest I could find to what you asked; it does not settle " + ", ".join(short) + ".")
        return template.format(title=Utils._observed_title(Utils._observations(engine), ref[0]) or ref[0],
                               sku=ref[1], price=state.get("price"), currency=state.get("currency") or "",
                               because=because)

    @staticmethod
    def _unanswered_message(engine: Engine) -> str:
        owed = ""
        for entry in engine.dialogue:
            if any(call["function"]["name"] == MESSAGE_TOOL for call in entry.get("tool_calls") or []):
                owed = ""
            content = ((entry.get("environment_result") or {}).get("user_message") or {}).get("content") or ""
            if content.strip():
                owed = content
        return owed

    @staticmethod
    def _order_calls(engine: Engine, ref: tuple[str, str], call_id: str) -> list[dict[str, Any]]:
        if ORDER_TOOL not in Utils._tool_names(engine):
            return []
        return [Utils._tool_call(ORDER_TOOL, Utils._ids(ref), f"{call_id}-{ORDER_TOOL}")]

    @staticmethod
    def _answer_before_order(engine: Engine, turn: int, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        order = next((index for index, call in enumerate(tool_calls) if call["function"]["name"] == ORDER_TOOL), None)
        if (order is None or len(tool_calls[order:]) >= engine.max_calls
                or MESSAGE_TOOL not in Utils._tool_names(engine)
                or any(call["function"]["name"] == MESSAGE_TOOL for call in tool_calls[:order])
                or not Utils._unanswered_message(engine)):
            return tool_calls
        arguments = Utils._arguments(tool_calls[order])
        content = Utils._choice_message(engine, (str(arguments.get("product_id")), str(arguments.get("sku"))),
                                        ORDER_MESSAGE)
        reply = Utils._tool_call(MESSAGE_TOOL, {"content": content}, f"{engine.problem_id}-turn-{turn}-{MESSAGE_TOOL}")
        room = max(0, engine.max_calls - 1 - len(tool_calls[order:]))
        return [*Utils._limit_calls(tool_calls[:order], room), reply, *tool_calls[order:]]

    @staticmethod
    def _track_progress(engine: Engine) -> None:
        seen = {ref[0] for ref in Utils._variant_states(engine)}
        cart, _ = Utils._cart_state(engine)
        requirements = tuple(sorted(r.check_key() for r in engine.requirements))
        met = sum(1 for verdicts in engine.candidates.values() if Utils._meets_all(verdicts))
        signature = (len(seen), tuple(cart), requirements, met)
        engine.stalled = getattr(engine, "stalled", 0) + 1 if signature == getattr(engine, "progress", None) else 0
        engine.progress = signature

    @staticmethod
    def _checked_since_add(engine: Engine, ref: tuple[str, str]) -> bool:
        bodies = [body for _, body in Utils._observations(engine)]
        added_at = max((place for place, body in enumerate(bodies)
                        if isinstance(added := body.get("added"), dict)
                        and (str(added.get("product_id")), str(added.get("sku"))) == ref), default=-1)
        return any(place > added_at and (str(body.get("product_id")), str(body.get("sku"))) == ref
                   for place, body in enumerate(bodies))

    @staticmethod
    def _recover_calls(engine: Engine, turn: int) -> list[dict[str, Any]]:
        cart, _ = Utils._cart_state(engine)
        tools = Utils._tool_names(engine)
        if not cart or not engine.requirements or not {INSPECT_TOOL, REMOVE_TOOL} <= tools:
            return []
        ref = cart[-1]
        arguments = Utils._ids(ref)
        call_id = f"{engine.problem_id}-turn-{turn}"
        verdicts = engine.candidates.get(ref) or {}
        if Utils._meets_all(verdicts):
            if Utils._checked_since_add(engine, ref):
                return Utils._order_calls(engine, ref, call_id)
            return [Utils._tool_call(INSPECT_TOOL, arguments, f"{call_id}-{INSPECT_TOOL}")]
        calls = [Utils._tool_call(REMOVE_TOOL, arguments, f"{call_id}-{REMOVE_TOOL}")]
        return Utils._limit_calls(calls + Utils._replacement_calls(engine, turn, cart=[]), engine.max_calls)

    @staticmethod
    def _replacement_calls(engine: Engine, turn: int, cart: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
        adds = Utils._add_calls(engine, turn, cart, new_product_only=True)
        if adds:
            return adds
        compares = Utils._compare_calls(engine, turn)
        searches = Utils._new_search_calls(engine, turn)
        if compares:
            return Utils._limit_calls(compares + searches, engine.max_calls)
        return (
            Utils._limit_calls(searches + Utils._recovery_search_calls(engine, turn), engine.max_calls)
            or Utils._add_calls(engine, turn, cart)
        )

    @staticmethod
    def _recovery_search_calls(engine: Engine, turn: int) -> list[dict[str, Any]]:
        _, added = Utils._cart_state(engine)
        if not added:
            return []
        title = Utils._observed_title(Utils._observations(engine), added[-1][0])
        return Utils._new_search_calls(engine, turn, title) if title else []

    @staticmethod
    def _choose_calls(engine: Engine, tools: list[dict[str, Any]] | None = None) -> tuple[str, list[dict[str, Any]]]:
        messages, dialogue = engine.messages, engine.dialogue
        for _attempt in range(2):
            assistant = Utils._llm(messages, tools=tools or engine.tools, tool_choice="required")
            assistant_content = assistant.get("content") or ""
            tool_calls = Utils._limit_calls(assistant.get("tool_calls") or [], engine.max_calls)
            if tool_calls:
                return assistant_content, tool_calls
            dialogue.append({"role": "assistant", "content": assistant_content})
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": Prompt.CONTINUE},
                ]
            )
        raise RuntimeError("model returned no tool call before completion")

    @staticmethod
    def _send_turn(
        engine: Engine, turn: int, assistant_content: str, tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        tool_calls = Utils._answer_before_order(engine, turn, tool_calls)
        messages, dialogue = engine.messages, engine.dialogue
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{engine.problem_id}-turn-{turn}"
        envelope = {
            **engine.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": Utils._arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            name = tool_call["function"]["name"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(call_result["observation"]),
                }
            )
        if result.get("user_message"):
            messages.append(
                {"role": "user", "content": result["user_message"]["content"]}
            )
        return result

    @staticmethod
    def _llm_arguments(prompt: str, schema: dict[str, Any], name: str, label: str) -> dict[str, Any] | None:
        try:
            message = Utils._llm(prompt, tools=[schema], tool_choice="required")
            call = next(c for c in message.get("tool_calls") or [] if c["function"]["name"] == name)
            return Utils._arguments(call)
        except StopIteration:
            pass
        except (RuntimeError, ValueError, KeyError, TypeError):
            pass
        return None

    @staticmethod
    def _json(prompt: str) -> dict[str, Any]:
        try:
            text = Utils._llm(prompt).get("content") or ""
            value = json.loads(text[text.find("{") : text.rfind("}") + 1])
        except (RuntimeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _llm(prompt: str | list[dict[str, Any]], **params: Any) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        for model in Utils.MODELS:
            started = time.monotonic()
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={**params, "model": model, "messages": messages, "temperature": 0},
            )
            try:
                message = inference["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                continue
            if message.get("content") or message.get("tool_calls"):
                return message
        raise RuntimeError("inference request failed on every model")


@dataclass
class Requirement:
    kind: str
    value: str | None
    hidden: bool
    turn: int = 0
    amount: str | None = None
    currency: str | None = None
    id: str | None = None
    replaces: str | None = None

    def verdict_key(self) -> str:
        return f"{self.id} [{self.kind}]"

    def check_key(self) -> tuple[Any, ...]:
        def text(value: str | None) -> str | None:
            return " ".join(value.casefold().split()) if value is not None else None

        if self.kind == Kind.BUDGET:
            return self.kind, Decimal(self.amount) if self.amount else None, self.currency
        if self.kind == Kind.STOCK:
            return (self.kind,)
        return self.kind, text(self.value)


class Engine:
    marker = ""
    row_checked: tuple[str, ...] = (Kind.BRAND,)

    def __init__(self, problem_data: dict[str, Any]) -> None:
        self.environment = problem_data["environment"]
        self.binding = self.environment["binding"]
        self.policy = self.environment["policy_view"]
        self.problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.query = self.policy["query"]
        self.tools = self.policy["tools"]
        self.max_steps = int(self.policy["max_steps"])
        self.max_calls = int(self.policy.get("max_calls_per_turn", 1))
        self.messages: list[dict[str, Any]] = [
            {"role": "user", "content": Prompt.SHOP.format(query=self.query)},
        ]
        self.dialogue: list[dict[str, Any]] = []
        self.started = time.monotonic()

    def run(self) -> list[dict[str, Any]]:
        for turn in range(1, self.max_steps + 1):
            content, tool_calls = Utils._choose_calls(self)
            result = Utils._send_turn(self, turn, content, tool_calls)
            if any((call.get("observation") or {}).get("done") for call in result["calls"]):
                break
        return self.dialogue

class CartEngine(Engine):
    RESCUE_TURNS = 4

    def _code_calls(self, turn: int) -> list[dict[str, Any]]:
        return (self._rescue_calls(turn) or Utils._recover_calls(self, turn)
                or Utils._replacement_calls(self, turn))

    def _fallback_calls(self, turn: int) -> tuple[str, list[dict[str, Any]]]:
        try:
            return self._model_calls()
        except RuntimeError:
            return "", self._rescue_calls(turn, forced=True) or self._waiting_calls()

    def _waiting_calls(self) -> list[dict[str, Any]]:
        return ([Utils._tool_call(INSPECT_CART_TOOL, {}, f"{self.problem_id}-waiting")]
                if INSPECT_CART_TOOL in Utils._tool_names(self) else [])

    def _rescue_due(self, turn: int) -> bool:
        if self.max_steps - turn < self.RESCUE_TURNS:
            return True
        _, added = Utils._cart_state(self)
        return bool(added) and getattr(self, "stalled", 0) >= STALL_TURNS

    def _rescue_calls(self, turn: int, forced: bool = False) -> list[dict[str, Any]]:
        tools = Utils._tool_names(self)
        if not (forced or self._rescue_due(turn)) or not {ADD_TOOL, INSPECT_TOOL, ORDER_TOOL} <= tools:
            return []
        self.rescues = getattr(self, "rescues", 0) + 1
        ref = self._closest_ref()
        if ref is None or self.rescues > self.RESCUE_TURNS + 2:
            return []
        cart, added = Utils._cart_state(self)
        if not added and self.max_steps - turn >= self.RESCUE_TURNS and not Utils._meets_all(self.candidates.get(ref)):
            return []
        call_id = f"{self.problem_id}-turn-{turn}-rescue"
        if ref in cart:
            if Utils._checked_since_add(self, ref):
                return Utils._order_calls(self, ref, call_id)
            return [Utils._tool_call(INSPECT_TOOL, Utils._ids(ref), f"{call_id}-{INSPECT_TOOL}")]
        held = [Utils._tool_call(REMOVE_TOOL, Utils._ids(other), f"{call_id}-{REMOVE_TOOL}-{index}")
                for index, other in enumerate(cart, start=1)] if REMOVE_TOOL in tools else []
        return Utils._limit_calls(
            [*held, Utils._tool_call(ADD_TOOL, Utils._ids(ref), f"{call_id}-{ADD_TOOL}"),
             Utils._tool_call(INSPECT_TOOL, Utils._ids(ref), f"{call_id}-{INSPECT_TOOL}")], self.max_calls)

    def _closest_ref(self) -> tuple[str, str] | None:
        states = Utils._variant_states(self)
        stocked = lambda refs: [ref for ref in refs if (states.get(ref) or {}).get("in_stock") is not False]
        pool = (stocked(ref for ref, verdicts in self.candidates.items()
                        if verdicts and Verdict.UNMET not in verdicts.values())
                or stocked(self.candidates)
                or [ref for ref, state in states.items() if state.get("in_stock") is True])
        if not pool:
            return None
        cart, _ = Utils._cart_state(self)
        def rank(ref: tuple[str, str]) -> tuple[int, int, bool, float]:
            verdicts = self.candidates.get(ref) or {}
            unmet = sum(1 for verdict in verdicts.values() if verdict == Verdict.UNMET)
            unsettled = sum(1 for verdict in verdicts.values() if verdict != Verdict.MET) if verdicts else 1
            price = (states.get(ref) or {}).get("price")
            return unmet, unsettled, ref not in cart, float("inf") if price is None else float(price)
        ordered = sorted(pool, key=rank)
        return Utils._choose_candidate(self, ordered) or ordered[0]

    def _model_calls(self) -> tuple[str, list[dict[str, Any]]]:
        return Utils._choose_calls(self, Utils._lookup_tools(self.tools))

    def _opening_calls(self, turn: int, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tool_calls

    def _after_result(self, turn: int, result: dict[str, Any]) -> None:
        pass

    def run(self) -> list[dict[str, Any]]:
        self.requirements = Utils._merge_requirements([], Utils._extract_requirements(Utils._split_query(self.query)[0], [], 0))
        self.candidates: dict[tuple[str, str], dict[str, str]] = {}
        self.stated: dict[tuple[str, str], int] = {}
        self.category_labelled = True
        for turn in range(1, self.max_steps + 1):
            content, tool_calls = "", self._code_calls(turn)
            if not tool_calls:
                content, tool_calls = self._fallback_calls(turn)
            if not tool_calls:
                break
            tool_calls = self._opening_calls(turn, tool_calls)
            result = Utils._send_turn(self, turn, content, tool_calls)
            previous = self.requirements
            if ((result.get("user_message") or {}).get("content") or "").strip():
                updates = Utils._extract_requirements(result["user_message"]["content"], self.requirements, turn)
                self.requirements = Utils._merge_requirements(self.requirements, updates)
            self._after_result(turn, result)
            labelled, listed = Utils._category_context(self)
            relabelled, self.category_labelled = labelled and not self.category_labelled, labelled
            settled = Utils._settled_specs(self, previous)
            if {r.check_key() for r in previous} == {r.check_key() for r in self.requirements} and not relabelled:
                reviewed, stated = Utils._review_candidates(self.requirements, result, labelled, listed, settled)
                self.candidates.update(reviewed)
                self.stated.update(stated)
            else:
                self.candidates, stated = Utils._review_candidates(
                    self.requirements,
                    {"calls": [c for entry in self.dialogue for c in (entry.get("environment_result") or {}).get("calls") or []]},
                    labelled,
                    listed,
                    settled,
                )
                self.stated = {**self.stated, **stated}
            Utils._refresh_candidate_states(self)
            if any((call.get("observation") or {}).get("done") for call in result["calls"]):
                break
            Utils._track_progress(self)
        return self.dialogue


class IntentDecomposition(Engine):
    marker = "mix of firm requirements and nice-to-haves"
    MAX_SEARCHES = 3
    RESCUE_TURNS = 4
    JUDGE_BATCH = 40

    def run(self) -> list[dict[str, Any]]:
        self.brief = Utils._json(Prompt.INTENT_BRIEF.format(query=self.query))
        try:
            self.budget = float(str(self.brief["budget"]).replace(",", ""))
        except (KeyError, TypeError, ValueError):
            return super().run()
        self.keywords = [q for q in self.brief.get("keywords") or [] if isinstance(q, str)]
        self.valid: dict[tuple[str, str], tuple[bool, bool]] = {}
        self.doubtful: dict[tuple[str, str], tuple[bool, bool]] = {}
        self.judged: set[tuple[str, str]] = set()
        self.opened: set[str] = set()
        self.sent: set[tuple[str, float]] = set()
        self.heard: list[str] = []
        self.phase, self.searches = "search", 0
        self.rescue = self.budget_cut = self.waited = False

        for turn in range(1, self.max_steps + 1):
            calls = self._plan(turn)
            if not calls:
                break
            result = self._send(turn, calls)
            if any((call.get("observation") or {}).get("done") for call in result["calls"]):
                break
        return self.dialogue

    def _plan(self, turn: int) -> list[tuple[str, dict[str, Any]]]:
        if self.max_steps - turn < self.RESCUE_TURNS:
            self.rescue = True
            if self.phase in ("search", "compare"):
                self.phase = "commit"
        phases = {"search": self._search, "compare": self._compare, "commit": self._commit, "final": self._final}
        for _ in range(len(phases)):
            phase = phases.get(self.phase)
            planned = phase() if phase else []
            if planned:
                return planned
        return []

    def _search(self) -> list[tuple[str, dict[str, Any]]]:
        cap = {"max_price": self.budget, "in_stock": True, "k": Utils._page_size(self.tools, SEARCH_TOOL)}
        queries = []
        for query in self.keywords:
            if (query.casefold(), self.budget) in self.sent or len(queries) == self.max_calls - 2:
                continue
            self.sent.add((query.casefold(), self.budget))
            queries.append(query)
        calls = [(SEARCH_TOOL, {"query": q, **cap}) for q in queries]
        if (FILTER_TOOL, self.budget) not in self.sent:
            self.sent.add((FILTER_TOOL, self.budget))
            shelf = {"category": self.brief.get("category"), **cap}
            if self.brief.get("brand"):
                calls.append((FILTER_TOOL, {**shelf, "brand": self.brief["brand"]}))
            calls.append((FILTER_TOOL, shelf))
        self.searches += 1
        self.budget_cut = False
        self.phase = "compare"
        return calls

    def _compare(self) -> list[tuple[str, dict[str, Any]]]:
        brand = str(self.brief.get("brand") or "").casefold()
        rows = Utils._rows(self)
        ids = [pid for pid, row in rows.items()
               if pid not in self.opened and row.get("in_stock") and self._number(row.get("price")) <= self.budget]
        ids.sort(key=lambda pid: not brand or brand not in str(rows[pid].get("brand") or "").casefold())
        ids = ids[: COMPARE_CAP * (self.max_calls - 1)]
        self.opened.update(ids)
        self.phase = "commit"
        return [(COMPARE_TOOL, {"product_ids": ids[i : i + COMPARE_CAP]}) for i in range(0, len(ids), COMPARE_CAP)]

    def _commit(self) -> list[tuple[str, dict[str, Any]]]:
        self._judge()
        if Utils._cart_state(self)[0]:
            self.phase = "final"
            return []
        pick = self._top(k for k, (preferred, _) in self.valid.items() if preferred and self._live(k))
        backup = self._backup(pick)
        if not (pick and backup) and self._can_search():
            self.phase = "search"
            return []
        pick = pick or backup or self._top(k for k in self.valid if self._live(k))
        if not pick:
            return self._wait()
        self.phase = "final"
        return [(ADD_TOOL, Utils._ids(pick))]

    def _final(self) -> list[tuple[str, dict[str, Any]]]:
        final = self._best()
        if not final:
            if self._can_search():
                self.phase = "search"
                return []
            return self._wait()
        if final not in Utils._cart_state(self)[0]:
            return [(ADD_TOOL, Utils._ids(final)), (INSPECT_TOOL, Utils._ids(final))]
        if not Utils._checked_since_add(self, final):
            return [(INSPECT_TOOL, Utils._ids(final))]
        self.phase = "done"
        return [(ORDER_TOOL, Utils._ids(final))]

    def _judge(self) -> None:
        variants = Utils._variants(self)
        new = [ref for ref in variants if ref not in self.judged]
        self.judged.update(new)
        fields = ("title", "brand", "category_path", "facts", "options")
        for start in range(0, len(new), self.JUDGE_BATCH):
            batch = {f"{ref[0]}::{ref[1]}": ref for ref in new[start : start + self.JUDGE_BATCH]}
            verdict = Utils._json(Prompt.INTENT_JUDGE.format(
                requirements=json.dumps(self.brief, ensure_ascii=False),
                variants=json.dumps({key: {f: variants[ref].get(f) for f in fields} for key, ref in batch.items()},
                                    ensure_ascii=False),
            ))
            for key, view in (verdict.get("items") or {}).items():
                ref = batch.get(key)
                if ref is None or not isinstance(view, dict) or view.get("valid") is not True:
                    continue
                pool = self.valid if self._amount_matches(view.get("spec_amount")) else self.doubtful
                pool[ref] = (view.get("preferred") is True, view.get("clear") is True)
            self.keywords += [q for q in verdict.get("keywords") or [] if isinstance(q, str)]
        if new:
            self.dialogue.append({"role": "judge", "judged": len(new),
                                  "valid": {f"{ref[0]}::{ref[1]}": list(view) for ref, view in self.valid.items()}})

    def _send(self, turn: int, calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        tool_calls = [
            Utils._tool_call(name, arguments, f"{self.problem_id}-turn-{turn}-{index}")
            for index, (name, arguments) in enumerate(calls, start=1)
        ]
        result = Utils._send_turn(self, turn, "", Utils._limit_calls(tool_calls, self.max_calls))
        self._hear(((result.get("user_message") or {}).get("content") or ""))
        return result

    def _hear(self, text: str) -> None:
        if not text:
            return
        self.waited = False
        self.heard.append(text)
        current = {k: v for k, v in self.brief.items() if k != "keywords"}
        revised = Utils._json(Prompt.INTENT_UPDATE.format(
            requirements=json.dumps(current, ensure_ascii=False), messages=json.dumps(self.heard, ensure_ascii=False),
        )) or {}
        budget = self._number(str(revised.get("budget") or "").replace(",", ""))
        if 0 < budget < self.budget:
            self.budget, self.budget_cut = budget, True
        spec = str(revised.get("spec") or "")
        if not spec or Utils._same_spec(spec, str(current.get("spec") or "")):
            return
        # A replaced spec voids every verdict and the listings found for it: judge afresh on a new search.
        self.brief["spec"] = spec
        self.keywords += [q for q in revised.get("keywords") or [] if isinstance(q, str)]
        self.keywords.append(f"{self.brief.get('category') or ''} {spec}".strip())
        self.valid.clear()
        self.doubtful.clear()
        self.judged.clear()
        self.phase, self.budget_cut = "search", True

    def _wait(self) -> list[tuple[str, dict[str, Any]]]:
        """With nothing to buy and nothing to search, one idle turn lets a pending shopper message arrive.
        After that, the judge's verdicts the amount check overruled are the last candidates before stopping."""
        if not (self.rescue or self.waited):
            self.waited = True
            return [(INSPECT_CART_TOOL, {})]
        self.valid.update(self.doubtful)
        self.doubtful.clear()
        return []

    def _amount_matches(self, amount: Any) -> bool:
        """Only an explicitly different amount overrides the verdict; a missing or unreadable one leaves it standing."""
        wanted = Utils._number_tokens(str(self.brief.get("spec") or ""))
        stated = Utils._number_tokens(str(amount if amount is not None else ""))
        return len(wanted) != 1 or len(stated) != 1 or stated[0] == wanted[0]

    def _can_search(self) -> bool:
        fresh = (FILTER_TOOL, self.budget) not in self.sent or any(
            (q.casefold(), self.budget) not in self.sent for q in self.keywords)
        return (self.searches < self.MAX_SEARCHES or self.budget_cut) and not self.rescue and fresh

    def _price(self, ref: tuple[str, str]) -> float:
        return self._number((Utils._variant_states(self).get(ref) or {}).get("price"))

    def _live(self, ref: tuple[str, str]) -> bool:
        state = Utils._variant_states(self).get(ref) or {}
        return state.get("in_stock") is True and self._number(state.get("price")) <= self.budget

    def _top(self, refs: Any) -> tuple[str, str] | None:
        return min(refs, key=lambda ref: (not self.valid[ref][1], self._price(ref)), default=None)

    def _backup(self, pick: tuple[str, str] | None) -> tuple[str, str] | None:
        return self._top(ref for ref in self.valid if self._live(ref) and (not pick or ref[0] != pick[0]))

    def _best(self) -> tuple[str, str] | None:
        moved = Utils._event_target(self)
        cart = Utils._cart_state(self)[0]
        refs = [ref for ref in self.valid if self._live(ref) and (moved is None or ref[0] != moved[0])]
        return min(refs, key=lambda ref: (not self.valid[ref][1], ref not in cart, self._price(ref)), default=None)

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")


class RetrievalRecall(CartEngine):
    marker = "search the catalog for what the user described"
    row_checked = (*Engine.row_checked, Kind.PHRASE)
    QUERY_BUDGET = 3

    def _after_result(self, turn: int, result: dict[str, Any]) -> None:
        if turn == 1 and (feedback := self._unmatched_feedback()):
            updates = Utils._extract_requirements(Utils._split_query(self.query)[0], self.requirements, 0, feedback)
            self.requirements = Utils._merge_requirements(self.requirements, updates)

    def _unmatched_feedback(self) -> str:
        category = Utils._value(self.requirements, Kind.CATEGORY)
        phrase = Utils._value(self.requirements, Kind.PHRASE)
        entry, lines = self.dialogue[-1], []
        for call, call_result in zip(entry["tool_calls"], (entry.get("environment_result") or {}).get("calls") or []):
            envelope = call_result.get("observation") or {}
            rows = (Utils._body(call_result) or {}).get("results")
            if envelope.get("error") or not isinstance(rows, list):
                continue
            name, arguments = call["function"]["name"], Utils._arguments(call)
            if name == FILTER_TOOL and category and arguments.get("category") == category and not arguments.get("brand") and not rows:
                lines.append(f"- category {category!r}: a filter for it returned no listings")
            if name == SEARCH_TOOL and phrase and arguments.get("query") == Utils._unquoted(phrase) and not any(
                Utils._words_match(phrase, row.get("title")) == Verdict.MET for row in rows
            ):
                lines.append(f"- phrase {phrase!r}: no search result title contains all of its words")
        if not lines:
            return ""
        return ("These condition values matched nothing when used as written. Check each against the text and report "
                "every condition again:\n" + "\n".join(lines))

    def _queries(self) -> set[str]:
        return {
            str(Utils._arguments(call).get("query"))
            for message in self.messages if message.get("role") == "assistant"
            for call in message.get("tool_calls") or [] if call["function"]["name"] == SEARCH_TOOL
        }

    def _within_query_budget(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        queries, kept = self._queries(), []
        for call in calls:
            query = str(Utils._arguments(call).get("query")) if call["function"]["name"] == SEARCH_TOOL else None
            if query is not None and query not in queries:
                if len(queries) >= self.QUERY_BUDGET:
                    continue
                queries.add(query)
            kept.append(call)
        return kept

    def _code_calls(self, turn: int) -> list[dict[str, Any]]:
        calls = super()._code_calls(turn)
        if not calls and not self._queries():
            calls = Utils._new_search_calls(self, turn, Utils._split_query(self.query)[0])
        calls = self._needed_compares(turn, calls)
        return self._within_query_budget(Utils._limit_calls([*calls, *self._coverage_calls(turn, calls)], self.max_calls))

    def _needed_compares(self, turn: int, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        met = {ref[0] for ref, verdicts in self.candidates.items() if Utils._meets_all(verdicts)}
        others = [call for call in calls if call["function"]["name"] != COMPARE_TOOL]
        found = [call for call in calls if call["function"]["name"] == COMPARE_TOOL]
        compares = found[:1]
        if len(met) >= 2 and others:
            compares = []
        elif len(met) < 2 and not compares and any(call["function"]["name"] == ADD_TOOL for call in others):
            compares = Utils._compare_calls(self, turn)[:1]
        return [*others, *compares]

    def _coverage_calls(self, turn: int, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _, added = Utils._cart_state(self)
        phrase = Utils._value(self.requirements, Kind.PHRASE)
        free = self.QUERY_BUDGET - len(self._queries())
        if added or phrase is None or free <= 0 or any(
            call["function"]["name"] in CART_TOOLS for call in calls
        ):
            return []

        phrase = Utils._unquoted(phrase)
        titles = {
            str(row.get("product_id")): Utils._words(row.get("title"))
            for _, body in Utils._observations(self) for row in body.get("results") or []
        }
        counts: dict[str, int] = {}
        for title in titles.values():
            if Utils._words(phrase) <= title:
                for word in title - Utils._words(phrase):
                    if len(word) > 1:
                        counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts, key=lambda word: (-counts[word], word))
        return [call for word in ranked[:free] for call in Utils._new_search_calls(self, turn, f"{phrase} {word}")]

    def _model_calls(self) -> tuple[str, list[dict[str, Any]]]:
        lookup = Utils._lookup_tools(self.tools)
        without_search = [tool for tool in lookup if tool["function"]["name"] != SEARCH_TOOL]
        content, calls = Utils._choose_calls(self, lookup if len(self._queries()) < self.QUERY_BUDGET else without_search)
        kept = self._within_query_budget(calls)
        if not kept:
            content, kept = Utils._choose_calls(self, without_search)
        return content, kept


class ConstraintSatisfaction(CartEngine):
    marker = "the user gives firm, non-negotiable requirements"
    REQUIREMENT_QUESTION = (
        "Before I choose a product, could you tell me your firm requirements, "
        "including any specification the item must have?"
    )

    def _opening_calls(self, turn: int, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if turn != 1:
            return tool_calls
        question = Utils._tool_call(MESSAGE_TOOL, {"content": self.REQUIREMENT_QUESTION},
                                    f"{self.problem_id}-requirement-question")
        return [question, *Utils._limit_calls(tool_calls, self.max_calls - 1)]


class PreferenceReasoning(Engine):
    marker = "authorizes one test order"

    def run(self) -> list[dict[str, Any]]:
        return super().run()

class Ranking(Engine):
    marker = "has a stated tradeoff priority"

    WAIT_SECONDS = 40
    MAX_ASKS = 6
    MODEL_TIME_BUDGET = 150
    MAX_LISTINGS = 15
    DEBUG = False

    SCREEN = """
Choose which catalog listings to open for a shopper, using only their search rows.

Request and rules:
{query}

Kind of product wanted: {kind}
{focus}

Search rows, one per listing:
{rows}

Return one JSON object: {{"open": ["<listing label>", "..."]}}
List up to {limit} listings whose row describes the kind of product wanted itself (skip accessories, parts, cases and
other items made for it), most promising for this request first.
""".strip()

    def _screen(self, groups: list[tuple[str, ...]], texts: dict[tuple[str, ...], str], focus: str, limit: int = 0) -> list[tuple[str, ...]]:
        if len(groups) <= 1:
            return list(groups)
        labels = {f"g{i + 1}": g for i, g in enumerate(groups[:60])}
        rows = {label: texts.get(group, "")[:160] for label, group in labels.items()}
        answer = self._ask(self.SCREEN.format(
            query=self.query, kind=self.kind, focus=focus, rows=json.dumps(rows, ensure_ascii=False), limit=min(60, limit or 2 * self.MAX_LISTINGS),
        ))
        picked = list(dict.fromkeys(k for x in answer.get("open") or [] if (k := self._label_key(labels, x)) is not None))
        self._note("screen", {"asked": len(labels), "picked": len(picked)})
        return list(dict.fromkeys(picked))

    def _more_keys(self, keys: list[tuple[str, ...]], dropped: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
        reserve = getattr(self, "reserve", [])
        if reserve and self.turn < self.max_steps - 3:
            self.reserve = []
            self._read_all(self._detail_calls(reserve), turns=1)
            keys.extend(k for k in self.seen if self._group_of(k) in set(reserve) and k not in keys)
        return [k for k in keys if k not in dropped]

    def _row_text(self, outer: list[str], own: list[str]) -> str:
        return " | ".join(line for line in outer + own if line.partition(": ")[0] not in self.ids)

    def _note(self, kind: str, data: Any) -> None:
        if self.DEBUG:
            self.dialogue.append({"role": "debug", "kind": kind, "data": data})

    @staticmethod
    def _norm(text: Any) -> str:
        return " ".join(str(text).casefold().split())

    @staticmethod
    def _numbers(text: Any) -> list[float]:
        import re

        found = []
        for token in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text)):
            try:
                found.append(float(token.replace(",", "")))
            except ValueError:
                continue
        return found

    def _ask(self, prompt: str) -> dict[str, Any]:
        import re
        import threading
        import time

        self.asks = getattr(self, "asks", 0) + 1
        spent = getattr(self, "model_seconds", 0.0)
        if self.asks > self.MAX_ASKS or spent >= self.MODEL_TIME_BUDGET:
            return {}
        wait = max(1.0, min(self.WAIT_SECONDS, self.MODEL_TIME_BUDGET - spent))
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                box["answer"] = Utils._llm([{"role": "user", "content": prompt}])
            except Exception:
                box["answer"] = None

        started = time.monotonic()
        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        worker.join(wait)
        self.model_seconds = spent + (time.monotonic() - started)
        answer = box.get("answer")
        if worker.is_alive() or not isinstance(answer, dict):
            return {}
        text = answer.get("content") or ""
        if not isinstance(text, str):
            return {}
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        chunk = text[start : end + 1]
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                value = json.loads(candidate)
            except ValueError:
                continue
            return value if isinstance(value, dict) else {}
        return {}

    def _schema(self, name: Any) -> dict[str, Any] | None:
        for tool in self.tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and function.get("name") == name:
                return function.get("parameters") or {}
        return None

    def _tool(self, name: Any) -> str | None:
        return name if isinstance(name, str) and self._schema(name) is not None else None

    def _props(self, name: Any) -> dict[str, Any]:
        return (self._schema(name) or {}).get("properties") or {}

    def _fits(self, name: str, args: dict[str, Any]) -> bool:
        schema = self._schema(name) or {}
        props = schema.get("properties") or {}
        return all(k in props for k in args) and all(k in args for k in schema.get("required") or [])

    def _tools_text(self) -> str:
        described = []
        for tool in self.tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict):
                described.append({"name": function.get("name"), "description": function.get("description", ""), "parameters": function.get("parameters", {})})
        return json.dumps(described, ensure_ascii=False)

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    def _plan_tools(self, raw: dict[str, Any]) -> bool:
        names = raw.get("tools") if isinstance(raw.get("tools"), dict) else {}
        self.details = self._tool(names.get("details"))
        self.many = self._tool(names.get("details_many"))
        self.many_param = names.get("details_many_param") if names.get("details_many_param") in self._props(self.many) else None
        limit = self._number(names.get("details_many_limit"))
        self.many_limit = max(1, int(limit)) if limit else 1
        self.stock = self._tool(names.get("stock"))
        self.add = self._tool(names.get("add"))
        self.remove = self._tool(names.get("remove"))
        self.order = self._tool(names.get("order"))
        add_props = self._props(self.add)
        wanted = [p for p in raw.get("item_params") or [] if isinstance(p, str) and p in add_props]
        self.ids = wanted or [p for p in (self._schema(self.add) or {}).get("required") or [] if p in add_props]
        needed = list((self._schema(self.details) or {}).get("required") or [])
        usable = bool(needed) and set(needed) <= set(self.ids)
        self.group = needed if usable else list(self.ids)
        if not usable:
            self.details = None
        if not (self.many and self.many_param and len(self.group) == 1):
            self.many = None
        self.budget = self._number(raw.get("budget"))
        self.need_stock = raw.get("must_be_in_stock") is not False
        self.kind = str(raw.get("kind") or "")
        self.seen: dict[tuple[str, ...], dict[str, Any]] = {}
        self.price_label: str | None = None
        self.stock_label: str | None = None
        return bool(self.add and self.order and self.ids and (self.details or self.many))

    def _read_plan(self, prompt: str) -> dict[str, Any]:
        for _attempt in range(2):
            raw = self._ask(prompt)
            if raw and self._plan_tools(raw):
                return raw
        return {}

    def _learn_state(self, observation: dict[str, Any]) -> None:
        rows = self._rows(observation)
        if len(rows) != 1:
            return
        flags = [line for line in rows[0][2] if self._norm(line.partition(": ")[2]) in ("true", "false")]
        amounts = [
            line for line in rows[0][2]
            if line.partition(": ")[0] not in self.ids and self._number(line.partition(": ")[2]) is not None
        ]
        if len(flags) == 1:
            self.stock_label = flags[0].partition(": ")[0]
            self.state_learned = True
        if len(amounts) == 1:
            self.price_label = amounts[0].partition(": ")[0]
            self.state_learned = True

    def _send(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        calls = calls[: self.max_calls]
        if not calls or self.done or self.turn >= self.max_steps:
            return []
        turn = self.turn + 1
        group_id = f"{self.problem_id}-turn-{turn}"
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [{"call_id": f"{group_id}-{i}", "action": {"name": n, "args": a}} for i, (n, a) in enumerate(calls, start=1)],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            self.done = True
            return []
        self.turn = turn
        tool_calls = [
            {"id": f"{group_id}-{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(calls, start=1)
        ]
        self.dialogue.append({"role": "assistant", "content": "", "tool_calls": tool_calls, "environment_result": result})
        observations = []
        for call in result.get("calls") or []:
            wrapper = call.get("observation") if isinstance(call, dict) else None
            wrapper = wrapper if isinstance(wrapper, dict) else {}
            if wrapper.get("done"):
                self.done = True
            inner = wrapper.get("observation")
            observations.append(inner if isinstance(inner, dict) else {})
        return observations + [{}] * (len(calls) - len(observations))

    def _rows(self, node: Any) -> list[tuple[tuple[str, ...], list[str], list[str]]]:
        ids = self.ids

        def holds(value: Any) -> bool:
            if isinstance(value, dict):
                return any(p in value for p in ids) or any(holds(v) for v in value.values())
            if isinstance(value, list):
                return any(holds(v) for v in value)
            return False

        def lines(level: dict[str, Any]) -> list[str]:
            out: list[str] = []

            def flat(value: Any, label: str) -> None:
                if isinstance(value, dict):
                    for key, inner in value.items():
                        flat(inner, str(key))
                elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
                    for inner in value:
                        flat(inner, label)
                else:
                    shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    out.append(f"{label}: {shown}"[:400])

            flat(level, "")
            return out

        found: list[tuple[tuple[str, ...], list[str], list[str]]] = []

        def walk(value: Any, known: dict[str, str], outer: list[str]) -> None:
            if isinstance(value, list):
                for inner in value:
                    walk(inner, known, outer)
                return
            if not isinstance(value, dict):
                return
            own = {p: str(value[p]) for p in ids if isinstance(value.get(p), (str, int)) and not isinstance(value.get(p), bool)}
            known = {**known, **own}
            plain = {k: v for k, v in value.items() if not holds(v)}
            if own and all(p in known for p in ids):
                found.append((tuple(known[p] for p in ids), outer, lines(plain)))
            deeper = outer + lines(plain) if own else outer
            for inner in value.values():
                if isinstance(inner, (dict, list)):
                    walk(inner, known, deeper)

        walk(node, {}, [])
        return found

    def _record(self, observation: dict[str, Any]) -> None:
        for key, outer, own in self._rows(observation):
            entry = self.seen.setdefault(key, {"first": len(self.seen), "outer": [], "own": []})
            for part, fresh in (("outer", outer), ("own", own)):
                merged = {line.partition(": ")[0]: line for line in entry[part]}
                merged.update({line.partition(": ")[0]: line for line in fresh})
                entry[part] = list(merged.values())

    def _group_of(self, key: tuple[str, ...]) -> tuple[str, ...]:
        values = dict(zip(self.ids, key))
        return tuple(values.get(p, "") for p in self.group)

    def _args(self, name: str | None, key: tuple[str, ...]) -> dict[str, Any]:
        props = self._props(name)
        return {p: v for p, v in zip(self.ids, key) if p in props}

    def _detail_calls(self, groups: list[tuple[str, ...]]) -> list[tuple[str, dict[str, Any]]]:
        groups = [g for g in dict.fromkeys(groups) if all(g)]
        if self.many and (len(groups) > 1 or not self.details):
            return [
                (self.many, {self.many_param: [g[0] for g in groups[i : i + self.many_limit]]})
                for i in range(0, len(groups), self.many_limit)
            ]
        if self.details:
            props = self._props(self.details)
            return [(self.details, {p: v for p, v in zip(self.group, g) if p in props}) for g in groups]
        return []

    def _read_all(self, calls: list[tuple[str, dict[str, Any]]], turns: int) -> None:
        for start in range(0, len(calls), self.max_calls):
            if turns <= 0 or self.done or self.turn >= self.max_steps - 2:
                return
            turns -= 1
            chunk = calls[start : start + self.max_calls]
            for (name, _args), observation in zip(chunk, self._send(chunk)):
                if name not in (self.add, self.remove, self.order):
                    self._record(observation)

    def _item_text(self, key: tuple[str, ...]) -> list[str]:
        entry = self.seen.get(key) or {}
        return list(entry.get("own", [])) + list(entry.get("outer", []))

    def _shown(self, key: tuple[str, ...], part: Any) -> str | None:
        if not isinstance(part, dict) or not isinstance(part.get("quote"), str) or not part["quote"].strip():
            return None
        quote = self._norm(part["quote"].replace("\\", ""))
        return next((line for line in self._item_text(key) if quote in self._norm(line.replace("\\", ""))), None)

    def _learn(self, key: tuple[str, ...], part: Any, attr: str) -> None:
        line = self._shown(key, part)
        if not line or ": " not in line or getattr(self, attr) is not None or getattr(self, "state_learned", False):
            return
        value = line.partition(": ")[2]
        flag = self._norm(value) in ("true", "false")
        if (attr == "stock_label" and flag) or (attr == "price_label" and not flag and self._number(value) is not None):
            setattr(self, attr, line.partition(": ")[0])

    def _state(self, key: tuple[str, ...]) -> tuple[float | None, bool | None]:
        price = stock = None
        for line in self._item_text(key):
            label, _, value = line.partition(": ")
            if price is None and label == self.price_label:
                numbers = self._numbers(value)
                price = numbers[0] if numbers else None
            if stock is None and label == self.stock_label:
                text = self._norm(value)
                stock = True if text == "true" else False if text == "false" else None
        return price, stock

    def _ok(self, key: tuple[str, ...]) -> bool:
        price, stock = self._state(key)
        if self.need_stock and stock is False:
            return False
        return self.budget is None or price is None or price <= self.budget + 1e-9

    def _evidence(self, keys: list[tuple[str, ...]], per_listing: int = 6) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
        labels: dict[str, tuple[str, ...]] = {}
        listings: dict[tuple[str, ...], dict[str, Any]] = {}
        for key in keys:
            entry = self.seen.get(key)
            if not entry:
                continue
            listing = listings.setdefault(self._group_of(key), {"listing": [line[:300] for line in entry["outer"][:20]], "items": {}})
            if len(listing["items"]) >= per_listing:
                continue
            label = f"i{len(labels) + 1}"
            labels[label] = key
            listing["items"][label] = [line[:300] for line in entry["own"][:12]]
        return labels, list(listings.values())

    def _learn_row(self, lines: list[str]) -> None:
        if getattr(self, "price_label", None) and getattr(self, "stock_label", None):
            return
        flags = {l.partition(": ")[0] for l in lines if ": " in l and self._norm(l.partition(": ")[2]) in ("true", "false")}
        amounts = {
            l.partition(": ")[0] for l in lines
            if ": " in l and l.partition(": ")[0] not in self.ids and self._number(l.partition(": ")[2]) is not None
        }
        if len(flags) == 1 and len(amounts) == 1:
            self.price_label = getattr(self, "price_label", None) or next(iter(amounts))
            self.stock_label = getattr(self, "stock_label", None) or next(iter(flags))

    def _row_price(self, lines: list[str]) -> float | None:
        label = getattr(self, "price_label", None)
        return next((self._number(l.partition(": ")[2]) for l in lines if label and l.partition(": ")[0] == label), None)

    def _over_budget(self, prices: list[float | None]) -> int:
        known = [p for p in prices if p is not None]
        return 1 if self.budget is not None and known and len(known) == len(prices) and min(known) > self.budget + 1e-9 else 0

    @staticmethod
    def _listy(value: str) -> bool:
        text = value.strip()
        if not text or text[0] not in "[{" or text[-1] not in "]}":
            return False
        try:
            return isinstance(json.loads(text), (list, dict))
        except ValueError:
            return False

    def _label_key(self, labels: dict[str, tuple[str, ...]], name: Any) -> tuple[str, ...] | None:
        import re

        if not isinstance(name, str) or not name.strip():
            return None
        if name.strip() in labels:
            return labels[name.strip()]
        pieces = set(re.split(r"[^\w\-]+", name))
        named = [key for key in dict.fromkeys(labels.values()) if [p for p in key if p] and all(p in pieces for p in key if p)]
        return named[0] if len(named) == 1 else None

    def _fallback_commit(
        self, ordered: list[tuple[str, ...]], dropped: set[tuple[str, ...]], committed: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        pool = list(dict.fromkeys(k for k in [*ordered, *self.seen] if k not in dropped and k != committed))
        for _attempt in range(2):
            choice = next((k for k in pool if self._ok(k)), None)
            if choice is None or self.done or self.turn >= self.max_steps - 1:
                break
            calls = [(self.add, self._args(self.add, choice))]
            if committed is not None and committed != choice and self.remove:
                calls.append((self.remove, self._args(self.remove, committed)))
            if self.stock and self._fits(self.stock, self._args(self.stock, choice)):
                calls.append((self.stock, self._args(self.stock, choice)))
            observations = self._send(calls[: self.max_calls])
            for (name, _args), observation in zip(calls, observations):
                if name == self.stock:
                    self._record(observation)
                    self._learn_state(observation)
            pool.remove(choice)
            if observations and not (observations[0] or {}).get("error"):
                committed = choice
                if self._ok(choice):
                    break
        return committed

    def _commit_and_order(self, ordered: list[tuple[str, ...]], order_args: Any, rejudge: Any = None) -> None:
        ordered = list(dict.fromkeys(ordered))
        committed: tuple[str, ...] | None = None
        dropped: set[tuple[str, ...]] = set()
        for _attempt in range(4):
            if self.done or self.turn >= self.max_steps - 1:
                break
            choice = next((k for k in ordered if k not in dropped and self._ok(k)), None)
            if choice is None and rejudge is not None:
                extra, rejudge = rejudge(set(dropped)), None
                ordered = list(dict.fromkeys([*ordered, *extra]))
                choice = next((k for k in ordered if k not in dropped and self._ok(k)), None)
            if choice is None:
                break
            calls = [(self.add, self._args(self.add, choice))]
            if committed is not None and committed != choice and self.remove:
                calls.append((self.remove, self._args(self.remove, committed)))
            backups = [k for k in ordered if k not in dropped and k != choice][:3]
            if self.stock and self._fits(self.stock, self._args(self.stock, choice)):
                reads = [(self.stock, self._args(self.stock, k)) for k in [choice, *backups]]
            else:
                reads = self._detail_calls([self._group_of(k) for k in [choice, *backups]])
            calls = (calls + reads)[: self.max_calls]
            observations = self._send(calls)
            added = observations[0] if observations else {}
            for (name, _args), observation in zip(calls, observations):
                if name not in (self.add, self.remove, self.order):
                    self._record(observation)
                    if name == self.stock:
                        self._learn_state(observation)
            if not observations or added.get("error"):
                dropped.add(choice)
                continue
            committed = choice
            if self._ok(choice):
                break
            dropped.add(choice)
        if (committed is None or not self._ok(committed)) and not self.done and self.turn < self.max_steps - 1:
            committed = self._fallback_commit(ordered, dropped, committed)
        if committed is not None and not self.done and self.turn < self.max_steps:
            self._note("order", {
                "item": list(committed), "state": self._state(committed),
                "asks": getattr(self, "asks", 0), "model_seconds": round(getattr(self, "model_seconds", 0.0), 1),
            })
            self._send([(self.order, order_args(committed))])

    READ = """
You plan a shopping agent's work from the shopper's request and the tools it may call. Answer with one JSON object.

Tools (name, description, parameters):
{tools}

Request and rules:
{query}

Return:
{{
  "tools": {{"details": "tool showing one listing's facts and variants",
            "details_many": "tool showing several listings at once, or null",
            "details_many_param": "that tool's list parameter, or null",
            "details_many_limit": "how many ids that tool accepts per call, as a number, or null",
            "stock": "tool showing one exact item's current price and availability, or null",
            "add": "tool putting one exact item in the cart",
            "remove": "tool taking an item out of the cart, or null",
            "order": "tool placing the final order"}},
  "item_params": ["the add tool's parameters that identify one exact item"],
  "candidates": [{{"<item parameter>": "value copied exactly from the request"}}],
  "budget": "the most the shopper will pay, as a number, or null",
  "must_be_in_stock": true,
  "kind": "the kind of product wanted, in the request's words",
  "criteria": [{{"name": "one ranking criterion in the request's words", "direction": "higher or lower"}}]
}}
List every candidate item the request names, and every ranking criterion in the stated order, including a stated final
tie-break.
""".strip()

    JUDGE = """
Read each candidate's lines and report the values needed to rank them. Use only the lines shown and copy every quote
exactly from that candidate's lines. Listing lines apply to every item of that listing; an item's own lines describe that
exact item, and when an own line and a listing line disagree, the own line decides.

Request and rules:
{query}

Kind of product wanted: {kind}
Criteria in order: {criteria}

Candidates:
{listings}

Return one JSON object:
{{"candidates": {{"<item label>": {{
  "right_kind": "yes" | "no" | "unsure",
  "price": {{"number": number, "quote": "..."}},
  "available": {{"value": true | false, "quote": "..."}}
}}}},
 "measures": {{"<item label>": [{{"value": "the item's value as written", "amount": number or null, "unit": "the unit of amount, or null"}}]}}}}
"measures" gives, for every item, one entry per criterion in the same order. "amount" puts every item's value for that
criterion on one common scale in one unit, the same "unit" for every item (convert units). Use null only when none of the
item's lines state it.
When an item's lines state a criterion in more than one place, take the value in this order: a fact line named for that
property, then the item's own option line, then the title. Apply this same order to every item.
A grade is converted to the measure it stands for before comparing, never ranked
by its label: labels naming the same measure get the same amount, and a larger measure always gets a larger amount.
""".strip()

    def run(self) -> list[dict[str, Any]]:
        self.turn, self.done = 0, False
        raw = self._read_plan(self.READ.format(tools=self._tools_text(), query=self.query))
        if not raw:
            return super().run()
        self._note("read", raw)
        try:
            self._shop(raw)
        except Exception as error:
            self._note("error", repr(error))
        return self.dialogue

    def _by_label(self, labels: dict[str, tuple[str, ...]], mapping: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not isinstance(mapping, dict):
            return out
        names = {key: label for label, key in labels.items()}
        for name, value in mapping.items():
            label = names.get(self._label_key(labels, name))
            if label is not None:
                out.setdefault(label, value)
        return out

    def _own_amount(self, key: tuple[str, ...], part: dict[str, Any], amount: float | None) -> float | None:
        import re

        value = str(part.get("value") or "")
        pattern = r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*([^\W\d_]{2,})"
        stated = re.search(pattern, value)
        unit = stated.group(2).casefold() if stated and amount is not None and abs(float(stated.group(1)) - amount) < 1e-9 else ""
        if amount is not None and not unit:
            for line in (self.seen.get(key) or {}).get("outer", []):
                match = next((m for m in re.findall(pattern, line.partition(": ")[2]) if abs(float(m[0]) - amount) < 1e-9), None)
                if match:
                    unit = match[1].casefold()
                    break
        if amount is None or not unit:
            return amount
        found = set()
        for line in (self.seen.get(key) or {}).get("own", []):
            for number, other in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*([^\W\d_]{2,})(?![\w])", line.partition(": ")[2]):
                short, long = sorted((unit, other.casefold()), key=len)
                if long.startswith(short):
                    found.add(float(number))
        return found.pop() if len(found) == 1 and amount not in found else amount

    def _units(self, measures: dict[str, Any], index: int) -> set[str]:
        """Distinct units reported for one criterion; more than one means the amounts are not comparable."""
        units = set()
        for row in measures.values():
            part = row[index] if isinstance(row, list) and index < len(row) and isinstance(row[index], dict) else {}
            if self._number(part.get("amount")) is not None:
                units.add(" ".join(str(part.get("unit") or "").casefold().split()))
        return units

    def _amount(self, measures: dict[str, Any], label: str, index: int) -> float | None:
        row = measures.get(label) if isinstance(measures.get(label), list) else []
        part = row[index] if index < len(row) and isinstance(row[index], dict) else {}
        return self._number(part.get("amount"))

    def _names_item(self, group: Any, label: str, key: tuple[str, ...]) -> bool:
        import re

        entries = group if isinstance(group, list) else [group]
        for entry in entries:
            text = self._norm(json.dumps(entry, ensure_ascii=False) if isinstance(entry, (dict, list)) else entry)
            if text == label or re.search(r"(?<![a-z0-9])" + re.escape(label) + r"(?![a-z0-9])", text):
                return True
            if any(value and self._norm(value) in text for value in key):
                return True
        return False

    def _candidate_refs(self, raw: dict[str, Any]) -> list[dict[str, str]]:
        refs = []
        for ref in raw.get("candidates") or []:
            if not isinstance(ref, dict):
                continue
            clean = {p: str(v).strip() for p, v in ref.items() if p in self.ids and str(v).strip() and str(v).strip() in self.query}
            # A catalog key "<product>::<variant>" carries every identifying parameter at once.
            whole = next((v.split("::") for v in clean.values() if len(self.ids) > 1 and len(v.split("::")) == len(self.ids)), None)
            if whole or clean:
                refs.append(dict(zip(self.ids, whole)) if whole else clean)
        return refs

    def _matches(self, key: tuple[str, ...], ref: dict[str, str]) -> bool:
        values = dict(zip(self.ids, key))
        return all(values.get(p) == v for p, v in ref.items())

    def _resolve(self, refs: list[dict[str, str]]) -> list[dict[str, str]]:
        """A candidate the listings do not know is a name: search it and keep the row sharing most of its words."""
        unknown = [ref for ref in refs if not any(self._matches(key, ref) for key in self.seen)]
        search = self._tool(SEARCH_TOOL)
        if not unknown or not search:
            return refs
        names = [" ".join(ref.values()) for ref in unknown]
        found = []
        for name, observation in zip(names, self._send([(search, {"query": name}) for name in names])):
            self._record(observation)
            wanted = Utils._words(name)

            def shared(row: tuple[tuple[str, ...], list[str], list[str]]) -> int:
                return len(wanted & Utils._words(" ".join(row[1] + row[2])))

            best = max(self._rows(observation), key=shared, default=None)
            if best is not None and shared(best) >= 0.6 * len(wanted):
                found.append(dict(zip(self.ids, best[0])))
        self._read_all(self._detail_calls([tuple(ref.get(p, "") for p in self.group) for ref in found]), turns=1)
        return [ref for ref in refs if ref not in unknown] + found

    def _shop(self, raw: dict[str, Any]) -> None:
        refs = self._candidate_refs(raw)
        self._read_all(self._detail_calls([tuple(ref.get(p, "") for p in self.group) for ref in refs]), turns=2)
        refs = self._resolve(refs)
        keys = [k for k in self.seen if any(self._matches(k, ref) for ref in refs)] or list(self.seen)
        criteria = [c for c in raw.get("criteria") or [] if isinstance(c, dict) and c.get("name")]
        labels, listings = self._evidence(keys, per_listing=20)
        answer: dict[str, Any] = {}
        for _attempt in range(3):
            answer = self._ask(self.JUDGE.format(
                query=self.query, kind=self.kind, criteria=json.dumps(criteria, ensure_ascii=False),
                listings=json.dumps(listings, ensure_ascii=False),
            ))
            if self._by_label(labels, answer.get("measures")):
                break
        measures = self._by_label(labels, answer.get("measures"))
        gaps = sorted({
            label for index in range(len(criteria)) for label in labels
            if self._amount(measures, label, index) is None
            and any(self._amount(measures, other, index) is not None for other in labels)
        })
        mixed = [str(criterion["name"]) for index, criterion in enumerate(criteria) if len(self._units(measures, index)) > 1]
        if gaps or mixed:
            notes = []
            if gaps:
                notes.append("These items got no amount for a criterion that other items have: " + ", ".join(gaps)
                             + ". Read every line of those items again (title, options and facts) and give the amount their lines state.")
            if mixed:
                notes.append("These criteria were given in more than one unit: " + ", ".join(mixed)
                             + ". Convert every item's amount for them to one shared unit.")
            retry = self._ask(self.JUDGE.format(
                query=self.query, kind=self.kind, criteria=json.dumps(criteria, ensure_ascii=False),
                listings=json.dumps(listings, ensure_ascii=False),
            ) + "\n\n" + "\n".join(notes))
            if self._by_label(labels, retry.get("measures")):
                answer = retry
        self._note("judge", answer)
        found = self._by_label(labels, answer.get("candidates"))
        measures = self._by_label(labels, answer.get("measures"))
        info: dict[tuple[str, ...], tuple[Any, ...]] = {}
        wrong: set[tuple[str, ...]] = set()
        for label, key in labels.items():
            entry = found.get(label) if isinstance(found.get(label), dict) else {}
            self._learn(key, entry.get("price"), "price_label")
            self._learn(key, entry.get("available"), "stock_label")
            if entry.get("right_kind") == "no":
                wrong.add(key)
            row = measures.get(label) if isinstance(measures.get(label), list) else []
            places = []
            for index, criterion in enumerate(criteria):
                part = row[index] if index < len(row) and isinstance(row[index], dict) else {}
                amount = self._own_amount(key, part, self._number(part.get("amount")))
                if amount is None:
                    places.append((1, 0.0))
                else:
                    lower = str(criterion.get("direction", "")).casefold().startswith("low")
                    places.append((0, amount if lower else -amount))
            info[key] = (tuple(places), self.seen[key]["first"])
        ordered = sorted([k for k in info if k not in wrong], key=lambda k: info[k]) or sorted(info, key=lambda k: info[k])
        self._note("ranked", [[list(k), info[k]] for k in ordered])
        self._commit_and_order(ordered, lambda k: self._args(self.order, k))

class Recovery(Engine):
    marker = "critical constraints are category, budget, and in-stock availability"

    WAIT_SECONDS = 40
    MAX_ASKS = 6
    MODEL_TIME_BUDGET = 150
    MAX_LISTINGS = 15
    DEBUG = False

    SCREEN = """
Choose which catalog listings to open for a shopper, using only their search rows.

Request and rules:
{query}

Kind of product wanted: {kind}
{focus}

Search rows, one per listing:
{rows}

Return one JSON object: {{"open": ["<listing label>", "..."]}}
List up to {limit} listings whose row describes the kind of product wanted itself (skip accessories, parts, cases and
other items made for it), most promising for this request first.
""".strip()

    def _screen(self, groups: list[tuple[str, ...]], texts: dict[tuple[str, ...], str], focus: str, limit: int = 0) -> list[tuple[str, ...]]:
        if len(groups) <= 1:
            return list(groups)
        labels = {f"g{i + 1}": g for i, g in enumerate(groups[:60])}
        rows = {label: texts.get(group, "")[:160] for label, group in labels.items()}
        answer = self._ask(self.SCREEN.format(
            query=self.query, kind=self.kind, focus=focus, rows=json.dumps(rows, ensure_ascii=False), limit=min(60, limit or 2 * self.MAX_LISTINGS),
        ))
        picked = list(dict.fromkeys(k for x in answer.get("open") or [] if (k := self._label_key(labels, x)) is not None))
        self._note("screen", {"asked": len(labels), "picked": len(picked)})
        return list(dict.fromkeys(picked))

    def _more_keys(self, keys: list[tuple[str, ...]], dropped: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
        reserve = getattr(self, "reserve", [])
        if reserve and self.turn < self.max_steps - 3:
            self.reserve = []
            self._read_all(self._detail_calls(reserve), turns=1)
            keys.extend(k for k in self.seen if self._group_of(k) in set(reserve) and k not in keys)
        return [k for k in keys if k not in dropped]

    def _row_text(self, outer: list[str], own: list[str]) -> str:
        return " | ".join(line for line in outer + own if line.partition(": ")[0] not in self.ids)

    def _note(self, kind: str, data: Any) -> None:
        if self.DEBUG:
            self.dialogue.append({"role": "debug", "kind": kind, "data": data})

    @staticmethod
    def _norm(text: Any) -> str:
        return " ".join(str(text).casefold().split())

    @staticmethod
    def _numbers(text: Any) -> list[float]:
        import re

        found = []
        for token in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text)):
            try:
                found.append(float(token.replace(",", "")))
            except ValueError:
                continue
        return found

    def _ask(self, prompt: str) -> dict[str, Any]:
        import re
        import threading
        import time

        self.asks = getattr(self, "asks", 0) + 1
        spent = getattr(self, "model_seconds", 0.0)
        if self.asks > self.MAX_ASKS or spent >= self.MODEL_TIME_BUDGET:
            return {}
        wait = max(1.0, min(self.WAIT_SECONDS, self.MODEL_TIME_BUDGET - spent))
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                box["answer"] = Utils._llm([{"role": "user", "content": prompt}])
            except Exception:
                box["answer"] = None

        started = time.monotonic()
        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        worker.join(wait)
        self.model_seconds = spent + (time.monotonic() - started)
        answer = box.get("answer")
        if worker.is_alive() or not isinstance(answer, dict):
            return {}
        text = answer.get("content") or ""
        if not isinstance(text, str):
            return {}
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        chunk = text[start : end + 1]
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                value = json.loads(candidate)
            except ValueError:
                continue
            return value if isinstance(value, dict) else {}
        return {}

    def _schema(self, name: Any) -> dict[str, Any] | None:
        for tool in self.tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and function.get("name") == name:
                return function.get("parameters") or {}
        return None

    def _tool(self, name: Any) -> str | None:
        return name if isinstance(name, str) and self._schema(name) is not None else None

    def _props(self, name: Any) -> dict[str, Any]:
        return (self._schema(name) or {}).get("properties") or {}

    def _fits(self, name: str, args: dict[str, Any]) -> bool:
        schema = self._schema(name) or {}
        props = schema.get("properties") or {}
        return all(k in props for k in args) and all(k in args for k in schema.get("required") or [])

    def _tools_text(self) -> str:
        described = []
        for tool in self.tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict):
                described.append({"name": function.get("name"), "description": function.get("description", ""), "parameters": function.get("parameters", {})})
        return json.dumps(described, ensure_ascii=False)

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    def _plan_tools(self, raw: dict[str, Any]) -> bool:
        names = raw.get("tools") if isinstance(raw.get("tools"), dict) else {}
        self.details = self._tool(names.get("details"))
        self.many = self._tool(names.get("details_many"))
        self.many_param = names.get("details_many_param") if names.get("details_many_param") in self._props(self.many) else None
        limit = self._number(names.get("details_many_limit"))
        self.many_limit = max(1, int(limit)) if limit else 1
        self.stock = self._tool(names.get("stock"))
        self.add = self._tool(names.get("add"))
        self.remove = self._tool(names.get("remove"))
        self.order = self._tool(names.get("order"))
        add_props = self._props(self.add)
        wanted = [p for p in raw.get("item_params") or [] if isinstance(p, str) and p in add_props]
        self.ids = wanted or [p for p in (self._schema(self.add) or {}).get("required") or [] if p in add_props]
        needed = list((self._schema(self.details) or {}).get("required") or [])
        usable = bool(needed) and set(needed) <= set(self.ids)
        self.group = needed if usable else list(self.ids)
        if not usable:
            self.details = None
        if not (self.many and self.many_param and len(self.group) == 1):
            self.many = None
        self.budget = self._number(raw.get("budget"))
        self.need_stock = raw.get("must_be_in_stock") is not False
        self.kind = str(raw.get("kind") or "")
        self.seen: dict[tuple[str, ...], dict[str, Any]] = {}
        self.price_label: str | None = None
        self.stock_label: str | None = None
        return bool(self.add and self.order and self.ids and (self.details or self.many))

    def _read_plan(self, prompt: str) -> dict[str, Any]:
        for _attempt in range(2):
            raw = self._ask(prompt)
            if raw and self._plan_tools(raw):
                return raw
        return {}

    def _learn_state(self, observation: dict[str, Any]) -> None:
        rows = self._rows(observation)
        if len(rows) != 1:
            return
        flags = [line for line in rows[0][2] if self._norm(line.partition(": ")[2]) in ("true", "false")]
        amounts = [
            line for line in rows[0][2]
            if line.partition(": ")[0] not in self.ids and self._number(line.partition(": ")[2]) is not None
        ]
        if len(flags) == 1:
            self.stock_label = flags[0].partition(": ")[0]
            self.state_learned = True
        if len(amounts) == 1:
            self.price_label = amounts[0].partition(": ")[0]
            self.state_learned = True

    def _send(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        calls = calls[: self.max_calls]
        if not calls or self.done or self.turn >= self.max_steps:
            return []
        turn = self.turn + 1
        group_id = f"{self.problem_id}-turn-{turn}"
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [{"call_id": f"{group_id}-{i}", "action": {"name": n, "args": a}} for i, (n, a) in enumerate(calls, start=1)],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            self.done = True
            return []
        self.turn = turn
        tool_calls = [
            {"id": f"{group_id}-{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(calls, start=1)
        ]
        self.dialogue.append({"role": "assistant", "content": "", "tool_calls": tool_calls, "environment_result": result})
        observations = []
        for call in result.get("calls") or []:
            wrapper = call.get("observation") if isinstance(call, dict) else None
            wrapper = wrapper if isinstance(wrapper, dict) else {}
            if wrapper.get("done"):
                self.done = True
            inner = wrapper.get("observation")
            observations.append(inner if isinstance(inner, dict) else {})
        return observations + [{}] * (len(calls) - len(observations))

    def _rows(self, node: Any) -> list[tuple[tuple[str, ...], list[str], list[str]]]:
        ids = self.ids

        def holds(value: Any) -> bool:
            if isinstance(value, dict):
                return any(p in value for p in ids) or any(holds(v) for v in value.values())
            if isinstance(value, list):
                return any(holds(v) for v in value)
            return False

        def lines(level: dict[str, Any]) -> list[str]:
            out: list[str] = []

            def flat(value: Any, label: str) -> None:
                if isinstance(value, dict):
                    for key, inner in value.items():
                        flat(inner, str(key))
                elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
                    for inner in value:
                        flat(inner, label)
                else:
                    shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    out.append(f"{label}: {shown}"[:400])

            flat(level, "")
            return out

        found: list[tuple[tuple[str, ...], list[str], list[str]]] = []

        def walk(value: Any, known: dict[str, str], outer: list[str]) -> None:
            if isinstance(value, list):
                for inner in value:
                    walk(inner, known, outer)
                return
            if not isinstance(value, dict):
                return
            own = {p: str(value[p]) for p in ids if isinstance(value.get(p), (str, int)) and not isinstance(value.get(p), bool)}
            known = {**known, **own}
            plain = {k: v for k, v in value.items() if not holds(v)}
            if own and all(p in known for p in ids):
                found.append((tuple(known[p] for p in ids), outer, lines(plain)))
            deeper = outer + lines(plain) if own else outer
            for inner in value.values():
                if isinstance(inner, (dict, list)):
                    walk(inner, known, deeper)

        walk(node, {}, [])
        return found

    def _record(self, observation: dict[str, Any]) -> None:
        for key, outer, own in self._rows(observation):
            entry = self.seen.setdefault(key, {"first": len(self.seen), "outer": [], "own": []})
            for part, fresh in (("outer", outer), ("own", own)):
                merged = {line.partition(": ")[0]: line for line in entry[part]}
                merged.update({line.partition(": ")[0]: line for line in fresh})
                entry[part] = list(merged.values())

    def _group_of(self, key: tuple[str, ...]) -> tuple[str, ...]:
        values = dict(zip(self.ids, key))
        return tuple(values.get(p, "") for p in self.group)

    def _args(self, name: str | None, key: tuple[str, ...]) -> dict[str, Any]:
        props = self._props(name)
        return {p: v for p, v in zip(self.ids, key) if p in props}

    def _detail_calls(self, groups: list[tuple[str, ...]]) -> list[tuple[str, dict[str, Any]]]:
        groups = [g for g in dict.fromkeys(groups) if all(g)]
        if self.many and (len(groups) > 1 or not self.details):
            return [
                (self.many, {self.many_param: [g[0] for g in groups[i : i + self.many_limit]]})
                for i in range(0, len(groups), self.many_limit)
            ]
        if self.details:
            props = self._props(self.details)
            return [(self.details, {p: v for p, v in zip(self.group, g) if p in props}) for g in groups]
        return []

    def _read_all(self, calls: list[tuple[str, dict[str, Any]]], turns: int) -> None:
        for start in range(0, len(calls), self.max_calls):
            if turns <= 0 or self.done or self.turn >= self.max_steps - 2:
                return
            turns -= 1
            chunk = calls[start : start + self.max_calls]
            for (name, _args), observation in zip(chunk, self._send(chunk)):
                if name not in (self.add, self.remove, self.order):
                    self._record(observation)

    def _item_text(self, key: tuple[str, ...]) -> list[str]:
        entry = self.seen.get(key) or {}
        return list(entry.get("own", [])) + list(entry.get("outer", []))

    def _shown(self, key: tuple[str, ...], part: Any) -> str | None:
        if not isinstance(part, dict) or not isinstance(part.get("quote"), str) or not part["quote"].strip():
            return None
        quote = self._norm(part["quote"].replace("\\", ""))
        return next((line for line in self._item_text(key) if quote in self._norm(line.replace("\\", ""))), None)

    def _learn(self, key: tuple[str, ...], part: Any, attr: str) -> None:
        line = self._shown(key, part)
        if not line or ": " not in line or getattr(self, attr) is not None or getattr(self, "state_learned", False):
            return
        value = line.partition(": ")[2]
        flag = self._norm(value) in ("true", "false")
        if (attr == "stock_label" and flag) or (attr == "price_label" and not flag and self._number(value) is not None):
            setattr(self, attr, line.partition(": ")[0])

    def _state(self, key: tuple[str, ...]) -> tuple[float | None, bool | None]:
        price = stock = None
        for line in self._item_text(key):
            label, _, value = line.partition(": ")
            if price is None and label == self.price_label:
                numbers = self._numbers(value)
                price = numbers[0] if numbers else None
            if stock is None and label == self.stock_label:
                text = self._norm(value)
                stock = True if text == "true" else False if text == "false" else None
        return price, stock

    def _ok(self, key: tuple[str, ...]) -> bool:
        price, stock = self._state(key)
        if self.need_stock and stock is False:
            return False
        return self.budget is None or price is None or price <= self.budget + 1e-9

    def _evidence(self, keys: list[tuple[str, ...]], per_listing: int = 6) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
        labels: dict[str, tuple[str, ...]] = {}
        listings: dict[tuple[str, ...], dict[str, Any]] = {}
        for key in keys:
            entry = self.seen.get(key)
            if not entry:
                continue
            listing = listings.setdefault(self._group_of(key), {"listing": [line[:300] for line in entry["outer"][:20]], "items": {}})
            if len(listing["items"]) >= per_listing:
                continue
            label = f"i{len(labels) + 1}"
            labels[label] = key
            listing["items"][label] = [line[:300] for line in entry["own"][:12]]
        return labels, list(listings.values())

    def _learn_row(self, lines: list[str]) -> None:
        if getattr(self, "price_label", None) and getattr(self, "stock_label", None):
            return
        flags = {l.partition(": ")[0] for l in lines if ": " in l and self._norm(l.partition(": ")[2]) in ("true", "false")}
        amounts = {
            l.partition(": ")[0] for l in lines
            if ": " in l and l.partition(": ")[0] not in self.ids and self._number(l.partition(": ")[2]) is not None
        }
        if len(flags) == 1 and len(amounts) == 1:
            self.price_label = getattr(self, "price_label", None) or next(iter(amounts))
            self.stock_label = getattr(self, "stock_label", None) or next(iter(flags))

    def _row_price(self, lines: list[str]) -> float | None:
        label = getattr(self, "price_label", None)
        return next((self._number(l.partition(": ")[2]) for l in lines if label and l.partition(": ")[0] == label), None)

    def _over_budget(self, prices: list[float | None]) -> int:
        known = [p for p in prices if p is not None]
        return 1 if self.budget is not None and known and len(known) == len(prices) and min(known) > self.budget + 1e-9 else 0

    @staticmethod
    def _listy(value: str) -> bool:
        text = value.strip()
        if not text or text[0] not in "[{" or text[-1] not in "]}":
            return False
        try:
            return isinstance(json.loads(text), (list, dict))
        except ValueError:
            return False

    def _label_key(self, labels: dict[str, tuple[str, ...]], name: Any) -> tuple[str, ...] | None:
        import re

        if not isinstance(name, str) or not name.strip():
            return None
        if name.strip() in labels:
            return labels[name.strip()]
        pieces = set(re.split(r"[^\w\-]+", name))
        named = [key for key in dict.fromkeys(labels.values()) if [p for p in key if p] and all(p in pieces for p in key if p)]
        return named[0] if len(named) == 1 else None

    def _fallback_commit(
        self, ordered: list[tuple[str, ...]], dropped: set[tuple[str, ...]], committed: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        pool = list(dict.fromkeys(k for k in [*ordered, *self.seen] if k not in dropped and k != committed))
        for _attempt in range(2):
            choice = next((k for k in pool if self._ok(k)), None)
            if choice is None or self.done or self.turn >= self.max_steps - 1:
                break
            calls = [(self.add, self._args(self.add, choice))]
            if committed is not None and committed != choice and self.remove:
                calls.append((self.remove, self._args(self.remove, committed)))
            if self.stock and self._fits(self.stock, self._args(self.stock, choice)):
                calls.append((self.stock, self._args(self.stock, choice)))
            observations = self._send(calls[: self.max_calls])
            for (name, _args), observation in zip(calls, observations):
                if name == self.stock:
                    self._record(observation)
                    self._learn_state(observation)
            pool.remove(choice)
            if observations and not (observations[0] or {}).get("error"):
                committed = choice
                if self._ok(choice):
                    break
        return committed

    def _commit_and_order(self, ordered: list[tuple[str, ...]], order_args: Any, rejudge: Any = None) -> None:
        ordered = list(dict.fromkeys(ordered))
        committed: tuple[str, ...] | None = None
        dropped: set[tuple[str, ...]] = set()
        for _attempt in range(4):
            if self.done or self.turn >= self.max_steps - 1:
                break
            choice = next((k for k in ordered if k not in dropped and self._ok(k)), None)
            if choice is None and rejudge is not None:
                extra, rejudge = rejudge(set(dropped)), None
                ordered = list(dict.fromkeys([*ordered, *extra]))
                choice = next((k for k in ordered if k not in dropped and self._ok(k)), None)
            if choice is None:
                break
            calls = [(self.add, self._args(self.add, choice))]
            if committed is not None and committed != choice and self.remove:
                calls.append((self.remove, self._args(self.remove, committed)))
            backups = [k for k in ordered if k not in dropped and k != choice][:3]
            if self.stock and self._fits(self.stock, self._args(self.stock, choice)):
                reads = [(self.stock, self._args(self.stock, k)) for k in [choice, *backups]]
            else:
                reads = self._detail_calls([self._group_of(k) for k in [choice, *backups]])
            calls = (calls + reads)[: self.max_calls]
            observations = self._send(calls)
            added = observations[0] if observations else {}
            for (name, _args), observation in zip(calls, observations):
                if name not in (self.add, self.remove, self.order):
                    self._record(observation)
                    if name == self.stock:
                        self._learn_state(observation)
            if not observations or added.get("error"):
                dropped.add(choice)
                continue
            committed = choice
            if self._ok(choice):
                break
            dropped.add(choice)
        if (committed is None or not self._ok(committed)) and not self.done and self.turn < self.max_steps - 1:
            committed = self._fallback_commit(ordered, dropped, committed)
        if committed is not None and not self.done and self.turn < self.max_steps:
            self._note("order", {
                "item": list(committed), "state": self._state(committed),
                "asks": getattr(self, "asks", 0), "model_seconds": round(getattr(self, "model_seconds", 0.0), 1),
            })
            self._send([(self.order, order_args(committed))])

    READ = """
You plan a shopping agent's work from the shopper's request and the tools it may call. Answer with one JSON object.

Tools (name, description, parameters):
{tools}

Request and rules:
{query}

Return:
{{
  "tools": {{"details": "tool showing one listing's facts and variants",
            "details_many": "tool showing several listings at once, or null",
            "details_many_param": "that tool's list parameter, or null",
            "details_many_limit": "how many ids that tool accepts per call, as a number, or null",
            "stock": "tool showing one exact item's current price and availability, or null",
            "add": "tool putting one exact item in the cart",
            "remove": "tool taking an item out of the cart, or null",
            "order": "tool placing the final order"}},
  "item_params": ["the add tool's parameters that identify one exact item"],
  "budget": "the most the shopper will pay, as a number, or null",
  "must_be_in_stock": true,
  "kind": "the kind of product wanted, in the request's words",
  "requirements": ["each firm requirement on what the product itself must be, in the request's words (not price, availability or the preferences)"],
  "must_show": ["each word or hyphenated term the request uses for the form or type the product must have beyond its kind, alternatives listed separately"],
  "must_not_show": ["each word or hyphenated term the request uses for what the product must not be, alternatives listed separately"],
  "preferences": [{{"property": "one preferred property", "value": "its preferred value, copied from the request"}}],
  "searches": [{{"tool": "a search or filter tool", "args": {{}}}}]
}}
"preferences": most important first, meaning the one the shopper gives up last comes first.
"searches": up to {limit} varied catalog look-ups likely to find the preferred item and close alternatives (for example
the preferred values alone, written a few ways, and the kind of product), each using only arguments its tool accepts,
including the price limit and availability when the tool accepts them.
""".strip()

    JUDGE = """
Pick shopping items for a request, using only each item's lines, and copy every quote exactly from that item's lines.
Listing lines apply to every item of that listing; an item's own lines describe that exact item, and when an own line and
a listing line disagree, the own line decides. Follow the request's own rules for which line decides a property.
Evidence rules: a property counts as shown only when every line of this item that states it agrees on one value
(its own option lines, its fact lines, and its title when the title names that property too); a line naming two
values, or two lines naming different values, means it is not shown. A requirement the item's lines do not visibly
state is not met. The kind of product must be visibly stated by the item's own title, type lines or options, with no
line naming a different kind, and lines stating capabilities that only fit a different kind of product make the kind
unproven.

Request and rules:
{query}

Kind of product wanted: {kind}
Price limit: {budget}
Firm requirements: {requirements}
Preferences, most important first: {preferences}

Listings:
{listings}

Return one JSON object:
{{"shortlist": [{{
  "item": "<item label>",
  "preferences": [{{"verdict": "yes" | "no" | "unsure", "quote": "..."}}],
  "price": {{"number": number, "quote": "..."}},
  "available": {{"value": true | false, "quote": "..."}}
}}]}}
List up to 8 items that are genuinely the kind of product wanted, within the price limit and available, including items
that miss some preferences; the preferences only order the list, best match first. "preferences" has one entry per preference, in the same order. Decide each preference from the
line that the request's rules say decides it; "yes" only when that deciding line shows exactly that value for this
item, and "unsure" when that line is missing, unclear or disagrees with another deciding line.
""".strip()

    def run(self) -> list[dict[str, Any]]:
        self.turn, self.done = 0, False
        raw = self._read_plan(self.READ.format(tools=self._tools_text(), query=self.query, limit=max(2, min(8, self.max_calls))))
        if not raw:
            return super().run()
        self._note("read", raw)
        try:
            self._shop(raw)
        except Exception as error:
            self._note("error", repr(error))
        return self.dialogue

    def _shop(self, raw: dict[str, Any]) -> None:
        wants = [
            w for w in raw.get("preferences") or []
            if isinstance(w, dict) and isinstance(w.get("value"), str) and w["value"].strip() and self._norm(w["value"]) in self._norm(self.query)
        ]
        import re

        for want in wants:
            for other in wants:
                if other is want or len(other["value"].strip()) >= len(want["value"].strip()):
                    continue
                trimmed = " ".join(re.sub(r"(?<![^\W_])" + re.escape(other["value"].strip()) + r"(?![^\W_])", " ", want["value"], flags=re.I).split())
                if trimmed:
                    want["value"] = trimmed
        searches = [
            (s["tool"], s["args"]) for s in raw.get("searches") or []
            if isinstance(s, dict) and self._tool(s.get("tool")) and isinstance(s.get("args"), dict) and self._fits(s["tool"], s["args"])
        ][: self.max_calls]
        if wants:
            top = self._norm(wants[0]["value"])
            base = next(((tool, args, name) for tool, args in searches for name, value in args.items()
                         if isinstance(value, str) and top in self._norm(value)), None)
            if base is not None:
                tool, args, name = base
                values = [w["value"] for w in wants]
                kind = str(self.kind or "").strip()
                asked = {self._norm(a.get(name)) for t, a in searches if t == tool and isinstance(a.get(name), str)}
                for text in (" ".join([*values, kind]).strip(), " ".join(reversed(values)), f"{values[0]} {kind}".strip(), values[0]):
                    extra = {**args, name: text}
                    if self._norm(text) not in asked and len(searches) < self.max_calls and self._fits(tool, extra):
                        asked.add(self._norm(text))
                        searches.append((tool, extra))
        import re

        query_words = set(re.findall(r"[^\W_]+", self._norm(self.query)))
        self.needs = []
        for need in raw.get("requirements") or []:
            words = re.findall(r"[^\W_]+", self._norm(need)) if isinstance(need, str) else []
            if words and sum(w in query_words for w in words) >= len(words) / 2:
                self.needs.append(need)
        spoken = " " + " ".join(re.findall(r"[^\W_]+", self._norm(self.query))) + " "
        kind_words = set(re.findall(r"[^\W_]+", self._norm(self.kind)))
        self.terms = {}
        for name in ("must_show", "must_not_show"):
            found: list[str] = []
            for text in raw.get(name) or []:
                if not isinstance(text, str):
                    continue
                for piece in (text, *re.findall(r"[^\W_]+(?:-[^\W_]+)+", text)):
                    term = " ".join(re.findall(r"[^\W_]+", self._norm(piece)))
                    if not term or term in found or f" {term} " not in spoken:
                        continue
                    if name == "must_show" and set(term.split()) <= kind_words:
                        continue
                    found.append(term)
            self.terms[name] = found
        hits: list[tuple[tuple[str, ...], str]] = []
        texts: dict[tuple[str, ...], str] = {}
        best: dict[tuple[str, ...], int] = {}
        prices: dict[tuple[str, ...], list[float | None]] = {}
        observations = self._send(searches)
        for position, observation in enumerate(observations):
            asked = self._norm(json.dumps(searches[position][1], ensure_ascii=False)) if len(observations) == len(searches) else ""
            local: list[tuple[str, ...]] = []
            for key, outer, own in self._rows(observation):
                hits.append((key, self._norm(" ".join(outer + own))))
                texts.setdefault(self._group_of(key), self._row_text(outer, own))
                self._learn_row(outer + own)
                prices.setdefault(self._group_of(key), []).append(self._row_price(outer + own))
                if all(self._group_of(key)) and self._group_of(key) not in local:
                    local.append(self._group_of(key))
            if wants and self._norm(wants[0]["value"]) in asked:
                for rank, group in enumerate(local):
                    best[group] = min(best.get(group, rank), rank)
        ranked_hits = sorted(range(len(hits)), key=lambda i: (-sum(self._norm(w["value"]) in hits[i][1] for w in wants), i))
        groups: list[tuple[str, ...]] = []
        for index in ranked_hits:
            group = self._group_of(hits[index][0])
            if all(group) and group not in groups:
                groups.append(group)
        focus = "Preferences, most important first: " + json.dumps([{"property": w.get("property"), "value": w["value"]} for w in wants], ensure_ascii=False)
        picked = self._screen(groups, texts, focus, limit=len(groups)) or groups
        screened = [
            *sorted((group for group in picked if group in best), key=best.get),
            *sorted((group for group in best if best[group] < 8 and group not in picked), key=best.get),
            *(group for group in picked if group not in best),
        ]
        screened = sorted(screened, key=lambda group: self._over_budget(prices.get(group, [])))
        groups, self.reserve = screened[: self.MAX_LISTINGS], screened[self.MAX_LISTINGS : 2 * self.MAX_LISTINGS]
        self._read_all(self._detail_calls(groups), turns=2)
        keys = [k for k in self.seen if self._group_of(k) in groups]
        ordered = self._pick(keys, wants)
        self._commit_and_order(
            ordered, lambda k: self._args(self.order, k),
            rejudge=lambda dropped: self._pick(self._more_keys(keys, dropped), wants),
        )

    def _pick(self, keys: list[tuple[str, ...]], wants: list[dict[str, Any]]) -> list[tuple[str, ...]]:
        import re

        preferences = json.dumps([{"property": w.get("property"), "value": w["value"]} for w in wants], ensure_ascii=False)
        answer: dict[str, Any] = {}
        labels: dict[str, tuple[str, ...]] = {}
        for share, per_listing in ((1, 6), (2, 3)):
            labels, listings = self._evidence(keys[: max(1, len(keys) // share)], per_listing=per_listing)
            if not labels:
                return []
            answer = self._ask(self.JUDGE.format(
                query=self.query, kind=self.kind, budget=self.budget, preferences=preferences,
                requirements=json.dumps(getattr(self, "needs", []), ensure_ascii=False),
                listings=json.dumps(listings, ensure_ascii=False),
            ))
            if isinstance(answer.get("shortlist"), list) and answer["shortlist"]:
                break
        self._note("judge", answer)
        info: dict[tuple[str, ...], tuple[Any, ...]] = {}
        for position, pick in enumerate(answer.get("shortlist") or []):
            key = self._label_key(labels, pick.get("item")) if isinstance(pick, dict) else None
            if key is None or key in info:
                continue
            self._learn(key, pick.get("price"), "price_label")
            self._learn(key, pick.get("available"), "stock_label")
            verdicts = pick.get("preferences") if isinstance(pick.get("preferences"), list) else []
            missed, loose = [], []
            def exact_in(line: str | None, target: str) -> bool:
                value = self._norm(line.partition(": ")[2] if line and ": " in line else (line or ""))
                if not value:
                    return False
                if len(value.split()) > 3:
                    return bool(re.search(r"(?<![^\W_])" + re.escape(target) + r"(?![^\W_])", value))
                if target in [self._norm(p) for p in re.split(r"[/,;|()\-]+", value)]:
                    return True
                if target.replace(" ", "").isalpha() and any(
                    re.search(r"(?<![^\W_])" + re.escape(target) + r"(?![^\W_])", self._norm(piece))
                    and len(self._norm(piece).split()) <= len(target.split()) + 2
                    for piece in re.split(r"[/,;|()\-]+", value)
                ):
                    return True
                words = value.split()
                if len(words) > 1 and self._norm(" ".join(words[1:])) == target:
                    others = self._norm(" ".join(l for l in self._item_text(key) if l != line))
                    return bool(re.search(r"(?<![^\W_])" + re.escape(words[0]) + r"(?![^\W_])", others))
                return False

            for index, want in enumerate(wants):
                part = verdicts[index] if index < len(verdicts) else None
                line = self._shown(key, part) if isinstance(part, dict) and part.get("verdict") in ("yes", "unsure") else None
                target = self._norm(want["value"])
                exact = exact_in(line, target)
                if line and not exact and not re.search(r"(?<![^\W_])" + re.escape(target) + r"(?![^\W_])", self._norm(line)):
                    entry = self.seen.get(key) or {}
                    line = next((l for l in [*entry.get("own", []), *entry.get("outer", [])] if exact_in(l, target)), line)
                    exact = exact_in(line, target)
                overruled = False
                prop = " ".join(re.findall(r"[^\W_]+", self._norm(want.get("property") or "")))
                own_lines = (self.seen.get(key) or {}).get("own", [])
                if line and exact and line not in own_lines and prop:
                    named = [
                        l for l in self._item_text(key)
                        if l != line and " ".join(re.findall(r"[^\W_]+", self._norm(l.partition(": ")[0]))) == prop
                    ]
                    if named and not any(exact_in(l, target) for l in named):
                        exact, overruled = False, True
                clearly_other = overruled or (
                    isinstance(part, dict) and part.get("verdict") == "no" and self._shown(key, part) is not None
                )
                missed.append(0 if line and exact else 1 if clearly_other else 2)
                loose.append(0 if line and exact else 1)
            own_words = " " + " ".join(re.findall(r"[^\W_]+", self._norm(" ".join(
                line.partition(": ")[2] for line in self._item_text(key)
                if ": " in line and not self._listy(line.partition(": ")[2])
            )))) + " "
            terms = getattr(self, "terms", {})
            if any(f" {term} " in own_words for term in terms.get("must_not_show", [])):
                continue
            unproven = 1 if terms.get("must_show") and not any(f" {term} " in own_words for term in terms["must_show"]) else 0
            info[key] = (unproven, tuple(missed), tuple(loose), position)
        pool = list(info)
        ordered = sorted(info, key=lambda k: info[k])
        self._note("ranked", [[list(k), info[k]] for k in ordered])
        return ordered

class Justification(Engine):
    marker = "a valid product plus a trustworthy reason"

    WAIT_SECONDS = 40
    MAX_ASKS = 6
    MODEL_TIME_BUDGET = 150
    MAX_LISTINGS = 15
    DEBUG = False

    SCREEN = """
Choose which catalog listings to open for a shopper, using only their search rows.

Request and rules:
{query}

Kind of product wanted: {kind}
{focus}

Search rows, one per listing:
{rows}

Return one JSON object: {{"open": ["<listing label>", "..."]}}
List up to {limit} listings whose row describes the kind of product wanted itself (skip accessories, parts, cases and
other items made for it), most promising for this request first.
""".strip()

    def _screen(self, groups: list[tuple[str, ...]], texts: dict[tuple[str, ...], str], focus: str, limit: int = 0) -> list[tuple[str, ...]]:
        if len(groups) <= 1:
            return list(groups)
        labels = {f"g{i + 1}": g for i, g in enumerate(groups[:60])}
        rows = {label: texts.get(group, "")[:160] for label, group in labels.items()}
        answer = self._ask(self.SCREEN.format(
            query=self.query, kind=self.kind, focus=focus, rows=json.dumps(rows, ensure_ascii=False), limit=min(60, limit or 2 * self.MAX_LISTINGS),
        ))
        picked = list(dict.fromkeys(k for x in answer.get("open") or [] if (k := self._label_key(labels, x)) is not None))
        self._note("screen", {"asked": len(labels), "picked": len(picked)})
        return list(dict.fromkeys(picked))

    def _more_keys(self, keys: list[tuple[str, ...]], dropped: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
        reserve = getattr(self, "reserve", [])
        if reserve and self.turn < self.max_steps - 3:
            self.reserve = []
            self._read_all(self._detail_calls(reserve), turns=1)
            keys.extend(k for k in self.seen if self._group_of(k) in set(reserve) and k not in keys)
        return [k for k in keys if k not in dropped]

    def _row_text(self, outer: list[str], own: list[str]) -> str:
        return " | ".join(line for line in outer + own if line.partition(": ")[0] not in self.ids)

    def _note(self, kind: str, data: Any) -> None:
        if self.DEBUG:
            self.dialogue.append({"role": "debug", "kind": kind, "data": data})

    @staticmethod
    def _norm(text: Any) -> str:
        return " ".join(str(text).casefold().split())

    @staticmethod
    def _numbers(text: Any) -> list[float]:
        import re

        found = []
        for token in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text)):
            try:
                found.append(float(token.replace(",", "")))
            except ValueError:
                continue
        return found

    def _ask(self, prompt: str) -> dict[str, Any]:
        import re
        import threading
        import time

        self.asks = getattr(self, "asks", 0) + 1
        spent = getattr(self, "model_seconds", 0.0)
        if self.asks > self.MAX_ASKS or spent >= self.MODEL_TIME_BUDGET:
            return {}
        wait = max(1.0, min(self.WAIT_SECONDS, self.MODEL_TIME_BUDGET - spent))
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                box["answer"] = Utils._llm([{"role": "user", "content": prompt}])
            except Exception:
                box["answer"] = None

        started = time.monotonic()
        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        worker.join(wait)
        self.model_seconds = spent + (time.monotonic() - started)
        answer = box.get("answer")
        if worker.is_alive() or not isinstance(answer, dict):
            return {}
        text = answer.get("content") or ""
        if not isinstance(text, str):
            return {}
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        chunk = text[start : end + 1]
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                value = json.loads(candidate)
            except ValueError:
                continue
            return value if isinstance(value, dict) else {}
        return {}

    def _schema(self, name: Any) -> dict[str, Any] | None:
        for tool in self.tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and function.get("name") == name:
                return function.get("parameters") or {}
        return None

    def _tool(self, name: Any) -> str | None:
        return name if isinstance(name, str) and self._schema(name) is not None else None

    def _props(self, name: Any) -> dict[str, Any]:
        return (self._schema(name) or {}).get("properties") or {}

    def _fits(self, name: str, args: dict[str, Any]) -> bool:
        schema = self._schema(name) or {}
        props = schema.get("properties") or {}
        return all(k in props for k in args) and all(k in args for k in schema.get("required") or [])

    def _tools_text(self) -> str:
        described = []
        for tool in self.tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict):
                described.append({"name": function.get("name"), "description": function.get("description", ""), "parameters": function.get("parameters", {})})
        return json.dumps(described, ensure_ascii=False)

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    def _plan_tools(self, raw: dict[str, Any]) -> bool:
        names = raw.get("tools") if isinstance(raw.get("tools"), dict) else {}
        self.details = self._tool(names.get("details"))
        self.many = self._tool(names.get("details_many"))
        self.many_param = names.get("details_many_param") if names.get("details_many_param") in self._props(self.many) else None
        limit = self._number(names.get("details_many_limit"))
        self.many_limit = max(1, int(limit)) if limit else 1
        self.stock = self._tool(names.get("stock"))
        self.add = self._tool(names.get("add"))
        self.remove = self._tool(names.get("remove"))
        self.order = self._tool(names.get("order"))
        add_props = self._props(self.add)
        wanted = [p for p in raw.get("item_params") or [] if isinstance(p, str) and p in add_props]
        self.ids = wanted or [p for p in (self._schema(self.add) or {}).get("required") or [] if p in add_props]
        needed = list((self._schema(self.details) or {}).get("required") or [])
        usable = bool(needed) and set(needed) <= set(self.ids)
        self.group = needed if usable else list(self.ids)
        if not usable:
            self.details = None
        if not (self.many and self.many_param and len(self.group) == 1):
            self.many = None
        self.budget = self._number(raw.get("budget"))
        self.need_stock = raw.get("must_be_in_stock") is not False
        self.kind = str(raw.get("kind") or "")
        self.seen: dict[tuple[str, ...], dict[str, Any]] = {}
        self.price_label: str | None = None
        self.stock_label: str | None = None
        return bool(self.add and self.order and self.ids and (self.details or self.many))

    def _read_plan(self, prompt: str) -> dict[str, Any]:
        for _attempt in range(2):
            raw = self._ask(prompt)
            if raw and self._plan_tools(raw):
                return raw
        return {}

    def _learn_state(self, observation: dict[str, Any]) -> None:
        rows = self._rows(observation)
        if len(rows) != 1:
            return
        flags = [line for line in rows[0][2] if self._norm(line.partition(": ")[2]) in ("true", "false")]
        amounts = [
            line for line in rows[0][2]
            if line.partition(": ")[0] not in self.ids and self._number(line.partition(": ")[2]) is not None
        ]
        if len(flags) == 1:
            self.stock_label = flags[0].partition(": ")[0]
            self.state_learned = True
        if len(amounts) == 1:
            self.price_label = amounts[0].partition(": ")[0]
            self.state_learned = True

    def _send(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        calls = calls[: self.max_calls]
        if not calls or self.done or self.turn >= self.max_steps:
            return []
        turn = self.turn + 1
        group_id = f"{self.problem_id}-turn-{turn}"
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [{"call_id": f"{group_id}-{i}", "action": {"name": n, "args": a}} for i, (n, a) in enumerate(calls, start=1)],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            self.done = True
            return []
        self.turn = turn
        tool_calls = [
            {"id": f"{group_id}-{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(calls, start=1)
        ]
        self.dialogue.append({"role": "assistant", "content": "", "tool_calls": tool_calls, "environment_result": result})
        observations = []
        for call in result.get("calls") or []:
            wrapper = call.get("observation") if isinstance(call, dict) else None
            wrapper = wrapper if isinstance(wrapper, dict) else {}
            if wrapper.get("done"):
                self.done = True
            inner = wrapper.get("observation")
            observations.append(inner if isinstance(inner, dict) else {})
        return observations + [{}] * (len(calls) - len(observations))

    def _rows(self, node: Any) -> list[tuple[tuple[str, ...], list[str], list[str]]]:
        ids = self.ids

        def holds(value: Any) -> bool:
            if isinstance(value, dict):
                return any(p in value for p in ids) or any(holds(v) for v in value.values())
            if isinstance(value, list):
                return any(holds(v) for v in value)
            return False

        def lines(level: dict[str, Any]) -> list[str]:
            out: list[str] = []

            def flat(value: Any, label: str) -> None:
                if isinstance(value, dict):
                    for key, inner in value.items():
                        flat(inner, str(key))
                elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
                    for inner in value:
                        flat(inner, label)
                else:
                    shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    out.append(f"{label}: {shown}"[:400])

            flat(level, "")
            return out

        found: list[tuple[tuple[str, ...], list[str], list[str]]] = []

        def walk(value: Any, known: dict[str, str], outer: list[str]) -> None:
            if isinstance(value, list):
                for inner in value:
                    walk(inner, known, outer)
                return
            if not isinstance(value, dict):
                return
            own = {p: str(value[p]) for p in ids if isinstance(value.get(p), (str, int)) and not isinstance(value.get(p), bool)}
            known = {**known, **own}
            plain = {k: v for k, v in value.items() if not holds(v)}
            if own and all(p in known for p in ids):
                found.append((tuple(known[p] for p in ids), outer, lines(plain)))
            deeper = outer + lines(plain) if own else outer
            for inner in value.values():
                if isinstance(inner, (dict, list)):
                    walk(inner, known, deeper)

        walk(node, {}, [])
        return found

    def _record(self, observation: dict[str, Any]) -> None:
        for key, outer, own in self._rows(observation):
            entry = self.seen.setdefault(key, {"first": len(self.seen), "outer": [], "own": []})
            for part, fresh in (("outer", outer), ("own", own)):
                merged = {line.partition(": ")[0]: line for line in entry[part]}
                merged.update({line.partition(": ")[0]: line for line in fresh})
                entry[part] = list(merged.values())

    def _group_of(self, key: tuple[str, ...]) -> tuple[str, ...]:
        values = dict(zip(self.ids, key))
        return tuple(values.get(p, "") for p in self.group)

    def _args(self, name: str | None, key: tuple[str, ...]) -> dict[str, Any]:
        props = self._props(name)
        return {p: v for p, v in zip(self.ids, key) if p in props}

    def _detail_calls(self, groups: list[tuple[str, ...]]) -> list[tuple[str, dict[str, Any]]]:
        groups = [g for g in dict.fromkeys(groups) if all(g)]
        if self.many and (len(groups) > 1 or not self.details):
            return [
                (self.many, {self.many_param: [g[0] for g in groups[i : i + self.many_limit]]})
                for i in range(0, len(groups), self.many_limit)
            ]
        if self.details:
            props = self._props(self.details)
            return [(self.details, {p: v for p, v in zip(self.group, g) if p in props}) for g in groups]
        return []

    def _read_all(self, calls: list[tuple[str, dict[str, Any]]], turns: int) -> None:
        for start in range(0, len(calls), self.max_calls):
            if turns <= 0 or self.done or self.turn >= self.max_steps - 2:
                return
            turns -= 1
            chunk = calls[start : start + self.max_calls]
            for (name, _args), observation in zip(chunk, self._send(chunk)):
                if name not in (self.add, self.remove, self.order):
                    self._record(observation)

    def _item_text(self, key: tuple[str, ...]) -> list[str]:
        entry = self.seen.get(key) or {}
        return list(entry.get("own", [])) + list(entry.get("outer", []))

    def _shown(self, key: tuple[str, ...], part: Any) -> str | None:
        if not isinstance(part, dict) or not isinstance(part.get("quote"), str) or not part["quote"].strip():
            return None
        quote = self._norm(part["quote"].replace("\\", ""))
        return next((line for line in self._item_text(key) if quote in self._norm(line.replace("\\", ""))), None)

    def _learn(self, key: tuple[str, ...], part: Any, attr: str) -> None:
        line = self._shown(key, part)
        if not line or ": " not in line or getattr(self, attr) is not None or getattr(self, "state_learned", False):
            return
        value = line.partition(": ")[2]
        flag = self._norm(value) in ("true", "false")
        if (attr == "stock_label" and flag) or (attr == "price_label" and not flag and self._number(value) is not None):
            setattr(self, attr, line.partition(": ")[0])

    def _state(self, key: tuple[str, ...]) -> tuple[float | None, bool | None]:
        price = stock = None
        for line in self._item_text(key):
            label, _, value = line.partition(": ")
            if price is None and label == self.price_label:
                numbers = self._numbers(value)
                price = numbers[0] if numbers else None
            if stock is None and label == self.stock_label:
                text = self._norm(value)
                stock = True if text == "true" else False if text == "false" else None
        return price, stock

    def _ok(self, key: tuple[str, ...]) -> bool:
        price, stock = self._state(key)
        if self.need_stock and stock is False:
            return False
        return self.budget is None or price is None or price <= self.budget + 1e-9

    def _evidence(self, keys: list[tuple[str, ...]], per_listing: int = 6) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
        labels: dict[str, tuple[str, ...]] = {}
        listings: dict[tuple[str, ...], dict[str, Any]] = {}
        for key in keys:
            entry = self.seen.get(key)
            if not entry:
                continue
            listing = listings.setdefault(self._group_of(key), {"listing": [line[:300] for line in entry["outer"][:20]], "items": {}})
            if len(listing["items"]) >= per_listing:
                continue
            label = f"i{len(labels) + 1}"
            labels[label] = key
            listing["items"][label] = [line[:300] for line in entry["own"][:12]]
        return labels, list(listings.values())

    def _learn_row(self, lines: list[str]) -> None:
        if getattr(self, "price_label", None) and getattr(self, "stock_label", None):
            return
        flags = {l.partition(": ")[0] for l in lines if ": " in l and self._norm(l.partition(": ")[2]) in ("true", "false")}
        amounts = {
            l.partition(": ")[0] for l in lines
            if ": " in l and l.partition(": ")[0] not in self.ids and self._number(l.partition(": ")[2]) is not None
        }
        if len(flags) == 1 and len(amounts) == 1:
            self.price_label = getattr(self, "price_label", None) or next(iter(amounts))
            self.stock_label = getattr(self, "stock_label", None) or next(iter(flags))

    def _row_price(self, lines: list[str]) -> float | None:
        label = getattr(self, "price_label", None)
        return next((self._number(l.partition(": ")[2]) for l in lines if label and l.partition(": ")[0] == label), None)

    def _over_budget(self, prices: list[float | None]) -> int:
        known = [p for p in prices if p is not None]
        return 1 if self.budget is not None and known and len(known) == len(prices) and min(known) > self.budget + 1e-9 else 0

    @staticmethod
    def _listy(value: str) -> bool:
        text = value.strip()
        if not text or text[0] not in "[{" or text[-1] not in "]}":
            return False
        try:
            return isinstance(json.loads(text), (list, dict))
        except ValueError:
            return False

    def _label_key(self, labels: dict[str, tuple[str, ...]], name: Any) -> tuple[str, ...] | None:
        import re

        if not isinstance(name, str) or not name.strip():
            return None
        if name.strip() in labels:
            return labels[name.strip()]
        pieces = set(re.split(r"[^\w\-]+", name))
        named = [key for key in dict.fromkeys(labels.values()) if [p for p in key if p] and all(p in pieces for p in key if p)]
        return named[0] if len(named) == 1 else None

    def _fallback_commit(
        self, ordered: list[tuple[str, ...]], dropped: set[tuple[str, ...]], committed: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        pool = list(dict.fromkeys(k for k in [*ordered, *self.seen] if k not in dropped and k != committed))
        for _attempt in range(2):
            choice = next((k for k in pool if self._ok(k)), None)
            if choice is None or self.done or self.turn >= self.max_steps - 1:
                break
            calls = [(self.add, self._args(self.add, choice))]
            if committed is not None and committed != choice and self.remove:
                calls.append((self.remove, self._args(self.remove, committed)))
            if self.stock and self._fits(self.stock, self._args(self.stock, choice)):
                calls.append((self.stock, self._args(self.stock, choice)))
            observations = self._send(calls[: self.max_calls])
            for (name, _args), observation in zip(calls, observations):
                if name == self.stock:
                    self._record(observation)
                    self._learn_state(observation)
            pool.remove(choice)
            if observations and not (observations[0] or {}).get("error"):
                committed = choice
                if self._ok(choice):
                    break
        return committed

    def _commit_and_order(self, ordered: list[tuple[str, ...]], order_args: Any, rejudge: Any = None) -> None:
        ordered = list(dict.fromkeys(ordered))
        committed: tuple[str, ...] | None = None
        dropped: set[tuple[str, ...]] = set()
        for _attempt in range(4):
            if self.done or self.turn >= self.max_steps - 1:
                break
            choice = next((k for k in ordered if k not in dropped and self._ok(k)), None)
            if choice is None and rejudge is not None:
                extra, rejudge = rejudge(set(dropped)), None
                ordered = list(dict.fromkeys([*ordered, *extra]))
                choice = next((k for k in ordered if k not in dropped and self._ok(k)), None)
            if choice is None:
                break
            calls = [(self.add, self._args(self.add, choice))]
            if committed is not None and committed != choice and self.remove:
                calls.append((self.remove, self._args(self.remove, committed)))
            backups = [k for k in ordered if k not in dropped and k != choice][:3]
            if self.stock and self._fits(self.stock, self._args(self.stock, choice)):
                reads = [(self.stock, self._args(self.stock, k)) for k in [choice, *backups]]
            else:
                reads = self._detail_calls([self._group_of(k) for k in [choice, *backups]])
            calls = (calls + reads)[: self.max_calls]
            observations = self._send(calls)
            added = observations[0] if observations else {}
            for (name, _args), observation in zip(calls, observations):
                if name not in (self.add, self.remove, self.order):
                    self._record(observation)
                    if name == self.stock:
                        self._learn_state(observation)
            if not observations or added.get("error"):
                dropped.add(choice)
                continue
            committed = choice
            if self._ok(choice):
                break
            dropped.add(choice)
        if (committed is None or not self._ok(committed)) and not self.done and self.turn < self.max_steps - 1:
            committed = self._fallback_commit(ordered, dropped, committed)
        if committed is not None and not self.done and self.turn < self.max_steps:
            self._note("order", {
                "item": list(committed), "state": self._state(committed),
                "asks": getattr(self, "asks", 0), "model_seconds": round(getattr(self, "model_seconds", 0.0), 1),
            })
            self._send([(self.order, order_args(committed))])

    READ = """
You plan a shopping agent's work from the shopper's request and the tools it may call. Answer with one JSON object.

Tools (name, description, parameters):
{tools}

Request and rules:
{query}

Return:
{{
  "tools": {{"details": "tool showing one listing's facts and variants",
            "details_many": "tool showing several listings at once, or null",
            "details_many_param": "that tool's list parameter, or null",
            "details_many_limit": "how many ids that tool accepts per call, as a number, or null",
            "stock": "tool showing one exact item's current price and availability, or null",
            "add": "tool putting one exact item in the cart",
            "remove": "tool taking an item out of the cart, or null",
            "order": "tool placing the final order"}},
  "item_params": ["the add tool's parameters that identify one exact item"],
  "budget": "the most the shopper will pay, as a number, or null",
  "must_be_in_stock": true,
  "kind": "the kind of product wanted, in the request's words",
  "specification": "the specification the product must have, copied from the request",
  "brand": "the brand the request requires, copied from the request, or null",
  "searches": [{{"tool": "a search or filter tool", "args": {{}}}}]
}}
"searches": up to {limit} varied catalog look-ups likely to find products of that kind with that specification, each
using only arguments its tool accepts, including the price limit and availability when the tool accepts them.
""".strip()

    JUDGE = """
Pick shopping items for a request, using only each item's lines, and copy every quote exactly from that item's lines.
Listing lines apply to every item of that listing; an item's own lines describe that exact item, and when an own line and
a listing line disagree, the own line decides.
Evidence rules: a property counts as shown only when every line of this item that states it agrees on one value
(its own option lines, its fact lines, and its title when the title names that property too); a line naming two
values, or two lines naming different values, means it is not shown. A requirement the item's lines do not visibly
state is not met. The kind of product must be visibly stated by the item's own title, type lines or options, with no
line naming a different kind, and lines stating capabilities that only fit a different kind of product make the kind
unproven.

Request and rules:
{query}

Kind of product wanted: {kind}
Specification required: {specification}
Price limit: {budget}

Listings:
{listings}

Return one JSON object:
{{"shortlist": [{{
  "item": "<item label>",
  "specification": [{{"quote": "..."}}],
  "price": {{"number": number, "quote": "..."}},
  "available": {{"value": true | false, "quote": "..."}},
  "reasons": [{{"quote": "..."}}]
}}]}}
List every item (up to 8) that is genuinely the kind of product wanted, has the specification, is within the price
limit and available, best proven first; more than one when more qualify, so a backup exists. "specification" quotes every
line of this item that states the required specification (its title too when the title states it). The specification
counts as proven only by a line that states it plainly for this item;
a line that names different values, examples or choices, or that another line of the same item contradicts, does not
prove it. "reasons" quotes up to 3 more lines of this item, each a plain, true, positive fact that differs from the
specification, price and availability lines; each value must be one plain value stating what its label names, not a
list, a path of categories, a range or a code, and not a figure that the item's own option lines state differently.
""".strip()

    def run(self) -> list[dict[str, Any]]:
        self.turn, self.done = 0, False
        raw = self._read_plan(self.READ.format(tools=self._tools_text(), query=self.query, limit=max(2, min(8, self.max_calls))))
        if not raw:
            return super().run()
        self._note("read", raw)
        try:
            self._shop(raw)
        except Exception as error:
            self._note("error", repr(error))
        return self.dialogue

    def _shop(self, raw: dict[str, Any]) -> None:
        wanted = str(raw.get("specification") or "")
        import re

        brand = raw.get("brand") if isinstance(raw.get("brand"), str) else ""
        self.brand_words = (
            set(re.findall(r"[^\W_]+", self._norm(brand))) if brand.strip() and self._norm(brand) in self._norm(self.query) else set()
        )
        searches = [
            (s["tool"], s["args"]) for s in raw.get("searches") or []
            if isinstance(s, dict) and self._tool(s.get("tool")) and isinstance(s.get("args"), dict) and self._fits(s["tool"], s["args"])
        ][: self.max_calls]
        groups: list[tuple[str, ...]] = []
        texts: dict[tuple[str, ...], str] = {}
        row_lines: dict[tuple[str, ...], list[str]] = {}
        prices: dict[tuple[str, ...], list[float | None]] = {}
        for observation in self._send(searches):
            for key, outer, own in self._rows(observation):
                group = self._group_of(key)
                texts.setdefault(group, self._row_text(outer, own))
                row_lines.setdefault(group, outer + own)
                self._learn_row(outer + own)
                prices.setdefault(group, []).append(self._row_price(outer + own))
                if all(group) and group not in groups:
                    groups.append(group)
        picked = self._screen(groups, texts, f"Specification required: {wanted}") or groups
        screened = [
            *sorted(picked, key=lambda group: self._unstated_lines(row_lines.get(group, []))),
            *(group for group in groups if group not in picked and not self._unstated_lines(row_lines.get(group, []))),
        ]
        screened = sorted(screened, key=lambda group: self._over_budget(prices.get(group, [])))
        groups, self.reserve = screened[: self.MAX_LISTINGS], screened[self.MAX_LISTINGS : 2 * self.MAX_LISTINGS]
        self._read_all(self._detail_calls(groups), turns=2)
        keys = [k for k in self.seen if self._group_of(k) in groups]
        self.claim_lines: dict[tuple[str, ...], list[str]] = {}
        self.spec_labels: dict[str, int] = {}
        ordered = self._pick(keys, wanted)
        weak = [k for k in ordered if self._unstated(k) or self._kind_mismatch(k) or self._brand_miss(k)]
        backup = self._backups([k for k in keys if k not in ordered], wanted)

        def cleanest(first: list[tuple[str, ...]], second: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
            pool = list(dict.fromkeys([*first, *second]))
            return sorted(pool, key=lambda k: (self._brand_miss(k), self._unclaimable(k), 0 if self._clean(k, wanted) else 1))

        def replacement(dropped: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
            more = self._more_keys(keys, dropped)
            extra = self._pick(more, wanted)
            spare = self._backups([k for k in more if k not in extra], wanted)
            return [*cleanest([k for k in extra if not (self._unstated(k) or self._kind_mismatch(k) or self._brand_miss(k))], spare), *weak, *extra]

        self._commit_and_order(cleanest([k for k in ordered if k not in weak], backup) or ordered, self._order_args, rejudge=replacement)

    def _pick(self, keys: list[tuple[str, ...]], wanted: str) -> list[tuple[str, ...]]:
        import re

        keys = sorted(keys, key=self._unstated)
        answer: dict[str, Any] = {}
        labels: dict[str, tuple[str, ...]] = {}
        for share, per_listing in ((1, 6), (2, 3)):
            labels, listings = self._evidence(keys[: max(1, len(keys) // share)], per_listing=per_listing)
            if not labels:
                return []
            answer = self._ask(self.JUDGE.format(
                query=self.query, kind=self.kind, specification=wanted, budget=self.budget,
                listings=json.dumps(listings, ensure_ascii=False),
            ))
            if isinstance(answer.get("shortlist"), list) and answer["shortlist"]:
                break
        self._note("judge", answer)
        words = {w for w in re.findall(r"[^\W_]+", self._norm(wanted)) if len(w) > 1}
        asked = set(self._figures(wanted))
        info: dict[tuple[str, ...], tuple[Any, ...]] = {}
        for position, pick in enumerate(answer.get("shortlist") or []):
            key = self._label_key(labels, pick.get("item")) if isinstance(pick, dict) else None
            if key is None or key in info:
                continue
            quotes = pick.get("specification") if isinstance(pick.get("specification"), list) else [pick.get("specification")]
            plain = list(dict.fromkeys(
                line for line in (self._shown(key, part) for part in quotes)
                if line and ": " in line and not self._listy(line.partition(": ")[2])
            ))
            if not plain:
                continue
            own = set((self.seen.get(key) or {}).get("own", []))
            claimable = [line for line in plain if line not in own and len(line.partition(": ")[2].split()) <= 8]
            proof = min(claimable or plain, key=lambda line: len(line.partition(": ")[2]))
            if proof in own or len(proof.partition(": ")[2].split()) > 8:
                proof = min(
                    (line for line in self._item_text(key) if line not in own and self._states_spec(line, words, asked)),
                    key=lambda line: len(line.partition(": ")[2]), default=proof,
                )
            value = proof.partition(": ")[2]
            if len(value.split()) <= 8:
                self.spec_labels[proof.partition(": ")[0]] = self.spec_labels.get(proof.partition(": ")[0], 0) + 1
            tokens = set(re.findall(r"[^\W_]+", self._norm(value)))
            alpha = {t for t in words if not any(ch.isdigit() for ch in t)}
            stated = set(self._figures(value))
            if asked and stated and not asked & stated:
                continue
            if len(alpha) > 1 and alpha & tokens and not alpha <= tokens and not asked & stated:
                continue
            missing = len(alpha - tokens) + len(asked - stated)
            phrase = self._norm(value)
            backed = any(
                line != proof and (
                    re.search(r"(?<![^\W_])" + re.escape(phrase) + r"(?![^\W_])", self._norm(line.partition(": ")[2]))
                    or (words and words <= set(re.findall(r"[^\W_]+", self._norm(line.partition(": ")[2]))))
                )
                for line in self._item_text(key)
            )
            self._learn(key, pick.get("price"), "price_label")
            self._learn(key, pick.get("available"), "stock_label")
            variant_figures: set[float] = set()
            for own_line in self.seen.get(key, {}).get("own", []):
                own_value = own_line.partition(": ")[2]
                if self._number(own_value) is None:
                    variant_figures |= set(self._figures(own_value))
            offered = pick.get("reasons") if isinstance(pick.get("reasons"), list) else [pick.get("reason")]
            reason = None
            for part in offered:
                line = self._shown(key, part)
                if not line or line == proof or ": " not in line:
                    continue
                figures = set(self._figures(line.partition(": ")[2]))
                if figures and not figures <= variant_figures:
                    continue
                reason = line
                break
            reason = next(iter(self._stated_lines(key)), reason)
            self.claim_lines[key] = [line for line in (proof, reason) if line and ": " in line]
            exact = 0 if tokens and tokens <= set(re.findall(r"[^\W_]+", self._norm(wanted))) else 1
            info[key] = (self._kind_mismatch(key), self._brand_miss(key), self._unstated(key), self._unclaimable(key), 0 if backed else 1, exact, missing, position)
        ordered = sorted(info, key=lambda k: info[k])
        self._note("ranked", [[list(k), info[k], self.claim_lines.get(k)] for k in ordered])
        return ordered

    @staticmethod
    def _figures(text: Any) -> list[float]:
        text, found, index = str(text or ""), [], 0
        while index < len(text):
            char = text[index]
            previous = text[index - 1] if index else ""
            dimension = previous in "xX" and index >= 2 and text[index - 2].isdigit()
            if not char.isdigit() or (previous and (previous.isdigit() or previous in "._" or (previous.isalpha() and not dimension))):
                index += 1
                continue
            end = index
            while end < len(text) and (text[end].isdigit() or (text[end] in ".," and text[end + 1 : end + 2].isdigit())):
                end += 1
            tail = end
            while tail < len(text) and text[tail].isalpha():
                tail += 1
            if tail > end and tail < len(text) and text[tail].isdigit() and text[end:tail].casefold() != "x":
                while tail < len(text) and text[tail].isalnum():
                    tail += 1
                index = tail
                continue
            try:
                found.append(float(text[index:end].replace(",", "")))
            except ValueError:
                pass
            index = end
        return found

    def _brand_miss(self, key: tuple[str, ...]) -> int:
        import re

        wanted = getattr(self, "brand_words", set())
        if not wanted:
            return 0
        for line in self._stated_lines(key):
            if wanted <= set(re.findall(r"[^\W_]+", self._norm(line.partition(": ")[2]))):
                return 0
        return 1

    def _unclaimable(self, key: tuple[str, ...]) -> int:
        lines = self.claim_lines.get(key) or []
        proof = lines[0] if lines else ""
        own = set((self.seen.get(key) or {}).get("own", []))
        return 1 if not proof or proof in own or len(proof.partition(": ")[2].split()) > 8 else 0

    def _clean(self, key: tuple[str, ...], wanted: str) -> bool:
        import re

        lines = self.claim_lines.get(key) or []
        value = lines[0].partition(": ")[2] if lines and ": " in lines[0] else ""
        words = {w for w in re.findall(r"[^\W_]+", self._norm(wanted)) if len(w) > 1}
        alpha = {t for t in words if not any(ch.isdigit() for ch in t)}
        asked = set(self._figures(wanted))
        parts = [part for part in re.split(r"[;,]", value) if part.strip()]
        return bool(parts) and all(
            bool(alpha and alpha <= set(re.findall(r"[^\W_]+", self._norm(part))))
            or bool(asked and asked <= set(self._figures(part)))
            for part in parts
        )

    def _backups(self, keys: list[tuple[str, ...]], wanted: str) -> list[tuple[str, ...]]:
        import re

        if not getattr(self, "spec_labels", None):
            return []
        label = max(self.spec_labels, key=self.spec_labels.get)
        words = {w for w in re.findall(r"[^\W_]+", self._norm(wanted)) if len(w) > 1}
        alpha = {t for t in words if not any(ch.isdigit() for ch in t)}
        asked = set(self._figures(wanted))
        found: list[tuple[int, int, tuple[str, ...]]] = []
        for position, key in enumerate(keys):
            if key in self.claim_lines or self._kind_mismatch(key) or self._brand_miss(key) or self._unstated(key) or not self._ok(key):
                continue
            line = next((l for l in self._item_text(key) if l.partition(": ")[0] == label), None)
            value = line.partition(": ")[2] if line else ""
            if not value or self._listy(value):
                continue
            tokens = set(re.findall(r"[^\W_]+", self._norm(value)))
            numbers = set(self._figures(value))
            if not ((alpha and alpha <= tokens) or (asked and asked <= numbers)):
                continue
            found.append((1 if asked and numbers and not asked <= numbers else 0, position, key))
            self.claim_lines[key] = [line, *self._stated_lines(key)[:1]]
        return [key for _clash, _position, key in sorted(found)]

    def _filter_fields(self) -> set[str]:
        names = [t["function"].get("name") for t in self.tools if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        return {
            field for name in names for field, spec in self._props(name).items()
            if isinstance(spec, dict) and spec.get("type") == "string" and field not in self.ids
        }

    def _stated_lines(self, key: tuple[str, ...]) -> list[str]:
        fields = self._filter_fields()
        return [
            line for line in self._item_text(key)
            if line.partition(": ")[0] in fields and line.partition(": ")[2].strip().casefold() not in ("", "null", "none")
            and not self._listy(line.partition(": ")[2])
        ]

    def _states_spec(self, line: str, words: set[str], asked: set[float]) -> bool:
        import re

        label, _, value = line.partition(": ")
        if not value or len(value.split()) > 8 or self._listy(value):
            return False
        if label in (self.price_label, self.stock_label) or label in self.ids:
            return False
        tokens = set(re.findall(r"[^\W_]+", self._norm(value)))
        numbers = set(self._figures(value))
        alpha = {t for t in words if not any(ch.isdigit() for ch in t)}
        if asked and numbers and not asked & numbers:
            return False
        return bool((asked and asked <= numbers) or (alpha and alpha <= tokens))

    def _unstated(self, key: tuple[str, ...]) -> int:
        return self._unstated_lines(self._item_text(key))

    def _unstated_lines(self, lines: list[str]) -> int:
        fields = self._filter_fields()
        return sum(
            1 for line in lines
            if line.partition(": ")[0] in fields and line.partition(": ")[2].strip().casefold() in ("", "null", "none")
        )

    def _kind_mismatch(self, key: tuple[str, ...]) -> int:
        import re

        def stems(text: str) -> set[str]:
            return {w.rstrip("s") for w in re.findall(r"[^\W_]+", self._norm(text)) if len(w) > 2}

        kind = stems(self.kind)
        paths = []
        for line in self._item_text(key):
            value = line.partition(": ")[2].strip()
            if not value.startswith("["):
                continue
            try:
                parts = json.loads(value)
            except ValueError:
                continue
            if isinstance(parts, list) and parts and all(isinstance(p, str) for p in parts):
                paths.append([stems(p) for p in parts])
        if not kind or not paths:
            return 0
        return 0 if any(kind in levels or kind <= levels[-1] for levels in paths) else 1

    def _order_args(self, key: tuple[str, ...]) -> dict[str, Any]:
        args = self._args(self.order, key)
        schema = self._schema(self.order) or {}
        props = schema.get("properties") or {}
        container = next((n for n in schema.get("required") or [] if n not in args and n in props), None)
        if container is None:
            return args
        box = props[container] or {}
        arrays = [n for n, s in (box.get("properties") or {}).items() if (s or {}).get("type") == "array"]
        if not arrays:
            return args
        item = ((box.get("properties") or {})[arrays[0]] or {}).get("items") or {}
        names = list(item.get("required") or (item.get("properties") or {}).keys())[:2]
        if len(names) < 2:
            return args
        lines = list(self.claim_lines.get(key, []))
        for label in (self.price_label, self.stock_label):
            line = next((l for l in self._item_text(key) if label and l.partition(": ")[0] == label), None)
            if line:
                lines.append(line)
        claims, used = [], set()
        for line in lines:
            label, _, value = line.partition(": ")
            if label and value and label not in used and not self._listy(value):
                used.add(label)
                claims.append({names[0]: label, names[1]: value})
        args[container] = {arrays[0]: claims}
        return args

FAMILIES: tuple[type[Engine], ...] = (
    IntentDecomposition,
    RetrievalRecall,
    ConstraintSatisfaction,
    PreferenceReasoning,
    Ranking,
    Recovery,
    Justification,
)


def clarify(query: str) -> type[Engine]:
    rules = " ".join(Utils._split_query(query)[1].split()).casefold()
    for family in FAMILIES:
        if family.marker in rules:
            return family

    try:
        answer = Utils._llm(Prompt.CLARIFY.format(query=query), max_tokens=16).get("content") or ""
    except Exception:
        return Engine
    family = next((family for family in FAMILIES if family.__name__ in answer), Engine)
    return family


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    family = clarify(problem_data["environment"]["policy_view"]["query"])
    return family(problem_data).run()


__all__ = ["agent_main"]
