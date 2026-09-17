"""A shopping agent: it reads the request, works the tools the task offers, and buys one item."""

from __future__ import annotations

import json
import re
import threading
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
    STATE = (BUDGET, STOCK)  # what a market event can change, so its verdict is read again every turn


class Source:
    OPTIONS = "options"
    FACTS = "facts"
    TITLE = "title"
    PATH = "category_path"


ORDER_MESSAGE = "I am ordering {title} ({named}) at {price} {currency}. {because}"
STALL_TURNS = 5  # turns in a row that brought nothing new before committing to what there is
MODEL_SECONDS = 60  # one model that stops answering is dropped for the next rather than held for
SPEC_JUDGE_BATCH = 5
CONDITION_ID = "C{index}"


class Prompt:
    SHOP = """
Use the supplied shopping tools to satisfy the shopper. Continue until an observation reports done=true.

{query}
""".strip()

    PLAN = """
You plan a shopping agent's work from the tools it may call. Answer with one JSON object and nothing else.

Tools (name, description, parameters):
{tools}

Return:
{{
  "tools": {{"details": "tool showing one listing's facts and variants",
            "details_many": "tool showing several listings at once, or null",
            "details_many_param": "that tool's list parameter, or null",
            "details_many_limit": "how many ids that tool accepts per call, as a number, or null",
            "list": "tool listing items from values already in hand, or null",
            "find": "tool listing items from words, or null",
            "stock": "tool showing one exact item's current price and availability, or null",
            "cart": "tool showing what the cart holds, or null",
            "add": "tool putting one exact item in the cart",
            "remove": "tool taking an item out of the cart, or null",
            "order": "tool placing the final order",
            "message": "tool saying something to the shopper, or null"}},
  "item_params": ["the add tool's parameters that identify one exact item"]
}}
""".strip()

    CONTINUE = """
The environment has not reported done=true. Choose one of the supplied tools to continue.
""".strip()

    CLARIFY = """
Read what a shopper asks for and decide which kind of shopping work it is, judging only by the shopper's own words.
First note what the request holds, then name the kind. Reply with one JSON object and nothing else:
{{"listed_items": "the items the shopper limits the choice to, listed one by one, or none",
 "asks_why": "the shopper's words asking to be told why the pick is right, or none",
 "wanted_model": "the one particular model the shopper wants and what they say to do if that exact model is gone, or none",
 "words_to_match": "words the shopper says a listing has to match, as distinct from the kind of product they want, or none",
 "use": "what the shopper says the item is for and which qualities matter more for it, or none",
 "optional": "something the shopper would like but says is not required, or none",
 "required": "the conditions the shopper says the item must meet",
 "kind": "the name of the first kind below that fits"}}

Kinds, in order:
- StatedTradeoff: the shopper limits the choice to items they list and says how to order them.
- ExplainedBuy: the shopper asks to be told why the pick is right.
- NamedItem: the shopper wants one particular model and says what to do if that exact model is gone.
- WideSearch: the shopper asks for a listing matching given words.
- OpenPriorities: the shopper describes the use of the item and which qualities matter more for it.
- SoftPreference: besides required conditions, the shopper names something they would like but do not require.
- FirmConditions: every condition the shopper names is required.

Shopping request:
{query}
""".strip()

    INTENT_BRIEF = """
You read a shopping request and split it into firm requirements and a soft brand preference. Reply with one JSON
object and nothing else:
{{"category": "the catalog category name for the kind of product asked for",
 "spec": "the one required product spec as the shopper wrote it, with its number and unit",
 "budget": the maximum price as a number,
 "currency": "the currency code",
 "brand": "the preferred brand, or null",
 "keywords": ["every short catalog search query that could find this product, one for each distinct wording"]}}

Shopping request:
{query}
""".strip()

    INTENT_UPDATE = """
You keep a shopper's requirements current. Below are the requirements as they stand and a message the shopper has just
sent. Reply with one JSON object and nothing else, holding each of those fields as it stands after the message, and
"keywords": every short catalog search query that could find a product meeting those requirements, one for each
distinct wording.
- A value the message states for a field replaces that field's current value, whichever way it moves.
- A field the message gives no new value for keeps its current value, also when the message only points back to it.
- A question, a remark, a request to check again or a thanks changes nothing.
- The brand stays a preference, never a firm requirement.

Current requirements:
{current}

Message:
{message}
""".strip()

    INTENT_JUDGE = """
You check catalog variants against a shopper's firm requirements. Reply with one JSON object and nothing else:
{{"items": {{"<variant key>": {{"valid": true if the variant meets EVERY firm requirement: the right category, the required spec shown in this variant's options or in the product facts (the title alone is not enough), and the product itself rather than an accessory or a part made for it,
                             "preferred": true if its brand is the preferred brand,
                             "clear": true if the required spec value is stated plainly (options, facts, or a normally written title), false if it is only implied, garbled or guessed}}}},
 "keywords": ["new catalog search queries, one for each distinct wording, only when fewer than two valid products are
 listed, so the shopper has one to buy and one to fall back on"]}}
Price and stock are checked separately; ignore them.

Requirements:
{requirements}

Variants:
{variants}
""".strip()

    REQUIREMENTS = """
Report every condition the ordered item must meet that the text below states, using report_requirements.
Report brand and category as separate conditions. Copy each value verbatim from the text.
A current condition the text only points back to, without stating its value, is unchanged: leave it out.
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

    # -- the query, tool calls and tool lists --------------------------------------------------------------------------

    @staticmethod
    def _split_query(query: str) -> tuple[str, str]:
        """Shopper request and Task rules paragraph of a policy query."""
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
        """The row count the tool contract allows a search or filter to return."""
        schema = next((tool["function"] for tool in tools if tool["function"]["name"] == name), {})
        return (((schema.get("parameters") or {}).get("properties") or {}).get("k") or {}).get("maximum")

    @staticmethod
    def _tool_names(engine: Engine) -> set[str]:
        """The names of the tools this session offers."""
        return {tool["function"]["name"] for tool in engine.tools}

    @staticmethod
    def _value(requirements: list[Requirement], kind: str) -> str | None:
        """The value of the first requirement of this kind that states one."""
        return next((r.value for r in requirements if r.kind == kind and r.value), None)

    @staticmethod
    def _unquoted(value: str) -> str:
        """value without the quotation marks a copied condition can carry."""
        return value.strip().strip("\"'“”")

    @staticmethod
    def _ref(engine: Engine, *sources: Any) -> tuple[str, ...]:
        """The item these observation parts name, in the order the session's cart add names its parts."""
        found = []
        for name in engine.ids:
            value = next((part[name] for part in sources if isinstance(part, dict) and part.get(name) is not None), "")
            found.append(str(value))
        return tuple(found)

    @staticmethod
    def _one(engine: Engine, source: Any) -> str:
        """The first of the parts that name an item, as this source writes it."""
        name = engine.ids[0] if engine.ids else ""
        return str((source or {}).get(name, "")) if isinstance(source, dict) else ""

    @staticmethod
    def _named(engine: Engine, source: Any) -> bool:
        """Whether this row names every part of an item."""
        return isinstance(source, dict) and all(source.get(name) is not None for name in engine.ids)

    @staticmethod
    def _ids(engine: Engine, ref: tuple[str, ...]) -> dict[str, str]:
        """One item named the way this session's own tools name it."""
        return dict(zip(engine.ids, (str(part) for part in ref)))

    @staticmethod
    def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

    @staticmethod
    def _limit_calls(calls: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """The first limit calls, for a turn that may send no more than limit of them."""
        return calls[:limit]

    @staticmethod
    def _lookup_tools(engine: Engine) -> list[dict[str, Any]]:
        """The tools that only read, for the turns the model is asked what to look at next."""
        writes = {engine.add, engine.remove, engine.order}
        return [tool for tool in engine.tools if tool["function"]["name"] not in writes]

    # -- text and numbers ----------------------------------------------------------------------------------------------

    @staticmethod
    def _grounded(text: str, source: str) -> bool:
        """True when text appears in source, or every word of it does ("6.9 inch" vs "6.9-inch")."""

        wanted = Utils._words(text)
        if not wanted:
            return False
        if " ".join(str(text).casefold().split()) in " ".join(str(source).casefold().split()):
            return True
        return wanted <= Utils._words(source)

    @staticmethod
    def _grounded_limit(text: str, source: str) -> bool:
        """True when the figure a limit names, and any currency beside it, are written in source."""
        numbers = set(Utils._number_tokens(text))
        if not numbers:
            return False
        words = Utils._words(source)
        return numbers <= set(Utils._number_tokens(source)) and {
            word for word in Utils._words(text) if len(word) == 3 and word.isalpha()} <= words

    @staticmethod
    def _words(value: Any, plural: bool = False) -> set[str]:
        """The case-folded letter and digit words of value; with plural, a trailing "s" dropped from words over three letters."""
        tokens = "".join(char if char.isalnum() else " " for char in str(value or "").casefold()).split()
        return {t[:-1] if plural and len(t) > 3 and t.endswith("s") else t for t in tokens}

    @staticmethod
    def _words_match(wanted: str, available: Any, plural: bool = False) -> str:
        """Every word of wanted appears in available; a plural "s" is ignored when plural is set."""

        wanted_words = Utils._words(wanted, plural)
        if not wanted_words:
            return Verdict.UNVERIFIED
        return Verdict.MET if wanted_words <= Utils._words(available, plural) else Verdict.UNMET

    @staticmethod
    def _contains_words(text: Any, quote: Any) -> bool:
        """The quote's words appear in text in the same order, ignoring case, symbols and character width."""

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
        """A category label as the catalog compares labels: ASCII letters and digits in lower case, single-spaced."""
        return " ".join("".join(char if char.isascii() and char.isalnum() else " " for char in str(text or "").casefold()).split())

    @staticmethod
    def _number_tokens(text: str) -> list[Decimal]:
        """Decimal numbers written in text; a comma between digits is a thousands separator ("11,699.10")."""
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
        """(number, unit) for each standalone number in text written before a unit."""
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
        """Units spelled alike, or one a longer form of the other that starts with its first two or more letters."""
        short, long = sorted((unit, other), key=len)
        return unit == other or (len(short) > 1 and long.startswith(short))

    @staticmethod
    def _states_spec(text: Any, spec: str) -> bool:
        """Whether text states spec: every number the spec writes with a unit appears in text before the same unit."""
        wanted = Utils._number_units(spec)
        if not wanted:
            return Utils._words_match(spec, text) == Verdict.MET
        found = Utils._number_units(text)
        return all(any(number == got and Utils._same_unit(unit, have) for got, have in found) for number, unit in wanted)

    # -- requirements --------------------------------------------------------------------------------------------------

    @staticmethod
    def _budget_terms(value: str) -> tuple[str | None, str | None, str | None]:
        """The limit a budget phrase states, and the problem that rejects it; the smallest of several numbers is the
        limit, and a lone ISO code is taken with it."""
        numbers = Utils._number_tokens(value)
        if not numbers:
            return None, None, "no number in the budget phrase"
        codes = {word for word in value.replace(",", " ").split()
                 if len(word) == 3 and word.isalpha() and word.isupper()}
        return f"{min(numbers)}", codes.pop() if len(codes) == 1 else None, None

    @staticmethod
    def _extract_requirements(text: str, current: list[Requirement], turn: int, feedback: str = "") -> list[Requirement]:
        """Conditions the shopper request (turn 0) or a shopper message (turn N) states, kept only when grounded in it."""
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
        return requirements

    @staticmethod
    def _merge_requirements(requirements: list[Requirement], updates: list[Requirement]) -> list[Requirement]:
        """Current conditions after updates, each keeping one id; an update older than the condition it would replace is
        dropped."""
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

    # -- candidate verdicts --------------------------------------------------------------------------------------------

    @staticmethod
    def _meets_all(verdicts: dict[str, str] | None) -> bool:
        """True when a variant was judged and every verdict it carries is met."""
        return bool(verdicts) and all(verdict == Verdict.MET for verdict in verdicts.values())

    @staticmethod
    def _check_variant(
        requirements: list[Requirement],
        product: dict[str, Any],
        variant: dict[str, Any],
        judged: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Per requirement: Verdict.MET, Verdict.UNMET, or Verdict.UNVERIFIED when it could not be checked; a hidden
        requirement is not judged."""

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
        """Price check: the requirement's operator on decimal amounts, and the currency codes when both name one."""
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
    def _settled_specs(engine: Engine, previous: list[Requirement]) -> dict[tuple[str, str], dict[str, str]]:
        """Spec verdicts already read off a listing that the current requirements leave standing; a requirement the
        shopper altered is read again."""
        keys = {r.check_key() for r in previous}
        kept = {r.verdict_key() for r in engine.requirements if r.kind == Kind.SPEC and r.check_key() in keys}
        return {ref: {key: verdict for key, verdict in cell.items()
                      if key in kept and verdict in Verdict.ALL}
                for ref, cell in engine.candidates.items()}

    @staticmethod
    def _review_candidates(
        engine: Engine, requirements: list[Requirement], result: dict[str, Any], labelled: bool = True,
        listed: set[str] | None = None, settled: dict[tuple[str, str], dict[str, str]] | None = None,
    ) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], int]]:
        """Verdicts for the variants this turn showed, and how many specs each was read off its own options for; settled
        verdicts are not read again."""
        products = list({Utils._one(engine, p): p for p in Utils._observed_products(result)}.values())
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
                if Utils._one(engine, product) not in (listed or set()):
                    continue
                for variant in product.get("variants") or []:
                    judged.setdefault(Utils._ref(engine, variant, product), {}).update(
                        {r.verdict_key(): Verdict.MET for r in categories}
                    )
        to_judge = [
            product for product in passing
            if any(Verdict.UNMET not in Utils._check_variant(
                categories, product, variant,
                judged.get(Utils._ref(engine, variant, product))).values()
                for variant in product.get("variants") or [{}])
        ]
        jobs = [lambda batch=batch: Utils._judge_specs(engine, requirements, batch, settled)
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
                ref = Utils._ref(engine, variant, product)
                verdicts = (Utils._check_variant(requirements, product, variant, judged.get(ref))
                            if requirements else {"requirements": Verdict.UNVERIFIED})
                reviewed[ref] = verdicts
        return reviewed, stated

    @staticmethod
    def _judge_specs(
        engine: Engine, requirements: list[Requirement], products: list[dict[str, Any]],
        settled: dict[tuple[str, str], dict[str, str]] | None = None,
    ) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], int]]:
        """Spec verdicts per (product_id, sku), and how many were read off the variant's own options; a variant already
        read for a requirement is left out."""
        specs = [r for r in requirements if r.kind == Kind.SPEC and r.value is not None]
        known = lambda ref, r: (settled or {}).get(ref, {}).get(r.verdict_key()) in Verdict.ALL
        variants = {
            f"{product.get('product_id')}::{variant.get('sku')}": (product, variant)
            for product in products for variant in product.get("variants") or []
            if not all(known(Utils._ref(engine, variant, product), r) for r in specs)
        }
        verdicts = {
            Utils._ref(engine, variant, product): {r.verdict_key(): Verdict.UNVERIFIED for r in specs}
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
            ref = Utils._ref(engine, variant, product)
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
        """The verdict from the reported lines whose quote code finds in the named field, and the side that decided it."""
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
        """A reported line's verdict after the number check, or None when it states no value; a quote with no number
        leaves the model's verdict standing."""
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
        """Whether the catalog's own name for the required category is known; the model picks it from the category-path
        names observed so far."""
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
        engine.named_from = (label, labels)  # the catalog's own name needs no naming
        engine.requirements = [replace(r, value=label) if r is requirement else r for r in engine.requirements]
        return True

    @staticmethod
    def _category_context(engine: Engine) -> tuple[bool, set[str]]:
        """Whether the category requirement is a label on some observed category path, and the products filter calls for
        it returned."""
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
                if call["function"]["name"] == engine.lister and Utils._label(Utils._arguments(call).get("category")) == wanted:
                    rows = (Utils._body(call_result) or {}).get("results") or []
                    listed.update(Utils._one(engine, row) for row in rows)
        return labelled, listed

    # -- what the dialogue observed ------------------------------------------------------------------------------------

    @staticmethod
    def _body(call_result: dict[str, Any]) -> dict[str, Any] | None:
        """The observation body one call result carries, or None when it carries none."""
        body = (call_result.get("observation") or {}).get("observation")
        return body if isinstance(body, dict) else None

    @staticmethod
    def _products(body: dict[str, Any]) -> list[dict[str, Any]]:
        """The products with variants an observation body holds: those compared, or the one product viewed."""
        return body.get("compared") or ([body] if body.get("variants") else [])

    @staticmethod
    def _observations(engine: Engine) -> list[tuple[int, dict[str, Any]]]:
        """(dialogue index, observation body) for every environment result, in order."""
        return [
            (index, body)
            for index, entry in enumerate(engine.dialogue)
            for call_result in (entry.get("environment_result") or {}).get("calls") or []
            if (body := Utils._body(call_result)) is not None
        ]

    @staticmethod
    def _observed_products(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Products with variants returned by view or compare calls in one environment result."""
        products = []
        for call in result.get("calls") or []:
            body = Utils._body(call)
            if body is not None:
                products += Utils._products(body)
        return products

    @staticmethod
    def _observed_title(engine: Engine, observations: list[tuple[int, dict[str, Any]]], first: str) -> str | None:
        """The latest title a search row, compared product or viewed product showed for product_id."""
        titles = [
            row["title"] for _, body in observations
            for row in [*(body.get("results") or []), *(body.get("compared") or []), body]
            if isinstance(row, dict) and Utils._one(engine, row) == first and row.get("title")
        ]
        return titles[-1] if titles else None

    @staticmethod
    def _product_rows(engine: Engine) -> dict[str, dict[str, Any]]:
        """Per product, the first search or filter row the dialogue showed for it."""
        rows: dict[str, dict[str, Any]] = {}
        for _, body in Utils._observations(engine):
            for row in body.get("results") or []:
                rows.setdefault(Utils._one(engine, row), row)
        return rows

    @staticmethod
    def _variants(engine: Engine) -> dict[tuple[str, str], dict[str, Any]]:
        """Per variant seen by compare or view, its own fields under the fields of its product."""
        variants: dict[tuple[str, str], dict[str, Any]] = {}
        for _, body in Utils._observations(engine):
            for product in Utils._products(body):
                product_fields = {f: product.get(f) for f in (*engine.ids, "title", "brand", "category_path", "facts")}
                for variant in product.get("variants") or []:
                    variants[Utils._ref(engine, variant, product)] = {**product_fields, **variant}
        return variants

    @staticmethod
    def _variant_states(engine: Engine) -> dict[tuple[str, str], dict[str, Any]]:
        """Latest price, currency and stock observed for each variant; a later observation replaces an earlier one."""
        states: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in engine.dialogue:
            for call_result in (entry.get("environment_result") or {}).get("calls") or []:
                body = Utils._body(call_result)
                if body is None:
                    continue
                rows = [*(body.get("results") or []), *(body.get("cart") or []), body.get("added"), body]
                rows += [
                    {**variant, **{name: product.get(name) for name in engine.ids if name not in variant}}
                    for product in Utils._observed_products({"calls": [call_result]})
                    for variant in product.get("variants") or []
                ]
                for row in rows:
                    if Utils._named(engine, row) and "in_stock" in row:
                        states[Utils._ref(engine, row)] = {
                            "price": row.get("current_price", row.get("price")),
                            "currency": row.get("currency"),
                            "in_stock": row.get("in_stock"),
                        }
        return states

    @staticmethod
    def _refresh_candidate_states(engine: Engine) -> None:
        """Re-check the budget and stock verdicts of reviewed variants against their latest observed state."""
        requirements = [r for r in engine.requirements if r.kind in Kind.STATE]
        suffixes = tuple(f"[{kind}]" for kind in Kind.STATE)
        states = Utils._variant_states(engine)
        for ref, verdicts in engine.candidates.items():
            if ref not in states:
                continue
            for key in [key for key in verdicts if key.endswith(suffixes)]:
                verdicts.pop(key)  # a restated condition keeps its verdict under the earlier name otherwise
            for key, verdict in Utils._check_variant(requirements, {}, states[ref]).items():
                verdicts[key] = verdict

    @staticmethod
    def _cart_state(engine: Engine) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """The cart as its latest observation shows it, and every variant a cart add accepted, in order."""
        cart: list[tuple[str, str]] = []
        added: list[tuple[str, str]] = []
        for _, body in Utils._observations(engine):
            if isinstance(body.get("added"), dict):
                added.append(Utils._ref(engine, body["added"]))
            if isinstance(body.get("cart"), list):
                cart = [Utils._ref(engine, item) for item in body["cart"] if isinstance(item, dict)]
        return cart, added

    # -- the calls code sends, and the model and environment -----------------------------------------------------------

    @staticmethod
    def _search_calls(
        engine: Engine, requirements: list[Requirement], call_id: str, query: str | None = None
    ) -> list[dict[str, Any]]:
        """filter and search tool calls built from the requirements; a given query replaces the one built from them."""
        schemas = {
            tool["function"]["name"]: (tool["function"].get("parameters") or {}).get("properties") or {}
            for tool in engine.tools if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }

        budget = next((r for r in requirements if r.kind == Kind.BUDGET and r.amount), None)
        max_price = float(budget.amount) if budget else None
        in_stock = True if any(r.kind == Kind.STOCK for r in requirements) else None
        brand, category = Utils._value(requirements, Kind.BRAND), Utils._value(requirements, Kind.CATEGORY)
        phrase = Utils._value(requirements, Kind.PHRASE)
        specs = [r.value for r in requirements if r.kind == Kind.SPEC and r.value is not None]
        wanted = [
            ("list", engine.lister, {"category": category, "brand": brand, "max_price": max_price}),
            ("find", engine.finder, {"query": query or (Utils._unquoted(phrase) if phrase
                       else " ".join(v for v in (brand, category, *specs) if v) or None)}),
        ]
        calls = []
        for job, name, arguments in wanted:
            properties = schemas.get(name) if name else None
            if properties is None or all(value is None for value in arguments.values()):
                continue
            arguments = {**arguments, "max_price": max_price, "in_stock": in_stock,
                         "k": (properties.get("k") or {}).get("maximum")}
            arguments = {key: value for key, value in arguments.items()
                         if value is not None and key in properties}
            if not arguments:
                continue
            calls.append({"id": f"{call_id}-{job}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(arguments)}})
        return calls

    @staticmethod
    def _new_search_calls(engine: Engine, turn: int, query: str | None = None) -> list[dict[str, Any]]:
        """Search calls whose tool and arguments were not sent earlier in this dialogue; given a query, only the search
        using it."""
        sent = {
            (call["function"]["name"], json.dumps(Utils._arguments(call), sort_keys=True))
            for message in engine.messages if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        }
        call_id = f"{engine.problem_id}-turn-{turn}" + ("-query" if query else "")
        calls = [
            call for call in Utils._search_calls(engine, engine.requirements, call_id, query)
            if (query is None or Utils._arguments(call).get("query") == query)
            and (call["function"]["name"], json.dumps(Utils._arguments(call), sort_keys=True)) not in sent
        ]
        return calls

    @staticmethod
    def _compare_calls(engine: Engine, turn: int) -> list[dict[str, Any]]:
        """compare calls, as many products each as that tool takes, for rows not viewed or compared yet whose row fails no requirement
        in engine.row_checked."""
        if not (engine.many and engine.many_param):
            return []
        rows, observed = [], set()
        for entry in engine.dialogue:
            result = entry.get("environment_result")
            if not result:
                continue
            for call, call_result in zip(entry["tool_calls"], result.get("calls") or []):
                observation = call_result.get("observation") or {}
                if call["function"]["name"] in {engine.lister, engine.finder}:
                    rows += (Utils._body(call_result) or {}).get("results") or []
                elif call["function"]["name"] == engine.many and not observation.get("error"):
                    product_ids = Utils._arguments(call).get(engine.many_param)
                    if isinstance(product_ids, list):
                        observed.update(str(product_id) for product_id in product_ids)
            observed.update(Utils._one(engine, product) for product in Utils._observed_products(result))
        row_requirements = [r for r in engine.requirements if r.kind in engine.row_checked]
        product_ids = list(dict.fromkeys(
            Utils._one(engine, row) for row in rows
            if Utils._one(engine, row) not in observed
            and Verdict.UNMET not in Utils._check_variant(row_requirements, row, row).values()
        ))
        return [
            Utils._tool_call(engine.many, {engine.many_param: product_ids[start:start + engine.many_limit]},
                             f"{engine.problem_id}-turn-{turn}-details-many-{index}")
            for index, start in enumerate(range(0, len(product_ids), engine.many_limit), start=1)
        ]

    @staticmethod
    def _add_calls(
        engine: Engine, turn: int, cart: list[tuple[str, str]] | None = None, new_product_only: bool = False
    ) -> list[dict[str, Any]]:
        """add_to_cart for a variant meeting every requirement while the cart is empty: a product not committed to
        before, then more specs read off its own options, then cheapest."""
        observed_cart, added = Utils._cart_state(engine)
        if (observed_cart if cart is None else cart) or not engine.add:
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

        ref = min(met, key=rank)
        return [Utils._tool_call(engine.add, Utils._ids(engine, ref), f"{engine.problem_id}-turn-{turn}-add")]

    @staticmethod
    def _choice_message(engine: Engine, ref: tuple[str, str], template: str) -> str:
        """One variant described for the shopper; the line saying every condition holds is written only where the
        verdicts say it does."""
        state = Utils._variant_states(engine).get(ref) or {}
        candidates = getattr(engine, "candidates", None)
        short = [key for key, verdict in ((candidates or {}).get(ref) or {}).items() if verdict != Verdict.MET]
        because = ("" if candidates is None else "It meets every condition you gave." if not short else
                   "It is the nearest I could find to what you asked; it does not settle " + ", ".join(short) + ".")
        return template.format(title=Utils._observed_title(engine, Utils._observations(engine), ref[0]) or ref[0],
                               named=", ".join(f"{name} {part}" for name, part in
                                               zip(engine.ids[1:], ref[1:])) or ref[0],
                               price=state.get("price"), currency=state.get("currency") or "",
                               because=because).strip()

    @staticmethod
    def _speech(engine: Engine, text: str) -> dict[str, str] | None:
        texts = [name for name, spec in engine._props(engine.speak).items() if isinstance(spec, dict) and spec.get("type") == "string"]
        required = [name for name in (engine._schema(engine.speak) or {}).get("required") or [] if name in texts]
        field = required[0] if len(required) == 1 else texts[0] if len(texts) == 1 else None
        return {field: text} if field else None

    @staticmethod
    def _unanswered_message(engine: Engine) -> str:
        """The shopper's last message that no message of ours has answered yet; empty when the shopper is owed nothing."""
        owed = ""
        for entry in engine.dialogue:
            if any(call["function"]["name"] == engine.speak for call in entry.get("tool_calls") or []):
                owed = ""
            content = ((entry.get("environment_result") or {}).get("user_message") or {}).get("content") or ""
            if content.strip():
                owed = content
        return owed

    @staticmethod
    def _order_calls(engine: Engine, ref: tuple[str, str], call_id: str) -> list[dict[str, Any]]:
        """place_test_order for ref, alone: nothing follows the order, and a cart write cannot share its turn."""
        if not engine.order:
            return []
        return [Utils._tool_call(engine.order, Utils._ids(engine, ref), f"{call_id}-order")]

    @staticmethod
    def _answer_before_order(engine: Engine, turn: int, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The shopper's answer, put before the order; the shopper hears nothing after it, so one message answers every
        question they asked."""
        order = next((index for index, call in enumerate(tool_calls) if call["function"]["name"] == engine.order), None)
        if (order is None or len(tool_calls[order:]) >= engine.max_calls
                or not engine.speak
                or any(call["function"]["name"] == engine.speak for call in tool_calls[:order])
                or not Utils._unanswered_message(engine)):
            return tool_calls
        arguments = Utils._arguments(tool_calls[order])
        speech = Utils._speech(engine, Utils._choice_message(engine, Utils._ref(engine, arguments), ORDER_MESSAGE))
        if speech is None:
            return tool_calls
        reply = Utils._tool_call(engine.speak, speech, f"{engine.problem_id}-turn-{turn}-message")
        room = max(0, engine.max_calls - 1 - len(tool_calls[order:]))
        return [*Utils._limit_calls(tool_calls[:order], room), reply, *tool_calls[order:]]

    @staticmethod
    def _track_progress(engine: Engine) -> None:
        """Count the turns in a row that brought nothing new."""
        seen = {ref[0] for ref in Utils._variant_states(engine)}
        cart, _ = Utils._cart_state(engine)
        requirements = tuple(sorted(r.check_key() for r in engine.requirements))
        met = sum(1 for verdicts in engine.candidates.values() if Utils._meets_all(verdicts))
        signature = (len(seen), tuple(cart), requirements, met)
        engine.stalled = getattr(engine, "stalled", 0) + 1 if signature == getattr(engine, "progress", None) else 0
        engine.progress = signature

    @staticmethod
    def _checked_since_add(engine: Engine, ref: tuple[str, str]) -> bool:
        """Whether an observation of ref arrived after it was added, so the order follows a turn that read."""
        bodies = [body for _, body in Utils._observations(engine)]  # call order, since one turn holds several calls
        added_at = max((place for place, body in enumerate(bodies)
                        if isinstance(added := body.get("added"), dict)
                        and Utils._ref(engine, added) == ref), default=-1)
        return any(place > added_at and Utils._ref(engine, body) == ref
                   for place, body in enumerate(bodies))

    @staticmethod
    def _recover_calls(engine: Engine, turn: int) -> list[dict[str, Any]]:
        """For the variant in the cart: inspect_stock then the order while it meets every requirement, else remove it and
        work toward a replacement."""
        cart, _ = Utils._cart_state(engine)
        if not cart or not engine.requirements or not (engine.stock and engine.remove):
            return []
        ref = cart[-1]
        arguments = Utils._ids(engine, ref)
        call_id = f"{engine.problem_id}-turn-{turn}"
        verdicts = engine.candidates.get(ref) or {}
        if Utils._meets_all(verdicts):
            if Utils._checked_since_add(engine, ref):
                return Utils._order_calls(engine, ref, call_id)
            return [Utils._tool_call(engine.stock, arguments, f"{call_id}-stock")]
        calls = [Utils._tool_call(engine.remove, arguments, f"{call_id}-remove")]
        return Utils._limit_calls(calls + Utils._replacement_calls(engine, turn, cart=[]), engine.max_calls)

    @staticmethod
    def _replacement_calls(engine: Engine, turn: int, cart: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
        """Calls toward the next variant to commit to: a judged candidate from an uncommitted product, then compare of
        unjudged rows, then new retrieval, then an add from a product committed to before."""
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
        """A search for the observed title of the last variant a cart add committed to, unless it was already sent."""
        _, added = Utils._cart_state(engine)
        if not added:
            return []
        title = Utils._observed_title(engine, Utils._observations(engine), added[-1][0])
        return Utils._new_search_calls(engine, turn, title) if title else []

    @staticmethod
    def _choose_calls(engine: Engine, tools: list[dict[str, Any]] | None = None) -> tuple[str, list[dict[str, Any]]]:
        """The model's content and tool calls for one turn, asking once more when it returns none; tools, when given,
        replaces engine.tools."""
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
        """Record the calls, run them in the environment, then record the results and any shopper message."""
        tool_calls = [call for call in tool_calls if call["function"]["name"]]
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
        """The arguments of the model's call to the named tool, or None when it answered without one."""
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
        """One model call answered with a JSON object; {} when nothing readable came back."""
        try:
            text = Utils._llm(prompt).get("content") or ""
        except RuntimeError:
            return {}
        start, end = text.find("{"), text.rfind("}")
        if not isinstance(text, str) or start < 0 or end <= start:
            return {}
        chunk = text[start : end + 1]
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                value = json.loads(candidate)
            except ValueError:
                continue
            return value if isinstance(value, dict) else {}
        return {}

    @staticmethod
    def _bounded(call: Any, seconds: int) -> Any:
        """What call returns within seconds, or None when it raised or was still running."""
        box: dict[str, Any] = {}

        def attempt() -> None:
            try:
                box["value"] = call()
            except Exception:
                box["value"] = None

        worker = threading.Thread(target=attempt, daemon=True)
        worker.start()
        worker.join(seconds)
        return None if worker.is_alive() else box.get("value")

    @staticmethod
    def _llm(prompt: str | list[dict[str, Any]], **params: Any) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        for model in Utils.MODELS:
            inference = Utils._bounded(lambda model=model: _proxy.post(
                "/inference/chat/completions",
                json_data={**params, "model": model, "messages": messages, "temperature": 0},
            ), MODEL_SECONDS)
            try:
                message = inference["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                continue
            if message.get("content") or message.get("tool_calls"):
                return message
        raise RuntimeError("inference request failed on every model")


@dataclass
class Requirement:
    """One condition the ordered variant must meet, from the query (turn 0) or a shopper message."""

    kind: str
    value: str | None
    hidden: bool
    turn: int = 0
    amount: str | None = None
    currency: str | None = None
    id: str | None = None
    replaces: str | None = None

    def verdict_key(self) -> str:
        """The key a variant's verdict for this condition is stored under; the id stays when the condition is restated."""
        return f"{self.id} [{self.kind}]"

    def check_key(self) -> tuple[Any, ...]:
        """The fields the variant check reads, so a restated condition keys the same as the original."""

        def text(value: str | None) -> str | None:
            return " ".join(value.casefold().split()) if value is not None else None

        if self.kind == Kind.BUDGET:
            return self.kind, Decimal(self.amount) if self.amount else None, self.currency
        if self.kind == Kind.STOCK:
            return (self.kind,)
        return self.kind, text(self.value)


class Engine:
    """Shared main loop and the tools every strategy works through. Each strategy overrides run()."""

    # kinds checked on search rows; a row is one exact item, so only values every item of a listing shares
    row_checked: tuple[str, ...] = (Kind.BRAND,)
    READ: str | None = None

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
        self.details = self.many = self.many_param = self.stock = self.cart = None
        self.lister = self.finder = self.speak = self.add = self.remove = self.order = None
        self.ids: list[str] = []
        self.group: list[str] = []
        self.many_limit = 1

    def run(self) -> list[dict[str, Any]]:
        """Shop to the plan this strategy reads, or, without one to read, let the model choose each action."""
        if self.READ:
            self.turn, self.done = 0, False
            raw = self._read_plan(self.READ.format(tools=self._tools_text(), query=self.query,
                                                   limit=self.max_calls))
            if raw:
                try:
                    self._shop(raw)
                except Exception:
                    pass
                return self.dialogue
        return self._choose_each_turn()

    def _choose_each_turn(self) -> list[dict[str, Any]]:
        """Every action chosen by the model from the tools this session offers."""
        for turn in range(1, self.max_steps + 1):
            content, tool_calls = Utils._choose_calls(self)
            result = Utils._send_turn(self, turn, content, tool_calls)
            if any((call.get("observation") or {}).get("done") for call in result["calls"]):
                break
        return self.dialogue

    MAX_LISTINGS = 15
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
        """One small call over search rows: the listings that are really the kind of product wanted, best first."""
        if len(groups) <= 1:
            return list(groups)
        labels = {f"g{i + 1}": g for i, g in enumerate(groups)}
        rows = {label: texts.get(group, "") for label, group in labels.items()}
        answer = self._ask(self.SCREEN.format(
            query=self.query, kind=self.kind, focus=focus, rows=json.dumps(rows, ensure_ascii=False), limit=limit or 2 * self.MAX_LISTINGS,
        ))
        picked = list(dict.fromkeys(k for x in answer.get("open") or [] if (k := self._label_key(labels, x)) is not None))
        return list(dict.fromkeys(picked))
    def _more_keys(self, keys: list[tuple[str, ...]], dropped: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
        """Items for a replacement round: open the reserved listings first, then everything not rejected."""
        reserve = getattr(self, "reserve", [])
        if reserve and self.turn < self.max_steps - 3:
            self.reserve = []
            self._read_all(self._detail_calls(reserve), turns=1)
            keys.extend(k for k in self.seen if self._group_of(k) in set(reserve) and k not in keys)
        return [k for k in keys if k not in dropped]
    def _row_text(self, outer: list[str], own: list[str]) -> str:
        return " | ".join(line for line in outer + own if line.partition(": ")[0] not in self.ids)
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
        """One reading of the request, while the turns this task allows still leave room for one."""
        self.asks = getattr(self, "asks", 0) + 1
        return Utils._json(prompt) if self.asks <= self.max_steps else {}
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
        self.finder = self._tool(names.get("find"))
        self.lister = self._tool(names.get("list"))
        self.cart = self._tool(names.get("cart"))
        self.speak = self._tool(names.get("message"))
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
        """Read the task once more when the first answer is missing or unusable."""
        for _attempt in range(2):
            raw = self._ask(prompt)
            if raw and self._plan_tools(raw):
                return raw
        return {}
    def _learn_state(self, observation: dict[str, Any]) -> None:
        """From a live-state read of one item: its single yes/no value is availability, its single number is price."""
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
        """One turn with at most max_calls calls; returns each call's observation (empty dicts when nothing came back)."""
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
        said = ((result.get("user_message") or {}).get("content") or "").strip()
        if said:
            limits = [r.amount for r in Utils._extract_requirements(said, [], turn) if r.kind == Kind.BUDGET and r.amount]
            if limits:
                self.budget = float(limits[-1])
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
        """Every exact item in a result: (its id values, listing lines around it, its own lines)."""
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
                    out.append(f"{label}: {shown}")

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
    def _read_all(self, calls: list[tuple[str, dict[str, Any]]], turns: int, kept: int = 2) -> None:
        for start in range(0, len(calls), self.max_calls):
            if turns <= 0 or self.done or self.turn >= self.max_steps - kept:
                return
            turns -= 1
            chunk = calls[start : start + self.max_calls]
            for (name, _args), observation in zip(chunk, self._send(chunk)):
                if name not in (self.add, self.remove, self.order):
                    self._record(observation)
                    if name == self.stock:
                        self._learn_state(observation)
    def _item_text(self, key: tuple[str, ...]) -> list[str]:
        entry = self.seen.get(key) or {}
        return list(entry.get("own", [])) + list(entry.get("outer", []))
    def _shown(self, key: tuple[str, ...], part: Any) -> str | None:
        """The item's own line holding a model quote (None when the quote is not in the item's lines)."""
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
    def _evidence(self, keys: list[tuple[str, ...]]) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
        labels: dict[str, tuple[str, ...]] = {}
        listings: dict[tuple[str, ...], dict[str, Any]] = {}
        for key in keys:
            entry = self.seen.get(key)
            if not entry:
                continue
            listing = listings.setdefault(self._group_of(key), {"listing": list(entry["outer"]), "items": {}})
            label = f"i{len(labels) + 1}"
            labels[label] = key
            listing["items"][label] = list(entry["own"])
        return labels, list(listings.values())
    def _learn_row(self, lines: list[str]) -> None:
        """Price and availability labels read off one search row: its only plain number and its only true/false value."""
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
        """1 when every price seen for a listing's rows is known and above the price limit."""
        known = [p for p in prices if p is not None]
        return 1 if self.budget is not None and known and len(known) == len(prices) and min(known) > self.budget + 1e-9 else 0
    @staticmethod
    def _listy(value: str) -> bool:
        """Whether a line value is a list or object written as JSON (a path of categories, say), not plain text."""
        text = value.strip()
        if not text or text[0] not in "[{" or text[-1] not in "]}":
            return False
        try:
            return isinstance(json.loads(text), (list, dict))
        except ValueError:
            return False
    def _label_key(self, labels: dict[str, tuple[str, ...]], name: Any) -> tuple[str, ...] | None:
        """The item a model answer names: its label, or else the one item whose own ids all appear in that name."""
        import re

        if not isinstance(name, str) or not name.strip():
            return None
        if name.strip() in labels:
            return labels[name.strip()]
        pieces = set(re.split(r"[^\w\-]+", name))
        named = [key for key in dict.fromkeys(labels.values()) if [p for p in key if p] and all(p in pieces for p in key if p)]
        return named[0] if len(named) == 1 else None
    def _removed(self, calls: list[tuple[str, dict[str, Any]]], observations: list[dict[str, Any]]) -> bool:
        return (len(observations) > 1 and calls[1][0] == self.remove and bool(observations[1])
                and not observations[1].get("error"))
    def _fallback_commit(
        self, ordered: list[tuple[str, ...]], dropped: set[tuple[str, ...]], committed: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """Last resort when the rounds above ran out: add the next judged item that is in stock and within the price
        limit, read it, and return it."""
        pool = [k for k in ordered if k not in dropped and k != committed]
        for _attempt in range(2):
            choice = next((k for k in pool if self._ok(k)), None)
            if choice is None or self.done or self.turn >= self.max_steps - 1:
                break
            calls = [(self.add, self._args(self.add, choice))]
            if committed is not None and committed != choice and self.remove:
                calls.append((self.remove, self._args(self.remove, committed)))
            if self.stock and self._fits(self.stock, self._args(self.stock, choice)):
                calls.append((self.stock, self._args(self.stock, choice)))
            if getattr(self, "added", False):
                calls += self._detail_calls([self._group_of(choice)])
            observations = self._send(calls[: self.max_calls])
            for (name, _args), observation in zip(calls, observations):
                if name not in (self.add, self.remove):
                    self._record(observation)
                    if name == self.stock:
                        self._learn_state(observation)
            pool.remove(choice)
            if self._removed(calls, observations) and (observations[0] or {}).get("error"):
                committed = None
            if observations and not (observations[0] or {}).get("error"):
                self.added = True
                self._read_all(calls[self.max_calls :], turns=len(calls), kept=1)
                committed = choice
                if self._ok(choice):
                    break
        return committed
    def _commit_and_order(self, ordered: list[tuple[str, ...]], order_args: Any, rejudge: Any = None) -> None:
        """Add the best judged item and read it with its backups in one turn, then order; the best other item seen when
        nothing judged is orderable."""
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
            spare: list[tuple[str, dict[str, Any]]] = []
            if self.stock and self._fits(self.stock, self._args(self.stock, choice)):
                calls.append((self.stock, self._args(self.stock, choice)))
                if getattr(self, "added", False):
                    calls += self._detail_calls([self._group_of(choice)])
                spare = [(self.stock, self._args(self.stock, k)) for k in backups]
            else:
                calls += self._detail_calls([self._group_of(choice)])
                spare = self._detail_calls([self._group_of(k) for k in backups if self._group_of(k) != self._group_of(choice)])
            observations = self._send([*calls, *spare])
            added = observations[0] if observations else {}
            for (name, _args), observation in zip([*calls, *spare], observations):
                if name not in (self.add, self.remove, self.order):
                    self._record(observation)
                    if name == self.stock:
                        self._learn_state(observation)
            if not observations or added.get("error"):
                committed = None if self._removed(calls, observations) else committed
                dropped.add(choice)
                continue
            self.added = True
            self._read_all(calls[self.max_calls :], turns=len(calls), kept=1)
            committed = choice
            if self._ok(choice):
                break
            dropped.add(choice)
        if (committed is None or not self._ok(committed)) and not self.done and self.turn < self.max_steps - 1:
            committed = self._fallback_commit(ordered, dropped, committed)
        if committed is not None and self._ok(committed) and not self.done and self.turn < self.max_steps:
            calls = [(self.order, order_args(committed))]
            owed = bool(self.speak) and bool(Utils._unanswered_message(self))
            speech = Utils._speech(self, Utils._choice_message(self, committed, ORDER_MESSAGE)) if owed else None
            if speech:
                reply = (self.speak, speech)
                if self.max_calls > 1:
                    calls.insert(0, reply)
                elif self.turn < self.max_steps - 1:
                    self._send([reply])
            self._send(calls)

class CartEngine(Engine):
    """Turn loop for the strategies that work from the conditions the shopper states."""
    RESCUE_TURNS = 4  # turns kept for committing to the best variant found, rather than looking for a better one

    def _code_calls(self, turn: int) -> list[dict[str, Any]]:
        """The calls code sends this turn, empty when it has none and the model fallback decides."""
        return (self._rescue_calls(turn) or Utils._recover_calls(self, turn)
                or Utils._replacement_calls(self, turn))

    def _fallback_calls(self, turn: int) -> tuple[str, list[dict[str, Any]]]:
        """What to send on a turn code found nothing for: the model's lookups, else a commit to what there is, else a
        read of the cart."""
        try:
            return self._model_calls()
        except RuntimeError:
            return "", self._rescue_calls(turn, forced=True) or self._waiting_calls(turn)

    def _waiting_calls(self, turn: int) -> list[dict[str, Any]]:
        """A read of the cart for a turn code and the model both came up empty on."""
        return ([Utils._tool_call(self.cart, {}, f"{self.problem_id}-turn-{turn}-cart")] if self.cart else [])

    def _rescue_due(self, turn: int) -> bool:
        """Whether to stop looking for a better variant, read from the turns the task allows and what they have brought."""
        if self.max_steps - turn < self.RESCUE_TURNS:
            return True
        _, added = Utils._cart_state(self)
        return bool(added) and getattr(self, "stalled", 0) >= STALL_TURNS

    def _rescue_calls(self, turn: int, forced: bool = False) -> list[dict[str, Any]]:
        """Commit to the variant the observations best support instead of looking for a better one; bounded so it does
        not repeat itself."""
        tools = Utils._tool_names(self)
        if not (forced or self._rescue_due(turn)) or not (self.add and self.stock and self.order):
            return []
        self.rescues = getattr(self, "rescues", 0) + 1
        ref = self._closest_ref()
        if ref is None or self.rescues > self.RESCUE_TURNS + 2:
            return []
        cart, added = Utils._cart_state(self)
        if not added and self.max_steps - turn >= self.RESCUE_TURNS and not Utils._meets_all(self.candidates.get(ref)):
            return []  # the shopper is committed by the first add, so it is not spent while the turns still allow better
        call_id = f"{self.problem_id}-turn-{turn}-rescue"
        if ref in cart:
            if Utils._checked_since_add(self, ref):
                return Utils._order_calls(self, ref, call_id)
            return [Utils._tool_call(self.stock, Utils._ids(self, ref), f"{call_id}-stock")]
        held = [Utils._tool_call(self.remove, Utils._ids(self, other), f"{call_id}-remove-{index}")
                for index, other in enumerate(cart, start=1)] if self.remove else []
        return Utils._limit_calls(
            [*held, Utils._tool_call(self.add, Utils._ids(self, ref), f"{call_id}-add"),
             Utils._tool_call(self.stock, Utils._ids(self, ref), f"{call_id}-stock")], self.max_calls)

    def _closest_ref(self) -> tuple[str, str] | None:
        """The variant to commit to when the search has to stop: fewest unmet, then fewest unsettled, then already in the
        cart, then cheapest."""
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
        return min(pool, key=rank)

    def _model_calls(self) -> tuple[str, list[dict[str, Any]]]:
        """What to look at next, asked of the model on a turn the conditions gave nothing to send."""
        return Utils._choose_calls(self, Utils._lookup_tools(self))

    def _opening_calls(self, turn: int, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """This turn's calls, for a strategy that opens some turn with a call of its own; the turn stands as chosen."""
        return tool_calls

    def _after_result(self, turn: int, result: dict[str, Any]) -> None:
        """What the strategy reads from this turn's result beyond the shopper's requirement updates."""

    def run(self) -> list[dict[str, Any]]:
        if not self._read_plan(Prompt.PLAN.format(tools=self._tools_text())):
            return super().run()
        self.requirements = Utils._merge_requirements(
            [], Utils._extract_requirements(Utils._split_query(self.query)[0], [], 0))
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
                reviewed, stated = Utils._review_candidates(self, self.requirements, result, labelled, listed, settled)
                self.candidates.update(reviewed)
                self.stated.update(stated)
            else:
                self.candidates, stated = Utils._review_candidates(
                    self,
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


# -- strategies ----------------------------------------------------------------


class SoftPreference(Engine):
    """Brief → search → compare → judge and commit → final pick; shopper messages interrupt any turn."""

    RESCUE_TURNS = 4
    JUDGE_BATCH = 40

    def run(self) -> list[dict[str, Any]]:
        if not self._read_plan(Prompt.PLAN.format(tools=self._tools_text())):
            return super().run()
        self.brief = Utils._json(Prompt.INTENT_BRIEF.format(query=self.query))
        try:
            self.budget = float(str(self.brief["budget"]).replace(",", ""))
        except (KeyError, TypeError, ValueError):
            return super().run()
        self.keywords = [q for q in self.brief.get("keywords") or [] if isinstance(q, str)]
        self.valid: dict[tuple[str, str], tuple[bool, bool]] = {}  # judged variant -> (preferred brand, spec stated clearly)
        self.judged: set[tuple[str, str]] = set()
        self.opened: set[str] = set()
        self.sent: set[tuple[str, float]] = set()  # (query or the listing job, budget) already searched
        self.phase, self.searches = "search", 0
        self.rescue = self.budget_changed = False

        for turn in range(1, self.max_steps + 1):
            calls = [(name, arguments) for name, arguments in self._plan(turn) if name]
            if not calls:
                break
            result = self._send_calls(turn, calls)
            if any((call.get("observation") or {}).get("done") for call in result["calls"]):
                break
        return self.dialogue

    def _plan(self, turn: int) -> list[tuple[str, dict[str, Any]]]:
        """This turn's calls: whatever the current phase asks for."""
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
        cap = {"max_price": self.budget, "in_stock": True, "k": Utils._page_size(self.tools, self.finder)}
        queries = []
        for query in self.keywords:  # the brief, the judge and every re-read add queries, and they repeat each other
            if (query.casefold(), self.budget) in self.sent or len(queries) == self.max_calls - 2:
                continue
            self.sent.add((query.casefold(), self.budget))
            queries.append(query)
        calls = [(self.finder, {"query": q, **cap}) for q in queries]
        if ("list", self.budget) not in self.sent:
            self.sent.add(("list", self.budget))
            shelf = {"category": self.brief.get("category"), **cap}
            if self.brief.get("brand"):
                calls.append((self.lister, {**shelf, "brand": self.brief["brand"]}))
            calls.append((self.lister, shelf))
        self.searches += 1
        self.budget_changed = False
        self.phase = "compare"
        return calls

    def _compare(self) -> list[tuple[str, dict[str, Any]]]:
        brand = str(self.brief.get("brand") or "").casefold()
        rows = Utils._product_rows(self)
        ids = [pid for pid, row in rows.items()
               if pid not in self.opened and row.get("in_stock") and self._money(row.get("price")) <= self.budget]
        ids.sort(key=lambda pid: not brand or brand not in str(rows[pid].get("brand") or "").casefold())
        ids = ids[: self.many_limit * (self.max_calls - 1)]
        self.opened.update(ids)
        self.phase = "commit"
        return [(self.many, {self.many_param: ids[i : i + self.many_limit]})
                for i in range(0, len(ids), self.many_limit)]

    def _commit(self) -> list[tuple[str, dict[str, Any]]]:
        """Judge what compare returned, then add the one item to buy."""
        self._judge()
        if Utils._cart_state(self)[0]:  # committed earlier: a budget-cut search only refreshed the candidates
            self.phase = "final"
            return []
        pick = self._top(k for k, (preferred, _) in self.valid.items() if preferred and self._live(k))
        backup = self._backup(pick)
        if not (pick and backup) and self._can_search():
            self.phase = "search"
            return []
        pick = pick or backup or self._top(k for k in self.valid if self._live(k))
        if not pick:  # nothing valid yet: wait a few turns for shopper updates
            return [] if self.rescue else [(self.cart, {})]
        self.phase = "final"
        return [(self.add, Utils._ids(self, pick))]

    def _final(self) -> list[tuple[str, dict[str, Any]]]:
        """Order the best valid item the market left alone: add it in place of anything the cart still holds, read it
        back, order it, one turn each."""
        final = self._best()
        if not final:
            if self._can_search():  # e.g. a budget cut left no valid backup
                self.phase = "search"
                return []
            # nothing to order yet: keep the episode open a few turns so shopper updates can arrive
            return [] if self.rescue else [(self.cart, {})]
        cart = Utils._cart_state(self)[0]
        if final not in cart:
            held = [(self.remove, Utils._ids(self, ref)) for ref in cart] if self.remove else []
            return [*held, (self.add, Utils._ids(self, final)), (self.stock, Utils._ids(self, final))]
        if not Utils._checked_since_add(self, final):
            return [(self.stock, Utils._ids(self, final))]
        self.phase = "done"
        return [(self.order, Utils._ids(self, final))]

    def _judge(self) -> None:
        """Model check of newly compared variants against the firm requirements and the brand."""
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
                if key in batch and isinstance(view, dict) and view.get("valid") is True:
                    self.valid[batch[key]] = (view.get("preferred") is True, view.get("clear") is True)
            self.keywords += [q for q in verdict.get("keywords") or [] if isinstance(q, str)]

    def _send_calls(self, turn: int, calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        """Send one turn of planned calls, then learn from the observations and the shopper."""
        tool_calls = [
            Utils._tool_call(name, arguments, f"{self.problem_id}-turn-{turn}-{index}")
            for index, (name, arguments) in enumerate(calls, start=1)
        ]
        result = Utils._send_turn(self, turn, "", Utils._limit_calls(tool_calls, self.max_calls))
        self._hear(((result.get("user_message") or {}).get("content") or ""))
        return result

    def _hear(self, text: str) -> None:
        """A shopper message applied to the requirements as they stand, never to the request they started from, so a
        later message cannot undo an earlier change. A changed product requirement or brand leaves every verdict stale;
        a changed budget leaves them standing, since the verdicts never read the price."""
        if not text:
            return
        fields = ("category", "spec", "budget", "currency", "brand")
        current = {field: self.brief.get(field) for field in fields}
        revised = Utils._json(Prompt.INTENT_UPDATE.format(
            current=json.dumps({**current, "budget": self.budget}, ensure_ascii=False), message=text)) or {}
        self.keywords += [q for q in revised.get("keywords") or [] if isinstance(q, str)]
        changed = set()
        spec, held = str(revised.get("spec") or ""), str(current["spec"] or "")
        if spec and not (Utils._states_spec(spec, held) or Utils._states_spec(held, spec)):
            self.brief["spec"] = spec
            changed.add("spec")
        category = revised.get("category")
        if category and Utils._label(category) != Utils._label(current["category"] or ""):
            self.brief["category"] = category
            changed.add("category")
        if "brand" in revised and str(revised["brand"] or "").casefold() != str(current["brand"] or "").casefold():
            self.brief["brand"] = revised["brand"]
            changed.add("brand")
        budget = self._money(str(revised.get("budget") or "").replace(",", ""))
        if budget != float("inf") and abs(budget - self.budget) > 1e-9:
            self.budget = self.brief["budget"] = budget
            self.budget_changed = True
            changed.add("budget")
        if changed & {"spec", "category", "brand"}:
            self.valid.clear()
            self.judged.clear()
        if changed:
            self.phase = "commit"

    def _can_search(self) -> bool:
        """Another round of looking: none once the shopper has one product to buy and another to fall back on, and none
        when there is no wording left to send or a rescue is under way. Before the first cart choice, the one to buy is a
        product of the brand they named, since that is the choice they asked to have honoured; any other valid product
        only settles the looking once no wording is left to find theirs. A budget the shopper changed reopens the
        looking, since what fits has changed with it."""
        fresh = ("list", self.budget) not in self.sent or any(
            (q.casefold(), self.budget) not in self.sent for q in self.keywords)
        valid = {ref[0] for ref in self.valid}
        named = {ref[0] for ref, (preferred, _) in self.valid.items() if preferred}
        first = bool(self.brief.get("brand")) and not Utils._cart_state(self)[0]
        settled = len(valid) >= 2 and (bool(named) or not first)
        return ((not settled and self.searches < self.max_calls) or self.budget_changed) and not self.rescue and fresh

    def _price(self, ref: tuple[str, str]) -> float:
        return self._money((Utils._variant_states(self).get(ref) or {}).get("price"))

    def _live(self, ref: tuple[str, str]) -> bool:
        state = Utils._variant_states(self).get(ref) or {}
        return state.get("in_stock") is True and self._money(state.get("price")) <= self.budget

    def _top(self, refs: Any) -> tuple[str, str] | None:
        """Best candidate: spec clearly stated first, then cheapest."""
        return min(refs, key=lambda ref: (not self.valid[ref][1], self._price(ref)), default=None)

    def _backup(self, pick: tuple[str, str] | None) -> tuple[str, str] | None:
        """Best valid, live variant of another product."""
        return self._top(ref for ref in self.valid if self._live(ref) and (not pick or ref[0] != pick[0]))

    def _best(self) -> tuple[str, str] | None:
        """Best valid variant the latest observed price and stock still allow: clear spec first, then the brand the
        shopper named, then already in the cart, then cheapest. A market change is judged by where it left the price and
        stock, since only a move out of the budget or out of stock calls for a switch."""
        cart = Utils._cart_state(self)[0]
        refs = [ref for ref in self.valid if self._live(ref)]
        return min(refs, key=lambda ref: (not self.valid[ref][1], not self.valid[ref][0], ref not in cart,
                                          self._price(ref)), default=None)

    @staticmethod
    def _money(value: Any) -> float:
        """A written price or budget; a value that is not a number is beyond every budget."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")


class WideSearch(CartEngine):
    row_checked = (*Engine.row_checked, Kind.PHRASE)

    def _after_result(self, turn: int, result: dict[str, Any]) -> None:
        """On the first turn, read the requirements again when a value the code sent matched nothing."""
        if turn == 1 and (feedback := self._unmatched_feedback()):
            updates = Utils._extract_requirements(Utils._split_query(self.query)[0], self.requirements, 0, feedback)
            self.requirements = Utils._merge_requirements(self.requirements, updates)

    def _unmatched_feedback(self) -> str:
        """A note for re-extraction naming the category a filter found nothing for and the phrase no search row title
        contains."""
        category = Utils._value(self.requirements, Kind.CATEGORY)
        phrase = Utils._value(self.requirements, Kind.PHRASE)
        entry, lines = self.dialogue[-1], []
        for call, call_result in zip(entry["tool_calls"], (entry.get("environment_result") or {}).get("calls") or []):
            envelope = call_result.get("observation") or {}
            rows = (Utils._body(call_result) or {}).get("results")
            if envelope.get("error") or not isinstance(rows, list):
                continue
            name, arguments = call["function"]["name"], Utils._arguments(call)
            if name == self.lister and category and arguments.get("category") == category and not arguments.get("brand") and not rows:
                lines.append(f"- category {category!r}: a filter for it returned no listings")
            if name == self.finder and phrase and arguments.get("query") == Utils._unquoted(phrase) and not any(
                Utils._words_match(phrase, row.get("title")) == Verdict.MET for row in rows
            ):
                lines.append(f"- phrase {phrase!r}: no search result title contains all of its words")
        if not lines:
            return ""
        return ("These condition values matched nothing when used as written. Check each against the text and report "
                "every condition again:\n" + "\n".join(lines))

    def _queries(self) -> set[str]:
        """Distinct search queries sent so far."""
        return {
            str(Utils._arguments(call).get("query"))
            for message in self.messages if message.get("role") == "assistant"
            for call in message.get("tool_calls") or [] if call["function"]["name"] == self.finder
        }

    def _brought_new(self) -> bool:
        """Whether the last turn showed the shopper a product no earlier turn had and no stated condition rules out. A
        row the conditions already rule out is not something they have been shown, however new its id is."""
        checked = [r for r in self.requirements if r.kind in self.row_checked]
        seen: set[str] = set()
        fresh = True
        for _, body in Utils._observations(self):
            rows = {
                Utils._one(self, row) for row in body.get("results") or []
                if isinstance(row, dict) and Verdict.UNMET not in Utils._check_variant(checked, row, row).values()
            }
            fresh = bool(rows - seen)
            seen |= rows
        return fresh

    def _query_room(self) -> int:
        """Whether one more wording is worth sending: none once two products already have a variant meeting every
        condition, since the shopper has one to buy and one to fall back on, and none once the last wording brought
        nothing they had not been shown that could still be what they asked for."""
        met = {ref[0] for ref, verdicts in self.candidates.items() if Utils._meets_all(verdicts)}
        if len(met) >= 2 or (self._queries() and not self._brought_new()):
            return 0
        return 1

    def _within_query_room(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """calls without the searches whose wording there is no longer room for."""
        queries, room, kept = self._queries(), self._query_room(), []
        for call in calls:
            query = str(Utils._arguments(call).get("query")) if call["function"]["name"] == self.finder else None
            if query is not None and query not in queries:
                if room <= 0:
                    continue
                room -= 1
                queries.add(query)
            kept.append(call)
        return kept

    def _code_calls(self, turn: int) -> list[dict[str, Any]]:
        """The calls for one turn: the cart flow, a first search, the compares an add and its replacement need, and wider
        searches."""
        calls = super()._code_calls(turn)
        if not calls and not self._queries():
            calls = Utils._new_search_calls(self, turn, Utils._split_query(self.query)[0])
        calls = self._needed_compares(turn, calls)
        return self._within_query_room(Utils._limit_calls([*calls, *self._wider_searches(turn, calls)], self.max_calls))

    def _needed_compares(self, turn: int, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """calls with at most one compare batch: none while two products already meet every requirement, one more beside
        a cart add while fewer do."""
        met = {ref[0] for ref, verdicts in self.candidates.items() if Utils._meets_all(verdicts)}
        others = [call for call in calls if call["function"]["name"] != self.many]
        found = [call for call in calls if call["function"]["name"] == self.many]
        compares = found[:1]
        if len(met) >= 2 and others:
            compares = []
        elif len(met) < 2 and not compares and any(call["function"]["name"] == self.add for call in others):
            compares = Utils._compare_calls(self, turn)[:1]
        return [*others, *compares]

    def _wider_searches(self, turn: int, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Before the first cart add, searches for the phrase extended with the words most common in the observed titles
        holding it, while a further wording still has room to bring something new."""
        _, added = Utils._cart_state(self)
        phrase = Utils._value(self.requirements, Kind.PHRASE)
        free = self._query_room()
        if added or phrase is None or free <= 0 or any(
            call["function"]["name"] in {self.add, self.remove, self.order} for call in calls
        ):
            return []

        phrase = Utils._unquoted(phrase)
        titles = {
            Utils._one(self, row): Utils._words(row.get("title"))
            for _, body in Utils._observations(self) for row in body.get("results") or []
        }
        counts: dict[str, int] = {}
        for title in titles.values():
            if Utils._words(phrase) <= title:
                for word in title - Utils._words(phrase):
                    if len(word) > 1:
                        counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts, key=lambda word: (-counts[word], word))
        widened = [call for word in ranked for call in Utils._new_search_calls(self, turn, f"{phrase} {word}")]
        return widened[:free]

    def _model_calls(self) -> tuple[str, list[dict[str, Any]]]:
        """The model fallback, offered only lookup tools, and a search only while there is room for another wording."""
        lookup = Utils._lookup_tools(self)
        without_search = [tool for tool in lookup if tool["function"]["name"] != self.finder]
        content, calls = Utils._choose_calls(self, lookup if self._query_room() else without_search)
        kept = self._within_query_room(calls)
        if not kept:
            content, kept = Utils._choose_calls(self, without_search)
        return content, kept


class FirmConditions(CartEngine):
    REQUIREMENT_QUESTION = (
        "Before I choose a product, could you tell me your firm requirements, "
        "including any specification the item must have?"
    )

    def _opening_calls(self, turn: int, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The first turn opens with the requirement question, so the shopper states what the request withheld."""
        if turn != 1:
            return tool_calls
        speech = Utils._speech(self, self.REQUIREMENT_QUESTION) if self.speak else None
        if speech is None:
            return tool_calls
        question = Utils._tool_call(self.speak, speech,
                                    f"{self.problem_id}-requirement-question")
        return [question, *Utils._limit_calls(tool_calls, self.max_calls - 1)]


class OpenPriorities(Engine):
    """Every action chosen by the model, for a request that leaves the priorities to weigh."""
class StatedTradeoff(Engine):
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
            "order": "tool placing the final order",
            "message": "tool saying something to the shopper, or null"}},
  "item_params": ["the add tool's parameters that identify one exact item"],
  "candidates": [{{"<item parameter>": "value copied exactly from the request"}}],
  "budget": "the most the shopper will pay, as a number, or null",
  "must_be_in_stock": true,
  "kind": "the kind of product wanted, in the request's words",
  "criteria": [{{"name": "one criterion the request orders items by, in its words", "direction": "higher or lower"}}]
}}
List every candidate item the request names, and every criterion in the stated order, including a stated final
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
 "measures": {{"<item label>": [{{"value": "the item's value as written", "amount": number or null}}]}}}}
"measures" gives, for every item, one entry per criterion in the same order. "amount" puts every item's value for that
criterion on one common scale, where a larger amount means more of what the criterion measures (convert units, and turn
grades into amounts in their usual order, so a higher grade always gets a larger amount). Use null only when none of the item's lines
state it; when its lines give different values for it, use the one written next to that property's own name or unit.
""".strip()


    def _by_label(self, labels: dict[str, tuple[str, ...]], mapping: Any) -> dict[str, Any]:
        """A model answer keyed by item, re-keyed by item label however the model wrote each item's name."""
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
        """When this item's own lines state a single different number in the judged value's unit, the own line decides."""
        import re

        value = str(part.get("value") or "")
        pattern = r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*([^\W\d_]{2,})"
        stated = re.search(pattern, value)
        unit = stated.group(2).casefold() if stated and amount is not None and abs(float(stated.group(1)) - amount) < 1e-9 else ""
        if amount is not None and not unit:
            # the judged value names no unit: take the unit written next to that number on the listing's own lines
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

    def _amount(self, measures: dict[str, Any], label: str, index: int) -> float | None:
        row = measures.get(label) if isinstance(measures.get(label), list) else []
        part = row[index] if index < len(row) and isinstance(row[index], dict) else {}
        return self._number(part.get("amount"))


    def _shop(self, raw: dict[str, Any]) -> None:
        refs = []
        for ref in raw.get("candidates") or []:
            if isinstance(ref, dict):
                clean = {p: str(v) for p, v in ref.items() if p in self.ids and str(v).strip() and str(v) in self.query}
                if clean:
                    refs.append(clean)
        groups = [tuple(ref.get(p, "") for p in self.group) for ref in refs]
        self._read_all(self._detail_calls(groups), turns=2)
        keys = [
            k for k in self.seen
            if any(all(dict(zip(self.ids, k)).get(p) == v for p, v in ref.items()) for ref in refs)
        ] or list(self.seen)
        criteria = [c for c in raw.get("criteria") or [] if isinstance(c, dict) and c.get("name")]
        labels, listings = self._evidence(keys)
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
        if gaps:
            retry = self._ask(self.JUDGE.format(
                query=self.query, kind=self.kind, criteria=json.dumps(criteria, ensure_ascii=False),
                listings=json.dumps(listings, ensure_ascii=False),
            ) + "\n\nThese items got no amount for a criterion that other items have: " + ", ".join(gaps)
              + ". Read every line of those items again (title, options and facts) and give the amount their lines state.")
            if self._by_label(labels, retry.get("measures")):
                answer = retry
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
        self._commit_and_order(ordered, lambda k: self._args(self.order, k))

class NamedItem(Engine):
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
            "order": "tool placing the final order",
            "message": "tool saying something to the shopper, or null"}},
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
List every item that is genuinely the kind of product wanted, within the price limit and available, including items
that miss some preferences; the preferences only order the list, best match first. "preferences" has one entry per preference, in the same order. Decide each preference from the
line that the request's rules say decides it; "yes" only when that deciding line shows exactly that value for this
item, and "unsure" when that line is missing, unclear or disagrees with another deciding line.
""".strip()


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
        ]
        extras: list[tuple[str, dict[str, Any]]] = []
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
                    if self._norm(text) not in asked and self._fits(tool, extra):
                        asked.add(self._norm(text))
                        extras.append((tool, extra))
        searches = [*searches[: max(0, self.max_calls - len(extras))], *extras][: self.max_calls]
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
        if not getattr(self, "exact", None) and self.reserve:
            read = len(keys)
            more = self._more_keys(keys, set())
            if len(keys) > read:
                ordered = self._pick(more, wants)
        self._commit_and_order(
            ordered, lambda k: self._args(self.order, k),
            rejudge=(lambda dropped: self._pick(self._more_keys(keys, dropped), wants)) if self.reserve else None,
        )

    def _pick(self, keys: list[tuple[str, ...]], wants: list[dict[str, Any]]) -> list[tuple[str, ...]]:
        """One judge call over these items (a smaller retry when it gives nothing); best match first."""
        import re

        preferences = json.dumps([{"property": w.get("property"), "value": w["value"]} for w in wants], ensure_ascii=False)
        answer: dict[str, Any] = {}
        labels: dict[str, tuple[str, ...]] = {}
        for share in (1, 2):
            labels, listings = self._evidence(keys[: max(1, len(keys) // share)])
            if not labels:
                return []
            answer = self._ask(self.JUDGE.format(
                query=self.query, kind=self.kind, budget=self.budget, preferences=preferences,
                requirements=json.dumps(getattr(self, "needs", []), ensure_ascii=False),
                listings=json.dumps(listings, ensure_ascii=False),
            ))
            if isinstance(answer.get("shortlist"), list) and answer["shortlist"]:
                break
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
                    return True  # a word value named inside a longer option name (a shade name holding the colour word)
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
                        exact, overruled = False, True  # the item's own line for this property names another value
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
        self.exact = {key for key, (unproven, missed, _loose, _position) in info.items() if not unproven and not any(missed)}
        ordered = sorted(info, key=lambda k: info[k])
        return ordered

class ExplainedBuy(Engine):
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
            "order": "tool placing the final order",
            "message": "tool saying something to the shopper, or null"}},
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
List every item that is genuinely the kind of product wanted, has the specification, is within the price
limit and available, best proven first; more than one when more qualify, so a backup exists. "specification" quotes every
line of this item that states the required specification (its title too when the title states it). The specification
counts as proven only by a line that states it plainly for this item;
a line that names different values, examples or choices, or that another line of the same item contradicts, does not
prove it. "reasons" quotes each further line of this item that is a plain, true, positive fact that differs from the
specification, price and availability lines; each value must be one plain value stating what its label names, not a
list, a path of categories, a range or a code, and not a figure that the item's own option lines state differently.
""".strip()


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
        """One judge call over these items (a smaller retry when it gives nothing); best proven first."""
        import re

        keys = sorted(keys, key=self._unstated)
        answer: dict[str, Any] = {}
        labels: dict[str, tuple[str, ...]] = {}
        for share in (1, 2):
            labels, listings = self._evidence(keys[: max(1, len(keys) // share)])
            if not labels:
                return []
            answer = self._ask(self.JUDGE.format(
                query=self.query, kind=self.kind, specification=wanted, budget=self.budget,
                listings=json.dumps(listings, ensure_ascii=False),
            ))
            if isinstance(answer.get("shortlist"), list) and answer["shortlist"]:
                break
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
                    (line for line in self._item_text(key) if line not in own and self._line_states_spec(line, words, asked)),
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
        return ordered

    @staticmethod
    def _figures(text: Any) -> list[float]:
        """Standalone figures in text; a number glued inside a code states no value, and a dimension written number-x-
        number keeps both."""
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
                index = tail  # a code: no figure
                continue
            try:
                found.append(float(text[index:end].replace(",", "")))
            except ValueError:
                pass
            index = end
        return found

    def _brand_miss(self, key: tuple[str, ...]) -> int:
        """1 when the request requires a brand and none of this item's filterable text fields (its brand line) contains
        every word of that brand."""
        import re

        wanted = getattr(self, "brand_words", set())
        if not wanted:
            return 0
        for line in self._stated_lines(key):
            if wanted <= set(re.findall(r"[^\W_]+", self._norm(line.partition(": ")[2]))):
                return 0
        return 1

    def _unclaimable(self, key: tuple[str, ...]) -> int:
        """1 when this item's specification line is not a product fact a claim can cite."""
        lines = self.claim_lines.get(key) or []
        proof = lines[0] if lines else ""
        own = set((self.seen.get(key) or {}).get("own", []))
        return 1 if not proof or proof in own or len(proof.partition(": ")[2].split()) > 8 else 0

    def _clean(self, key: tuple[str, ...], wanted: str) -> bool:
        """Whether every separately listed part of this item's specification line states the whole specification."""
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
        """Items the judge did not list that state the whole specification under the label its proofs used, of the wanted
        kind, in stock and within price."""
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
        """Text fields the catalog tools accept as arguments, other than the item identifiers."""
        names = [t["function"].get("name") for t in self.tools if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        return {
            field for name in names for field, spec in self._props(name).items()
            if isinstance(spec, dict) and spec.get("type") == "string" and field not in self.ids
        }

    def _stated_lines(self, key: tuple[str, ...]) -> list[str]:
        """This item's plain lines for those fields, usable as a checkable reason."""
        fields = self._filter_fields()
        return [
            line for line in self._item_text(key)
            if line.partition(": ")[0] in fields and line.partition(": ")[2].strip().casefold() not in ("", "null", "none")
            and not self._listy(line.partition(": ")[2])
        ]

    def _line_states_spec(self, line: str, words: set[str], asked: set[float]) -> bool:
        """Whether one short line states the required specification by its own words or figures."""
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
        """How many text fields the catalog tools filter by that this item's lines leave empty."""
        return self._unstated_lines(self._item_text(key))

    def _unstated_lines(self, lines: list[str]) -> int:
        fields = self._filter_fields()
        return sum(
            1 for line in lines
            if line.partition(": ")[0] in fields and line.partition(": ")[2].strip().casefold() in ("", "null", "none")
        )

    def _kind_mismatch(self, key: tuple[str, ...]) -> int:
        """1 when the item has category path lines and none names the wanted kind (as a whole level or within its last level)."""
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
        """Identifier arguments plus the claims the order form requires, built only from this item's checked lines."""
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

STRATEGIES: tuple[type[Engine], ...] = (
    SoftPreference,
    WideSearch,
    FirmConditions,
    OpenPriorities,
    StatedTradeoff,
    NamedItem,
    ExplainedBuy,
)


# -- entry -----------------------------------------------------------------------

def clarify(query: str) -> type[Engine]:
    """The kind of shopping the request asks for, named by the model after it notes what the request holds."""
    kind = str(Utils._json(Prompt.CLARIFY.format(query=query)).get("kind") or "")
    return next((strategy for strategy in STRATEGIES if strategy.__name__ == kind.strip()), FirmConditions)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Read what kind of shopping the request asks for and run that strategy."""

    strategy = clarify(problem_data["environment"]["policy_view"]["query"])
    return strategy(problem_data).run()


__all__ = ["agent_main"]
