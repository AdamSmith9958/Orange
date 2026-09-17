"""ORO Bench validator scoring logic -- reference copy, not importable as-is.

Source: oro-env-runtime 0.2.18 from PyPI (MIT, ORO AI), the version pinned in
docker/validator/pyproject.toml. Each section below is one module copied verbatim;
modules still import each other through ``oro_env_runtime`` package paths.

Reading order:
  reward.py                         final paid reward: gates then family metric
  verify.py                         hard checks, budget cuts, interventions, claims gate
  families/base.py                  shared market-event closure
  families/intent_decomposition.py  TF1
  families/preference_reasoning.py  TF4
  families/*.py                     TF2, TF3, TF5, TF6, TF7
  acceptance.py / attributes.py     hard validity, variant attribute resolution
  claim_grounding.py                justification claim truthfulness
"""

from __future__ import annotations


####################################################################################################
# oro_env_runtime/reward.py
####################################################################################################

"""Fail-closed paid-reward contract for the seven shopping-agent families."""

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from oro_env_runtime.tf4_judge_contract import TF4_RELEASE_THRESHOLDS

REWARD_VERSION = "oro_gate_then_gradient_v2"
TF4_PREFERENCE_REWARD_VERSION = "oro_tf4_preference_reward_v3"
TF4_RELEASE_GATE_PATH = Path(__file__).with_name("tf4_hybrid_release_gate.json")


def tf4_release_gate_payload_passed(
    gate: dict[str, Any],
    *,
    judge_contract: dict[str, Any],
) -> bool:
    """Return true only for a fully passed TF4 gate with the exact judge contract."""

    gate_body = gate.get("gate")
    if not isinstance(gate_body, dict):
        return False
    checks = gate_body.get("checks")
    if not isinstance(checks, dict):
        return False
    sealed_inputs = gate.get("sealed_inputs")
    calibration = gate.get("calibration_result")
    holdout = gate.get("holdout")
    if not all(
        isinstance(value, dict) for value in (sealed_inputs, calibration, holdout)
    ):
        return False
    assert isinstance(sealed_inputs, dict)
    assert isinstance(calibration, dict)
    assert isinstance(holdout, dict)
    counts = (
        sealed_inputs.get("calibration_records"),
        sealed_inputs.get("calibration_g1_records"),
        sealed_inputs.get("holdout_records"),
        sealed_inputs.get("holdout_g1_records"),
    )
    hashes = (
        sealed_inputs.get("calibration_file_sha256"),
        sealed_inputs.get("calibration_seal_sha256"),
        sealed_inputs.get("holdout_file_sha256"),
        sealed_inputs.get("holdout_seal_sha256"),
    )
    required_predictions = calibration.get("required_predictions")
    holdout_metrics = holdout.get("metrics")
    holdout_required = holdout.get("required_predictions")
    if not isinstance(holdout_metrics, dict):
        return False
    metric_names = (
        "macro_f1",
        "cohen_kappa",
        "overall_false_positive_rate",
        "comparison_false_positive_rate",
    )
    if not all(
        isinstance(holdout_metrics.get(name), (int, float))
        and not isinstance(holdout_metrics.get(name), bool)
        and math.isfinite(float(holdout_metrics[name]))
        and 0.0 <= float(holdout_metrics[name]) <= 1.0
        for name in metric_names
    ):
        return False
    calibration_counts = (
        calibration.get("valid_predictions"),
        calibration.get("invalid_evidence_witnesses"),
        calibration.get("schema_or_runtime_errors"),
        calibration.get("incomplete_judgments"),
    )
    holdout_counts = (
        holdout.get("valid_predictions"),
        holdout.get("invalid_evidence_witnesses"),
        holdout.get("schema_or_runtime_errors"),
        holdout.get("incomplete_judgments"),
    )
    expected_checks = {
        "macro_f1": holdout_metrics["macro_f1"]
        >= TF4_RELEASE_THRESHOLDS["macro_f1_min"],
        "cohen_kappa": holdout_metrics["cohen_kappa"]
        >= TF4_RELEASE_THRESHOLDS["cohen_kappa_min"],
        "overall_false_positive_rate": holdout_metrics["overall_false_positive_rate"]
        <= TF4_RELEASE_THRESHOLDS["overall_false_positive_rate_max"],
        "comparison_false_positive_rate": holdout_metrics[
            "comparison_false_positive_rate"
        ]
        <= TF4_RELEASE_THRESHOLDS["comparison_false_positive_rate_max"],
        "zero_invalid_evidence_witnesses": holdout.get("invalid_evidence_witnesses")
        == TF4_RELEASE_THRESHOLDS["invalid_evidence_witnesses_max"],
        "full_prediction_coverage": holdout.get("valid_predictions")
        == holdout_required,
        "unseen_holdout_split": holdout.get("unseen_split") is True,
    }
    if not (
        all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in counts
        )
        and all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef")
            for value in hashes
        )
        and calibration.get("protocol_completed") is True
        and isinstance(required_predictions, int)
        and not isinstance(required_predictions, bool)
        and required_predictions == sealed_inputs.get("calibration_g1_records")
        and sealed_inputs["calibration_g1_records"]
        <= sealed_inputs["calibration_records"]
        and sealed_inputs["holdout_g1_records"] <= sealed_inputs["holdout_records"]
        and sealed_inputs["calibration_file_sha256"]
        != sealed_inputs["holdout_file_sha256"]
        and sealed_inputs["calibration_seal_sha256"]
        != sealed_inputs["holdout_seal_sha256"]
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in calibration_counts + holdout_counts
        )
        and calibration.get("valid_predictions") == required_predictions
        and calibration.get("invalid_evidence_witnesses") == 0
        and calibration.get("schema_or_runtime_errors") == 0
        and calibration.get("incomplete_judgments") == 0
        and holdout.get("human_labels_approved") is True
        and holdout.get("judge_run_started") is True
        and holdout.get("future_release_eligible") is True
        and isinstance(holdout_required, int)
        and not isinstance(holdout_required, bool)
        and holdout_required == sealed_inputs.get("holdout_g1_records")
        and holdout.get("schema_or_runtime_errors") == 0
        and holdout.get("incomplete_judgments") == 0
        and checks == expected_checks
        and all(expected_checks.values())
    ):
        return False
    return bool(
        gate.get("schema_version") == "tf4_hybrid_release_gate_v1"
        and gate.get("status") == "PASS"
        and gate.get("reward_mode") == "enabled"
        and gate.get("hybrid_enabled") is True
        and gate.get("judge_contract") == judge_contract
        and gate_body.get("eligible") is True
        and gate.get("required_gate")
        == {
            **TF4_RELEASE_THRESHOLDS,
            "full_prediction_coverage": True,
        }
    )


def tf4_release_gate_passed(
    path: Path = TF4_RELEASE_GATE_PATH,
    *,
    judge_contract: dict[str, Any],
) -> bool:
    try:
        gate = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return tf4_release_gate_payload_passed(gate, judge_contract=judge_contract)


def tf4_release_gate_fingerprint(gate: dict[str, Any]) -> str:
    payload = json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_paid_reward(
    *,
    family: str,
    event_required: bool,
    correct: bool,
    construct_success: bool | None,
    family_metric: Any,
    family_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Return the only scalar an external validator may pay.

    Hard correctness is necessary but not sufficient. Event-bearing families must prove
    policy-visible uptake of the state change; TF5 uses its stricter stateful rerank construct.
    The family metric remains graded after those gates, preserving useful reward differences.
    """
    metric_valid = (
        isinstance(family_metric, (int, float))
        and not isinstance(family_metric, bool)
        and math.isfinite(float(family_metric))
        and 0.0 <= float(family_metric) <= 1.0
    )
    exploit_free = family_metrics.get("sacrificial_exploit") is not True
    if family == "ranking":
        stateful_gate = family_metrics.get("stateful_recovery_success") is True
        stateful_gate_source = "stateful_recovery_success"
    elif event_required:
        stateful_gate = family_metrics.get("event_closed") is True
        stateful_gate_source = "event_closed"
    else:
        stateful_gate = True
        stateful_gate_source = "not_required"

    if not metric_valid:
        reason = "invalid_family_metric"
    elif not correct:
        reason = "hard_gate_failed"
    elif not exploit_free:
        reason = "exploit_detected"
    elif not stateful_gate:
        reason = "stateful_gate_failed"
    else:
        reason = "eligible"
    eligible = reason == "eligible"
    value = round(float(family_metric), 6) if eligible else 0.0
    return {
        "version": REWARD_VERSION,
        "value": value,
        "eligible": eligible,
        "reason": reason,
        "hard_gate_passed": bool(correct),
        "family_metric": float(family_metric) if metric_valid else None,
        "metric_valid": metric_valid,
        "event_required": bool(event_required),
        "stateful_gate": stateful_gate,
        "stateful_gate_source": stateful_gate_source,
        "construct_success": construct_success,
        "exploit_free": exploit_free,
    }


def build_tf4_preference_reward(
    *,
    base_reward_record: dict[str, Any],
    d_score: float | None,
    j_score: float | None,
    j_axes: dict[str, float],
    proof_valid: bool,
    judge_model: str,
    rubric_version: str,
    adjudication_version: str,
    proof_schema_version: str,
    prompt_version: str,
    release_gate_passed: bool,
) -> dict[str, Any]:
    """Build the automatic TF4 reward candidate and activate it only after calibration."""

    g = base_reward_record.get("eligible") is True
    d_valid = (
        isinstance(d_score, (int, float))
        and not isinstance(d_score, bool)
        and math.isfinite(float(d_score))
        and 0.0 <= float(d_score) <= 1.0
    )
    j_valid = (
        isinstance(j_score, (int, float))
        and not isinstance(j_score, bool)
        and math.isfinite(float(j_score))
        and 0.0 <= float(j_score) <= 1.0
        and proof_valid
    )
    if not g:
        status = "skipped_by_gate"
        value = 0.0
        paid_value = 0.0
    elif not d_valid:
        status = "invalid_d"
        value = None
        paid_value = None
    elif not j_valid:
        status = "judge_unavailable"
        value = None
        paid_value = None if release_gate_passed else float(base_reward_record["value"])
    else:
        status = "active" if release_gate_passed else "shadow"
        value = round(0.5 * float(d_score) + 0.5 * float(j_score), 6)
        paid_value = (
            value if release_gate_passed else float(base_reward_record["value"])
        )
    return {
        "version": TF4_PREFERENCE_REWARD_VERSION,
        "status": status,
        "value": value,
        "paid_value": paid_value,
        "eligible": g and paid_value is not None,
        "active": bool(release_gate_passed),
        "release_gate_passed": bool(release_gate_passed),
        "G": bool(g),
        "D": float(d_score) if d_valid else None,
        "J": float(j_score) if j_valid else None,
        "J_axes": dict(j_axes),
        "weights": {"D": 0.5, "J": 0.5},
        "proof_valid": bool(proof_valid),
        "judge_model": judge_model,
        "rubric_version": rubric_version,
        "adjudication_version": adjudication_version,
        "proof_schema_version": proof_schema_version,
        "prompt_version": prompt_version,
        "base_reward": dict(base_reward_record),
    }


__all__ = [
    "REWARD_VERSION",
    "TF4_PREFERENCE_REWARD_VERSION",
    "TF4_RELEASE_GATE_PATH",
    "build_paid_reward",
    "build_tf4_preference_reward",
    "tf4_release_gate_fingerprint",
    "tf4_release_gate_payload_passed",
    "tf4_release_gate_passed",
]


####################################################################################################
# oro_env_runtime/verify.py
####################################################################################################

"""Deterministic verifier for one episode ledger."""

import logging
import re

from .catalog import Catalog
from .claim_grounding import submitted_claims_truthful
from .families import Family, get_family
from .observations import candidate_key, visible_candidate_pairs
from .reward import build_paid_reward
from .schema import REQUIRED_CHECKS, CandidateRef, LedgerEntry, TaskSpec, VerifierResult

_PROVISIONAL_REFERENCE = 20
_LOG = logging.getLogger(__name__)
_REPLY_REQUEST = re.compile(
    r"(?:^|[.!?;]\s*)(?:please\s+)?(?:confirm|reply|respond|answer|tell me|let me know)\b",
    re.IGNORECASE,
)


def _find_order(ledger: list[LedgerEntry]) -> tuple[dict | None, int | None, str | None]:
    """Return only an order signal backed by the canonical cart/order transition."""

    if [entry.seq for entry in ledger] != list(range(len(ledger))):
        return None, None, "noncontiguous_ledger_sequence"

    for index, signal in enumerate(ledger):
        if signal.kind != "verifier_signal" or signal.actor != "harness":
            continue
        if index < 2 or index + 2 >= len(ledger):
            continue
        action = ledger[index - 1]
        observation = ledger[index + 1]
        termination = ledger[index + 2]
        if not (
            action.kind == "model_action"
            and action.actor == "agent"
            and action.payload.get("name") == "place_test_order"
            and action.turn == signal.turn == observation.turn == termination.turn
            and observation.kind == "observation"
            and observation.actor == "harness"
            and termination.kind == "termination"
            and termination.actor == "harness"
            and signal.state_hash == observation.state_hash == termination.state_hash
        ):
            continue

        order = signal.payload.get("order")
        if not isinstance(order, dict):
            continue
        target = order.get("target")
        if not isinstance(target, dict):
            continue
        target_pair = (target.get("product_id"), target.get("sku"))
        if not all(isinstance(value, str) and value for value in target_pair):
            continue

        public_order = observation.payload.get("order")
        if (
            observation.payload.get("status") != "placed"
            or not isinstance(public_order, dict)
            or (
                public_order.get("target", {}).get("product_id"),
                public_order.get("target", {}).get("sku"),
            )
            != target_pair
        ):
            continue

        cart_observation = next(
            (
                entry
                for entry in reversed(ledger[: index - 1])
                if entry.kind == "observation"
                and entry.actor == "harness"
                and isinstance(entry.payload.get("cart"), list)
            ),
            None,
        )
        if cart_observation is None or cart_observation.state_hash != action.state_hash:
            continue
        cart_pairs = [
            (row.get("product_id"), row.get("sku"))
            for row in cart_observation.payload["cart"]
            if isinstance(row, dict)
        ]
        args = action.payload.get("args") or {}
        named_pair = (args.get("product_id"), args.get("sku"))
        if all(named_pair):
            if tuple(str(value) for value in named_pair) != target_pair:
                continue
            if target_pair not in cart_pairs:
                continue
        elif not cart_pairs or cart_pairs[-1] != target_pair:
            continue
        return order, signal.seq, None
    return None, None, None


def _efficiency(
    ledger: list[LedgerEntry],
    observed_pairs: list[tuple[int, str]],
    reference: int | None,
) -> tuple[float, dict]:
    tool_calls = sum(1 for e in ledger if e.kind == "model_action")
    unique_observed = {key for _, key in observed_pairs}
    cost = tool_calls + len(unique_observed)
    ref = reference if reference else _PROVISIONAL_REFERENCE
    efficiency = min(1.0, ref / max(cost, 1))
    return efficiency, {
        "tool_calls": tool_calls,
        "unique_observed": len(unique_observed),
        "cost": cost,
        "reference_used": ref,
        "reference_is_provisional": reference is None,
    }


def _guarded_verify_extra(
    family: Family,
    task: TaskSpec,
    ledger: list[LedgerEntry],
    catalog: Catalog,
    checks: dict[str, bool | None],
    order: dict | None,
    order_seq: int | None,
    observed: list[tuple[int, str]],
) -> dict:
    """Run the family hook, then restore any REQUIRED_CHECKS it tried to move.

    Families extend ``checks`` with process signals, but the four hard gates alone
    decide ``correct``; no family may open or close one. Restoring here makes that
    a mechanical guarantee for future families.
    """
    before = {k: checks[k] for k in REQUIRED_CHECKS}
    metrics = family.verify_extra(
        task, ledger, catalog, checks, order, order_seq, observed=observed
    )
    violated = [k for k in REQUIRED_CHECKS if checks.get(k) != before[k]]
    if violated:
        _LOG.warning(
            "family %r verify_extra modified required checks %s; restoring hard gates",
            task.family,
            violated,
        )
        for k in violated:
            checks[k] = before[k]
    return metrics


def _interventions_answered(ledger: list[LedgerEntry], order_seq: int | None) -> bool | None:
    """Delivered public questions/response requests need a reply before the order.

    A reply is the solver's own text or a ``message`` action after the intervention.
    Pure constraint updates require compliance, checked by the budget/spec gates,
    not an unsolicited acknowledgement. None means no response was requested.
    Only public action/content determine this requirement, never expected_effect.
    """

    asked = [
        entry.seq
        for entry in ledger
        if entry.kind == "user_message"
        and (entry.payload.get("env_signal") or {}).get("kind") == "intervention"
        and (
            entry.payload.get("action") == "clarify_demand"
            or "?" in str(entry.payload.get("content") or "")
            or "？" in str(entry.payload.get("content") or "")
            or _REPLY_REQUEST.search(str(entry.payload.get("content") or ""))
        )
    ]
    if not asked:
        return None
    end = order_seq if order_seq is not None else len(ledger)
    replies = []
    for entry in ledger:
        if entry.actor != "agent":
            continue
        content = None
        if entry.kind == "model_message":
            content = entry.payload.get("content")
        elif entry.kind == "model_action" and entry.payload.get("name") == "message":
            content = (entry.payload.get("args") or {}).get("content")
        if isinstance(content, str) and content.strip():
            replies.append(entry.seq)
    return all(any(seq < reply < end for reply in replies) for seq in asked)


def _pop_family_gate(metrics: dict) -> bool | None:
    gate = metrics.pop("family_gate", None)
    if gate is None or isinstance(gate, bool):
        return gate
    _LOG.warning("ignoring non-bool family_gate value %r", gate)
    return None


def _pop_construct_success(metrics: dict) -> bool | None:
    construct_success = metrics.pop("construct_success", None)
    if construct_success is None or isinstance(construct_success, bool):
        return construct_success
    _LOG.warning("ignoring non-bool construct_success value %r", construct_success)
    return None


def verify(
    task: TaskSpec,
    ledger: list[LedgerEntry],
    catalog: Catalog,
    *,
    reference_count: int | None,
    render_budget: int | None,
) -> VerifierResult:
    """Deterministic verdict over one episode ledger.

    ``render_budget`` selects the observation surface for observed-candidate metrics:

    - an int (live episodes): the solver-visible rendered surface, re-derived
      deterministically via ``visible_candidate_pairs``;
    - ``None`` (scripted adversaries): rendered surface with complete visibility.

    ``checks`` values are ``bool | None`` — None means the fact was never evaluable
    (no order placed, or an event-conditioned check whose event never fired). ``correct``
    requires every REQUIRED_CHECK to be ``True``, so None gates exactly like False.
    """
    acceptance_keys = set(task.acceptance.acceptable_keys)
    order, order_seq, ledger_integrity_error = _find_order(ledger)
    family = get_family(task.family)
    observed_pairs = visible_candidate_pairs(ledger, render_budget)
    visibility_surface = "rendered"

    if ledger_integrity_error is not None:
        checks: dict[str, bool | None] = {k: None for k in REQUIRED_CHECKS}
        eff, detail = _efficiency(ledger, observed_pairs, reference_count)
        family_metrics = {
            "visibility_surface": visibility_surface,
            "acceptance_source": "acceptance_contract",
            "ledger_integrity_error": ledger_integrity_error,
        }
        reward_record = build_paid_reward(
            family=task.family,
            event_required=task.event_rule is not None,
            correct=False,
            construct_success=None,
            family_metric=None,
            family_metrics=family_metrics,
        )
        return VerifierResult(
            correct=False,
            construct_success=None,
            paid_reward=reward_record["value"],
            reward_record=reward_record,
            efficiency=eff,
            checks=checks,
            efficiency_detail=detail,
            family_metrics=family_metrics,
            family_gate=None,
            status="unavailable",
            explanation="invalid ledger: noncontiguous sequence",
        )

    if order is None:
        # No order placed: none of the hard-gate facts is evaluable — record None, not a
        # vacuous False. `correct` stays False because the gates only pass on `is True`.
        checks: dict[str, bool | None] = {k: None for k in REQUIRED_CHECKS}
        eff, detail = _efficiency(ledger, observed_pairs, reference_count)
        family_metrics = _guarded_verify_extra(
            family, task, ledger, catalog, checks, order, order_seq, observed_pairs
        )
        family_gate = _pop_family_gate(family_metrics)
        construct_success = _pop_construct_success(family_metrics)
        family_metrics["visibility_surface"] = visibility_surface
        family_metrics["acceptance_source"] = "acceptance_contract"
        reward_record = build_paid_reward(
            family=task.family,
            event_required=task.event_rule is not None,
            correct=False,
            construct_success=construct_success,
            family_metric=family_metrics.get("family_metric"),
            family_metrics=family_metrics,
        )
        return VerifierResult(
            correct=False,
            construct_success=construct_success,
            paid_reward=reward_record["value"],
            reward_record=reward_record,
            efficiency=eff,
            checks=checks,
            efficiency_detail=detail,
            family_metrics=family_metrics,
            family_gate=family_gate,
            status="live",
            explanation="no order placed",
        )

    tgt = order["target"]
    o_key = candidate_key(tgt["product_id"], tgt["sku"])
    price = order.get("price_at_order", 0.0)
    currency = order.get("currency")
    in_stock = order.get("in_stock_at_order", False)

    checks = {k: False for k in REQUIRED_CHECKS}
    checks["final_in_gold"] = o_key in acceptance_keys
    checks["final_in_stock"] = in_stock is True
    # A delivered add_constraint line tightens the budget for every later order.
    budget = task.hard.budget
    for entry in ledger:
        if order_seq is not None and entry.seq >= order_seq:
            break
        signal = entry.payload.get("env_signal") or {}
        if entry.kind == "user_message" and signal.get("kind") == "intervention":
            rule = task.interventions[int(signal.get("index", 0))]
            if rule.action == "add_constraint" and "budget" in rule.public_delta:
                budget = min(budget, float(rule.public_delta["budget"]))
    checks["within_budget"] = currency == task.hard.currency and price <= budget
    # Its own fact: did the order name a candidate that exists in the catalog?
    # Stock state belongs to final_in_stock and is not duplicated here.
    checks["no_illegal_side_effects"] = catalog.exists(
        CandidateRef(product_id=tgt["product_id"], sku=tgt["sku"])
    )
    family_metrics = _guarded_verify_extra(
        family, task, ledger, catalog, checks, order, order_seq, observed_pairs
    )
    family_gate = _pop_family_gate(family_metrics)
    construct_success = _pop_construct_success(family_metrics)
    claims_truthful = submitted_claims_truthful(
        task, catalog, ledger, order, order_seq, observed_pairs
    )
    if claims_truthful is not None:
        checks["submitted_claims_truthful"] = claims_truthful
        if claims_truthful is False:
            family_gate = False
    family_metrics["visibility_surface"] = visibility_surface
    family_metrics["acceptance_source"] = "acceptance_contract"
    if task.interventions:
        answered = _interventions_answered(ledger, order_seq)
        family_metrics["interventions_answered"] = answered
        if answered is False:
            family_gate = False

    correct = (
        all(checks[k] is True for k in REQUIRED_CHECKS) and family_gate is not False
    )
    # Family quality is computed before the cross-family intervention gate. Full task
    # success cannot survive an unanswered delivered requirement or any hard failure.
    if not correct:
        construct_success = False
    eff, detail = _efficiency(ledger, observed_pairs, reference_count)
    gate_text = f" family_gate={family_gate}" if family_gate is not None else ""
    expl = (
        f"ordered {o_key} @ {price:.2f} {currency or 'UNKNOWN'} "
        f"(budget {task.hard.budget:.2f} {task.hard.currency}); "
        f"in_gold={checks['final_in_gold']} in_stock={checks['final_in_stock']} "
        f"within_budget={checks['within_budget']}{gate_text} | process: family={task.family}"
    )
    reward_record = build_paid_reward(
        family=task.family,
        event_required=task.event_rule is not None,
        correct=correct,
        construct_success=construct_success,
        family_metric=family_metrics.get("family_metric"),
        family_metrics=family_metrics,
    )
    return VerifierResult(
        correct=correct,
        construct_success=construct_success,
        paid_reward=reward_record["value"],
        reward_record=reward_record,
        efficiency=eff,
        checks=checks,
        efficiency_detail=detail,
        family_metrics=family_metrics,
        family_gate=family_gate,
        status="live",
        explanation=expl,
    )


__all__ = ["verify"]


####################################################################################################
# oro_env_runtime/families/__init__.py
####################################################################################################

"""Family registry."""

from .base import Family
from .constraint_satisfaction import ConstraintSatisfactionFamily
from .intent_decomposition import IntentDecompositionFamily
from .justification import JustificationFamily
from .preference_reasoning import PreferenceReasoningFamily
from .ranking import RankingFamily
from .recovery import RecoveryFamily
from .retrieval_recall import RetrievalRecallFamily

FAMILIES: dict[str, Family] = {
    "intent_decomposition": IntentDecompositionFamily(),
    "retrieval_recall": RetrievalRecallFamily(),
    "constraint_satisfaction": ConstraintSatisfactionFamily(),
    "preference_reasoning": PreferenceReasoningFamily(),
    "ranking": RankingFamily(),
    "recovery": RecoveryFamily(),
    "justification": JustificationFamily(),
}


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(FAMILIES))
        raise KeyError(f"unknown family {name!r}; expected one of: {known}") from exc


__all__ = ["FAMILIES", "get_family", "Family"]


####################################################################################################
# oro_env_runtime/families/base.py
####################################################################################################

"""Family adapter contract for Oro environment compilers."""

import re
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from oro_env_runtime.catalog import Catalog
from oro_env_runtime.observations import candidate_key
from oro_env_runtime.schema import CandidateRef, LedgerEntry, TaskSpec

if TYPE_CHECKING:
    from oro_env_runtime.environment import Environment


def title_token_set(title: str | None) -> frozenset[str]:
    """Return normalized title tokens used for observable identity checks."""
    return frozenset(re.findall(r"[a-z0-9]+", (title or "").lower()))


_GOLD_MEMBER_CACHE: dict[
    int, tuple[weakref.ReferenceType[TaskSpec], frozenset[CandidateRef]]
] = {}


def gold_members(task: TaskSpec) -> frozenset[CandidateRef]:
    """Set view of task.gold_set for constant-time membership.

    CandidateRef is frozen over (product_id, sku), so set membership equals list
    membership. Weak identity references release completed sessions' task data;
    the identity guard also rejects stale entries if an object id is reused."""
    key = id(task)
    entry = _GOLD_MEMBER_CACHE.get(key)
    if entry is None or entry[0]() is not task:
        if len(_GOLD_MEMBER_CACHE) >= 4096:
            _GOLD_MEMBER_CACHE.clear()

        def discard(reference: weakref.ReferenceType[TaskSpec]) -> None:
            current = _GOLD_MEMBER_CACHE.get(key)
            if current is not None and current[0] is reference:
                _GOLD_MEMBER_CACHE.pop(key, None)

        entry = (weakref.ref(task, discard), frozenset(task.gold_set))
        _GOLD_MEMBER_CACHE[key] = entry
    return entry[1]


def first_committed_key(ledger: list[LedgerEntry]) -> str | None:
    """Key of the first successful cart add, the public event trigger."""
    for entry in ledger:
        if entry.kind == "observation":
            added = entry.payload.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                return f"{pid}::{sku}"
    return None


def first_committed_ref(ledger: list[LedgerEntry]) -> CandidateRef | None:
    """Return the first candidate that the public cart accepted."""
    key = first_committed_key(ledger)
    if key is None:
        return None
    product_id, separator, sku = key.partition("::")
    if not separator:
        return None
    return CandidateRef(product_id=product_id, sku=sku)


def event_observed_or_signaled(
    ledger: list[LedgerEntry], order_seq: int | None, *, index: int = 0
) -> bool:
    """True when the policy saw market event ``index`` before ordering.

    Evidence is either an in-band sim reaction (react_to_event) or a post-event observation
    of the event target carrying its changed price/stock. Under a blind environment the
    observation surface stays stale, so self-detection is structurally impossible there.

    On a chained task the evidence window for one event ends where the next event fires, so
    a reaction to the SECOND delta cannot retroactively prove the first one was noticed.
    Single-event ledgers have no next event, so the window is unchanged.
    """
    events = [entry for entry in ledger if entry.kind == "harness_event"]
    if index >= len(events):
        return False
    event = events[index]
    next_seq = events[index + 1].seq if index + 1 < len(events) else None
    order_entry = next((entry for entry in ledger if entry.seq == order_seq), None)
    order_turn = order_entry.turn if order_entry is not None else None
    target = event.payload.get("target") or {}
    target_key = candidate_key(
        str(target.get("product_id") or ""), str(target.get("sku") or "")
    )
    new_price = event.payload.get("new_price")
    for entry in ledger:
        if entry.seq <= event.seq:
            continue
        if next_seq is not None and entry.seq >= next_seq:
            continue
        if order_seq is not None and entry.seq >= order_seq:
            continue
        if order_turn is not None and entry.turn >= order_turn:
            continue
        if (
            entry.kind == "user_message"
            and entry.payload.get("action") == "react_to_event"
        ):
            return True
        if entry.kind != "observation":
            continue
        payload = entry.payload
        revealed_event = payload.get("market_event") or {}
        revealed_target = revealed_event.get("target") or {}
        if (
            candidate_key(
                str(revealed_target.get("product_id") or ""),
                str(revealed_target.get("sku") or ""),
            )
            == target_key
        ):
            return True
        rows = []
        if payload.get("product_id") and payload.get("sku"):
            rows.append(payload)
        rows.extend(payload.get("results") or [])
        rows.extend(payload.get("variants") or [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_key = candidate_key(
                str(row.get("product_id") or target.get("product_id") or ""),
                str(row.get("sku") or target.get("sku") or ""),
            )
            if row_key != target_key:
                continue
            if (
                new_price is not None
                and row.get("price") is not None
                and abs(float(row["price"]) - float(new_price)) < 0.01
            ):
                return True
            if row.get("in_stock") is False and event.payload.get("kind") == "stockout":
                return True
    return False


# Closure fields every event family stamps into family_metrics, in one canonical order.
CLOSURE_FIELDS = (
    "event_required",
    "event_fired",
    "intended_event_commitment",
    "event_noticed_or_self_detected",
    "post_event_valid_final",
    "final_not_event_target",
    "no_sacrificial_blind_recovery",
    "sacrificial_exploit",
    "event_closed",
)

# Extra fields a CHAINED task stamps alongside CLOSURE_FIELDS. Single-event families keep
# their metric shape unchanged.
CHAIN_CLOSURE_FIELDS = ("events_required", "events_fired", "events_closed")


def event_closure(
    task: TaskSpec,
    ledger: list[LedgerEntry],
    checks: dict[str, bool | None],
    order: dict | None,
    order_seq: int | None,
    *,
    intended_event_commitment: bool | None,
) -> dict[str, Any]:
    """Event-closure components for the official construct on event-carrying tasks.

    The construct must reward an intended public commitment and true post-event state use. A
    dummy first add or blind survivor can pass terminal hard gates without the intended family
    capability. ``event_closed`` is the AND a family's ``construct_success`` must include when
    ``task.event_rule`` exists. ``sacrificial_exploit`` is the separate exploit flag.

    A CHAINED task (``task.event_rules``) must close every event in the chain: each one fired,
    each one noticed inside its own evidence window, and the final order valid under the state
    the LAST event left and not any event's invalidated target.
    """
    event_required = task.event_rule is not None
    events_required = (1 + len(task.event_rules)) if event_required else 0
    events = [entry for entry in ledger if entry.kind == "harness_event"]
    event = events[0] if events else None
    events_fired = len(events)
    event_fired = event is not None
    noticed = event_observed_or_signaled(ledger, order_seq)
    first_commit = first_committed_key(ledger)

    order_key: str | None = None
    if order is not None:
        tgt = order["target"]
        order_key = candidate_key(
            str(tgt.get("product_id") or ""), str(tgt.get("sku") or "")
        )

    def _invalidated_target(entry: LedgerEntry) -> tuple[str, bool]:
        payload = entry.payload
        target = payload.get("target") or {}
        key = candidate_key(
            str(target.get("product_id") or ""), str(target.get("sku") or "")
        )
        if payload.get("kind") == "stockout":
            return key, True
        new_price = payload.get("new_price")
        return key, (
            float(new_price) > task.hard.budget if new_price is not None else False
        )

    def _final_not_target(entry: LedgerEntry) -> bool | None:
        """None until an order exists; else whether it avoided this event's dead target."""

        if order_key is None:
            return None
        key, invalidated = _invalidated_target(entry)
        return order_key != key if invalidated else True

    target_key: str | None = None
    if event is not None:
        target_key, _ = _invalidated_target(event)

    post_event_valid_final: bool | None = None
    if order is not None:
        # The order record is stamped from true post-event state by the environment, so these
        # two hard-gate facts are exactly "valid under true post-event state" -- for a chained
        # task, under the state left by the LAST event.
        post_event_valid_final = (
            checks.get("final_in_stock") is True and checks.get("within_budget") is True
        )
    # No event to dodge: an order that exists trivially avoided one, as before.
    final_not_event_target = (
        _final_not_target(event)
        if event is not None
        else (True if order_key is not None else None)
    )

    blind_recovery = bool(
        event_fired
        and order_key is not None
        and target_key is not None
        and order_key != target_key
        and not noticed
    )
    # A wrong or exploratory first add fails the commitment gate, but it is not by itself an
    # exploit. Reserve exploit telemetry for the specific blind-recovery shape.
    sacrificial_exploit = blind_recovery

    # Per-event closure: fired, noticed on its own evidence window, and not the final order.
    per_event = [
        bool(
            event_observed_or_signaled(ledger, order_seq, index=index)
            and _final_not_target(entry) is not False
        )
        for index, entry in enumerate(events)
    ]
    events_closed = sum(per_event)

    if not event_required:
        event_closed = True
    else:
        event_closed = bool(
            events_fired == events_required
            and intended_event_commitment is True
            and events_closed == events_required
            and post_event_valid_final is True
            and not sacrificial_exploit
        )

    return {
        "event_required": event_required,
        "event_fired": event_fired,
        "intended_event_commitment": intended_event_commitment,
        "event_noticed_or_self_detected": noticed,
        "post_event_valid_final": post_event_valid_final,
        "final_not_event_target": final_not_event_target,
        "no_sacrificial_blind_recovery": not sacrificial_exploit,
        "sacrificial_exploit": sacrificial_exploit,
        "event_closed": event_closed,
        "event_target_key": target_key,
        "first_committed_key": first_commit,
        "events_required": events_required,
        "events_fired": events_fired,
        "events_closed": events_closed,
    }


class Family(ABC):
    name: str
    agent_system: str
    metric_name: str  # the family's headline signal; single source of truth

    def validate_event_rule(self, task: TaskSpec, catalog: Catalog) -> bool:
        """Recompute whether the public event rule has the family-owned form."""

        return task.event_rule is None

    def validate_event_commitment(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef | None
    ) -> bool | None:
        """Recompute whether the first add has the family-owned public role."""

        if task.event_rule is None:
            return None
        return False

    def event_commitment_candidates(
        self, task: TaskSpec, catalog: Catalog
    ) -> Iterable[CandidateRef]:
        """Return the task-sealed candidate search space for event admission."""

        if task.acceptance is None:
            return ()
        return (
            CandidateRef(product_id=product_id, sku=sku)
            for key in task.acceptance.acceptable_keys
            for product_id, separator, sku in [key.partition("::")]
            if separator
        )

    def event_targets(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef
    ) -> list[str]:
        """Candidate keys the market delta invalidates when it fires on ``ref``.

        The delta hits the committed option alone unless a family owns a wider notion of
        "this option is gone": TF5 stocks out the whole tradeoff-tie class, since a twin the
        task's own ranking rule cannot tell apart would silently inherit the top slot.
        """

        return [ref.key()]

    @abstractmethod
    def configure_environment(self, env: "Environment", task: TaskSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_write(self, env: "Environment", ref) -> bool:
        raise NotImplementedError

    @abstractmethod
    def verify_extra(
        self,
        task: TaskSpec,
        ledger: list[LedgerEntry],
        catalog: Catalog,
        checks: dict[str, bool | None],
        order: dict | None,
        order_seq: int | None,
        *,
        observed: list[tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def user_sim_context(self, task: TaskSpec) -> str | dict | None:
        return None

    def user_sim_allow_pushback(self, task: TaskSpec) -> bool:
        return False


__all__ = [
    "CHAIN_CLOSURE_FIELDS",
    "CLOSURE_FIELDS",
    "Family",
    "event_closure",
    "event_observed_or_signaled",
    "first_committed_key",
    "first_committed_ref",
    "title_token_set",
]


####################################################################################################
# oro_env_runtime/families/intent_decomposition.py
####################################################################################################

"""Intent-decomposition family for hard gates vs soft preferences."""

from typing import Any

from oro_env_runtime.acceptance import (
    brand_matches,
    candidate_satisfies_hard,
    deterministic_event_multiplier,
)
from oro_env_runtime.attributes import resolve_variant_attribute
from oro_env_runtime.catalog import Catalog
from oro_env_runtime.schema import (
    REQUIRED_CHECKS,
    CandidateRef,
    HardConstraints,
    LedgerEntry,
    TaskSpec,
)

from .base import (
    CHAIN_CLOSURE_FIELDS,
    CLOSURE_FIELDS,
    Family,
    event_closure,
    first_committed_ref,
    gold_members,
)
from .constraint_satisfaction import _cat_word, _spec_phrase

INTENT_REVISION_VERSION = "tf1_public_attribute_revision_v1"
INTENT_REVISION_NOTICE = (
    "After your first cart choice, I will give you one requirement update; "
    "read it before placing the order."
)


def revised_hard_constraints(task: TaskSpec) -> HardConstraints | None:
    """The sealed later contract; callers must not expose it before delivery."""
    revision = task.family_payload.get("intent_revision") or {}
    if revision.get("version") != INTENT_REVISION_VERSION:
        return None
    key = revision.get("attribute")
    value = revision.get("value")
    if key not in task.hard.attributes or not isinstance(value, str) or not value:
        return None
    return task.hard.model_copy(
        update={"attributes": {**task.hard.attributes, key: value}}
    )


def public_revision_utterance(task: TaskSpec, key: str, value: str) -> str:
    change = (
        f"I need to change one requirement: it must now have {_spec_phrase(key, value)} "
        f"instead of {_spec_phrase(key, task.hard.attributes[key])}. "
        if value != task.hard.attributes[key]
        else f"To restate my requirement: it must have {_spec_phrase(key, value)}. "
    )
    return (
        change + f"Keep every other requirement, including my {task.hard.budget:.2f} "
        f"{task.hard.currency} budget."
    )


def intent_revision_errors(task: TaskSpec, catalog: Catalog) -> list[str]:
    """Validate the two public states without weakening other families' admission."""
    if "intent_revision" not in task.family_payload:
        return []
    hard = revised_hard_constraints(task)
    revision = task.family_payload["intent_revision"]
    if hard is None:
        return ["invalid TF1 public attribute revision"]
    if INTENT_REVISION_NOTICE not in task.goal_text:
        return ["TF1 goal must announce the requirement update"]
    rules = task.interventions
    if len(rules) != 1 or not (
        rules[0].action == "replace_constraint"
        and rules[0].trigger.kind == "cart_add_count"
        and rules[0].trigger.count == 1
        and rules[0].public_delta
        == {"attributes": {revision["attribute"]: revision["value"]}}
        and rules[0].utterance
        == public_revision_utterance(task, revision["attribute"], revision["value"])
    ):
        return ["TF1 revision must state the first-cart public replacement"]
    revised = {
        ref.product_id
        for ref in task.gold_set
        if candidate_satisfies_hard(catalog, ref, hard)
    }
    return (
        []
        if len(revised) >= 2
        else ["TF1 revision needs two valid revised alternatives"]
    )


AGENT_SYSTEM = (
    "You are a shopping agent. The user describes what they want as a MIX of firm requirements and "
    "nice-to-haves, and you must tell them apart. The firm, non-negotiable requirements are the "
    "category, one specific hardware spec (an amount of RAM, a screen size, a resolution, or a "
    "refresh rate), the budget, and in-stock availability. An explicitly named preferred brand "
    "is a SOFT preference, subordinate to every firm requirement. For your initial cart choice, "
    "honor that named brand when a fully compliant item is available. Your initial choice and "
    "your later recovery are assessed separately: a later correction does not repair an "
    "initial choice that ignored an available, fully compliant preferred-brand item. Do NOT buy "
    "an item that misses a firm requirement just because it is cheaper or nicer, and do NOT refuse "
    "an item that meets every firm requirement just because it lacks a soft preference. Prices and "
    "stock can change while you shop, so verify the current price and stock before ordering; if "
    "your pick is repriced above budget, switch to another item that meets every firm requirement, "
    "even a different brand. When you need input or want to recommend something, send a short "
    "message to the user. If the shopper explicitly replaces a requirement, use its latest value "
    "and preserve every requirement they did not change. Finish by placing the order for a valid, "
    "in-stock item that meets every firm requirement within budget. Once verified, place the "
    "order - do not wait for permission."
)


class IntentDecompositionFamily(Family):
    name = "intent_decomposition"
    agent_system = AGENT_SYSTEM
    metric_name = "decomposition_correct"

    def configure_environment(self, env, task: TaskSpec) -> None:
        return None

    def on_write(self, env, ref: CandidateRef) -> bool:
        return False

    def validate_event_rule(self, task: TaskSpec, catalog: Catalog) -> bool:
        if task.event_rule is None:
            return True
        # Every chained event is the sealed recovery-commitment stockout.
        if any(
            rule.kind != "stockout" or rule.trigger != "second_successful_cart_add"
            for rule in task.event_rules
        ):
            return False
        return (
            task.event_rule.kind == "price_change"
            and task.event_rule.price_multiplier
            == deterministic_event_multiplier(
                self.name,
                task.seed,
                minimum_hundredths=20,
                maximum_hundredths=40,
            )
        )

    def validate_event_commitment(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef | None
    ) -> bool | None:
        if task.event_rule is None:
            return None
        return bool(
            ref is not None
            and ref in gold_members(task)
            and candidate_satisfies_hard(catalog, ref, task.hard)
            and ref.product_id
            in set(
                task.family_payload.get(
                    "preferred_brand_gold",
                    task.family_payload.get("reco_gold", []),
                )
            )
        )

    def user_sim_context(self, task: TaskSpec) -> str | dict | None:
        fp = task.family_payload
        return {
            "persona": fp.get("shopper_persona"),
            "hard_facts": [
                f"a {_cat_word(fp['category'])} with {_spec_phrase(fp['gated_key'], fp['gated_value'])}",
                f"in stock, at or under {task.hard.budget:.2f} {task.hard.currency}",
            ],
            "soft_axes": [{"axis": pref} for pref in fp["soft_prefs"]],
            "tradeoff_priority": [
                "every firm requirement and the budget come first",
                "for the initial cart choice, honor my named preferred brand when a fully compliant item is available",
                "the soft extras are nice-to-haves you would happily give up",
            ],
            "notes": [
                "if the preferred brand cannot meet every firm requirement, an off-brand item that does is fine"
            ],
        }

    def user_sim_allow_pushback(self, task: TaskSpec) -> bool:
        return True

    def verify_extra(
        self,
        task: TaskSpec,
        ledger: list[LedgerEntry],
        catalog: Catalog,
        checks: dict[str, bool | None],
        order: dict | None,
        order_seq: int | None,
        *,
        observed: list[tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        key, wanted = (
            task.family_payload["gated_key"],
            task.family_payload["gated_value"],
        )
        revised_hard = revised_hard_constraints(task)
        revision_delivered = False
        if revised_hard is not None:
            revision_delivered = any(
                entry.kind == "user_message"
                and entry.actor == "user_sim"
                and (order_seq is None or entry.seq < order_seq)
                and entry.payload.get("env_signal")
                == {"kind": "intervention", "index": 0, "action": "replace_constraint"}
                and entry.payload.get("content") == task.interventions[0].utterance
                for entry in ledger
            )
            if revision_delivered:
                wanted = revised_hard.attributes[key]
        alternate_brand = set(
            task.family_payload.get(
                "alternate_brand_gold",
                task.family_payload.get("off_brand_gold", []),
            )
        )
        attribute_satisfied: bool | None = None
        ordered_off_brand: bool | None = None
        if order is not None:
            tgt = order["target"]
            ref = CandidateRef(product_id=tgt["product_id"], sku=tgt["sku"])
            if catalog.exists(ref):
                attribute_satisfied = (
                    resolve_variant_attribute(
                        catalog.by_id(ref.product_id), catalog.meta(ref).options, key
                    )
                    == wanted
                )
                ordered_off_brand = ref.product_id in alternate_brand
                if revised_hard is not None:
                    ordered_off_brand = not brand_matches(
                        task.family_payload.get("preferred_brand"),
                        catalog.meta(ref).brand,
                        title=str(catalog.by_id(ref.product_id).get("title") or ""),
                    )
            else:
                attribute_satisfied = False
        correct = all(checks.get(k) is True for k in REQUIRED_CHECKS)
        revision_complete = revised_hard is None or revision_delivered
        public_constraints_valid = True
        if revised_hard is not None and order is not None:
            public_constraints_valid = bool(
                catalog.exists(ref)
                and candidate_satisfies_hard(
                    catalog, ref, revised_hard if revision_delivered else task.hard
                )
            )
        checks["attribute_satisfied"] = attribute_satisfied
        closure = event_closure(
            task,
            ledger,
            checks,
            order,
            order_seq,
            intended_event_commitment=self.validate_event_commitment(
                task, catalog, first_committed_ref(ledger)
            ),
        )
        return {
            "construct_success": bool(
                correct
                and attribute_satisfied is True
                and public_constraints_valid
                and revision_complete
                and closure["event_closed"]
            ),
            "family_gate": attribute_satisfied is True and public_constraints_valid,
            "family_metric": 1.0
            if correct and attribute_satisfied is True and public_constraints_valid and revision_complete
            else 0.0,
            "attribute_satisfied": attribute_satisfied,
            "ordered_off_brand": ordered_off_brand,
            "gated_key": key,
            "gated_value": wanted,
            "difficulty": task.family_payload.get("difficulty"),
            "over_trap_present": bool(alternate_brand),
            **(
                {"intent_revision_delivered": revision_delivered}
                if revised_hard is not None
                else {}
            ),
            **{k: closure[k] for k in CLOSURE_FIELDS},
            # Chain counts are stamped only for chained tasks, so single-event verdicts
            # keep their exact family_metrics shape.
            **(
                {k: closure[k] for k in CHAIN_CLOSURE_FIELDS}
                if task.event_rules
                else {}
            ),
        }


__all__ = ["IntentDecompositionFamily"]


####################################################################################################
# oro_env_runtime/families/preference_reasoning.py
####################################################################################################

"""Preference-reasoning family: optimize hidden alignment from visible facts."""

import re
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

from oro_env_runtime.acceptance import (
    candidate_satisfies_hard,
    deterministic_event_multiplier,
)
from oro_env_runtime.catalog import Catalog
from oro_env_runtime.observations import visible_candidate_pairs_for_actions
from oro_env_runtime.product_facts import visible_product_facts
from oro_env_runtime.schema import REQUIRED_CHECKS, CandidateRef, LedgerEntry, TaskSpec

from .base import (
    CLOSURE_FIELDS,
    Family,
    event_closure,
    first_committed_ref,
    gold_members,
)
from .constraint_satisfaction import _cat_word

AGENT_SYSTEM = (
    "You are a shopping agent. The user's initial request authorizes one test order and gives a "
    "loose hard request plus soft preferences. The "
    "hard requirements are category, budget, current stock, and buying an actual device rather "
    "than an accessory or review page. The soft preferences matter: inspect raw product facts "
    "such as battery, water/sweat resistance, materials, connectivity, and use-specific "
    "features, then choose the item that best matches the user's intended use instead of "
    "blindly buying the cheapest valid item. Before the first cart addition, clarify the "
    "user's priorities and inspect or compare the visible facts to choose a strong match "
    "across several preference axes. That first cart addition must be a genuine intended "
    "purchase, not an exploratory or sacrificial placeholder: its preference fit is evaluated "
    "too, and a good later replacement does not repair a weak initial commitment. "
    "Prices can change while you shop, so verify "
    "current price and stock after committing; if your first pick is repriced above budget, "
    "recover to another valid in-stock option. When you need input or want to recommend "
    "something, send a short message to the user. Compare a focused shortlist rather than "
    "inspecting every option. Once you have verified your best valid option's current price "
    "and stock, place that one test order - do not ask for a second confirmation."
)

_GOOD_THRESHOLD = 0.65
_WRONG_THRESHOLD = 0.30
_SPORT_FIT_PATTERN = r"\b(sports?|running|gym|workout|fitness|ear ?hook|secure fit)\b"
_PROVENANCE_VISIBLE_FACT = "visible_fact"
_PROVENANCE_TITLE_INFERENCE = "title_inference"
_PROVENANCE_SERIES_INFERENCE = "series_inference"
_PROVENANCE_WORLD_KNOWLEDGE = "world_knowledge"
_PROVENANCE_NONE = "none"
_REVEAL_POLICY = {
    "type": "progressive",
    "rule": (
        "lead with your use case; go into specific preferences only when asked or when a "
        "recommendation clearly misses them"
    ),
}


@dataclass(frozen=True)
class PreferenceScore:
    score: float
    axes: dict[str, float]
    penalty: float
    provenance: dict[str, str] = field(default_factory=dict)


def _contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def _facts_text(catalog: Catalog, ref: CandidateRef) -> str:
    rec = catalog.by_id(ref.product_id)
    facts = visible_product_facts(rec)
    return " ".join(
        f"{key.replace('_', ' ')} {value}" for key, value in facts.items()
    ).casefold()


def _match_source(pattern: str, facts_text: str, title_text: str = "") -> str:
    if _contains(facts_text, pattern):
        return _PROVENANCE_VISIBLE_FACT
    if title_text and _contains(title_text, pattern):
        return _PROVENANCE_TITLE_INFERENCE
    return _PROVENANCE_NONE


def _duration_source(
    value: float, facts_text: str, title_text: str = "", *, short: bool = False
) -> str:
    if value <= 0:
        return _PROVENANCE_NONE
    axis = _short_duration_axis if short else _duration_axis
    if axis(facts_text) > 0:
        return _PROVENANCE_VISIBLE_FACT
    if title_text and axis(title_text) > 0:
        return _PROVENANCE_TITLE_INFERENCE
    return _PROVENANCE_SERIES_INFERENCE


def _duration_axis(text: str) -> float:
    days = [
        float(m.group(1))
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*days?", text, flags=re.I)
    ]
    hours = [
        float(m.group(1))
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hours?", text, flags=re.I)
    ]
    if days:
        best = max(days)
        if best >= 10:
            return 1.0
        if best >= 5:
            return 0.75
        if best >= 2:
            return 0.5
    if hours:
        best = max(hours)
        if best >= 120:
            return 1.0
        if best >= 72:
            return 0.75
        if best >= 36:
            return 0.5
    if "solar" in text or "battery life" in text:
        return 0.5
    return 0.0


def _short_duration_axis(text: str) -> float:
    """Battery tiers for devices measured in hours (headphones, tablets), not days."""
    hours = [
        float(m.group(1))
        for m in re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*\+?\s*(?:hours?|hrs?|h)\b", text, flags=re.I
        )
    ]
    if re.search(r"\d+(?:\.\d+)?\s*days?", text, flags=re.I):
        return 1.0
    if hours:
        best = max(hours)
        if best >= 15:
            return 1.0
        if best >= 10:
            return 0.75
        if best >= 6:
            return 0.5
    if "battery life" in text:
        return 0.25
    return 0.0


def _score_smartwatch(catalog: Catalog, ref: CandidateRef) -> PreferenceScore:
    rec = catalog.by_id(ref.product_id)
    meta = catalog.meta(ref)
    facts = visible_product_facts(rec)
    text = " ".join(
        f"{key.replace('_', ' ')} {value}" for key, value in facts.items()
    ).casefold()
    brand = str(meta.brand or "").casefold()

    navigation = (
        1.0
        if _contains(
            text,
            r"\b(gps|gnss|maps?|compass|barometer|altitude|tactical|trail|hiking|outdoor)\b",
        )
        else 0.0
    )
    rugged = 0.0
    if _contains(
        text,
        r"\b(10\s*atm|5\s*atm|50m|100m|ip68|ip69|mil-std|waterproof|water resistant|sapphire|"
        r"fiber[- ]reinforced|titanium|rugged|tactical|tank)\b",
    ):
        rugged = 1.0
    elif _contains(text, r"\b(water|swim|steel|aluminum)\b"):
        rugged = 0.5
    battery = _duration_axis(text)
    outdoor_brand = 1.0 if brand in {"garmin", "amazfit", "fitbit", "kospet"} else 0.0

    penalty = (
        0.2
        if _contains(text, r"\b(kid|kids|child|children|myfirst|watch phone)\b")
        else 0.0
    )
    raw = (
        (0.32 * navigation)
        + (0.26 * rugged)
        + (0.26 * battery)
        + (0.16 * outdoor_brand)
        - penalty
    )
    axes = {
        "navigation": navigation,
        "rugged_water_resistance": rugged,
        "battery_life": battery,
        "outdoor_watch_credibility": outdoor_brand,
    }
    provenance = {
        "navigation": _PROVENANCE_VISIBLE_FACT if navigation > 0 else _PROVENANCE_NONE,
        "rugged_water_resistance": _PROVENANCE_VISIBLE_FACT
        if rugged > 0
        else _PROVENANCE_NONE,
        "battery_life": _duration_source(battery, text),
        "outdoor_watch_credibility": _PROVENANCE_WORLD_KNOWLEDGE
        if outdoor_brand > 0
        else _PROVENANCE_NONE,
    }
    return PreferenceScore(
        score=round(max(0.0, min(1.0, raw)), 3),
        axes=axes,
        penalty=penalty,
        provenance=provenance,
    )


def _score_headphones(catalog: Catalog, ref: CandidateRef) -> PreferenceScore:
    rec = catalog.by_id(ref.product_id)
    title = str(rec.get("title") or "").casefold()
    facts_text = _facts_text(catalog, ref)
    text = facts_text + " " + title
    category_text = " ".join(catalog.meta(ref).category_path).casefold()
    wireless_facts = facts_text + " " + category_text

    water = (
        1.0
        if _contains(
            text,
            r"\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\b",
        )
        else 0.0
    )
    sport = 1.0 if _contains(text, _SPORT_FIT_PATTERN) else 0.0
    battery = _short_duration_axis(text)
    wireless = (
        1.0
        if _contains(
            wireless_facts + " " + title, r"\b(wireless|bluetooth|true wireless)\b"
        )
        else 0.0
    )

    raw = (0.30 * water) + (0.28 * sport) + (0.22 * battery) + (0.20 * wireless)
    axes = {
        "sweat_water_resistance": water,
        "sport_secure_fit": sport,
        "battery_life": battery,
        "wireless_freedom": wireless,
    }
    provenance = {
        "sweat_water_resistance": _match_source(
            r"\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\b",
            facts_text,
            title,
        ),
        "sport_secure_fit": _match_source(
            _SPORT_FIT_PATTERN,
            facts_text,
            title,
        ),
        "battery_life": _duration_source(battery, facts_text, title, short=True),
        "wireless_freedom": _match_source(
            r"\b(wireless|bluetooth|true wireless)\b", wireless_facts, title
        ),
    }
    return PreferenceScore(
        score=round(max(0.0, min(1.0, raw)), 3),
        axes=axes,
        penalty=0.0,
        provenance=provenance,
    )


def _score_tablet(catalog: Catalog, ref: CandidateRef) -> PreferenceScore:
    rec = catalog.by_id(ref.product_id)
    title = str(rec.get("title") or "").casefold()
    facts_text = _facts_text(catalog, ref)
    text = facts_text + " " + title

    kid_ready = (
        1.0
        if _contains(text, r"\b(kids?|child|children|drop ?proof|rugged case|bumper)\b")
        else 0.0
    )
    battery = _short_duration_axis(text)
    size_fit = 1.0 if _contains(text, r"\b(8|10|10\.\d|11)\s*(?:inch|in\b|\")") else 0.0
    wifi = 1.0 if _contains(text, r"\b(wi-?fi|wifi)\b") else 0.0

    raw = (0.40 * kid_ready) + (0.25 * battery) + (0.20 * size_fit) + (0.15 * wifi)
    axes = {
        "kid_ready_durability": kid_ready,
        "battery_life": battery,
        "kid_friendly_size": size_fit,
        "wifi_ready": wifi,
    }
    provenance = {
        "kid_ready_durability": _match_source(
            r"\b(kids?|child|children|drop ?proof|rugged case|bumper)\b",
            facts_text,
            title,
        ),
        "battery_life": _duration_source(battery, facts_text, title, short=True),
        "kid_friendly_size": _match_source(
            r"\b(8|10|10\.\d|11)\s*(?:inch|in\b|\")", facts_text, title
        ),
        "wifi_ready": _match_source(r"\b(wi-?fi|wifi)\b", facts_text, title),
    }
    return PreferenceScore(
        score=round(max(0.0, min(1.0, raw)), 3),
        axes=axes,
        penalty=0.0,
        provenance=provenance,
    )


@dataclass(frozen=True)
class PreferenceProfile:
    """Per-category preference scenario: compiler-owned WHAT, deterministic scorer, sim facts.

    The scorer is an additive weighted sum, so the scenario must never state a strict
    lexicographic rank (audit finding: a literal rank-follower can pick a single-axis item
    below the alignment gate). priority_facts carry the honest weight-grouped framing, each
    use case is paired with its salient scorer axes and the goal lead axis is drawn from that
    pairing, and a scenario note says several axes together beat any single one. Laptops and
    Smartphones were evaluated and excluded on evidence grounds (1 and 2 good-vibe products
    on the committed catalog).
    """

    category: str
    budgets: tuple[float, ...]
    hard_budget_max: float  # difficulty band: budget <= this -> "hard"
    scorer: Callable[[Catalog, CandidateRef], PreferenceScore]
    axis_order: tuple[str, ...]  # descending scorer weight
    axis_meanings: dict[str, str]
    priority_facts: tuple[
        str, ...
    ]  # weight-grouped, additive-honest priority statements
    use_cases: tuple[
        tuple[str, tuple[str, ...]], ...
    ]  # (narrative, salient scorer axes)
    pref_axes: tuple[str, ...]  # latent_prefs shown to the shadow judge
    unacceptable: tuple[str, ...]
    accessory_phrases: tuple[str, ...]
    device_signals: tuple[str, ...] = ()
    device_brands: frozenset[str] = frozenset()
    reach_queries: tuple[str, ...] = ()
    actual_device_label: str = ""
    bad_title_tokens: tuple[str, ...] = (
        "customer review",
        "customer rating",
        "search results",
        "results page",
        "refurb",
        "pre-owned",
        "restored",
        "renewed",
    )
    extra: dict[str, Any] = field(default_factory=dict)


_PROFILES: dict[str, PreferenceProfile] = {
    "Smartwatches": PreferenceProfile(
        category="Smartwatches",
        budgets=(100.0, 150.0, 200.0, 250.0, 300.0),
        hard_budget_max=150.0,
        scorer=_score_smartwatch,
        axis_order=(
            "navigation",
            "rugged_water_resistance",
            "battery_life",
            "outdoor_watch_credibility",
        ),
        axis_meanings={
            "navigation": "GPS/navigation that can be relied on off-grid",
            "rugged_water_resistance": "something that survives water, drops, and rough handling",
            "battery_life": "battery that lasts multi-day trips",
            "outdoor_watch_credibility": "a brand known for real outdoor/fitness watches",
        },
        priority_facts=(
            "gps/navigation counts most",
            "ruggedness/water resistance and battery life count equally next",
            "outdoor brand credibility counts least",
        ),
        use_cases=(
            (
                "a hiking-obsessed retiree who is out on trails most weekends",
                ("navigation", "rugged_water_resistance"),
            ),
            (
                "multi-day backpacking trips where charging is rarely possible",
                ("battery_life", "navigation"),
            ),
            (
                "trail running and open-water swims in all weather",
                ("rugged_water_resistance",),
            ),
            (
                "long off-grid camping trips that lean hard on navigation",
                ("navigation", "battery_life"),
            ),
            (
                "someone who fishes, hunts, and hikes year-round in rough conditions",
                ("rugged_water_resistance", "battery_life"),
            ),
        ),
        pref_axes=(
            "gps/navigation for outdoor use",
            "long battery life",
            "water resistance or rugged materials",
            "fitness/outdoor-watch credibility",
        ),
        unacceptable=(
            "the cheapest valid option when it is clearly worse on these preferences",
            "fragile fashion-first models that would not survive outdoor use",
        ),
        accessory_phrases=(
            "case for",
            "screen protector",
            "band for",
            "strap for",
            "charger for",
            "mudra band",
            "replacement band",
            "watch band",
        ),
        device_signals=(
            "smartwatch",
            "smart watch",
            "watch phone",
            "apple watch",
            "galaxy watch",
            "pixel watch",
            "garmin",
            "fitbit",
            "amazfit",
            "kospet",
        ),
        device_brands=frozenset({"garmin", "fitbit", "amazfit", "kospet"}),
        reach_queries=(
            "rugged gps smartwatch",
            "long battery waterproof smartwatch",
            "outdoor fitness gps watch",
        ),
        actual_device_label="smartwatch_not_accessory",
    ),
    "Headphones": PreferenceProfile(
        category="Headphones",
        budgets=(50.0, 100.0, 150.0, 200.0, 300.0),
        hard_budget_max=100.0,
        scorer=_score_headphones,
        axis_order=(
            "sweat_water_resistance",
            "sport_secure_fit",
            "battery_life",
            "wireless_freedom",
        ),
        axis_meanings={
            "sweat_water_resistance": "sweat and water resistance that survives hard workouts",
            "sport_secure_fit": "a secure sport fit that stays put while running",
            "battery_life": "battery that outlasts long gym sessions and runs",
            "wireless_freedom": "fully wireless so there is no cable to snag",
        },
        priority_facts=(
            "sweat/water resistance counts most, with a secure sport fit close behind",
            "battery life next",
            "wireless freedom counts least",
        ),
        use_cases=(
            (
                "daily runs in every kind of weather",
                ("sweat_water_resistance", "sport_secure_fit"),
            ),
            ("heavy gym sessions with a lot of sweat", ("sweat_water_resistance",)),
            (
                "marathon training with long weekend runs",
                ("battery_life", "sport_secure_fit"),
            ),
            (
                "outdoor bootcamp workouts year-round",
                ("sweat_water_resistance", "sport_secure_fit"),
            ),
        ),
        pref_axes=(
            "sweat/water resistance for workouts",
            "secure sport fit",
            "long battery life",
            "wireless freedom",
        ),
        unacceptable=(
            "the cheapest valid option when it is clearly worse on these preferences",
            "studio or fashion headphones that would die in a sweaty workout",
        ),
        accessory_phrases=(
            "case for",
            "ear pads",
            "earpads",
            "replacement pad",
            "cable for",
            "adapter",
            "stand for",
        ),
        device_signals=(
            "headphone",
            "headphones",
            "earbud",
            "earbuds",
            "earphone",
            "earphones",
            "headset",
        ),
        reach_queries=(
            "waterproof sport earbuds",
            "wireless running headphones",
            "gym workout earbuds ipx",
        ),
        actual_device_label="headphones_not_accessory",
    ),
    "Tablets": PreferenceProfile(
        category="Tablets",
        budgets=(80.0, 120.0, 200.0, 300.0, 450.0),
        hard_budget_max=120.0,
        scorer=_score_tablet,
        axis_order=(
            "kid_ready_durability",
            "battery_life",
            "kid_friendly_size",
            "wifi_ready",
        ),
        axis_meanings={
            "kid_ready_durability": "kid-ready durability (drop protection or a rugged case)",
            "battery_life": "battery that lasts a full travel day",
            "kid_friendly_size": "a size a kid can actually hold (8 to 11 inch)",
            "wifi_ready": "wifi-ready for downloads and streaming",
        },
        priority_facts=(
            "kid-ready durability counts most by far",
            "battery life next",
            "size and wifi count least",
        ),
        use_cases=(
            (
                "a 6-year-old who drops everything at least once a day",
                ("kid_ready_durability",),
            ),
            (
                "long car trips and flights with two kids sharing it",
                ("battery_life", "kid_friendly_size"),
            ),
            (
                "a kid's first tablet that has to survive a school bag",
                ("kid_ready_durability", "kid_friendly_size"),
            ),
            ("road-trip entertainment that cannot die mid-drive", ("battery_life",)),
        ),
        pref_axes=(
            "kid-ready durability",
            "long battery life",
            "kid-friendly size",
            "wifi-ready",
        ),
        unacceptable=(
            "the cheapest valid option when it is clearly worse on these preferences",
            "fragile bare-glass tablets with no protection for kids",
        ),
        accessory_phrases=(
            "case for",
            "keyboard for",
            "screen protector",
            "stylus",
            "pen for",
            "stand for",
            "charger for",
        ),
        device_signals=("tablet", "ipad", "galaxy tab", "fire hd", "fire max"),
        reach_queries=(
            "kids tablet case bundle",
            "kids tablet 10 inch",
            "wifi tablet long battery",
        ),
        actual_device_label="tablet_not_accessory",
    ),
}
_DEFAULT_CATEGORY = "Smartwatches"  # frozen pre-profile traces are all smartwatch tasks


def _profile_for(category: str | None) -> PreferenceProfile:
    return _PROFILES.get(str(category or ""), _PROFILES[_DEFAULT_CATEGORY])


_SCORE_CACHE: weakref.WeakKeyDictionary[
    Catalog, dict[tuple[str, str], PreferenceScore]
] = weakref.WeakKeyDictionary()


def preference_score(
    catalog: Catalog, ref: CandidateRef, *, category: str | None = None
) -> PreferenceScore:
    if not catalog.exists(ref):
        return PreferenceScore(score=0.0, axes={}, penalty=0.0)
    # The scorers read only sealed catalog fields (visible product facts and brand);
    # price and stock changes live in the environment overlay, never in the catalog.
    # So the score is fixed per (catalog, candidate, category) for the whole epoch.
    cache = _SCORE_CACHE.setdefault(catalog, {})
    cache_key = (ref.key(), str(category or ""))
    if cache_key not in cache:
        cache[cache_key] = _profile_for(category).scorer(catalog, ref)
    return cache[cache_key]


def _event_target_key(ledger: list[LedgerEntry]) -> str | None:
    for entry in ledger:
        if entry.kind == "harness_event":
            target = entry.payload.get("target") or {}
            if target.get("product_id") and target.get("sku"):
                return f"{target['product_id']}::{target['sku']}"
    return None


def _post_event_acceptable(
    task: TaskSpec, catalog: Catalog, ledger: list[LedgerEntry]
) -> list[CandidateRef]:
    excluded = _event_target_key(ledger)
    refs: list[CandidateRef] = []
    for key in task.acceptance.acceptable_keys:
        if key == excluded:
            continue
        pid, sku = key.split("::", 1)
        # both fields come from splitting the key, so validation is a no-op here
        ref = CandidateRef.model_construct(product_id=pid, sku=sku)
        if catalog.exists(ref):
            refs.append(ref)
    return refs


class PreferenceReasoningFamily(Family):
    name = "preference_reasoning"
    agent_system = AGENT_SYSTEM
    metric_name = "preference_alignment"

    def configure_environment(self, env, task: TaskSpec) -> None:
        return None

    def on_write(self, env, ref: CandidateRef) -> bool:
        return False

    def validate_event_rule(self, task: TaskSpec, catalog: Catalog) -> bool:
        if task.event_rule is None:
            return True
        return (
            task.event_rule.kind == "price_change"
            and task.event_rule.price_multiplier
            == deterministic_event_multiplier(
                self.name,
                task.seed,
                minimum_hundredths=20,
                maximum_hundredths=35,
            )
        )

    def event_commitment_candidates(
        self, task: TaskSpec, catalog: Catalog
    ) -> list[CandidateRef]:
        """Preference commitments are necessarily members of the sealed gold set."""

        return task.gold_set

    def validate_event_commitment(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef | None
    ) -> bool | None:
        if task.event_rule is None:
            return None
        category = str(task.hard.category or _DEFAULT_CATEGORY)
        return bool(
            ref is not None
            and ref in gold_members(task)
            and candidate_satisfies_hard(catalog, ref, task.hard)
            and preference_score(catalog, ref, category=category).score
            >= _GOOD_THRESHOLD
        )

    def user_sim_context(self, task: TaskSpec) -> str | dict | None:
        category = str(task.hard.category or _DEFAULT_CATEGORY)
        contract = task.family_payload.get("preference_contract")
        if isinstance(contract, dict):
            notes = (
                [contract["multi_axis_rule"]] if contract.get("multi_axis_rule") else []
            )
            return {
                "use_case": contract.get("use_case"),
                "hard_facts": [
                    f"an actual {_cat_word(category)} device (not an accessory), in stock, "
                    f"at or under {task.hard.budget:.2f} {task.hard.currency}"
                ],
                "soft_axes": contract.get("soft_axes"),
                "tradeoff_priority": contract.get("tradeoff_priority"),
                "unacceptable_shortcuts": contract.get("unacceptable_shortcuts"),
                "notes": notes,
                "reveal_policy": _REVEAL_POLICY,
            }
        profile = task.family_payload.get("shopper_profile")
        if not isinstance(profile, dict):
            return (
                f"your hard requirements are an actual {_cat_word(category)} product, in stock, "
                f"at or under {task.hard.budget:.2f} {task.hard.currency}. Your preferences are soft but "
                "important: rugged outdoor use, GPS/navigation, water resistance, and long "
                "battery life. A cheaper valid watch is acceptable only if it is not clearly "
                "worse on those preferences."
            )
        return {
            "use_case": profile.get("use_case"),
            "hard_facts": [
                f"an actual {_cat_word(category)} device (not an accessory), in stock, "
                f"at or under {task.hard.budget:.2f} {task.hard.currency}"
            ],
            "soft_axes": profile.get("soft_axes"),
            "tradeoff_priority": profile.get("tradeoff_priority"),
            "unacceptable_shortcuts": profile.get("unacceptable_shortcuts"),
            "notes": profile.get("notes"),
            "reveal_policy": profile.get("reveal_policy"),
        }

    def user_sim_allow_pushback(self, task: TaskSpec) -> bool:
        return True

    def verify_extra(
        self,
        task: TaskSpec,
        ledger: list[LedgerEntry],
        catalog: Catalog,
        checks: dict[str, bool | None],
        order: dict | None,
        order_seq: int | None,
        *,
        observed: list[tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        category = str(task.hard.category or _DEFAULT_CATEGORY)
        chosen_score: PreferenceScore | None = None
        chosen_key: str | None = None
        chosen_facts: dict[str, str] | None = None
        if order is not None:
            target = order["target"]
            ref = CandidateRef(product_id=target["product_id"], sku=target["sku"])
            chosen_key = ref.key()
            chosen_score = preference_score(catalog, ref, category=category)
            chosen_facts = (
                visible_product_facts(catalog.by_id(ref.product_id))
                if catalog.exists(ref)
                else None
            )

        event_seq = next(
            (entry.seq for entry in ledger if entry.kind == "harness_event"), None
        )
        event_target_key = _event_target_key(ledger)
        pre_event_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare"}),
            before_seq=event_seq,
        )
        pre_event_keys = {key for _, key in pre_event_pairs}
        comparison_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare"}),
            before_seq=order_seq,
        )
        comparison_keys = {key for _, key in comparison_pairs}
        accepted_keys = set(task.acceptance.acceptable_keys)
        observed_eligible_keys = comparison_keys & accepted_keys
        observed_eligible_sources = {
            catalog.source_listing_key(key.split("::", 1)[0])
            for key in observed_eligible_keys
        }
        chosen_observed = chosen_key in comparison_keys
        state_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare", "inspect_stock"}),
            after_seq=event_seq,
            before_seq=order_seq,
        )
        state_keys = {key for _, key in state_pairs}
        final_state_observed = chosen_key in state_keys
        event_target_observed = event_target_key in pre_event_keys
        recovery_source_changed = bool(
            event_target_key
            and chosen_key
            and catalog.source_listing_key(event_target_key.split("::", 1)[0])
            != catalog.source_listing_key(chosen_key.split("::", 1)[0])
        )
        preference_evidence_grounded = bool(
            event_target_observed
            and chosen_observed
            and len(observed_eligible_sources) >= 2
            and recovery_source_changed
            and final_state_observed
        )

        survivors = _post_event_acceptable(task, catalog, ledger)
        survivor_scores = [
            (ref, preference_score(catalog, ref, category=category))
            for ref in survivors
        ]
        best = max((score.score for _ref, score in survivor_scores), default=None)
        ordered_rank = None
        if chosen_key and survivor_scores:
            ordered = sorted(
                survivor_scores, key=lambda item: (-item[1].score, item[0].key())
            )
            for idx, (ref, _score) in enumerate(ordered, start=1):
                if ref.key() == chosen_key:
                    ordered_rank = idx
                    break

        correct = all(checks.get(key) is True for key in REQUIRED_CHECKS)
        score_value = chosen_score.score if chosen_score is not None else None
        family_metric = (
            score_value
            if correct and preference_evidence_grounded and score_value is not None
            else 0.0
            if order is not None
            else None
        )
        preference_aligned_success = (
            correct
            and preference_evidence_grounded
            and score_value is not None
            and score_value >= _GOOD_THRESHOLD
        )
        contract = task.family_payload.get("preference_contract")
        contract_id = (
            contract.get("contract_id") if isinstance(contract, dict) else None
        )
        contract_version = (
            contract.get("version") if isinstance(contract, dict) else None
        )
        first_cart_seq = next(
            (
                entry.seq
                for entry in ledger
                if entry.kind == "observation"
                and isinstance(entry.payload.get("added"), dict)
                and entry.payload["added"].get("product_id")
                and entry.payload["added"].get("sku")
            ),
            None,
        )
        # The simulator authenticates this authored reply; do not grade its wording.
        user_sim_preference_reveal = any(
            entry.kind == "user_message"
            and entry.actor == "user_sim"
            and first_cart_seq is not None
            and entry.seq < first_cart_seq
            and entry.payload.get("action") == "clarify"
            and entry.payload.get("reason") == "sealed_facts_answer"
            and bool(entry.payload.get("content"))
            for entry in ledger
        )
        clarification_required = bool(
            isinstance(contract, dict)
            and contract.get("requires_priority_clarification") is True
        )
        priority_clarification_satisfied = (
            not clarification_required or user_sim_preference_reveal
        )
        if not priority_clarification_satisfied and family_metric is not None:
            family_metric = 0.0
        closure = event_closure(
            task,
            ledger,
            checks,
            order,
            order_seq,
            intended_event_commitment=self.validate_event_commitment(
                task, catalog, first_committed_ref(ledger)
            ),
        )
        return {
            "construct_success": bool(
                preference_aligned_success
                and priority_clarification_satisfied
                and closure["event_closed"]
            ),
            "family_metric": family_metric,
            "item_selection_success": correct,
            "hard_valid_checkout": correct,
            "deterministic_preference_alignment": bool(preference_aligned_success),
            "user_sim_preference_reveal": user_sim_preference_reveal,
            "priority_clarification_satisfied": priority_clarification_satisfied,
            "judge_diagnostic_coverage": None,
            "preference_alignment_score": score_value,
            "preference_evidence_grounded": preference_evidence_grounded,
            "observed_eligible_sources": len(observed_eligible_sources),
            "event_target_observed": event_target_observed,
            "recovery_source_changed": recovery_source_changed,
            "chosen_observed": chosen_observed,
            "final_state_observed": final_state_observed,
            "preference_aligned_success": preference_aligned_success,
            "preference_evidence_provenance": chosen_score.provenance
            if chosen_score is not None
            else None,
            "preference_detail": {
                "chosen_score": score_value,
                "best_survivor_score": best,
                "preference_regret": round((best or 0.0) - (score_value or 0.0), 3)
                if best is not None and score_value is not None
                else None,
                "ordered_survivor_rank": ordered_rank,
                "chosen_axes": chosen_score.axes if chosen_score is not None else None,
                "chosen_penalty": chosen_score.penalty
                if chosen_score is not None
                else None,
                "evidence_provenance": chosen_score.provenance
                if chosen_score is not None
                else None,
                "chosen_visible_facts": chosen_facts,
                "score_source": "brand_visible_product_facts",
                "score_hidden_from_solver": True,
                "preference_contract_id": contract_id,
                "preference_contract_version": contract_version,
            },
            "good_threshold": _GOOD_THRESHOLD,
            "wrong_threshold": _WRONG_THRESHOLD,
            "preference_contract_id": contract_id,
            "preference_contract_version": contract_version,
            "category": category,
            "eligible_count": task.family_payload.get("eligible_count"),
            "good_count": task.family_payload.get("good_count"),
            "wrong_count": task.family_payload.get("wrong_count"),
            "difficulty": task.family_payload.get("difficulty"),
            **{k: closure[k] for k in CLOSURE_FIELDS},
        }


__all__ = ["PreferenceReasoningFamily"]


####################################################################################################
# oro_env_runtime/families/retrieval_recall.py
####################################################################################################

"""Retrieval-recall family."""

import math
import re
from typing import Any

from oro_env_runtime.acceptance import candidate_satisfies_hard
from oro_env_runtime.catalog import Catalog
from oro_env_runtime.observations import candidate_key, observed_candidate_pairs
from oro_env_runtime.product_facts import visible_product_facts
from oro_env_runtime.schema import REQUIRED_CHECKS, CandidateRef, LedgerEntry, TaskSpec

from .base import (
    CLOSURE_FIELDS,
    Family,
    event_closure,
    first_committed_ref,
)

AGENT_SYSTEM = (
    "You are a shopping agent. Search the catalog for what the user described. Reformulate "
    "queries and inspect candidates until you are confident, then buy the best in-stock item "
    "within the user's budget. Survey the realistic options before committing: a good shortlist "
    "covers several distinct products, not one lucky hit. When you need input or want to "
    "recommend something, send a short message to the user. Finish by placing the order for a "
    "valid, in-stock item within budget. Once an item is verified in stock and within budget, "
    "place the order - do not wait for permission or further confirmation."
)

CONSTRUCT_RECALL_AT_10_THRESHOLD = 0.5
# Retrieval construct floors: credit counts relevance clusters (product identities), not SKUs.
# A task must offer at least _MIN_CONSTRUCT_CLUSTERS distinct identities to evidence retrieval
# at all, and a passing episode must surface at least _MIN_OBSERVED_CLUSTERS of them before
# ordering — one observed item can never satisfy retrieval quality.
_MIN_CONSTRUCT_CLUSTERS = 5
_MIN_OBSERVED_CLUSTERS = 3


_MODEL_MIN_LEN = 5
_MODEL_MAX_LEN = 24
# split-safety: a carried listing may share a model but a different condition or bundling; those
# never enter the pool through model identity


def _model_tokens(catalog: Catalog, product_id: str) -> frozenset[str]:
    """Distinctive normalized model identifiers for one product (empty when none qualify).

    A token qualifies only when it is 5-24 alphanumerics AND carries a digit, so generic
    marketing names never create identity; sources are identifiers.model and the visible
    model fact.
    """
    rec = catalog.by_id(product_id)
    values: list[str] = []
    ident = rec.get("identifiers")
    if isinstance(ident, dict) and ident.get("model"):
        values.append(str(ident["model"]))
    fact_model = visible_product_facts(rec).get("model")
    if fact_model:
        values.append(str(fact_model))
    out: set[str] = set()
    for value in values:
        token = "".join(re.findall(r"[a-z0-9]+", value.casefold()))
        if _MODEL_MIN_LEN <= len(token) <= _MODEL_MAX_LEN and any(
            c.isdigit() for c in token
        ):
            out.add(token)
    return frozenset(out)


def relevance_clusters(
    catalog: Catalog, refs: list[CandidateRef]
) -> list[frozenset[str]]:
    """Partition eligible candidates into product-identity relevance clusters.

    Two candidates share a cluster when they are variants of one product_id, carry a shared
    distinctive model identifier (the cross-retailer same-product case), or are observable
    twins (same brand, identical normalized title-token set, price within $0.01). The
    retrieval construct counts clusters, so duplicate listings of one identity can never
    inflate coverage and a pool of near-duplicates cannot trivialize the construct.
    """
    from .base import title_token_set

    keys = sorted({r.key() for r in refs})
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_pid: dict[str, list[str]] = {}
    for key in keys:
        by_pid.setdefault(key.split("::", 1)[0], []).append(key)
    for members in by_pid.values():
        for other in members[1:]:
            union(members[0], other)

    model_anchor: dict[str, str] = {}
    twin_groups: dict[tuple[str, frozenset[str]], list[tuple[float, str]]] = {}
    for pid, members in sorted(by_pid.items()):
        anchor = members[0]
        for token in sorted(_model_tokens(catalog, pid)):
            if token in model_anchor:
                union(model_anchor[token], anchor)
            else:
                model_anchor[token] = anchor
        ref = CandidateRef(product_id=pid, sku=anchor.split("::", 1)[1])
        if not catalog.exists(ref):
            continue
        meta = catalog.meta(ref)
        sig = (
            (meta.brand or "").strip().lower(),
            title_token_set(str(catalog.by_id(pid).get("title") or "")),
        )
        twin_groups.setdefault(sig, []).append((meta.price, anchor))
    for rows in twin_groups.values():
        rows.sort()
        for (p_a, key_a), (p_b, key_b) in zip(rows, rows[1:]):
            if abs(p_a - p_b) <= 0.01:
                union(key_a, key_b)

    clusters: dict[str, set[str]] = {}
    for key in keys:
        clusters.setdefault(find(key), set()).add(key)
    return [frozenset(members) for _, members in sorted(clusters.items())]


def _required_observed_clusters(total_clusters: int) -> int:
    """Translate the verifier's floor and recall threshold into one admission count."""
    return max(
        _MIN_OBSERVED_CLUSTERS,
        math.ceil(min(total_clusters, 10) * CONSTRUCT_RECALL_AT_10_THRESHOLD),
    )


def required_observed_clusters(task: TaskSpec) -> int:
    """Return the public coverage target, preserving the legacy task floor."""
    total = len(task.family_payload.get("relevance_clusters") or [])
    minimum = _required_observed_clusters(total)
    requested = task.family_payload.get("required_observed_clusters", minimum)
    if type(requested) is not int or requested < minimum:
        raise ValueError("retrieval coverage target must be an integer at least the task floor")
    if "required_observed_clusters" in task.family_payload and requested > total:
        raise ValueError("retrieval coverage target exceeds available product identities")
    return requested


def _shortlist_precision(
    ledger: list[LedgerEntry],
    catalog: Catalog,
    pool_keys: set[str],
    pool_pids: set[str],
) -> float | None:
    """Return eligible-pool precision over viewed, compared, and carted items."""
    shortlist: dict[tuple[str, str], bool] = {}
    for e in ledger:
        if e.kind != "model_action":
            continue
        args = e.payload.get("args") or {}
        pid = args.get("product_id")
        name = e.payload.get("name")
        if name == "view" and pid and catalog.contains_product(str(pid)):
            shortlist[("view", str(pid))] = str(pid) in pool_pids
        elif name == "compare":
            for cpid in args.get("product_ids") or []:
                if cpid and catalog.contains_product(str(cpid)):
                    shortlist[("compare", str(cpid))] = str(cpid) in pool_pids
        elif name == "add_to_cart" and pid and args.get("sku"):
            key = candidate_key(str(pid), str(args["sku"]))
            shortlist[("cart", key)] = key in pool_keys
    if not shortlist:
        return None
    return round(sum(shortlist.values()) / len(shortlist), 3)


def _reformulation_overfetch(
    ledger: list[LedgerEntry],
) -> tuple[bool | None, bool | None]:
    """Return reformulation and overfetch telemetry."""
    from oro_env_runtime.environment import READ_CAP

    n_search = 0
    queries: list[str] = []
    oversized_k = False
    for e in ledger:
        if e.kind != "model_action" or e.payload.get("name") != "search":
            continue
        n_search += 1
        args = e.payload.get("args") or {}
        if args.get("query") is not None:
            queries.append(str(args["query"]))
        k = args.get("k", args.get("num_results"))
        if k is not None:
            try:
                oversized_k = oversized_k or int(k) > READ_CAP
            except (TypeError, ValueError):
                pass
    if n_search == 0:
        return None, None
    distinct = set(queries)
    return len(distinct) > 1, (oversized_k or len(queries) > len(distinct))


class RetrievalRecallFamily(Family):
    name = "retrieval_recall"
    agent_system = AGENT_SYSTEM
    metric_name = "retrieval_quality"

    def configure_environment(self, env, task: TaskSpec) -> None:
        return None

    def on_write(self, env, ref: CandidateRef) -> bool:
        return False

    def validate_event_rule(self, task: TaskSpec, catalog: Catalog) -> bool:
        if task.event_rule is None:
            return True
        return (
            task.event_rule.kind == "stockout"
            and task.event_rule.price_multiplier is None
        )

    def validate_event_commitment(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef | None
    ) -> bool | None:
        if task.event_rule is None:
            return None
        return bool(
            ref is not None
            and candidate_satisfies_hard(catalog, ref, task.hard)
            and ref.key()
            in {str(key) for key in task.family_payload.get("retrieval_pool", [])}
        )

    def event_commitment_candidates(
        self, task: TaskSpec, catalog: Catalog
    ) -> list[CandidateRef]:
        return [
            CandidateRef(product_id=product_id, sku=sku)
            for key in task.family_payload.get("retrieval_pool", [])
            for product_id, separator, sku in [str(key).partition("::")]
            if separator
        ]

    def user_sim_context(self, task: TaskSpec) -> str | dict | None:
        theme = task.family_payload.get("theme")
        if not theme:
            return None
        return {
            "use_case": f'you are after something like "{" ".join(theme)}" for a normal home setup',
            "hard_facts": [f"in stock and at or under {task.hard.budget:.2f} {task.hard.currency}"],
        }

    def verify_extra(
        self,
        task: TaskSpec,
        ledger: list[LedgerEntry],
        catalog: Catalog,
        checks: dict[str, bool | None],
        order: dict | None,
        order_seq: int | None,
        *,
        observed: list[tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        from oro_env_runtime.environment import READ_CAP

        # Two pools, two jobs. The reference pool is the frozen seed anchor. The eligible pool is
        # the stamped admitted retrieval denominator.
        reference_pool = {r.key() for r in task.gold_set}
        eligible_pool = {str(k) for k in task.family_payload["retrieval_pool"]}
        recall_pool_source = "retrieval_pool"
        eligible_pids = {k.split("::", 1)[0] for k in eligible_pool}

        if observed is None:
            observed_pairs = observed_candidate_pairs(ledger)
        else:
            observed_pairs = list(observed)
        before_order = [
            (seq, key)
            for seq, key in observed_pairs
            if order_seq is None or seq < order_seq
        ]
        observed_keys = {key for _, key in before_order}
        # search/filter observations are the only payloads carrying a results list; keys surfaced
        # there isolate query-formulation coverage from later view/compare/inspect_stock reads.
        result_seqs = {
            e.seq
            for e in ledger
            if e.kind == "observation" and isinstance(e.payload.get("results"), list)
        }
        search_surfaced = {key for seq, key in before_order if seq in result_seqs}

        reference_observed = len(reference_pool & observed_keys)
        eligible_observed = len(eligible_pool & observed_keys)
        reference_recall = (
            reference_observed / len(reference_pool) if reference_pool else 0.0
        )
        eligible_recall = (
            eligible_observed / len(eligible_pool) if eligible_pool else 0.0
        )
        # recall@10 keeps the metric comparable across tasks when a pool exceeds READ_CAP and raw
        # recall is bounded by observation; both sides capped by READ_CAP so it stays in [0, 1].
        candidate_recall_at_10 = (
            min(reference_observed, 10) / min(len(reference_pool), 10)
            if reference_pool
            else 0.0
        )
        eligible_recall_at_10 = (
            min(eligible_observed, 10) / min(len(eligible_pool), 10)
            if eligible_pool
            else 0.0
        )
        search_result_recall = (
            len(eligible_pool & search_surfaced) / len(eligible_pool)
            if eligible_pool
            else 0.0
        )
        queries = {
            str(e.payload.get("args", {}).get("query"))
            for e in ledger
            if e.kind == "model_action"
            and e.payload.get("name") == "search"
            and e.payload.get("args", {}).get("query") is not None
        }
        # Efficiency-normalized coverage; None (unevaluated) when the denominator never existed.
        recall_per_query = round(eligible_recall / len(queries), 3) if queries else None
        recall_per_visible_candidate = (
            round(eligible_recall / len(observed_keys), 3) if observed_keys else None
        )
        checks["observed_eligible_before_order"] = eligible_observed > 0
        correct = all(checks.get(k) is True for k in REQUIRED_CHECKS)
        # Cluster-level retrieval construct: coverage of distinct product identities under the
        # fixed search interface, backend-agnostic. Stamped clusters are the frozen contract.
        clusters = [
            frozenset(str(k) for k in cluster)
            for cluster in task.family_payload["relevance_clusters"]
        ]
        cluster_source = "stamped"
        total_clusters = len(clusters)
        observed_clusters = sum(1 for cluster in clusters if cluster & observed_keys)
        search_surfaced_clusters = sum(
            1 for cluster in clusters if cluster & search_surfaced
        )
        cluster_recall_at_10 = (
            min(observed_clusters, min(total_clusters, 10)) / min(total_clusters, 10)
            if total_clusters
            else 0.0
        )
        weak_retrieval_evidence = total_clusters < _MIN_CONSTRUCT_CLUSTERS
        required_clusters = required_observed_clusters(task)
        closure = event_closure(
            task,
            ledger,
            checks,
            order,
            order_seq,
            intended_event_commitment=self.validate_event_commitment(
                task, catalog, first_committed_ref(ledger)
            ),
        )
        construct_success = bool(
            correct
            and closure["event_closed"]
            and not weak_retrieval_evidence
            and observed_clusters >= required_clusters
            and cluster_recall_at_10 >= CONSTRUCT_RECALL_AT_10_THRESHOLD
            and search_surfaced_clusters >= 1
        )
        reformulated, overfetch_attempted = _reformulation_overfetch(ledger)
        shortlist_precision = _shortlist_precision(
            ledger, catalog, eligible_pool, eligible_pids
        )
        precision = (
            float(shortlist_precision) if shortlist_precision is not None else 0.0
        )
        query_efficiency = min(1.0, 3.0 / len(queries)) if queries else 0.0
        recall_precision_f1 = (
            2.0 * cluster_recall_at_10 * precision / (cluster_recall_at_10 + precision)
            if cluster_recall_at_10 + precision > 0.0
            else 0.0
        )
        retrieval_quality = round(recall_precision_f1 * query_efficiency, 6)
        family_metric = (
            retrieval_quality if correct else 0.0 if order is not None else None
        )
        return {
            "construct_success": construct_success,
            "family_metric": family_metric,
            "retrieval_quality": retrieval_quality,
            "query_efficiency": round(query_efficiency, 6),
            "relevance_cluster_count": total_clusters,
            "observed_eligible_clusters": observed_clusters,
            "search_surfaced_clusters": search_surfaced_clusters,
            "cluster_recall_at_10": round(cluster_recall_at_10, 3),
            "weak_retrieval_evidence": weak_retrieval_evidence,
            "cluster_source": cluster_source,
            "min_construct_clusters": _MIN_CONSTRUCT_CLUSTERS,
            "min_observed_clusters": _MIN_OBSERVED_CLUSTERS,
            "required_observed_clusters": required_clusters,
            "candidate_recall": reference_recall,
            "candidate_recall_at_10": candidate_recall_at_10,
            "reference_recall": reference_recall,
            "reference_observed": reference_observed,
            "eligible_recall": eligible_recall,
            "eligible_recall_at_10": eligible_recall_at_10,
            "eligible_observed": eligible_observed,
            "search_result_recall": search_result_recall,
            "recall_per_query": recall_per_query,
            "recall_per_visible_candidate": recall_per_visible_candidate,
            "recall_pool_source": recall_pool_source,
            "recall_pool_size": len(eligible_pool),
            "construct_recall_at_10_threshold": CONSTRUCT_RECALL_AT_10_THRESHOLD,
            "distinct_search_queries": len(queries),
            "shortlist_precision": shortlist_precision,
            "reformulated": reformulated,
            "overfetch_attempted": overfetch_attempted,
            "page_depth": READ_CAP,
            **{k: closure[k] for k in CLOSURE_FIELDS},
        }


__all__ = ["RetrievalRecallFamily"]


####################################################################################################
# oro_env_runtime/families/constraint_satisfaction.py
####################################################################################################

"""Constraint-satisfaction family with SKU-level hard gates."""

from typing import Any

from oro_env_runtime.attributes import resolve_variant_attribute
from oro_env_runtime.catalog import Catalog
from oro_env_runtime.device_semantics import SMARTPHONE_PUBLIC_RULE, smartphone_supported
from oro_env_runtime.schema import REQUIRED_CHECKS, CandidateRef, LedgerEntry, TaskSpec

from .base import (
    CHAIN_CLOSURE_FIELDS,
    CLOSURE_FIELDS,
    Family,
    event_closure,
    first_committed_ref,
    gold_members,
)

AGENT_SYSTEM = (
    "You are a shopping agent. The user gives firm, non-negotiable requirements: a category, a "
    "brand when specified, a specific product spec, "
    "a budget, and in-stock availability. EVERY requirement must hold for the EXACT variant you "
    "order - a product often sells the right spec and the wrong spec under the same title, so "
    "check the specific SKU's option, price, and stock before ordering. "
    "Stock can change while you "
    "shop; if the variant you wanted sells out, do NOT settle for a different variant of the same "
    "product that misses a requirement - find another product that meets all of them. When you "
    "need input or want to recommend something, send a short message to the user. Finish by "
    "placing the order for a valid, in-stock item that meets every requirement within budget. Once "
    "verified, place the order - do not wait for permission or further confirmation."
) + " " + SMARTPHONE_PUBLIC_RULE


def _options_value(options: str | None, key: str) -> str | None:
    """Return a variant value from SKU options only."""
    return resolve_variant_attribute({}, options, key)


def _find_event(ledger: list[LedgerEntry]) -> tuple[dict | None, int | None]:
    for e in ledger:
        if e.kind == "harness_event":
            return e.payload, e.seq
    return None, None


def _display_value(key: str, value: str) -> str:
    if key == "ram":
        return value.upper()
    if key == "refresh_rate":
        return value[:-2] + "Hz"
    if key == "screen_size":
        return value.replace("in", "-inch")
    if key == "capacity":
        return value.removesuffix("ml") + " mL"
    if key == "wattage":
        return value.removesuffix("w") + " W"
    if key == "warranty_duration":
        months = value.removesuffix("mo")
        return f"{months}-month"
    if key == "pack_quantity":
        count = value.removesuffix("ct")
        return f"{count}-piece"
    return {
        "4k": "4K",
        "qhd": "QHD",
        "fhd": "1080p Full HD",
        "hd": "720p HD",
    }.get(value, value.upper())


def _spec_phrase(key: str, value: str) -> str:
    d = _display_value(key, value)
    return {
        "ram": f"{d} of RAM",
        "resolution": f"a {d} display",
        "screen_size": f"a {d} screen",
        "refresh_rate": f"a {d} refresh rate",
        "capacity": f"a {d} capacity",
        "wattage": f"{d} power",
        "warranty_duration": f"a {d} warranty",
        "pack_quantity": f"a {d} pack",
    }[key]


def _cat_word(category: str) -> str:
    low = category.lower()
    return low[:-1] if low.endswith("s") else low


class ConstraintSatisfactionFamily(Family):
    name = "constraint_satisfaction"
    agent_system = AGENT_SYSTEM
    metric_name = "constraints_satisfied"

    @staticmethod
    def genuine_device_satisfied(task: TaskSpec, record: dict, options: str | None = None) -> bool:
        # This family-scoped safeguard must not change other categories' gates.
        return ((task.hard.category or "").casefold().rstrip("s") != "smartphone"
                or smartphone_supported(record, options=options))

    def configure_environment(self, env, task: TaskSpec) -> None:
        return None

    def on_write(self, env, ref: CandidateRef) -> bool:
        return False

    def validate_event_rule(self, task: TaskSpec, catalog: Catalog) -> bool:
        if task.event_rule is None:
            return True
        # Every chained event is the sealed recovery-commitment stockout.
        if any(
            rule.kind != "stockout" or rule.trigger != "second_successful_cart_add"
            for rule in task.event_rules
        ):
            return False
        return (
            task.event_rule.kind == "stockout"
            and task.event_rule.price_multiplier is None
        )

    def validate_event_commitment(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef | None
    ) -> bool | None:
        if task.event_rule is None:
            return None
        if ref is None or not catalog.exists(ref):
            return False
        key = task.family_payload["gated_key"]
        return bool(
            ref in gold_members(task)
            and self.genuine_device_satisfied(
                task, catalog.by_id(ref.product_id), catalog.meta(ref).options
            )
            and resolve_variant_attribute(
                catalog.by_id(ref.product_id), catalog.meta(ref).options, key
            )
            == task.family_payload["gated_value"]
        )

    def user_sim_context(self, task: TaskSpec) -> str | dict | None:
        fp = task.family_payload
        item = (
            f"a {fp['brand']} {_cat_word(fp['category'])}"
            if fp.get("brand")
            else f"a {_cat_word(fp['category'])}"
        )
        return {
            "hard_facts": [
                item,
                *(
                    []
                    if fp.get("underspecified")
                    else [f"{_spec_phrase(fp['gated_key'], fp['gated_value'])} on the exact variant you buy"]
                ),
                f"in stock, at or under {task.hard.budget:.2f} {task.hard.currency}",
            ],
            # Underspecified tasks: the firm spec is sealed and revealed only on a direct question.
            "reveal_only_if_asked": (
                [f"{_spec_phrase(fp['gated_key'], fp['gated_value'])} on the exact variant you buy"]
                if fp.get("underspecified")
                else []
            ),
            "unacceptable_shortcuts": [
                "an item that misses any one requirement (a near-miss); you would rather keep looking than settle",
            ],
        }

    def user_sim_allow_pushback(self, task: TaskSpec) -> bool:
        return False

    def verify_extra(
        self,
        task: TaskSpec,
        ledger: list[LedgerEntry],
        catalog: Catalog,
        checks: dict[str, bool | None],
        order: dict | None,
        order_seq: int | None,
        *,
        observed: list[tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        key, wanted = (
            task.family_payload["gated_key"],
            task.family_payload["gated_value"],
        )
        spec_satisfied: bool | None = None
        genuine_device_satisfied: bool | None = None
        reorder_same_product: bool | None = None
        moved_to_new_product: bool | None = None
        if order is not None:
            tgt = order["target"]
            ref = CandidateRef(product_id=tgt["product_id"], sku=tgt["sku"])
            if catalog.exists(ref):
                genuine_device_satisfied = self.genuine_device_satisfied(
                    task, catalog.by_id(ref.product_id), catalog.meta(ref).options
                )
                got = resolve_variant_attribute(
                    catalog.by_id(ref.product_id), catalog.meta(ref).options, key
                )
                spec_satisfied = got == wanted
            else:
                spec_satisfied = False
                genuine_device_satisfied = False
            event, _ = _find_event(ledger)
            if event is not None:
                ev = event["target"]
                reorder_same_product = (
                    tgt["product_id"] == ev["product_id"] and tgt["sku"] != ev["sku"]
                )
                moved_to_new_product = tgt["product_id"] != ev["product_id"]
        correct = all(checks.get(k) is True for k in REQUIRED_CHECKS)
        checks["spec_satisfied"] = spec_satisfied
        checks["genuine_device_satisfied"] = genuine_device_satisfied
        family_gate = spec_satisfied is True and genuine_device_satisfied is True
        closure = event_closure(
            task,
            ledger,
            checks,
            order,
            order_seq,
            intended_event_commitment=self.validate_event_commitment(
                task, catalog, first_committed_ref(ledger)
            ),
        )
        return {
            "construct_success": bool(
                correct and family_gate and closure["event_closed"]
            ),
            "family_gate": family_gate,
            "family_metric": 1.0 if correct and family_gate else 0.0,
            "spec_satisfied": spec_satisfied,
            "genuine_device_satisfied": genuine_device_satisfied,
            "gated_key": key,
            "gated_value": wanted,
            "within_product_trap": task.family_payload.get(
                "within_product_trap", False
            ),
            "reorder_same_product": reorder_same_product,
            "moved_to_new_product": moved_to_new_product,
            **{k: closure[k] for k in CLOSURE_FIELDS},
            # Chain counts are stamped only for chained tasks, so single-event verdicts
            # keep their exact family_metrics shape.
            **(
                {k: closure[k] for k in CHAIN_CLOSURE_FIELDS}
                if task.event_rules
                else {}
            ),
        }


__all__ = ["ConstraintSatisfactionFamily"]


####################################################################################################
# oro_env_runtime/families/ranking.py
####################################################################################################

"""Ranking/tradeoff family."""

import weakref
from dataclasses import dataclass
from typing import Any

from oro_env_runtime.attributes import resolve_attribute, resolve_variant_attribute
from oro_env_runtime.catalog import Catalog
from oro_env_runtime.observations import visible_candidate_pairs_for_actions
from oro_env_runtime.schema import REQUIRED_CHECKS, CandidateRef, LedgerEntry, TaskSpec

from .base import (
    Family,
    event_observed_or_signaled as _event_observed_or_signaled,
    first_committed_key as _first_add_key,
)
from .constraint_satisfaction import _cat_word

AGENT_SYSTEM = (
    "You are a shopping agent. The user names a product category and has a stated tradeoff "
    "priority: choose the highest advertised maximum refresh rate, including an explicitly "
    "advertised overclock mode; among ties, choose the largest screen; among "
    "remaining ties, choose the lowest price. Inspect product facts and variant options for "
    "refresh rate, screen size, price, and stock; do not choose an item that is strictly worse "
    "than another available option on every tradeoff axis. View or compare the best initial "
    "candidate's facts before adding it to the cart; the first cart choice must be the "
    "best-ranked available option, not a placeholder. Use an explicit selected-variant "
    "specification before a product-wide value. Do not assume a multi-model listing's "
    "maximum specification belongs to a different selected model. Prices or stock can change while "
    "you shop, so if the market changes after your first cart action, re-rank the live options "
    "before ordering, and inspect the replacement's current price and stock after the change. "
    "Compare a focused shortlist rather than inspecting every option. Once "
    "you have verified your best-ranked valid option's current price and stock, place the "
    "order - do not wait for permission or further confirmation."
)

_CATEGORY = (
    "Gaming Monitors"  # frozen-trace default; new tasks sample _CATEGORY_PROFILES
)
_TRADEOFF_CACHE: weakref.WeakKeyDictionary[
    Catalog, dict[tuple[str, float | None], TradeoffValues | None]
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class TradeoffValues:
    refresh_hz: float
    screen_size_in: float
    price: float


@dataclass(frozen=True)
class RankingScore:
    score: float
    distance: float
    components: dict[str, float]


def _as_float(value: str | None, suffix: str) -> float | None:
    if not value or not value.endswith(suffix):
        return None
    try:
        return float(value[: -len(suffix)])
    except ValueError:
        return None


def tradeoff_values(
    catalog: Catalog, ref: CandidateRef, *, price: float | None = None
) -> TradeoffValues | None:
    cache = _TRADEOFF_CACHE.setdefault(catalog, {})
    cache_key = (ref.key(), round(float(price), 2) if price is not None else None)
    if cache_key in cache:
        return cache[cache_key]
    if not catalog.exists(ref):
        cache[cache_key] = None
        return None
    rec = catalog.by_id(ref.product_id)
    meta = catalog.meta(ref)
    refresh = resolve_variant_attribute(
        rec, meta.options, "refresh_rate"
    ) or resolve_attribute(rec, "refresh_rate")
    size = resolve_variant_attribute(
        rec, meta.options, "screen_size"
    ) or resolve_attribute(rec, "screen_size")
    refresh_hz = _as_float(refresh, "hz")
    screen_size = _as_float(size, "in")
    if refresh_hz is None or screen_size is None:
        cache[cache_key] = None
        return None
    values = TradeoffValues(
        refresh_hz=refresh_hz,
        screen_size_in=screen_size,
        price=float(meta.price if price is None else price),
    )
    cache[cache_key] = values
    return values


def _price_override(
    price_overrides: dict[str, float] | None, ref: CandidateRef
) -> float | None:
    return (price_overrides or {}).get(ref.key())


def _ref_from_key(key: str | None) -> CandidateRef | None:
    if not key or "::" not in key:
        return None
    product_id, sku = key.split("::", 1)
    return CandidateRef(product_id=product_id, sku=sku)


def _dominates(
    catalog: Catalog,
    challenger: CandidateRef,
    candidate: CandidateRef,
    *,
    price_overrides: dict[str, float] | None = None,
) -> bool:
    a = tradeoff_values(
        catalog, challenger, price=_price_override(price_overrides, challenger)
    )
    b = tradeoff_values(
        catalog, candidate, price=_price_override(price_overrides, candidate)
    )
    if a is None or b is None:
        return False
    no_worse = (
        a.refresh_hz >= b.refresh_hz
        and a.screen_size_in >= b.screen_size_in
        and a.price <= b.price + 0.01
    )
    strictly_better = (
        a.refresh_hz > b.refresh_hz
        or a.screen_size_in > b.screen_size_in
        or a.price < b.price - 0.01
    )
    return no_worse and strictly_better


def dominated_by(
    catalog: Catalog,
    ref: CandidateRef,
    pool: list[CandidateRef],
    *,
    price_overrides: dict[str, float] | None = None,
) -> CandidateRef | None:
    scores = _ranking_scores(catalog, pool, price_overrides=price_overrides)
    dominators = [
        other
        for other in pool
        if other.key() != ref.key()
        and _dominates(catalog, other, ref, price_overrides=price_overrides)
    ]
    if not dominators:
        return None
    dominators.sort(key=lambda item: (-scores[item.key()].score, item.key()))
    return dominators[0]


def frontier_refs(
    catalog: Catalog,
    pool: list[CandidateRef],
    *,
    price_overrides: dict[str, float] | None = None,
) -> list[CandidateRef]:
    values = {
        ref.key(): tradeoff_values(
            catalog, ref, price=_price_override(price_overrides, ref)
        )
        for ref in pool
    }
    scores = _ranking_scores(catalog, pool, price_overrides=price_overrides)
    frontier = [
        ref
        for ref in pool
        if values[ref.key()] is not None
        and not any(
            other.key() != ref.key()
            and values[other.key()] is not None
            and _dominates_values(values[other.key()], values[ref.key()])  # type: ignore[arg-type]
            for other in pool
        )
    ]
    frontier.sort(key=lambda ref: (-scores[ref.key()].score, ref.key()))
    return frontier


def _dominates_values(challenger: TradeoffValues, candidate: TradeoffValues) -> bool:
    no_worse = (
        challenger.refresh_hz >= candidate.refresh_hz
        and challenger.screen_size_in >= candidate.screen_size_in
        and challenger.price <= candidate.price + 0.01
    )
    strictly_better = (
        challenger.refresh_hz > candidate.refresh_hz
        or challenger.screen_size_in > candidate.screen_size_in
        or challenger.price < candidate.price - 0.01
    )
    return no_worse and strictly_better


def _norm_high(value: float, values: list[float]) -> float:
    lo, hi = min(values), max(values)
    return 1.0 if hi == lo else (value - lo) / (hi - lo)


def _norm_price(value: float, values: list[float]) -> float:
    lo, hi = min(values), max(values)
    return 1.0 if hi == lo else (hi - value) / (hi - lo)


def ranking_score(
    catalog: Catalog,
    ref: CandidateRef,
    pool: list[CandidateRef],
    *,
    price_overrides: dict[str, float] | None = None,
) -> RankingScore:
    return _ranking_scores(catalog, pool, price_overrides=price_overrides).get(
        ref.key(), RankingScore(score=0.0, distance=1.0, components={})
    )


def _ranking_scores(
    catalog: Catalog,
    pool: list[CandidateRef],
    *,
    price_overrides: dict[str, float] | None = None,
) -> dict[str, RankingScore]:
    chosen_by_key = {
        item.key(): tradeoff_values(
            catalog, item, price=_price_override(price_overrides, item)
        )
        for item in pool
    }
    vals = [value for value in chosen_by_key.values() if value is not None]
    if not vals:
        return {}
    refreshes = [v.refresh_hz for v in vals]
    sizes = [v.screen_size_in for v in vals]
    prices = [v.price for v in vals]
    ordered_vectors = sorted(
        {(-value.refresh_hz, -value.screen_size_in, value.price) for value in vals}
    )
    denominator = max(1, len(ordered_vectors) - 1)
    scores: dict[str, RankingScore] = {}
    for key, chosen in chosen_by_key.items():
        if chosen is None:
            continue
        components = {
            "refresh_rate": round(_norm_high(chosen.refresh_hz, refreshes), 3),
            "screen_size": round(_norm_high(chosen.screen_size_in, sizes), 3),
            "price": round(_norm_price(chosen.price, prices), 3),
        }
        vector = (-chosen.refresh_hz, -chosen.screen_size_in, chosen.price)
        rank_index = ordered_vectors.index(vector)
        score = round(1.0 - rank_index / denominator, 3)
        scores[key] = RankingScore(
            score=score,
            distance=round(1.0 - score, 3),
            components=components,
        )
    return scores


def _ranked_pool(
    catalog: Catalog,
    pool: list[CandidateRef],
    *,
    score_pool: list[CandidateRef] | None = None,
    price_overrides: dict[str, float] | None = None,
) -> list[CandidateRef]:
    values = {
        ref.key(): tradeoff_values(
            catalog, ref, price=_price_override(price_overrides, ref)
        )
        for ref in pool
    }
    return sorted(
        pool,
        key=lambda ref: (
            -values[ref.key()].refresh_hz,  # type: ignore[union-attr]
            -values[ref.key()].screen_size_in,  # type: ignore[union-attr]
            values[ref.key()].price,  # type: ignore[union-attr]
            ref.key(),
        ),
    )


def tie_class_keys(
    catalog: Catalog, pool: list[CandidateRef], ref: CandidateRef
) -> list[str]:
    """Keys in ``pool`` the stated ranking rule cannot tell apart from ``ref``.

    The rule is lexicographic over (refresh, size, price), so two candidates with an
    identical vector are the same answer. They must be invalidated together and accepted
    together, or the event proves nothing and the verifier picks a winner by key order.
    """

    values = tradeoff_values(catalog, ref)
    if values is None:
        return [ref.key()]
    keys = {
        item.key() for item in pool if tradeoff_values(catalog, item) == values
    }
    keys.add(ref.key())
    return sorted(keys)


def _event_state(ledger: list[LedgerEntry]) -> tuple[set[str], dict[str, float]]:
    price_overrides: dict[str, float] = {}
    excluded: set[str] = set()
    for entry in ledger:
        if entry.kind != "harness_event":
            continue
        target = entry.payload.get("target") or {}
        if not target.get("product_id") or not target.get("sku"):
            continue
        # A stockout invalidates every key the delta covered, which is the committed ref plus
        # the rest of its tie class when the family widened the event.
        keys = [str(key) for key in (entry.payload.get("targets") or [])] or [
            f"{target['product_id']}::{target['sku']}"
        ]
        if entry.payload.get("kind") == "stockout":
            excluded.update(keys)
        elif (
            entry.payload.get("kind") == "price_change"
            and entry.payload.get("new_price") is not None
        ):
            for key in keys:
                price_overrides[key] = float(entry.payload["new_price"])
    return excluded, price_overrides


def true_state_pool(
    task: TaskSpec, catalog: Catalog, ledger: list[LedgerEntry]
) -> tuple[list[CandidateRef], dict[str, float]]:
    excluded, price_overrides = _event_state(ledger)
    pool: list[CandidateRef] = []
    for ref in task.gold_set:
        if ref.key() in excluded:
            continue
        values = tradeoff_values(
            catalog, ref, price=_price_override(price_overrides, ref)
        )
        if values is None or values.price > task.hard.budget:
            continue
        pool.append(ref)
    return pool, price_overrides


class RankingFamily(Family):
    name = "ranking"
    agent_system = AGENT_SYSTEM
    metric_name = "ranking_priority_distance"

    def configure_environment(self, env, task: TaskSpec) -> None:
        return None

    def on_write(self, env, ref: CandidateRef) -> bool:
        return False

    def validate_event_rule(self, task: TaskSpec, catalog: Catalog) -> bool:
        if task.event_rule is None:
            return True
        return (
            task.event_rule.kind == "stockout"
            and task.event_rule.price_multiplier is None
        )

    def validate_event_commitment(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef | None
    ) -> bool | None:
        if task.event_rule is None:
            return None
        return bool(
            ref is not None
            and ref.key() == task.family_payload.get("pre_event_top_key")
        )

    def event_targets(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef
    ) -> list[str]:
        # The class is sealed at compile time. Recomputing it here would make event execution
        # read the evaluator's private gold set, which noninterference forbids, and it would
        # let the width of the delta depend on state the policy cannot see. A commitment that
        # is not the sealed pre-event top gets the plain single-key delta.
        if ref.key() != task.family_payload.get("pre_event_top_key"):
            return [ref.key()]
        sealed = [
            str(key) for key in (task.family_payload.get("pre_event_top_class") or [])
        ]
        return sealed or [ref.key()]

    def user_sim_context(self, task: TaskSpec) -> str | dict | None:
        return {
            "hard_facts": [
                f"an in-stock {_cat_word(str(task.hard.category or _CATEGORY))} at or under {task.hard.budget:.2f} {task.hard.currency}",
            ],
            "tradeoff_priority": [
                "choose the highest advertised maximum refresh rate, including an explicitly advertised overclock mode",
                "among ties, choose the largest screen",
                "among remaining ties, choose the lowest price",
            ],
            "unacceptable_shortcuts": [
                "an option that is strictly worse than another available one on every tradeoff axis",
            ],
        }

    def user_sim_allow_pushback(self, task: TaskSpec) -> bool:
        return True

    def verify_extra(
        self,
        task: TaskSpec,
        ledger: list[LedgerEntry],
        catalog: Catalog,
        checks: dict[str, bool | None],
        order: dict | None,
        order_seq: int | None,
        *,
        observed: list[tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        pool, price_overrides = true_state_pool(task, catalog, ledger)
        frontier = frontier_refs(catalog, pool, price_overrides=price_overrides)
        chosen_key: str | None = None
        chosen_values: TradeoffValues | None = None
        chosen_score: RankingScore | None = None
        chosen_rank: int | None = None
        gate = None
        dominated_ref: CandidateRef | None = None
        if order is not None:
            target = order["target"]
            ref = CandidateRef(product_id=target["product_id"], sku=target["sku"])
            chosen_key = ref.key()
            chosen_values = tradeoff_values(
                catalog, ref, price=_price_override(price_overrides, ref)
            )
            chosen_score = ranking_score(
                catalog, ref, pool, price_overrides=price_overrides
            )
            on_frontier = chosen_key in {item.key() for item in frontier}
            if not on_frontier and ref in pool:
                dominated_ref = dominated_by(
                    catalog, ref, pool, price_overrides=price_overrides
                )
            ranked = (
                _ranked_pool(catalog, pool, price_overrides=price_overrides)
                if pool
                else []
            )
            for idx, item in enumerate(ranked, start=1):
                if item.key() == chosen_key:
                    chosen_rank = idx
                    break
            # Rank 1 inside a tie class is decided by key order, which the task's own rule
            # never mentions. Gate on the class instead: a candidate whose live
            # (refresh, size, price) vector equals the top's is the same answer.
            top_values = (
                tradeoff_values(
                    catalog, ranked[0], price=_price_override(price_overrides, ranked[0])
                )
                if ranked
                else None
            )
            gate = bool(
                chosen_rank is not None
                and chosen_values is not None
                and chosen_values == top_values
            )

        best_score = max(
            (
                ranking_score(catalog, ref, pool, price_overrides=price_overrides).score
                for ref in frontier
            ),
            default=None,
        )
        base_correct = all(checks.get(key) is True for key in REQUIRED_CHECKS)
        on_frontier = (
            chosen_key in {item.key() for item in frontier}
            if chosen_key is not None
            else False
        )
        event_seq = next(
            (entry.seq for entry in ledger if entry.kind == "harness_event"), None
        )
        pre_event_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare"}),
            before_seq=event_seq,
        )
        pre_event_keys = {key for _, key in pre_event_pairs}
        comparison_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare"}),
            before_seq=order_seq,
        )
        comparison_keys = {key for _, key in comparison_pairs}
        accepted_keys = set(task.acceptance.acceptable_keys)
        observed_eligible_keys = comparison_keys & accepted_keys
        observed_eligible_products = {
            key.split("::", 1)[0] for key in observed_eligible_keys
        }
        state_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare", "inspect_stock"}),
            after_seq=event_seq,
            before_seq=order_seq,
        )
        state_keys = {key for _, key in state_pairs}
        pre_event_top_key = task.family_payload.get("pre_event_top_key")
        post_event_top_key = task.family_payload.get("post_event_top_key")
        # The sealed class is the set of post-event answers the stated rule cannot separate;
        # post_event_top_key stays its canonical member for reporting and for the harness.
        post_event_top_class = {
            str(key) for key in (task.family_payload.get("post_event_top_class") or [])
        }
        if post_event_top_key is not None:
            post_event_top_class.add(str(post_event_top_key))
        pre_event_top_observed = pre_event_top_key in pre_event_keys
        post_event_top_observed = bool(state_keys & post_event_top_class)
        chose_post_event_top = (
            chosen_key is not None and chosen_key in post_event_top_class
        )
        ranking_evidence_grounded = bool(
            pre_event_top_observed
            and chose_post_event_top
            and post_event_top_observed
        )
        metric = (
            chosen_score.score
            if base_correct
            and gate is True
            and ranking_evidence_grounded
            and chosen_score is not None
            else 0.0
            if order is not None
            else None
        )
        event_fired = any(entry.kind == "harness_event" for entry in ledger)
        event_noticed_or_self_detected = _event_observed_or_signaled(ledger, order_seq)
        first_committed_key = _first_add_key(ledger)
        intended_event_commitment = bool(
            event_fired
            and first_committed_key is not None
            and first_committed_key == pre_event_top_key
        )
        # A wrong first rank choice fails the commitment gate. Mark an exploit only when the
        # policy changes its choice without observing the event that caused the rerank.
        sacrificial_exploit = bool(
            event_fired
            and chosen_key is not None
            and first_committed_key is not None
            and chosen_key != first_committed_key
            and not event_noticed_or_self_detected
        )
        ranking_aligned_success = bool(
            base_correct and gate is True and ranking_evidence_grounded
        )
        event_visible_to_policy = event_noticed_or_self_detected
        cart_state_uptake = bool(
            event_fired
            and event_visible_to_policy
            and intended_event_commitment
            and chosen_key is not None
            and first_committed_key != chosen_key
        )
        market_rerank_after_event = bool(
            ranking_aligned_success and chose_post_event_top
        )
        stateful_recovery_success = bool(
            cart_state_uptake and market_rerank_after_event
        )
        static_rank1_selection = bool(
            ranking_aligned_success and not stateful_recovery_success
        )
        return {
            "family_gate": gate,
            "construct_success": stateful_recovery_success,
            "family_metric": metric,
            "ranking_aligned_success": ranking_aligned_success,
            "stateful_recovery_success": stateful_recovery_success,
            "static_rank1_selection": static_rank1_selection,
            "cart_state_uptake": cart_state_uptake,
            "market_rerank_after_event": market_rerank_after_event,
            "ranking_evidence_grounded": ranking_evidence_grounded,
            "observed_eligible_products": len(observed_eligible_products),
            "pre_event_top_observed": pre_event_top_observed,
            "post_event_top_observed": post_event_top_observed,
            "event_visible_to_policy": event_visible_to_policy,
            "intended_event_commitment": intended_event_commitment,
            "sacrificial_exploit": sacrificial_exploit,
            "selected_key": chosen_key,
            "pre_event_top_key": pre_event_top_key,
            "post_event_top_key": post_event_top_key,
            "dominated_by": dominated_ref.key() if dominated_ref is not None else None,
            "event_noticed_or_self_detected": event_noticed_or_self_detected,
            "ranking_detail": {
                "chosen_key": chosen_key,
                "selected_key": chosen_key,
                "first_committed_key": first_committed_key,
                "pre_event_top_key": pre_event_top_key,
                "post_event_top_key": post_event_top_key,
                "post_event_top_class": sorted(post_event_top_class),
                "chosen_values": chosen_values.__dict__
                if chosen_values is not None
                else None,
                "chosen_priority_score": chosen_score.score
                if chosen_score is not None
                else None,
                "chosen_priority_distance": chosen_score.distance
                if chosen_score is not None
                else None,
                "chosen_components": chosen_score.components
                if chosen_score is not None
                else None,
                "best_frontier_score": best_score,
                "priority_regret": round(
                    (best_score or 0.0) - (chosen_score.score if chosen_score else 0.0),
                    3,
                )
                if best_score is not None and chosen_score is not None
                else None,
                "ordered_true_state_rank": chosen_rank,
                "not_dominated": on_frontier,
                "top_ranked_frontier": bool(on_frontier and chosen_rank == 1)
                if chosen_rank is not None
                else None,
                "dominated_by": dominated_ref.key()
                if dominated_ref is not None
                else None,
                "cart_state_uptake": cart_state_uptake,
                "market_rerank_after_event": market_rerank_after_event,
                "event_visible_to_policy": event_visible_to_policy,
                "current_frontier_count": len(frontier),
                "current_frontier_keys": [ref.key() for ref in frontier],
                "priority": ["refresh_rate", "screen_size", "price"],
                "score_hidden_from_solver": False,
            },
            "eligible_count": task.family_payload.get("eligible_count"),
            "frontier_count": task.family_payload.get("frontier_count"),
            "dominated_count": task.family_payload.get("dominated_count"),
            "difficulty": task.family_payload.get("difficulty"),
        }


__all__ = ["RankingFamily"]


####################################################################################################
# oro_env_runtime/families/recovery.py
####################################################################################################

"""Recovery-by-relaxation family."""

import re
import weakref
from dataclasses import dataclass
from typing import Any, Callable

from oro_env_runtime.acceptance import (
    category_matches,
)
from oro_env_runtime.catalog import CandidateMeta, Catalog
from oro_env_runtime.device_semantics import (
    SMARTPHONE_PUBLIC_RULE,
    smartphone_supported,
)
from oro_env_runtime.observations import visible_candidate_pairs_for_actions
from oro_env_runtime.product_facts import visible_product_facts
from oro_env_runtime.schema import CandidateRef, LedgerEntry, TaskSpec

from .base import CLOSURE_FIELDS, Family, event_closure, first_committed_ref

AGENT_SYSTEM = (
    "You are a shopping agent. The shopper's critical constraints are category, budget, and "
    "in-stock availability; never relax those. The shopper may also name preferred model and color. "
    "Those preferences are relaxable only after the exact item becomes unavailable: first try the "
    "same model in another color, and only then consider changing models. Prices or stock can "
    "change while you shop, so if the exact item disappears after a cart action, recover by finding "
    "the least-relaxed in-stock option within budget before ordering. "
    "Judge a listing's model and color from the selected variant's own product facts, not from "
    "the title or a category label alone; when the evidence is missing, contradictory, or "
    "ambiguous, treat the identity as unresolved and choose an option that is not. "
    "Any eligible listing matching the preferred model and color is a valid first choice. "
    "Inspect that actual choice before adding it, and inspect the replacement's identity "
    "and current stock after the event."
    " " + SMARTPHONE_PUBLIC_RULE
)

_INVALID_TITLE_PREFIXES = ("customer ratings", "customer reviews")
_INVALID_TITLE_FRAGMENTS = (
    "join bild expo",
    "earbud",
    "in-ear",
    "in ear",
    "earphone",
    "open-ear",
    "true wireless",
)

_COLOR_PHRASES = (
    "midnight blue",
    "titanium black",
    "matte black",
    "triple black",
    "frost white",
    "white smoke",
    "cloud grey",
    "moonlight grey",
    "space gray",
    "street gray",
    "driftwood sand",
    "cypress green",
    "tidal blue",
    "light blue",
    "sky blue",
    "indigo blue",
    "canvas white",
    "platinum silver",
    "starlight",
    "graphite",
    "sandstone",
    "transparent",
    "silver",
    "beige",
    "black",
    "blue",
    "white",
    "gray",
    "grey",
    "green",
    "pink",
    "violet",
    "purple",
    "red",
    "sage",
    "gold",
    "khaki",
    "peach",
)

_MODEL_PATTERNS: tuple[tuple[str, Any], ...] = (
    (r"wh[-\s]?1000xm\s*([456])", lambda m: f"WH-1000XM{m.group(1)}"),
    (r"wf[-\s]?1000xm\s*([45])", lambda m: f"WF-1000XM{m.group(1)}"),
    (
        r"quietcomfort ultra(?: headphones)?[^\n]{0,80}\b2nd gen\b",
        lambda _m: "QuietComfort Ultra 2nd Gen",
    ),
    (r"quietcomfort ultra", lambda _m: "QuietComfort Ultra"),
    (r"quietcomfort(?: headphones| wireless)?", lambda _m: "QuietComfort"),
    (r"headphones 700", lambda _m: "Headphones 700"),
    (r"beats studio pro|studio pro", lambda _m: "Studio Pro"),
    (r"beats studio buds \+|studio buds \+", lambda _m: "Studio Buds Plus"),
    (r"beats studio buds|studio buds", lambda _m: "Studio Buds"),
    (r"beats fit pro", lambda _m: "Fit Pro"),
    (r"momentum true wireless 4|mtw4|momentum 4", lambda _m: "Momentum 4"),
    (r"accentum", lambda _m: "Accentum"),
    (r"px7\s*s2e", lambda _m: "Px7 S2e"),
    (r"px7\s*s3", lambda _m: "Px7 S3"),
    (r"elite 10", lambda _m: "Elite 10"),
    (r"jbuds lux anc|go lux", lambda _m: "JBuds Lux ANC"),
)


@dataclass(frozen=True)
class RecoveryCandidate:
    ref: CandidateRef
    price: float
    model: str
    color: str
    title: str
    category: str = "Headphones"


_ROW_CACHE: "weakref.WeakKeyDictionary[Catalog, tuple[RecoveryCandidate, ...]]" = (
    weakref.WeakKeyDictionary()
)
_ROW_MAP_CACHE: "weakref.WeakKeyDictionary[Catalog, dict[str, RecoveryCandidate]]" = (
    weakref.WeakKeyDictionary()
)


def _phrase_in(text: str, phrase: str) -> bool:
    words = re.findall(r"[a-z0-9]+", phrase.casefold())
    if not words:
        return False
    pattern = (
        r"(?<![a-z0-9])" + r"\W+".join(re.escape(w) for w in words) + r"(?![a-z0-9])"
    )
    return re.search(pattern, text.casefold()) is not None


# Accessories catalogued under the headphones path are not headphones (an AirPods case was
# sealed as a recovery gold family on the corrected pack). Whole-word so "standard" survives.
_ACCESSORY_RE = re.compile(
    r"\b(?:case|cases|cover|covers|sleeve|sleeves|skin|skins|pouch|protector|cable|cables|"
    r"strap|straps|ear ?tips?|ear ?hooks?|adapter|holder|stand|replacement|carabiner)\b"
)
_OVER_OR_ON_EAR_RE = re.compile(r"\b(?:over|on)[-\s]?ear\b", re.IGNORECASE)

# Mirrors oro_env_gen.families._shared.is_requested_device's feature-phone gate. Duplicated
# (not imported) because oro-env-runtime is the lower layer: oro_env_gen depends on it, never
# the reverse, and this rule is small and self-contained enough that duplication beats a new
# cross-package edge.
_FEATURE_PHONE_RE = re.compile(r"\b(?:feature|basic|dumb|button|keypad) ?phones?\b")


def _valid_title(title: str) -> bool:
    low = title.casefold().strip()
    return (
        bool(low)
        and not low.startswith(_INVALID_TITLE_PREFIXES)
        and not any(fragment in low for fragment in _INVALID_TITLE_FRAGMENTS)
        and not _ACCESSORY_RE.search(low)
    )


def _valid_smartphone_title(title: str) -> bool:
    low = title.casefold().strip()
    return (
        bool(low)
        and not low.startswith(_INVALID_TITLE_PREFIXES)
        and not _ACCESSORY_RE.search(low)
        and _FEATURE_PHONE_RE.search(low) is None
    )


def _visible_form_factor(rec: dict, options: str | None) -> bool:
    """Return whether a public read states the required headphone form."""
    surface = " ".join((str(rec.get("title") or ""), str(options or "")))
    return _OVER_OR_ON_EAR_RE.search(surface) is not None


def _no_extra_gate(_rec: dict, _options: str | None) -> bool:
    return True


@dataclass(frozen=True)
class CategoryProfile:
    """Per-category rules for recovery candidate admission and wording.

    ``device_word`` is the bare noun used in search queries and the "genuine X" goal clause;
    callers that need an article (goal text, the budget-pushback utterance) branch on it
    themselves since "headphones" takes none and "smartphone" does.
    """

    category: str
    device_word: str
    max_budget: float | None
    valid_title: Callable[[str], bool]
    extra_gate: Callable[[dict, str | None], bool]


_HEADPHONES_PROFILE = CategoryProfile(
    category="Headphones",
    device_word="headphones",
    max_budget=500.0,
    valid_title=_valid_title,
    extra_gate=_visible_form_factor,
)
_SMARTPHONES_PROFILE = CategoryProfile(
    category="Smartphones",
    device_word="smartphone",
    max_budget=None,
    valid_title=_valid_smartphone_title,
    extra_gate=_no_extra_gate,
)
# Fixed enumeration order: a compile's seed indexes into the union of both categories' plans
# (see oro_env_gen.families.recovery._build_plans), so this order is part of the seed contract.
_CATEGORY_PROFILES: tuple[CategoryProfile, ...] = (
    _HEADPHONES_PROFILE,
    _SMARTPHONES_PROFILE,
)
_CATEGORY_PROFILE_BY_NAME = {
    profile.category: profile for profile in _CATEGORY_PROFILES
}


def profile_for_category(category: str | None) -> CategoryProfile:
    """Look up a category's profile, defaulting to Headphones for unknown/missing categories."""
    return _CATEGORY_PROFILE_BY_NAME.get(category or "", _HEADPHONES_PROFILE)


def _color_label(rec: dict, options: str | None) -> str | None:
    facts = visible_product_facts(rec)
    for text in (options or "", facts.get("color") or "", str(rec.get("title") or "")):
        matches: list[tuple[int, int, str]] = []
        for color in (*_COLOR_PHRASES, "sage green"):
            pattern = (
                r"(?<![a-z0-9])"
                + r"\W+".join(re.escape(word) for word in color.split())
                + r"(?![a-z0-9])"
            )
            matches.extend(
                (
                    match.start(),
                    match.end(),
                    "green" if color == "sage green" else color,
                )
                for match in re.finditer(pattern, text.casefold())
            )
        if not matches:
            continue
        # A longest named shade owns its span: 'midnight blue' does not also
        # contribute a separate 'blue'. Disjoint colors remain ambiguous.
        longest = [
            match
            for match in matches
            if not any(
                other[0] <= match[0]
                and other[1] >= match[1]
                and (other[1] - other[0]) > (match[1] - match[0])
                for other in matches
            )
        ]
        labels = {match[2] for match in longest}
        return next(iter(labels)) if len(labels) == 1 else None
    return None


_TITLE_MODEL_CODE = re.compile(
    r"\b[a-z]*\d{2,5}[a-z0-9]*(?:-[a-z0-9]+)*\b", re.IGNORECASE
)
_NON_MODEL_CODE = re.compile(
    r"^(?:android|bluetooth|wifi|ips|ddr|lpddr|mtk)|"
    r"(?:gb|tb|mb|mah|hz|khz|mhz|ghz|mp|fps|inch|inches|p|g)$|^\d+x\d+$",
    re.IGNORECASE,
)


def _model_code(raw: str) -> str | None:
    """Normalize a stated model, never silently dropping a variant suffix."""
    words = re.findall(r"[a-z0-9]+", raw.casefold())
    while words and words[-1] in _COLOR_PHRASES:
        words.pop()
    code = "".join(words)
    return (
        code.upper()
        if 3 <= len(code) <= 24 and any(c.isdigit() for c in code)
        else None
    )


def _named_model(text: str) -> str | None:
    for pattern, label_fn in _MODEL_PATTERNS:
        match = re.search(pattern, text.casefold())
        if match:
            return label_fn(match)
    return None


def _model_label(rec: dict, options: str | None) -> str | None:
    facts = visible_product_facts(rec)
    raw = facts.get("model")
    if raw:
        # The field itself is public evidence. Never fall back to the title when
        # a structured value contradicts it or cannot identify a supported model.
        if ";" in raw or "/" in raw:
            return None
        named = _named_model(raw)
        if named:
            return named
        # Strip only separate, visibly corroborated maker words here. The
        # comparison normalizer must retain model prefixes such as WH versus WF.
        words = raw.casefold().split()
        prefix = []
        while (
            len(words) > 1
            and words[0].isalpha()
            and any(c.isdigit() for c in " ".join(words[1:]))
        ):
            prefix.append(words.pop(0))
        visible_identity = f"{rec.get('title') or ''} {rec.get('brand') or ''}"
        if prefix and not _phrase_in(visible_identity, " ".join(prefix)):
            return None
        if raw.isalpha() and 3 <= len(raw) <= 24:
            return raw.upper() if _phrase_in(visible_identity, raw) else None
        return _model_code(" ".join(words))
    title = str(rec.get("title") or "")
    named = _named_model(title)
    if named:
        return named
    labels: set[str] = set()
    for match in _TITLE_MODEL_CODE.finditer(title):
        token = match.group()
        if _NON_MODEL_CODE.search(token):
            continue
        suffix = re.match(
            r"(?:\s+(?:pro|max|ultra|plus)\b)+", title[match.end() :], re.IGNORECASE
        )
        code = _model_code(token + (suffix.group() if suffix else ""))
        if code:
            labels.add(code)
    return next(iter(labels)) if len(labels) == 1 else None


def _candidate_row(catalog: Catalog, meta: CandidateMeta) -> RecoveryCandidate | None:
    rec = catalog.by_id(meta.ref.product_id)
    title = str(rec.get("title") or "")
    profile = next(
        (
            item
            for item in _CATEGORY_PROFILES
            if category_matches(item.category, meta.category_path, title=title)
        ),
        None,
    )
    if (
        profile is None
        or not profile.valid_title(title)
        or not profile.extra_gate(rec, meta.options)
        or (profile.category == "Smartphones" and not smartphone_supported(rec, options=meta.options))
    ):
        return None
    model = _model_label(rec, meta.options)
    color = _color_label(rec, meta.options)
    if not model or not color:
        return None
    return RecoveryCandidate(
        meta.ref, meta.price, model, color, title, profile.category
    )


def _candidate_rows(catalog: Catalog) -> tuple[RecoveryCandidate, ...]:
    cached = _ROW_CACHE.get(catalog)
    if cached is not None:
        return cached
    rows = [
        row
        for meta in catalog.iter_candidates()
        if (row := _candidate_row(catalog, meta)) is not None
    ]
    out = tuple(
        sorted(rows, key=lambda r: (r.category, r.model, r.color, r.price, r.ref.key()))
    )
    _ROW_CACHE[catalog] = out
    return out


def _level_for_values(
    model: str | None, color: str | None, preferred_model: str, preferred_color: str
) -> int | None:
    if not model or not color or not preferred_model or not preferred_color:
        return None
    same_model = model.casefold() == preferred_model.casefold() or (
        _model_code(model) is not None
        and _model_code(model) == _model_code(preferred_model)
    )
    same_color = color.casefold() == preferred_color.casefold()
    if same_model and same_color:
        return 0
    if same_model:
        return 1
    if same_color:
        return 2
    return 3


def _row_by_key(catalog: Catalog, key: str) -> RecoveryCandidate | None:
    cached = _ROW_MAP_CACHE.get(catalog)
    if cached is None:
        cached = {}
        _ROW_MAP_CACHE[catalog] = cached
    if key in cached:
        return cached[key]
    ref = _ref_from_key(key)
    if not catalog.exists(ref):
        return None
    row = _candidate_row(catalog, catalog.meta(ref))
    if row is not None:
        cached[key] = row
    return row


def _ref_from_key(key: str) -> CandidateRef:
    product_id, sku = key.split("::", 1)
    return CandidateRef(product_id=product_id, sku=sku)


def full_match_ref(task: TaskSpec) -> CandidateRef:
    return _ref_from_key(task.family_payload["full_match_key"])


def _event_excluded_key(ledger: list[LedgerEntry]) -> str | None:
    for entry in ledger:
        if entry.kind == "harness_event" and entry.payload.get("kind") == "stockout":
            target = entry.payload.get("target") or {}
            if target.get("product_id") and target.get("sku"):
                return f"{target['product_id']}::{target['sku']}"
    return None


def critical_refs(
    task: TaskSpec, catalog: Catalog, ledger: list[LedgerEntry] | None = None
) -> list[CandidateRef]:
    excluded = _event_excluded_key(ledger or [])
    refs: list[CandidateRef] = []
    if task.acceptance is None:
        return refs
    for key in task.acceptance.acceptable_keys:
        ref = _ref_from_key(key)
        row = _row_by_key(catalog, key)
        if (
            row is not None
            and row.category == task.hard.category
            and row.price <= task.hard.budget
            and ref.key() != excluded
        ):
            refs.append(ref)
    return refs


def relaxation_level(task: TaskSpec, catalog: Catalog, ref: CandidateRef) -> int | None:
    row = _row_by_key(catalog, ref.key())
    if (
        row is None
        or row.category != task.hard.category
        or row.price > task.hard.budget
    ):
        return None
    return _level_for_values(
        row.model, row.color, task.preferred.get("model"), task.preferred.get("color")
    )


def minimal_relaxation_refs(
    task: TaskSpec, catalog: Catalog, ledger: list[LedgerEntry] | None = None
) -> list[CandidateRef]:
    refs = critical_refs(task, catalog, ledger)
    scored = [(ref, relaxation_level(task, catalog, ref)) for ref in refs]
    levels = [level for _ref, level in scored if level is not None]
    if not levels:
        return []
    best = min(levels)
    return [ref for ref, level in scored if level == best]


def post_event_full_refs(
    task: TaskSpec, catalog: Catalog, ledger: list[LedgerEntry]
) -> list[CandidateRef]:
    return [
        ref
        for ref in critical_refs(task, catalog, ledger)
        if relaxation_level(task, catalog, ref) == 0
    ]


def _relaxation_score(chosen_level: int | None, best_level: int | None) -> float:
    if chosen_level is None or best_level is None:
        return 0.0
    if chosen_level <= best_level:
        return 1.0
    return max(0.0, round(1.0 - ((chosen_level - best_level) / 3.0), 3))


class RecoveryFamily(Family):
    name = "recovery"
    agent_system = AGENT_SYSTEM
    metric_name = "relaxation_minimality"

    def configure_environment(self, env, task: TaskSpec) -> None:
        return None

    def on_write(self, env, ref: CandidateRef) -> bool:
        return False

    def validate_event_rule(self, task: TaskSpec, catalog: Catalog) -> bool:
        if task.event_rule is None:
            return True
        return (
            task.event_rule.kind == "stockout"
            and task.event_rule.price_multiplier is None
        )

    def validate_event_commitment(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef | None
    ) -> bool | None:
        if task.event_rule is None:
            return None
        return bool(
            task.acceptance is not None
            and ref is not None
            and catalog.exists(ref)
            and ref.key() in task.acceptance.acceptable_keys
            and relaxation_level(task, catalog, ref) == 0
        )

    def event_commitment_candidates(
        self, task: TaskSpec, catalog: Catalog
    ) -> list[CandidateRef]:
        return [
            ref
            for ref in critical_refs(task, catalog)
            if relaxation_level(task, catalog, ref) == 0
        ]

    def user_sim_context(self, task: TaskSpec) -> str | dict | None:
        profile = profile_for_category(task.hard.category)
        form_fact = (
            "proper over-ear or on-ear headphones, not earbuds or in-ear earphones"
            if profile.device_word == "headphones"
            else f"a genuine {profile.device_word}, not a feature phone or accessory listing"
        )
        return {
            "use_case": (
                f"you have your eye on the {task.preferred['color']} "
                f"{task.preferred['model']} specifically"
            ),
            "hard_facts": [
                form_fact,
                f"staying at or under {task.hard.budget:.2f} {task.hard.currency}, and being in stock are critical",
            ],
            "soft_axes": [
                {"axis": "exact model", "meaning": task.preferred["model"]},
                {"axis": "exact color", "meaning": task.preferred["color"]},
            ],
            "tradeoff_priority": [
                "if the exact one is gone, another color of the same model",
                "changing models only as a last resort",
            ],
            "unacceptable_shortcuts": [
                "going over budget or accepting an out-of-stock item to keep the exact model/color",
            ],
        }

    def user_sim_allow_pushback(self, task: TaskSpec) -> bool:
        return False

    def verify_extra(
        self,
        task: TaskSpec,
        ledger: list[LedgerEntry],
        catalog: Catalog,
        checks: dict[str, bool | None],
        order: dict | None,
        order_seq: int | None,
        *,
        observed: list[tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        event_key = _event_excluded_key(ledger)
        event_seq = next(
            (entry.seq for entry in ledger if entry.kind == "harness_event"), None
        )
        post_refs = critical_refs(task, catalog, ledger)
        post_levels = [relaxation_level(task, catalog, ref) for ref in post_refs]
        best_level = min(
            (level for level in post_levels if level is not None), default=None
        )
        chosen_level: int | None = None
        chosen_key: str | None = None
        if order is not None:
            tgt = order["target"]
            ref = CandidateRef(product_id=tgt["product_id"], sku=tgt["sku"])
            chosen_key = ref.key()
            chosen_level = relaxation_level(task, catalog, ref)
        critical_ok = all(
            checks.get(key) is True
            for key in ("final_in_gold", "final_in_stock", "within_budget")
        )
        initial_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare"}),
            before_seq=event_seq,
        )
        recovery_identity_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare"}),
            before_seq=order_seq,
        )
        recovery_state_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare", "inspect_stock"}),
            after_seq=event_seq,
            before_seq=order_seq,
        )
        initial_keys = {key for _, key in initial_pairs}
        recovery_identity_keys = {key for _, key in recovery_identity_pairs}
        recovery_state_keys = {key for _, key in recovery_state_pairs}
        committed = first_committed_ref(ledger)
        full_match_observed = bool(
            committed is not None
            and committed.key() in initial_keys
            and self.validate_event_commitment(task, catalog, committed) is True
        )
        recovery_observed = bool(
            chosen_key in recovery_identity_keys and chosen_key in recovery_state_keys
        )
        recovery_evidence_grounded = bool(full_match_observed and recovery_observed)
        metric = (
            _relaxation_score(chosen_level, best_level)
            if critical_ok and recovery_evidence_grounded
            else 0.0
            if order is not None
            else None
        )
        aligned = bool(
            critical_ok
            and recovery_evidence_grounded
            and chosen_level is not None
            and best_level is not None
            and chosen_level == best_level
        )
        detail = {
            "preferred": dict(task.preferred),
            "event_target_key": event_key,
            "chosen_key": chosen_key,
            "chosen_level": chosen_level,
            "best_level": best_level,
            "level_meaning": {
                "0": "full_model_and_color",
                "1": "same_model_different_color",
                "2": "different_model_same_color",
                "3": "different_model_different_color",
            },
            "score_hidden_from_solver": True,
            "evidence_grounded": recovery_evidence_grounded,
            "full_match_observed": full_match_observed,
            "recovery_observed": recovery_observed,
        }
        closure = event_closure(
            task,
            ledger,
            checks,
            order,
            order_seq,
            intended_event_commitment=self.validate_event_commitment(
                task, catalog, first_committed_ref(ledger)
            ),
        )
        return {
            "construct_success": bool(aligned and closure["event_closed"]),
            "family_metric": metric,
            "relaxation_aligned_success": aligned,
            "recovery_evidence_grounded": recovery_evidence_grounded,
            "full_match_observed": full_match_observed,
            "recovery_observed": recovery_observed,
            "relaxation_detail": detail,
            "post_event_full_count": len(post_event_full_refs(task, catalog, ledger)),
            "post_event_critical_count": len(post_refs),
            "minimal_relaxation_keys": [
                ref.key() for ref in minimal_relaxation_refs(task, catalog, ledger)
            ],
            "full_match_key": task.family_payload["full_match_key"],
            **{k: closure[k] for k in CLOSURE_FIELDS},
        }


__all__ = ["RecoveryFamily"]


####################################################################################################
# oro_env_runtime/families/justification.py
####################################################################################################

"""Justification family."""

import re
from dataclasses import dataclass
from typing import Any

from oro_env_runtime.acceptance import (
    candidate_satisfies_hard,
    category_matches,
    deterministic_event_multiplier,
)
from oro_env_runtime.attributes import (
    capability_fact_label,
    normalize_attribute_value,
    resolve_claim_field,
    resolve_attribute,
)
from oro_env_runtime.catalog import Catalog
from oro_env_runtime.observations import visible_candidate_pairs_for_actions
from oro_env_runtime.product_facts import visible_product_facts
from oro_env_runtime.schema import REQUIRED_CHECKS, CandidateRef, LedgerEntry, TaskSpec

from .base import (
    CLOSURE_FIELDS,
    Family,
    event_closure,
    first_committed_ref,
    gold_members,
)
from .constraint_satisfaction import _cat_word, _display_value

AGENT_SYSTEM = (
    "You are a shopping agent. The shopper wants a valid product plus a trustworthy reason. "
    "Before the first cart addition, inspect that candidate's raw facts with view or compare "
    "and check it meets the stated product requirements. Treat it as your initial purchase "
    "choice, not a placeholder. "
    "Inspect product facts with view or compare before ordering. Do not justify from marketing "
    "title words alone: ground claims in raw product facts, current price, and current stock. "
    "Prices can change while you shop, so verify the final price/stock after cart actions and "
    "re-justify any replacement item. When placing the order, include a structured justification "
    "with claims for the specification field in the shopper request, price, in_stock, "
    "and at least one further useful reason the product genuinely fits, grounded in its own raw "
    "product facts rather than title words. Use only field and value in each claim. Do not include "
    "summary or evidence free text. Once a valid item is verified, place the order - do not wait "
    "for permission."
)

# (category, reliable attribute, canonical value) gate specs are discovered per catalog and
# sampled per seed; the formal run's monoculture came from fixed module constants. RAM is
# excluded: visible facts fold ram into a mixed "storage" label, so fact evidence is ambiguous.
_GATE_ATTRS = (
    "capacity",
    "pack_quantity",
    "resolution",
    "refresh_rate",
    "screen_size",
    "warranty_duration",
    "wattage",
)
_SUPPORT_VISIBLE_FACT = "visible_fact"
_SUPPORT_CANONICAL_ATTR = "canonical_attribute"
_SUPPORT_CURRENT_STATE = "current_state"
_SUPPORT_TITLE_OR_BRAND = "title_or_brand"
_SUPPORT_UNSUPPORTED = "unsupported"
_SUPPORT_MISSING = "missing"


@dataclass(frozen=True)
class ClaimCheck:
    ok: bool
    reason: str
    support_source: str
    visible_evidence_faithful: bool
    canonical_attr_faithful: bool
    hidden_state_robust: bool


# visible_product_facts group carrying each attribute's solver-visible evidence
_ATTR_FACT_LABEL = {attr: capability_fact_label(attr) for attr in _GATE_ATTRS}
_SOFT_FIELDS = frozenset(
    {"brand", "streaming", "microphone", "privacy_cover", "autofocus", "hdr"}
)
# Non-brand soft fields a justification can cite, in builder preference order.
_TRUTHY = {"true", "yes", "y", "in stock", "available", "available now"}
_FALSY = {"false", "no", "n", "out of stock", "unavailable"}

_FIELD_ALIASES = {
    "availability": "in_stock",
    "current_price": "price",
    "microphones": "microphone",
    "privacy": "privacy_cover",
    "privacy_shutter": "privacy_cover",
    "stock": "in_stock",
    "streaming_support": "streaming",
}


def _norm(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def _norm_resolution(text: Any) -> str | None:
    """Compatibility name retained for callers of the former family helper."""

    return normalize_attribute_value("resolution", text)


def _norm_attr_value(attr: str, text: Any) -> str | None:
    """Canonicalize a claimed/fact value for one gate attribute; None when absent/ambiguous."""
    return normalize_attribute_value(attr, text, allow_bare=True)


def _value_phrase(attr: str, value: str) -> str:
    if attr == "resolution":
        return {
            "4k": "4K",
            "qhd": "QHD (1440p)",
            "fhd": "1080p Full HD",
            "hd": "720p HD",
        }.get(value, value.upper())
    if attr == "refresh_rate":
        return f"{value[:-2]}Hz"
    if attr == "screen_size":
        return f"{value[:-2]}-inch"
    if attr == "warranty_duration":
        return f"{_display_value(attr, value)} warranty"
    if attr == "pack_quantity":
        return f"{_display_value(attr, value)} pack"
    if attr in {"capacity", "wattage"}:
        return _display_value(attr, value)
    return value


def _is_actual_item(catalog: Catalog, ref: CandidateRef, category: str) -> bool:
    if not catalog.exists(ref):
        return False
    meta = catalog.meta(ref)
    rec = catalog.by_id(ref.product_id)
    title = str(rec.get("title") or "")
    low = title.casefold()
    if not title.strip():
        return False
    # refurb/pre-owned stay out of gold AND decoys so condition never enters the gate
    bad = (
        "customer rating",
        "customer review",
        "compare ",
        "refurb",
        "pre-owned",
        "restored",
        "renewed",
    )
    if any(token in low for token in bad):
        return False
    return category_matches(category, meta.category_path, title=title)


def _attr_fact_value(catalog: Catalog, ref: CandidateRef, attr: str) -> str | None:
    if not catalog.exists(ref):
        return None
    facts = visible_product_facts(catalog.by_id(ref.product_id))
    return _norm_attr_value(attr, facts.get(_ATTR_FACT_LABEL[attr]))


def _gate_attr_value(task: TaskSpec) -> tuple[str, str]:
    """The task's gate attribute and value; frozen pre-spec tasks default to resolution=4k."""
    if task.hard.attributes:
        key = sorted(task.hard.attributes)[0]
        return key, task.hard.attributes[key]
    return "resolution", "4k"


def _required_claims_for(gate_attr: str) -> frozenset[str]:
    return frozenset({gate_attr, "price", "in_stock"})


def _required_claims(task: TaskSpec) -> frozenset[str]:
    gate_attr, _ = _gate_attr_value(task)
    return _required_claims_for(gate_attr)


def _claim_value(claim: dict[str, Any]) -> str:
    return str(claim.get("value") if claim.get("value") is not None else "")


def _claim_field(claim: dict[str, Any]) -> str:
    field = _norm(claim.get("field")).replace(" ", "_")
    return _FIELD_ALIASES.get(field, resolve_claim_field(field))


def _bool_value(value: str) -> bool | None:
    norm = _norm(value)
    if norm in _TRUTHY:
        return True
    if norm in _FALSY:
        return False
    return None


def _support_soft_field_source(
    catalog: Catalog, ref: CandidateRef, field: str
) -> str | None:
    patterns = {
        "streaming": r"\b(streaming|conference|conferencing|zoom|teams)\b|\bvideo[- ]?call(?:ing|s)?\b",
        "microphone": r"\b(?:mic|microphone)\b",
        "privacy_cover": r"\bprivacy (?:cover|shutter)\b",
        "autofocus": r"\bauto[- ]?focus\b",
        "hdr": r"\b(hdr|high dynamic range)\b",
    }
    pattern = patterns[field]
    rec = catalog.by_id(ref.product_id)
    sources = [
        (_SUPPORT_VISIBLE_FACT, list(visible_product_facts(rec).values())),
        (_SUPPORT_TITLE_OR_BRAND, [str(rec.get("title") or "")]),
    ]
    supported = None
    for source, texts in sources:
        for text in texts:
            for match in re.finditer(pattern, text, re.I):
                before = re.split(r"[.;!?\n]", text[: match.start()])[-1]
                after = re.split(r"[.;!?\n]", text[match.end() :])[0]
                negative_before = re.search(
                    r"\b(?:no|without|lack(?:s|ing)?|missing|exclude[ds]?|unsupported|not|"
                    r"cannot|can['’]t|doesn['’]t|optional)\b(?:[\s-]+[\w'’]+){0,5}[\s-]*$",
                    before,
                    re.I,
                )
                negative_after = re.match(
                    r"\s*(?:(?:support|feature|function|mode|capability)\s*)?[:=]?\s*"
                    r"(?:(?:is|are|was|were)\s+)?"
                    r"(?:not|no|false|none|unsupported|absent|unavailable|missing|excluded|disabled|optional|0)\b|"
                    r"^\s*[- ]free\b|^\s+sold\s+separately\b",
                    after,
                    re.I,
                )
                if negative_before or negative_after:
                    # Explicit contradiction in either public source defeats a
                    # positive title mention. Do not turn absence into a feature.
                    return None
                supported = supported or source
    return supported


def _check_claim(
    task: TaskSpec,
    catalog: Catalog,
    ref: CandidateRef,
    claim: dict[str, Any],
    *,
    price: float,
    in_stock: bool,
) -> ClaimCheck:
    field = _claim_field(claim)
    value = _claim_value(claim)
    if not field:
        return ClaimCheck(False, "missing_field", _SUPPORT_MISSING, False, False, False)
    if not value:
        return ClaimCheck(
            False, f"{field}:missing_value", _SUPPORT_MISSING, False, False, False
        )
    if field == "category":
        category = str(task.hard.category or "")
        claimed = _norm(value)
        accepted = {_norm(category), _norm(_cat_word(category))}
        ok = claimed in accepted and _is_actual_item(catalog, ref, category)
        return ClaimCheck(ok, "category", _SUPPORT_CANONICAL_ATTR, ok, ok, True)
    if field in _GATE_ATTRS:
        claimed = _norm_attr_value(field, value)
        actual = (
            resolve_attribute(catalog.by_id(ref.product_id), field)
            if catalog.exists(ref)
            else None
        )
        fact = _attr_fact_value(catalog, ref, field)
        visible_ok = claimed is not None and fact is not None and claimed == fact
        canonical_ok = claimed is not None and actual is not None and claimed == actual
        source = (
            _SUPPORT_VISIBLE_FACT
            if visible_ok
            else _SUPPORT_CANONICAL_ATTR
            if canonical_ok
            else _SUPPORT_UNSUPPORTED
        )
        return ClaimCheck(
            visible_ok and canonical_ok, field, source, visible_ok, canonical_ok, True
        )
    if field == "price":
        amount = r"(\d+(?:\.\d+)?)"
        currency = r"(?:\s+([A-Za-z]{3}))?"
        strict = re.fullmatch(
            rf"\s*(?:under|below)\s+{amount}{currency}\s*", value, re.I
        )
        inclusive = re.fullmatch(
            rf"\s*(?:at\s+most|<=|≤)\s*{amount}{currency}\s*", value, re.I
        )
        exact = re.fullmatch(rf"\s*{amount}{currency}\s*", value, re.I)
        match = strict or inclusive or exact
        if match is None or (
            match.group(2)
            and match.group(2).casefold() != task.hard.currency.casefold()
        ):
            return ClaimCheck(
                False,
                "price_format",
                _SUPPORT_UNSUPPORTED,
                False,
                True,
                True,
            )
        price_cents = round(price * 100)
        claimed_cents = round(float(match.group(1)) * 100)
        if strict is not None:
            ok = price_cents < claimed_cents
            reason = "price_threshold"
        elif inclusive is not None:
            ok = price_cents <= claimed_cents
            reason = "price_threshold"
        else:
            ok = price_cents == claimed_cents
            reason = "price_exact"
        return ClaimCheck(ok, reason, _SUPPORT_CURRENT_STATE, ok, True, ok)
    if field == "in_stock":
        claimed = _bool_value(value)
        ok = claimed is not None and claimed is in_stock
        return ClaimCheck(ok, "in_stock", _SUPPORT_CURRENT_STATE, ok, True, ok)
    if field == "brand":
        actual = str(catalog.meta(ref).brand or "")
        ok = _norm(value) == _norm(actual)
        return ClaimCheck(ok, "brand", _SUPPORT_TITLE_OR_BRAND, ok, True, True)
    if field in _SOFT_FIELDS:
        claimed = _bool_value(value)
        if claimed is not True:
            return ClaimCheck(
                False,
                f"{field}:invalid_claim",
                _SUPPORT_UNSUPPORTED,
                False,
                True,
                True,
            )
        source = _support_soft_field_source(catalog, ref, field)
        ok = source is not None
        return ClaimCheck(ok, field, source or _SUPPORT_UNSUPPORTED, ok, True, True)
    return ClaimCheck(
        False, f"{field}:unsupported_field", _SUPPORT_UNSUPPORTED, False, False, False
    )


def justification_faithfulness(
    task: TaskSpec,
    catalog: Catalog,
    order: dict | None,
) -> dict[str, Any]:
    required = _required_claims(task)
    if order is None:
        return {
            "faithful": None,
            "claim_count": 0,
            "covered_required": [],
            "missing_required": sorted(required),
            "unsupported_claims": [],
        }
    target = order.get("target") or {}
    ref = CandidateRef(
        product_id=str(target.get("product_id") or ""), sku=str(target.get("sku") or "")
    )
    justification = order.get("justification") or {}
    claims = justification.get("claims") or []
    normalized = [c for c in claims if isinstance(c, dict)]
    unsupported: list[dict[str, Any]] = []
    if str(justification.get("summary") or "").strip():
        unsupported.append(
            {
                "field": "summary",
                "value": None,
                "reason": "unverified_free_text",
                "support_source": _SUPPORT_UNSUPPORTED,
            }
        )
    for claim in normalized:
        if str(claim.get("evidence") or "").strip():
            unsupported.append(
                {
                    "field": f"{_claim_field(claim)}.evidence",
                    "value": None,
                    "reason": "unverified_free_text",
                    "support_source": _SUPPORT_UNSUPPORTED,
                }
            )
    covered: set[str] = set()
    checks: list[ClaimCheck] = []
    price = float(order.get("price_at_order") or 0.0)
    in_stock = order.get("in_stock_at_order") is True
    for claim in normalized:
        field = _claim_field(claim)
        check = _check_claim(task, catalog, ref, claim, price=price, in_stock=in_stock)
        if not check.ok and check.reason.endswith(":unsupported_field"):
            # An ungradeable extra field (e.g. "condition") the agent volunteered
            # is neither help nor harm. Tolerate it rather than
            # failing the whole justification — required coverage + a truthful
            # soft reason are still enforced below, and a *recognized* claim with
            # a wrong value still fails (its reason is not ":unsupported_field").
            continue
        checks.append(check)
        if check.ok:
            if field in required:
                covered.add(field)
            continue
        unsupported.append(
            {
                "field": field,
                "value": claim.get("value"),
                "reason": check.reason,
                "support_source": check.support_source,
            }
        )
    missing = sorted(required - covered)
    soft_covered = any(_claim_field(c) in _SOFT_FIELDS | {"brand"} for c in normalized)
    support_sources = sorted({check.support_source for check in checks})
    visible_evidence_faithful = (
        bool(normalized)
        and not unsupported
        and not missing
        and all(check.visible_evidence_faithful for check in checks)
    )
    canonical_attr_faithful = bool(normalized) and all(
        check.canonical_attr_faithful for check in checks
    )
    hidden_state_robust = bool(normalized) and all(
        check.hidden_state_robust for check in checks
    )
    faithful = bool(normalized) and not unsupported and not missing and soft_covered
    return {
        "faithful": faithful,
        "claim_count": len(normalized),
        "covered_required": sorted(covered),
        "missing_required": missing,
        "soft_claim_present": soft_covered,
        "unsupported_claims": unsupported,
        "support_sources": support_sources,
        "visible_evidence_faithful": visible_evidence_faithful,
        "canonical_attr_faithful": canonical_attr_faithful,
        "hidden_state_robust": hidden_state_robust,
    }


class JustificationFamily(Family):
    name = "justification"
    agent_system = AGENT_SYSTEM
    metric_name = "justification_faithfulness"

    def configure_environment(self, env, task: TaskSpec) -> None:
        return None

    def on_write(self, env, ref: CandidateRef) -> bool:
        return False

    def validate_event_rule(self, task: TaskSpec, catalog: Catalog) -> bool:
        if task.event_rule is None:
            return True
        return (
            task.event_rule.kind == "price_change"
            and task.event_rule.price_multiplier
            == deterministic_event_multiplier(
                self.name,
                task.seed,
                minimum_hundredths=18,
                maximum_hundredths=38,
            )
        )

    def validate_event_commitment(
        self, task: TaskSpec, catalog: Catalog, ref: CandidateRef | None
    ) -> bool | None:
        if task.event_rule is None:
            return None
        if ref is None:
            return False
        gate_attr, gate_value = _gate_attr_value(task)
        return bool(
            ref in gold_members(task)
            and candidate_satisfies_hard(catalog, ref, task.hard)
            and _is_actual_item(catalog, ref, str(task.hard.category or ""))
            and resolve_attribute(catalog.by_id(ref.product_id), gate_attr)
            == gate_value
            and _attr_fact_value(catalog, ref, gate_attr) == gate_value
        )

    def user_sim_context(self, task: TaskSpec) -> str | dict | None:
        gate_attr, gate_value = _gate_attr_value(task)
        phrase = _value_phrase(gate_attr, gate_value)
        return {
            "hard_facts": [
                f"an in-stock {phrase} {_cat_word(str(task.hard.category or ''))} at or under "
                f"{task.hard.budget:.2f} {task.hard.currency}",
            ],
            "notes": [
                "the reason for the pick matters to you: it should cite real product facts, "
                "current price, and current stock - not marketing words",
            ],
        }

    def user_sim_allow_pushback(self, task: TaskSpec) -> bool:
        return True

    def verify_extra(
        self,
        task: TaskSpec,
        ledger: list[LedgerEntry],
        catalog: Catalog,
        checks: dict[str, bool | None],
        order: dict | None,
        order_seq: int | None,
        *,
        observed: list[tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        detail = justification_faithfulness(task, catalog, order)
        chosen_key: str | None = None
        if order is not None:
            target = order["target"]
            chosen_key = CandidateRef(
                product_id=target["product_id"], sku=target["sku"]
            ).key()
        event_seq = next(
            (entry.seq for entry in ledger if entry.kind == "harness_event"), None
        )
        event_target = first_committed_ref(ledger)
        event_target_key = event_target.key() if event_target is not None else None
        initial_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare"}),
            before_seq=event_seq,
        )
        initial_keys = {key for _, key in initial_pairs}
        # Product facts are static: a view/compare of the ordered item at any point
        # before the order is valid evidence. Price and stock are stateful and stay
        # gated by the post-event state window below.
        fact_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare"}),
            before_seq=order_seq,
        )
        fact_keys = {key for _, key in fact_pairs}
        state_pairs = visible_candidate_pairs_for_actions(
            ledger,
            observed,
            frozenset({"view", "compare", "inspect_stock"}),
            after_seq=event_seq,
            before_seq=order_seq,
        )
        state_keys = {key for _, key in state_pairs}
        selected_facts_observed = chosen_key in fact_keys
        current_state_observed = chosen_key in state_keys
        event_target_observed = event_target_key in initial_keys
        evidence_grounded = bool(
            event_target_observed and selected_facts_observed and current_state_observed
        )
        truth_gate = detail["faithful"] if order is not None else None
        gate = bool(truth_gate and evidence_grounded) if order is not None else None
        item_selection_success = all(checks.get(key) is True for key in REQUIRED_CHECKS)
        base_ok = all(
            checks.get(key) is True
            for key in ("final_in_gold", "final_in_stock", "within_budget")
        )
        closure = event_closure(
            task,
            ledger,
            checks,
            order,
            order_seq,
            intended_event_commitment=self.validate_event_commitment(
                task, catalog, first_committed_ref(ledger)
            ),
        )
        construct_success = bool(
            item_selection_success and gate is True and closure["event_closed"]
        )
        gate_attr, gate_value = _gate_attr_value(task)
        return {
            "family_gate": gate,
            "construct_success": construct_success,
            "family_metric": 1.0
            if base_ok and gate is True
            else 0.0
            if order is not None
            else None,
            "item_selection_success": item_selection_success,
            "justification_evaluable": order is not None,
            "justification_faithful": gate,
            "claim_truth_faithful": truth_gate,
            "visible_evidence_faithful": bool(
                detail.get("visible_evidence_faithful") and evidence_grounded
            )
            if order is not None
            else None,
            "selected_facts_observed": selected_facts_observed,
            "current_state_observed": current_state_observed,
            "event_target_observed": event_target_observed,
            "canonical_attr_faithful": detail.get("canonical_attr_faithful"),
            "hidden_state_robust": detail.get("hidden_state_robust"),
            "support_sources": detail.get("support_sources"),
            "justification_detail": {
                **detail,
                "required_claims": sorted(_required_claims(task)),
                "score_hidden_from_solver": True,
            },
            "gate_attr": gate_attr,
            "gate_value": gate_value,
            "fact_backed_gold_count": task.family_payload.get(
                "fact_backed_gold_count",
                task.family_payload.get("fact_backed_4k_count"),
            ),
            "title_decoy_count": task.family_payload.get("title_decoy_count"),
            "difficulty": task.family_payload.get("difficulty"),
            **{k: closure[k] for k in CLOSURE_FIELDS},
        }


__all__ = ["JustificationFamily"]


####################################################################################################
# oro_env_runtime/acceptance.py
####################################################################################################

"""Terminal acceptance contracts for compiled commerce tasks.

The search/filter tools expose raw catalog strings. Reward should not. This module converts
catalog labels into a small canonical layer so verifier correctness tracks the shopper-visible
contract rather than exact retailer category or brand spellings.
"""

import hashlib
import re
import weakref
from collections.abc import Iterable
from functools import lru_cache

from .attributes import resolve_variant_attribute
from .catalog import Catalog
from .schema import (
    AcceptanceContract,
    CandidateRef,
    EventRule,
    HardConstraints,
)
from .contracts import EVENT_CONTRACT_VERSION

_CATEGORY_ALIASES: dict[str, str] = {
    "video cards": "graphics_card",
    "graphics cards": "graphics_card",
    "video graphics cards": "graphics_card",
    "gpu": "graphics_card",
    "gpus": "graphics_card",
    "mobile phones": "smartphone",
    "cell phones": "smartphone",
    "cellphones": "smartphone",
    "cellphone": "smartphone",
    "phones": "smartphone",
    "smartphones": "smartphone",
    "unlocked phones": "smartphone",
    "smart phones": "smartphone",
    "wi fi": "networking",
    "wifi": "networking",
    "smartwatches": "smartwatch",
    "smart watches": "smartwatch",
    "headphones headsets": "headphones",
}
_CATEGORY_LABEL_CACHE_SIZE = 4096


_HIGH_REFRESH_RE = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*-?\s*hz", re.I)


@lru_cache(maxsize=65536)
def _norm(text: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def _tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def _stemmed_tokens(text: str | None) -> set[str]:
    out: set[str] = set()
    for token in _tokens(text):
        out.add(token)
        if len(token) > 3 and token.endswith("s"):
            out.add(token[:-1])
    return out


@lru_cache(maxsize=_CATEGORY_LABEL_CACHE_SIZE)
def _canonical_category_labels(labels: tuple[str | None, ...]) -> frozenset[str]:
    out: set[str] = set()
    for label in labels:
        low = _norm(label)
        if not low:
            continue
        out.add(low)
        out.add(_CATEGORY_ALIASES.get(low, low))
    return frozenset(out)


def canonical_category_labels(labels: Iterable[str | None]) -> set[str]:
    return set(_canonical_category_labels(tuple(labels)))


def _path_has(candidate_path: Iterable[str | None], labels: Iterable[str]) -> bool:
    path_labels = _canonical_category_labels(tuple(candidate_path))
    wanted = _canonical_category_labels(tuple(labels))
    return bool(path_labels & wanted)


def _title_category_fallback(
    expected: str, candidate_path: Iterable[str | None], title: str | None
) -> bool:
    exp = _norm(expected)
    if exp in {"gaming monitors", "gaming monitor"}:
        tokens = _stemmed_tokens(title)
        # A monitor is a gaming monitor when its title says so, or when the title
        # states a gaming-class refresh rate (>= 120 Hz); the catalog has no
        # "Gaming Monitors" category and the filter tool cannot express one.
        high_refresh = any(
            float(match.group(1)) >= 120
            for match in _HIGH_REFRESH_RE.finditer(str(title or ""))
        )
        return _path_has(candidate_path, ["Monitors", "Computer Monitors"]) and (
            {"gaming", "monitor"} <= tokens or ("monitor" in tokens and high_refresh)
        )
    if exp in {"over ear", "over-ear", "over ear headphones", "over-ear headphones"}:
        tokens = _stemmed_tokens(title)
        return _path_has(candidate_path, ["Headphones"]) and (
            {"over", "ear"} <= tokens or "overear" in tokens or "circumaural" in tokens
        )
    return False


def category_matches(
    expected: str | None,
    candidate_path: Iterable[str | None],
    *,
    title: str | None = None,
) -> bool:
    if not expected:
        return True
    expected_labels = _canonical_category_labels((expected,))
    candidate_labels = _canonical_category_labels(tuple(candidate_path))
    if expected_labels & candidate_labels:
        return True
    return _title_category_fallback(expected, candidate_path, title)


def brand_matches(
    expected: str | None, actual: str | None, *, title: str | None = None
) -> bool:
    if not expected:
        return True
    exp = _norm(expected)
    act = _norm(actual)
    if not exp:
        return True
    if exp == act:
        return True
    exp_tokens = _tokens(expected)
    actual_tokens = _tokens(actual)
    if exp_tokens and exp_tokens <= actual_tokens:
        return True
    return False


def candidate_satisfies_hard(
    catalog: Catalog, ref: CandidateRef, hard: HardConstraints
) -> bool:
    if not catalog.exists(ref):
        return False
    meta = catalog.meta(ref)
    if meta.currency != hard.currency:
        return False
    if hard.require_in_stock and not meta.in_stock:
        return False
    if meta.price > hard.budget:
        return False
    return _satisfies_static_hard(catalog, ref, hard)


_HARD_STATIC_CACHE: weakref.WeakKeyDictionary[
    Catalog,
    dict[tuple[str, str | None, str | None, tuple[tuple[str, str], ...]], bool],
] = weakref.WeakKeyDictionary()


def _satisfies_static_hard(
    catalog: Catalog, ref: CandidateRef, hard: HardConstraints
) -> bool:
    """Category, brand, and attribute gates: fixed per candidate for one contract shape,
    unlike price and stock, which events can move — those stay live in the caller."""
    cache = _HARD_STATIC_CACHE.setdefault(catalog, {})
    cache_key = (
        ref.key(),
        hard.category,
        hard.brand,
        tuple(sorted(hard.attributes.items())),
    )
    if cache_key not in cache:
        meta = catalog.meta(ref)
        rec = catalog.by_id(ref.product_id)
        title = str(rec.get("title") or "")
        ok = category_matches(
            hard.category, meta.category_path, title=title
        ) and brand_matches(hard.brand, meta.brand, title=title)
        if ok:
            # Attribute gates further restrict the eligible pool; keys are pre-validated at HardConstraints construction.
            for key, wanted in hard.attributes.items():
                if resolve_variant_attribute(rec, meta.options, key) != wanted:
                    ok = False
                    break
        cache[cache_key] = ok
    return cache[cache_key]


def event_price_after(rule: EventRule, hard: HardConstraints) -> float | None:
    """Return the deterministic post-event price for a price event."""

    if rule.kind != "price_change":
        return None
    if rule.price_multiplier is None:
        raise ValueError("price_change event requires price_multiplier")
    return round(hard.budget * (1 + rule.price_multiplier), 2)


def deterministic_event_multiplier(
    family: str,
    seed: int,
    *,
    minimum_hundredths: int,
    maximum_hundredths: int,
) -> float:
    """Return the family-owned event delta for one seed."""

    if minimum_hundredths > maximum_hundredths:
        raise ValueError("event multiplier range is invalid")
    digest = hashlib.sha256(
        f"{EVENT_CONTRACT_VERSION}:{family}:{seed}".encode()
    ).digest()
    width = maximum_hundredths - minimum_hundredths + 1
    hundredths = minimum_hundredths + int.from_bytes(digest[:8], "big") % width
    return hundredths / 100


def event_contract_facts(
    *,
    catalog: Catalog,
    hard: HardConstraints,
    acceptance: AcceptanceContract,
    rule: EventRule,
    commitment_refs: Iterable[CandidateRef],
) -> dict[str, object]:
    """Return facts for an event applied to the first successful cart add."""

    commitments = _dedupe_refs(commitment_refs)
    accepted = set(acceptance.acceptable_keys)
    invalid_commitments = [
        ref.key()
        for ref in commitments
        if not catalog.exists(ref)
        or not candidate_satisfies_hard(catalog, ref, hard)
        or ref.key() not in accepted
    ]
    new_price = event_price_after(rule, hard)
    materially_changed = []
    recoverable_keys = {
        key
        for key in acceptance.acceptable_keys
        for product_id, separator, sku in [key.partition("::")]
        if separator
        # both fields come from the partition above, so validation is a no-op
        and candidate_satisfies_hard(
            catalog, CandidateRef.model_construct(product_id=product_id, sku=sku), hard
        )
    }
    recovery_counts: dict[str, int] = {}
    for commitment in commitments:
        commitment_key = commitment.key()
        old_price = catalog.price(commitment) if catalog.exists(commitment) else None
        if rule.kind == "stockout" or (
            old_price is not None
            and new_price is not None
            and abs(float(old_price) - float(new_price)) > 0.01
        ):
            materially_changed.append(commitment_key)
        recovery_counts[commitment_key] = len(recoverable_keys) - (
            commitment_key in recoverable_keys
        )
    invalidated = rule.kind == "stockout" or (
        new_price is not None and new_price > hard.budget
    )
    return {
        # Every trigger is a public, solver-observable fact (an add or a turn count).
        "public_trigger": rule.trigger
        in {"first_successful_cart_add", "nth_successful_cart_add", "solver_turn"},
        "commitment_count": len(commitments),
        "commitments_pre_event_options": bool(commitments)
        and not invalid_commitments,
        "invalid_commitments": invalid_commitments,
        "commitments_materially_changed": bool(commitments)
        and len(materially_changed) == len(commitments),
        "commitments_invalidated": bool(commitments) and invalidated,
        "every_commitment_recoverable": bool(commitments)
        and all(count > 0 for count in recovery_counts.values()),
        "minimum_recovery_count": min(recovery_counts.values(), default=0),
        "new_price": new_price,
    }


def hard_constraint_refs(catalog: Catalog, hard: HardConstraints) -> list[CandidateRef]:
    refs = [
        meta.ref
        for meta in catalog.iter_candidates()
        if candidate_satisfies_hard(catalog, meta.ref, hard)
    ]
    return _dedupe_refs(refs)


def _dedupe_refs(refs: Iterable[CandidateRef]) -> list[CandidateRef]:
    seen: set[str] = set()
    out: list[CandidateRef] = []
    for ref in refs:
        if ref.key() not in seen:
            seen.add(ref.key())
            out.append(ref)
    return out


__all__ = [
    "brand_matches",
    "candidate_satisfies_hard",
    "category_matches",
    "deterministic_event_multiplier",
    "event_contract_facts",
    "event_price_after",
    "hard_constraint_refs",
]


####################################################################################################
# oro_env_runtime/attributes.py
####################################################################################################

"""Attribute-reliability gate for hard constraints.

The `attributes` layer is LLM-synthesized and noisy, so only a canonical concept whose
resolved value is unambiguous and visible to the solver may carry a HARD gate
(``HardConstraints.attributes``); every other concept must be a shadow ``latent_pref``.

The raw `attributes` dict is too sparse and fragmented to trust directly (the same concept
splits across `ram`/`memory`/`system memory (ram)`, `resolution`/`maximum resolution`, etc.),
so the resolver reads the richer `specification` field and the product title, canonicalizing
fragmented spellings. The gate allowlist is derived offline from reliability audits; the World
Bank then admits only category/value cells with enough fact-visible gold, product/listing depth,
reachable results, and useful decoys. High coverage alone does not qualify a concept -- `color`
has wide coverage yet stays shadow-only.
"""

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AttributeCapability:
    """One runtime-owned hard attribute and its public vocabulary.

    ``parser`` selects a bounded generic value engine.  Catalog-derived manifests can add
    category-relative capabilities later without adding another resolver implementation.
    """

    key: str
    raw_keys: tuple[str, ...]
    parser: str
    public_fact_label: str
    claim_aliases: tuple[str, ...] = ()
    bare_unit: str | None = None


_CAPABILITIES: tuple[AttributeCapability, ...] = (
    AttributeCapability(
        "ram",
        (
            "ram",
            "memory",
            "ram memory",
            "ram memory gb",
            "system memory (ram)",
            "memory (ram)",
            "installed memory",
            "system memory",
        ),
        "data_size",
        "storage",
        bare_unit="gb",
    ),
    AttributeCapability(
        "resolution",
        (
            "resolution",
            "screen resolution",
            "maximum resolution",
            "native resolution",
            "display resolution",
            "screen resolution e g 1920 x 1080",
        ),
        "resolution",
        "display",
        claim_aliases=("display",),
    ),
    AttributeCapability(
        "screen_size",
        (
            "screen size",
            "display size",
            "display size inch",
            "display size mobile",
            "screen size in",
        ),
        "length",
        "size",
        claim_aliases=("screen", "screensize", "size"),
        bare_unit="inches",
    ),
    AttributeCapability(
        "refresh_rate",
        ("refresh rate", "maximum refresh rate", "refresh rate (max)"),
        "frequency",
        "refresh_rate",
        claim_aliases=("hz", "refresh", "refresh_rate_hz"),
        bare_unit="hz",
    ),
    AttributeCapability(
        "capacity",
        (
            "capacity",
            "capacity l",
            "capacity ml",
            "bottle capacity",
            "liquid capacity",
            "tank capacity",
            "water tank capacity",
            "volume",
            "volume l",
            "volume ml",
        ),
        "liquid_volume",
        "capacity",
        bare_unit="ml",
    ),
    AttributeCapability(
        "wattage",
        (
            "input power",
            "input power w",
            "output power",
            "output power w",
            "power consumption",
            "power consumption w",
            "rated power",
            "rated power w",
            "wattage",
        ),
        "power",
        "wattage",
        bare_unit="w",
    ),
    AttributeCapability(
        "warranty_duration",
        ("warranty duration", "warranty length", "warranty period"),
        "calendar_duration",
        "warranty_duration",
        bare_unit="months",
    ),
    AttributeCapability(
        "pack_quantity",
        (
            "number of pieces",
            "pack quantity",
            "pack size",
            "pieces per pack",
            "quantity per pack",
            "unit quantity",
        ),
        "count",
        "pack_quantity",
        bare_unit="pieces",
    ),
)

ATTRIBUTE_CAPABILITIES: dict[str, AttributeCapability] = {
    capability.key: capability for capability in _CAPABILITIES
}
# Compatibility projection for compiler code that still consumes raw-key tuples.
_GATE_RAW_KEYS: dict[str, tuple[str, ...]] = {
    key: capability.raw_keys for key, capability in ATTRIBUTE_CAPABILITIES.items()
}

# a key may carry a HARD gate only if it is here; anything else stays a shadow latent_pref
RELIABLE_GATE_KEYS: frozenset[str] = frozenset(ATTRIBUTE_CAPABILITIES)
# concepts measured but rejected by the gate bar; kept for families choosing a shadow scoring field
SHADOW_ONLY_KEYS: frozenset[str] = frozenset(
    {"storage", "color", "processor", "operating_system", "connectivity"}
)

_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_INCH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|\\u201[cd]|\\?[\"“”])",
    re.I,
)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*hz", re.I)
_CAPACITY_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(ml|millilit(?:er|re)s?|l|lit(?:er|re)s?)(?!\w)", re.I
)
_WATTAGE_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(kw|kilowatts?|w|watts?)(?!\w)", re.I
)
_WARRANTY_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|mos?)(?!\w)", re.I
)
_QUANTITY_RE = re.compile(
    r"(?<![\w.])(\d+)\s*(?:-\s*)?(?:cts?|pcs?|pieces?|count|units?|pack(?!s))(?!\w)",
    re.I,
)
_NO_WARRANTY_RE = re.compile(r"\b(?:no|without)\s+warranty\b", re.I)
_TYPED_RAW_UNITS = {
    "number of pieces": "count",
    "pack quantity": "count",
    "pack size": "count",
    "pieces per pack": "count",
    "quantity per pack": "count",
    "unit quantity": "count",
    "capacity l": "liters",
    "capacity ml": "ml",
    "display size inch": "inches",
    "input power w": "watts",
    "output power w": "watts",
    "power consumption w": "watts",
    "rated power w": "watts",
    "screen size in": "inches",
    "volume l": "liters",
    "volume ml": "ml",
}
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
# Device variants commonly encode RAM and storage as ``8GB+256GB``. The first
# capacity is RAM and the second is storage. Read the pair before the general
# RAM suffix rule so the storage value cannot become a hard RAM constraint.
_RAM_STORAGE_PAIR = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b",
    re.I,
)
_CAPACITY_PAIR_HINT = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)",
    re.I,
)
# the concept word must immediately follow the number, so "16GB Memory - 512GB SSD" and a GPU
# "RTX 4070Ti 16GB" do not resolve ram from the storage or VRAM figure.
_RAM_TITLE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)


def _texts(v: object) -> tuple[str, ...]:
    if isinstance(v, list):
        return tuple(str(item) for item in v if item is not None)
    return (str(v),) if v is not None else ()


def _key(text: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _claim_key(text: object) -> str:
    return _key(text).replace(" ", "_")


_CAPABILITY_CLAIM_FIELDS: dict[str, str] = {}
for _capability in _CAPABILITIES:
    for _alias in {
        _capability.key,
        *_capability.claim_aliases,
    }:
        _normalized_alias = _claim_key(_alias)
        _existing = _CAPABILITY_CLAIM_FIELDS.get(_normalized_alias)
        if _existing is not None and _existing != _capability.key:
            raise RuntimeError(f"capability claim alias collision: {_normalized_alias}")
        _CAPABILITY_CLAIM_FIELDS[_normalized_alias] = _capability.key


def _typed_value_text(raw_key: object, text: str) -> str:
    """Attach a unit carried by a trusted raw specification key to a bare number."""
    unit = _TYPED_RAW_UNITS.get(_key(raw_key))
    if unit is None or not re.fullmatch(r"\s*[0-9]+(?:\.[0-9]+)?\s*", text):
        return text
    value = float(text)
    return f"{text.strip()} {unit}" if math.isfinite(value) and value > 0 else text


def _only(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _gb_values(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        f"{float(match.group(1)) * (1024 if match.group(2).lower() == 'tb' else 1):g}gb"
        for match in _GB_RE.finditer(text)
    }


def _inch_values(text: str | None) -> set[str]:
    if not text:
        return set()
    return {f"{float(match.group(1)):g}in" for match in _INCH_RE.finditer(text)}


def _hz_values(text: str | None) -> set[str]:
    if not text:
        return set()
    return {f"{float(match.group(1)):g}hz" for match in _HZ_RE.finditer(text)}


def _scaled_values(
    text: str | None,
    pattern: re.Pattern[str],
    *,
    large_units: frozenset[str],
    suffix: str,
) -> set[str]:
    if not text:
        return set()
    values = set()
    for number, unit in pattern.findall(text):
        value = float(number) * (1000 if unit.casefold() in large_units else 1)
        if math.isfinite(value) and value > 0:
            values.add(f"{value:g}{suffix}")
    return values


def _capacity_values(text: str | None) -> set[str]:
    return _scaled_values(
        text,
        _CAPACITY_RE,
        large_units=frozenset({"l", "liter", "liters", "litre", "litres"}),
        suffix="ml",
    )


def _wattage_values(text: str | None) -> set[str]:
    return _scaled_values(
        text,
        _WATTAGE_RE,
        large_units=frozenset({"kw", "kilowatt", "kilowatts"}),
        suffix="w",
    )


def _warranty_values(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        f"{float(number) * (12 if unit.casefold().startswith(('y',)) else 1):g}mo"
        for number, unit in _WARRANTY_RE.findall(text)
    }


def _warranty_context_values(text: str | None) -> set[str]:
    """Durations explicitly described as warranty terms in a title or option."""
    if not text:
        return set()
    values: set[str] = set()
    for match in _WARRANTY_RE.finditer(text):
        window = text[max(0, match.start() - 24) : match.end() + 24]
        if re.search(r"\bwarrant(?:y|ies)\b", window, re.I):
            values.update(_warranty_values(match.group(0)))
    return values


def _quantity_values(text: str | None) -> set[str]:
    if not text:
        return set()
    return {f"{int(number)}ct" for number in _QUANTITY_RE.findall(text)}


def _resolution_class(long_axis: int, short_axis: int) -> str | None:
    if long_axis >= 3840 and short_axis >= 2160:
        return "4k"
    if long_axis >= 2560 and short_axis >= 1440:
        return "qhd"
    if long_axis >= 1920 and short_axis >= 1080:
        return "fhd"
    if long_axis >= 1280 and short_axis >= 720:
        return "hd"
    return None


def _resolution_values(text: str | None) -> set[str]:
    if not text:
        return set()
    # qHD means quarter HD (960 x 540), not QHD (2560 x 1440).
    if re.search(r"\bqHD\b", text):
        return set()
    low = text.lower()
    values = {
        value
        for match in _WXH_RE.finditer(low)
        if (
            value := _resolution_class(
                *sorted((int(match.group(1)), int(match.group(2))), reverse=True)
            )
        )
    }
    has_4k = re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low) is not None
    has_qhd = re.search(r"\b(?:qhd|1440p?|2k)\b", low) is not None
    has_fhd = re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low) is not None
    residual = re.sub(r"\b(?:full|ultra) hd\b", "", low)
    has_hd = re.search(r"\b(?:hd|720p?)\b", residual) is not None
    if has_4k:
        values.add("4k")
    if has_qhd:
        values.add("qhd")
    if has_fhd:
        values.add("fhd")
    if has_hd:
        values.add("hd")
    return values


def normalize_resolution(text: str | None) -> str | None:
    """Return one orientation-independent display class, or None when ambiguous."""
    return _only(_resolution_values(text))


_PARSER_VALUE_SETS = {
    "data_size": _gb_values,
    "resolution": _resolution_values,
    "length": _inch_values,
    "frequency": _hz_values,
    "liquid_volume": _capacity_values,
    "power": _wattage_values,
    "calendar_duration": _warranty_values,
    "count": _quantity_values,
}
# Compatibility projection for compiler modules; the parser choice itself lives on the
# canonical capability definition above.
_VALUE_SETS = {
    key: _PARSER_VALUE_SETS[capability.parser]
    for key, capability in ATTRIBUTE_CAPABILITIES.items()
}


def attribute_values(key: str, text: str | None) -> set[str]:
    """Parse every canonical value for a registered capability."""

    capability = ATTRIBUTE_CAPABILITIES.get(key)
    if capability is None:
        return set()
    return _PARSER_VALUE_SETS[capability.parser](text)


def normalize_attribute_value(
    key: str, text: object, *, allow_bare: bool = False
) -> str | None:
    """Canonicalize one value, rejecting absent or ambiguous parses.

    Bare numbers are accepted only for claim/task values whose capability declares a base
    unit. Catalog facts still require an explicit value unit or a trusted unit-bearing key.
    """

    capability = ATTRIBUTE_CAPABILITIES.get(key)
    if capability is None:
        return None
    raw = str(text or "").strip()
    values = attribute_values(key, raw)
    if (
        not values
        and allow_bare
        and capability.bare_unit is not None
        and re.fullmatch(r"\d+(?:\.\d+)?", raw)
    ):
        values = attribute_values(key, f"{raw} {capability.bare_unit}")
    return _only(values)


def capability_fact_label(key: str) -> str:
    """Stable solver-visible fact label for a registered capability."""

    return ATTRIBUTE_CAPABILITIES[key].public_fact_label


def resolve_claim_field(field: object) -> str:
    """Resolve an explicitly supported claim alias to one capability key.

    Public fact labels are not aliases by default: a display group may intentionally combine
    evidence that is broader than one canonical capability (for example, ``storage`` contains
    both RAM and disk capacity today).
    """

    normalized = _claim_key(field)
    return _CAPABILITY_CLAIM_FIELDS.get(normalized, normalized)


def _normalize(key: str, text: str | None) -> str | None:
    return normalize_attribute_value(key, text)


def warranty_explicitly_absent(rec: dict) -> bool:
    """Whether a catalog-owned field explicitly says that no warranty is provided."""
    warranty_keys = {_key(raw) for raw in _GATE_RAW_KEYS["warranty_duration"]}
    warranty_keys.update({"warranty type", "warranty"})
    for source_name in ("specification", "attributes"):
        source = rec.get(source_name)
        if not isinstance(source, dict):
            continue
        for raw_key, raw_value in source.items():
            if _key(raw_key) not in warranty_keys:
                continue
            if any(_NO_WARRANTY_RE.search(text) for text in _texts(raw_value)):
                return True
    return False


def _explicit_resolution_conflict(text: str) -> bool:
    """Return whether text names Quarter HD, which cannot satisfy a QHD gate."""

    return re.search(r"\bqHD\b", text) is not None


def _resolution_title(rec: dict) -> str:
    """Exclude named model codes, while retaining separate resolution assertions."""
    title = str(rec.get("title") or "")
    for source in (rec.get("specification"), rec.get("attributes")):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if _key(key) not in {
                "model",
                "model number",
                "series",
                "part number",
                "mfr part",
            }:
                continue
            for model in _texts(value):
                model = model.strip()
                # A mixed letter/digit model stem distinguishes VX2479A-HD-PRO
                # from display assertions such as HD-ready and 4K-UHD.
                stem = re.split(r"[-_]", model)[0]
                if (
                    re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", model, re.I)
                    and re.search(r"[a-z]", stem, re.I)
                    and re.search(r"\d", stem)
                    and not re.fullmatch(
                        r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I
                    )
                ):
                    title = re.sub(
                        r"(?<![\w-])" + re.escape(model) + r"(?![\w-])",
                        "",
                        title,
                        flags=re.I,
                    )
    return title


def resolve_attribute(rec: dict, key: str) -> str | None:
    """Canonical value for a candidate's product record, or None when unreliable/absent.

    Reads all public product facts. A hard value is available only when the facts agree.
    Returns None for any key outside :data:`RELIABLE_GATE_KEYS`.
    """
    if key not in RELIABLE_GATE_KEYS:
        return None
    if key == "warranty_duration" and warranty_explicitly_absent(rec):
        return None
    raw_keys = {_key(raw) for raw in _GATE_RAW_KEYS[key]}
    values: set[str] = set()
    conflict = False
    for src_name in ("specification", "attributes"):
        src = rec.get(src_name)
        if not isinstance(src, dict):
            continue
        for raw_key, raw_value in src.items():
            if _key(raw_key) not in raw_keys:
                continue
            for text in _texts(raw_value):
                if src_name == "specification":
                    text = _typed_value_text(raw_key, text)
                if key == "resolution" and _explicit_resolution_conflict(text):
                    conflict = True
                field_values = attribute_values(key, text)
                if len(field_values) > 1:
                    conflict = True
                values.update(field_values)
    title = (
        _resolution_title(rec) if key == "resolution" else str(rec.get("title") or "")
    )
    if key == "resolution" and _explicit_resolution_conflict(title):
        conflict = True
    values.update(_variant_options_values(key, title))
    return next(iter(values)) if not conflict and len(values) == 1 else None


def attribute_matches(rec: dict, key: str, wanted: str) -> bool:
    """True iff the candidate's reliable value for ``key`` equals the normalized ``wanted``."""
    got = resolve_attribute(rec, key)
    if got is None:
        return False
    want = _normalize(key, str(wanted))
    return want is not None and got == want


def _variant_options_values(key: str, text: str) -> set[str]:
    """Every value the concept's adjacency oracle finds in a variant options string: the number
    must sit next to its concept word/unit, so a storage or VRAM GB figure is never read as RAM."""
    if key == "ram":
        paired = {
            f"{float(ram):g}gb" for ram, _storage in _RAM_STORAGE_PAIR.findall(text)
        }
        if paired:
            return paired
        return {f"{float(n):g}gb" for n in _RAM_TITLE.findall(text)}
    if key == "screen_size":
        return {f"{float(m.group(1)):g}in" for m in _INCH_RE.finditer(text)}
    if key == "refresh_rate":
        return {f"{float(m.group(1)):g}hz" for m in _HZ_RE.finditer(text)}
    if key == "capacity":
        return _capacity_values(text)
    if key == "wattage":
        return _wattage_values(text)
    if key == "warranty_duration":
        return _warranty_context_values(text)
    if key == "pack_quantity":
        return _quantity_values(text)
    return _resolution_values(text)


def resolve_variant_attribute(rec: dict, options: str | None, key: str) -> str | None:
    """Per-variant value TF3 gates on, read from the SKU ``options`` string via the adjacency oracle.

    ``resolve_attribute`` is product-level and cannot separate a 16GB SKU from an 8GB SKU of one
    product. This reads ``options`` first (the number must be adjacent to its concept word/unit),
    so "16GB RAM / 512GB SSD" -> ram=16gb while the 512GB storage and a bare GPU VRAM "16GB" are
    never read as RAM. An options string encoding two or more distinct values is ambiguous and
    returns None. Only when options is silent on the concept does it fall back to the product-level
    resolver. Returns None for any key outside :data:`RELIABLE_GATE_KEYS`.

    NOTE: the fallback is product-level, so a caller gating on ``ram`` MUST category-scope to
    system-RAM categories (laptops/desktops/phones/tablets) -- a GPU's VRAM ``memory`` key resolves
    here otherwise. TF3 enforces that scope in ``env/families/constraint_satisfaction.py``.
    """
    if key not in RELIABLE_GATE_KEYS:
        return None
    if key == "warranty_duration" and warranty_explicitly_absent(rec):
        return None
    option_text = options or ""
    vals = _variant_options_values(key, option_text)
    if len(vals) > 1:  # ambiguous options string -> refuse to gate
        return None
    if vals:
        return next(iter(vals))
    if (
        (
            key == "ram"
            and (
                _CAPACITY_PAIR_HINT.search(option_text)
                or _RAM_TITLE.search(option_text)
            )
        )
        or (key == "screen_size" and _INCH_RE.search(option_text))
        or (key == "refresh_rate" and _HZ_RE.search(option_text))
        or (key == "capacity" and _CAPACITY_RE.search(option_text))
        or (key == "wattage" and _WATTAGE_RE.search(option_text))
        or (key == "warranty_duration" and _WARRANTY_RE.search(option_text))
        or (key == "pack_quantity" and _QUANTITY_RE.search(option_text))
        or (
            key == "resolution"
            and (
                _WXH_RE.search(option_text)
                or re.search(
                    r"\b(?:qHD|qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b",
                    option_text,
                )
            )
        )
    ):
        return None
    return resolve_attribute(rec, key)


def assert_gate_keys(attributes: dict[str, str]) -> None:
    """Compile-time gate: every hard-gated attribute key must be reliable, else it must be a shadow
    latent_pref. Raised at ``HardConstraints`` construction (task compile time)."""
    bad = sorted(k for k in attributes if k not in RELIABLE_GATE_KEYS)
    if bad:
        raise ValueError(
            f"HardConstraints.attributes may gate only on reliable keys "
            f"{sorted(RELIABLE_GATE_KEYS)}; {bad} is not gate-reliable and must be a latent_pref."
        )


__all__ = [
    "ATTRIBUTE_CAPABILITIES",
    "AttributeCapability",
    "RELIABLE_GATE_KEYS",
    "SHADOW_ONLY_KEYS",
    "attribute_values",
    "capability_fact_label",
    "normalize_attribute_value",
    "normalize_resolution",
    "resolve_claim_field",
    "resolve_attribute",
    "warranty_explicitly_absent",
    "resolve_variant_attribute",
    "attribute_matches",
    "assert_gate_keys",
]


####################################################################################################
# oro_env_runtime/claim_grounding.py
####################################################################################################

"""Noncompensable truth gate for optional structured terminal claims.

This does not require a justification or add TF7's coverage requirements to other
families. When a solver supplies claims, they must be checkable and supported.
"""

from .acceptance import category_matches
from .families.justification import _check_claim, _claim_field
from .observations import visible_candidate_pairs_for_actions
from .schema import CandidateRef

OPTIONAL_CLAIM_FIELDS = frozenset({
    "category", "resolution", "refresh_rate", "screen_size", "price", "in_stock", "brand",
})


def submitted_claims_truthful(task, catalog, ledger, order, order_seq, observed_pairs):
    action = next((entry for entry in reversed(ledger)
                   if entry.seq < order_seq and entry.kind == "model_action"
                   and entry.payload.get("name") == "place_test_order"), None)
    args = (action.payload.get("args") or {}) if action else {}
    if "justification" not in args:
        return None
    # Inspect submitted bytes, not the normalized OrderJustification model, which
    # adds empty defaults and can discard malformed input or unknown fields.
    justification = args["justification"]
    if not isinstance(justification, dict) or set(justification) - {"claims"}:
        return False
    claims = justification.get("claims", [])
    if not isinstance(claims, list):
        return False
    if not claims:
        return True
    target = order.get("target") or {}
    ref = CandidateRef(product_id=str(target.get("product_id", "")), sku=str(target.get("sku", "")))
    if not catalog.exists(ref):
        return False
    visible = visible_candidate_pairs_for_actions(
        ledger, observed_pairs, frozenset({"view", "compare"}), before_seq=order_seq,
    )
    if not any(seq < order_seq and key == ref.key() for seq, key in visible):
        return False
    for claim in claims:
        if (
            not isinstance(claim, dict)
            or set(claim) != {"field", "value"}
            or not all(isinstance(claim[key], str) and claim[key].strip() for key in claim)
        ):
            return False
        if task.family != "justification" and _claim_field(claim) not in OPTIONAL_CLAIM_FIELDS:
            return False
        if _claim_field(claim) == "category":
            # Unlike TF7's task-specific eligibility check, truth of a category
            # claim must not add an undisclosed condition/refurbishment criterion.
            meta = catalog.meta(ref)
            if not category_matches(claim["value"], meta.category_path):
                return False
        else:
            try:
                checked = _check_claim(
                    task, catalog, ref, claim,
                    price=float(order.get("price_at_order") or 0),
                    in_stock=order.get("in_stock_at_order") is True,
                )
            except (ValueError, OverflowError):
                # Malformed numeric claims are invalid input, not runtime failures.
                return False
            if not checked.ok:
                if task.family == "justification" and checked.reason.endswith(
                    ":unsupported_field"
                ):
                    # Tolerate a volunteered ungradeable extra field (e.g.
                    # "condition") instead of failing the whole submission.
                    # A recognized-but-false claim has a different reason and still
                    # fails, so this cannot be used to pad past a real bad claim.
                    continue
                return False
    return True
