"""Basic family-routed agent for validator-owned generated environments."""

from __future__ import annotations

import json
from typing import Any

from src.agent.proxy_client import ProxyClient

_proxy = ProxyClient(timeout=120, max_retries=2)


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
    def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
        raw = tool_call["function"].get("arguments", "{}")
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed

    @staticmethod
    def _llm(prompt: str | list[dict[str, Any]], **params: Any) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        for model in Utils.MODELS:
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


class Engine:
    """Shared main loop. Each family overrides run()."""

    marker = ""

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

    def run(self) -> list[dict[str, Any]]:
        """Let the model choose actions from the environment's dynamic tool list."""

        messages, dialogue = self.messages, self.dialogue
        for turn in range(1, self.max_steps + 1):
            for _attempt in range(2):
                assistant = Utils._llm(messages, tools=self.tools, tool_choice="required")
                assistant_content = assistant.get("content") or ""
                tool_calls = (assistant.get("tool_calls") or [])[: self.max_calls]
                if tool_calls:
                    break
                dialogue.append({"role": "assistant", "content": assistant_content})
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {"role": "user", "content": Prompt.CONTINUE},
                    ]
                )
            else:
                raise RuntimeError("model returned no tool call before completion")

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }
            )
            group_id = f"{self.problem_id}-turn-{turn}"
            envelope = {
                **self.binding,
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
            if any(call["observation"]["done"] for call in result["calls"]):
                break

        return dialogue

# -- families ------------------------------------------------------------------


class IntentDecomposition(Engine):
    marker = "mix of firm requirements and nice-to-haves"

    def run(self) -> list[dict[str, Any]]:
        return super().run()


class RetrievalRecall(Engine):
    marker = "search the catalog for what the user described"

    def run(self) -> list[dict[str, Any]]:
        return super().run()


class ConstraintSatisfaction(Engine):
    marker = "the user gives firm, non-negotiable requirements"

    def run(self) -> list[dict[str, Any]]:
        return super().run()


class PreferenceReasoning(Engine):
    marker = "authorizes one test order"

    def run(self) -> list[dict[str, Any]]:
        return super().run()


class Ranking(Engine):
    marker = "has a stated tradeoff priority"

    def run(self) -> list[dict[str, Any]]:
        return super().run()


class Recovery(Engine):
    marker = "critical constraints are category, budget, and in-stock availability"

    def run(self) -> list[dict[str, Any]]:
        return super().run()


class Justification(Engine):
    marker = "a valid product plus a trustworthy reason"

    def run(self) -> list[dict[str, Any]]:
        return super().run()


FAMILIES: tuple[type[Engine], ...] = (
    IntentDecomposition,
    RetrievalRecall,
    ConstraintSatisfaction,
    PreferenceReasoning,
    Ranking,
    Recovery,
    Justification,
)


# -- entry -----------------------------------------------------------------------

def clarify(query: str) -> type[Engine]:
    rules = " ".join(query.partition("Task rules:")[2].split()).casefold()
    for family in FAMILIES:
        if family.marker in rules:
            return family

    try:
        answer = Utils._llm(Prompt.CLARIFY.format(query=query), max_tokens=16).get("content") or ""
    except Exception:
        return Engine
    return next((family for family in FAMILIES if family.__name__ in answer), Engine)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Place the task in its family and run that family's engine."""

    family = clarify(problem_data["environment"]["policy_view"]["query"])
    return family(problem_data).run()


__all__ = ["agent_main"]
