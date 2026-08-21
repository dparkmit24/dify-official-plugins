import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from dify_plugin.entities.model.llm import LLMResult, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    TextPromptMessageContent,
)
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.interfaces.agent import AgentModelConfig

from strategies.self_refine import SelfRefineParams, SelfRefineStrategy

SATISFACTORY_EVALUATION = json.dumps(
    {"is_satisfactory": True, "issues": "", "score": 95}
)


def _llm_result(content) -> LLMResult:
    return LLMResult(
        model="test-model",
        message=AssistantPromptMessage(content=content),
        usage=LLMUsage.empty_usage(),
    )


def _drain(generator):
    """Consume a generator, returning (yielded messages, return value)."""
    messages = []
    while True:
        try:
            messages.append(next(generator))
        except StopIteration as stop:
            return messages, stop.value


class TestSelfRefineStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = SelfRefineStrategy(runtime=Mock(), session=Mock())

    @staticmethod
    def _parameters() -> dict:
        return {
            "query": "What is 2+2?",
            "instruction": "Answer arithmetic questions.",
            "model": AgentModelConfig(
                provider="test", model="test-model", mode="chat"
            ),
        }

    def test_invoke_completes_single_attempt(self):
        """Regression test for #3718: _invoke passed a `previous_output` kwarg
        that _execute_agent does not accept, so every attempt raised TypeError
        and the strategy always ended with "Error: Execution failed"."""
        self.strategy.session.model.llm.invoke = Mock(
            side_effect=[
                _llm_result("The answer is 4."),
                _llm_result(SATISFACTORY_EVALUATION),
            ]
        )

        messages, _ = _drain(self.strategy._invoke(self._parameters()))

        texts = [
            message.message.text
            for message in messages
            if message.type == ToolInvokeMessage.MessageType.TEXT
        ]
        self.assertIn("The answer is 4.", texts)
        for text in texts:
            self.assertNotIn("Error: Execution failed", text)

        error_logs = [
            message
            for message in messages
            if message.type == ToolInvokeMessage.MessageType.LOG
            and message.message.status
            == ToolInvokeMessage.LogMessage.LogStatus.ERROR
        ]
        self.assertEqual(error_logs, [])

    def test_success_logs_use_success_status(self):
        """Regression test for #3718: success-path logs used LogStatus.FINISH,
        which does not exist in dify_plugin (only START, SUCCESS, ERROR)."""
        self.strategy.session.model.llm.invoke = Mock(
            side_effect=[
                _llm_result("The answer is 4."),
                _llm_result(SATISFACTORY_EVALUATION),
            ]
        )

        messages, _ = _drain(self.strategy._invoke(self._parameters()))

        success_logs = [
            message
            for message in messages
            if message.type == ToolInvokeMessage.MessageType.LOG
            and message.message.status
            == ToolInvokeMessage.LogMessage.LogStatus.SUCCESS
        ]
        self.assertGreater(len(success_logs), 0)

    def test_execute_agent_handles_list_content(self):
        """Regression test for #3718: providers that emit list content
        (e.g. Gemini, OpenRouter) crashed with TypeError because
        AssistantPromptMessage.content was assumed to always be str."""
        self.strategy.session.model.llm.invoke = Mock(
            return_value=_llm_result(
                [
                    TextPromptMessageContent(data="part one, "),
                    TextPromptMessageContent(data="part two"),
                ]
            )
        )
        params = SelfRefineParams(**self._parameters())

        _, result = _drain(
            self.strategy._execute_agent(
                params=params, previous_critique=None, attempt_number=1
            )
        )

        self.assertEqual(result["output"], "part one, part two")

    def test_evaluate_output_handles_list_content(self):
        """Regression test for #3718: _evaluate_output called .find() on
        list content, crashing with AttributeError and falling back to the
        generic critique instead of the model's actual evaluation."""
        self.strategy.session.model.llm.invoke = Mock(
            return_value=_llm_result(
                [TextPromptMessageContent(data=SATISFACTORY_EVALUATION)]
            )
        )
        params = SelfRefineParams(**self._parameters())

        evaluation = self.strategy._evaluate_output(
            params=params, output="The answer is 4."
        )

        self.assertTrue(evaluation.is_satisfactory)
        self.assertEqual(evaluation.score, 95)


if __name__ == "__main__":
    unittest.main()
