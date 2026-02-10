"""Gemini APIを使用したLLMクライアントのテスト."""
# ruff: noqa: S101

from types import SimpleNamespace
from unittest.mock import patch

from app.domain.llm_client.llm_client_base import LLMResult, Prompt
from app.infra.llm_client.gemini import GeminiClient

# ================================
# GeminiClient
# ================================


class TestGeminiClient:
    """GeminiClientのテスト"""

    def test_init_default_parameters(self) -> None:
        """デフォルトパラメータで初期化できる"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            client = GeminiClient(api_key="test-api-key")

            mock_client_cls.assert_called_once_with(api_key="test-api-key")
            assert client.model == "gemini-2.5-flash-lite"
            assert client.include_thoughts is True

    def test_init_custom_parameters(self) -> None:
        """カスタムパラメータで初期化できる"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            client = GeminiClient(
                api_key="custom-key",
                model="gemini-pro",
                include_thoughts=False,
            )

            mock_client_cls.assert_called_once_with(api_key="custom-key")
            assert client.model == "gemini-pro"
            assert client.include_thoughts is False

    def test_predict_output_only(self) -> None:
        """出力のみのレスポンスを正しくパースし、thinkingはNoneになる"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            part = SimpleNamespace(text="回答テキスト", thought=False)
            response = SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
            )
            mock_client_cls.return_value.models.generate_content.return_value = response

            client = GeminiClient(api_key="test-key")
            result = client.predict(Prompt(text="テスト質問"))

            assert result.output == "回答テキスト"
            assert result.thinking is None

    def test_predict_with_thinking_and_output(self) -> None:
        """思考ログと出力の両方があるレスポンスを正しくパースする"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            thinking_part = SimpleNamespace(text="思考内容", thought=True)
            output_part = SimpleNamespace(text="回答テキスト", thought=False)
            response = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[thinking_part, output_part]),
                    ),
                ],
            )
            mock_client_cls.return_value.models.generate_content.return_value = response

            client = GeminiClient(api_key="test-key")
            result = client.predict(Prompt(text="テスト質問"))

            assert result.output == "回答テキスト"
            assert result.thinking == "思考内容"

    def test_predict_skips_parts_without_text_attribute(self) -> None:
        """text属性がないパートはスキップされる"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            part_no_text = SimpleNamespace()
            part_with_text = SimpleNamespace(text="回答", thought=False)
            response = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[part_no_text, part_with_text]),
                    ),
                ],
            )
            mock_client_cls.return_value.models.generate_content.return_value = response

            client = GeminiClient(api_key="test-key")
            result = client.predict(Prompt(text="質問"))

            assert result.output == "回答"
            assert result.thinking is None

    def test_predict_skips_parts_with_empty_text(self) -> None:
        """textが空文字のパートはスキップされる"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            part_empty = SimpleNamespace(text="", thought=False)
            part_valid = SimpleNamespace(text="有効な回答", thought=False)
            response = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[part_empty, part_valid]),
                    ),
                ],
            )
            mock_client_cls.return_value.models.generate_content.return_value = response

            client = GeminiClient(api_key="test-key")
            result = client.predict(Prompt(text="質問"))

            assert result.output == "有効な回答"

    def test_predict_multiple_thinking_and_output_parts(self) -> None:
        """複数の思考ログと出力パートが改行で結合される"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            parts = [
                SimpleNamespace(text="思考1", thought=True),
                SimpleNamespace(text="思考2", thought=True),
                SimpleNamespace(text="出力1", thought=False),
                SimpleNamespace(text="出力2", thought=False),
            ]
            response = SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
            )
            mock_client_cls.return_value.models.generate_content.return_value = response

            client = GeminiClient(api_key="test-key")
            result = client.predict(Prompt(text="質問"))

            assert result.thinking == "思考1\n思考2"
            assert result.output == "出力1\n出力2"

    def test_predict_returns_llm_result_instance(self) -> None:
        """戻り値がLLMResultインスタンスである"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            part = SimpleNamespace(text="回答", thought=False)
            response = SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
            )
            mock_client_cls.return_value.models.generate_content.return_value = response

            client = GeminiClient(api_key="test-key")
            result = client.predict(Prompt(text="質問"))

            assert isinstance(result, LLMResult)

    def test_predict_calls_api_with_correct_parameters(self) -> None:
        """APIが正しいモデル名とプロンプト内容で呼び出される"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            part = SimpleNamespace(text="回答", thought=False)
            response = SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
            )
            mock_generate = mock_client_cls.return_value.models.generate_content
            mock_generate.return_value = response

            client = GeminiClient(api_key="test-key", model="gemini-pro")
            client.predict(Prompt(text="テストプロンプト"))

            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args
            assert call_kwargs.kwargs["model"] == "gemini-pro"
            assert call_kwargs.kwargs["contents"] == ["テストプロンプト"]

    def test_predict_part_without_thought_attribute(self) -> None:
        """thought属性がないパートは出力として扱われる"""
        with patch("app.infra.llm_client.gemini.genai.Client") as mock_client_cls:
            part = SimpleNamespace(text="回答テキスト")
            response = SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
            )
            mock_client_cls.return_value.models.generate_content.return_value = response

            client = GeminiClient(api_key="test-key")
            result = client.predict(Prompt(text="質問"))

            assert result.output == "回答テキスト"
            assert result.thinking is None
