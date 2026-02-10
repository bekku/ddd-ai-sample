"""Gemini APIを使用したLLMクライアント"""

from google import genai

from app.domain.llm_client.llm_client_base import LLMClientBase, LLMResult, Prompt


class GeminiClient(LLMClientBase):
    """Gemini APIを使用したLLMクライアントクラス"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        *,
        include_thoughts: bool = True,
    ) -> None:
        """初期化"""
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.include_thoughts = include_thoughts

    def predict(
        self,
        prompt: Prompt,
    ) -> LLMResult:
        """Gemini APIを使用してプロンプトを実行する"""
        contents: list[str] = [prompt.text]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(
                    include_thoughts=self.include_thoughts,
                ),
            ),
        )

        thinking_logs: list[str] = []
        outputs: list[str] = []

        parts = response.candidates[0].content.parts
        for part in parts:
            text = getattr(part, "text", None)
            if not text:
                continue

            if getattr(part, "thought", False):
                thinking_logs.append(text)
            else:
                outputs.append(text)

        thinking = "\n".join(thinking_logs).strip() if thinking_logs else None
        output = "\n".join(outputs).strip()

        return LLMResult(thinking=thinking, output=output)
