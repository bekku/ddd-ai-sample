"""プロンプトを定義する"""


class QuestionAnswerPrompt:
    """質問に対する回答を生成するプロンプト"""

    def __init__(self, question: str) -> None:
        """初期化"""
        self.question = question

    def get_prompt(self) -> str:
        """プロンプトを取得"""
        return f"""あなたは,質問に対する回答を生成するためのAIアシスタントです。
        質問: {self.question}
        に対する回答を生成してください。
        """
