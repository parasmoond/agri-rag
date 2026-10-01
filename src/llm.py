import os
import time
from google import genai


class GeminiLLM:

    def __init__(
        self,
        model="gemini-3.8-flash"
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

    def generate(
        self,
        prompt: str
    ) -> str:

        max_retries = 4

        for attempt in range(max_retries):

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )

                return response.text

            except Exception as e:

                if attempt == max_retries - 1:
                    raise e

                wait_time = 2 ** attempt
                time.sleep(wait_time)