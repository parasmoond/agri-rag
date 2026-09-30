import requests


class OllamaLLM:

    def __init__(
        self,
        model="qwen3:1.7b",
        base_url="http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        return result["response"]