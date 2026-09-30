import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.llm import OllamaLLM


def main():

    llm = OllamaLLM(
        model="qwen3:1.7b"
    )

    prompt = """
You are an agricultural assistant.

Explain what rice blast disease is
in simple terms.
"""

    print("Sending request to Ollama...\n")

    answer = llm.generate(prompt)

    print("LLM Response:")
    print(answer)


if __name__ == "__main__":
    main()