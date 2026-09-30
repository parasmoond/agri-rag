import sys
import json
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.rag_pipeline import RAGPipeline


QUESTIONS_PATH = Path(
    "evaluation/answer_questions.json"
)


def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def main():

    questions = load_questions()

    rag = RAGPipeline(
        model_name="qwen3:1.7b"
    )

    answerable_total = 0
    unanswerable_total = 0

    answerable_success = 0
    unanswerable_success = 0

    print("=" * 80)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 80)

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]
        question_type = item["type"]

        print("\n")
        print("=" * 80)
        print(f"QUESTION {number}")
        print("=" * 80)

        print(f"\nQuestion:")
        print(question)

        print(
            f"\nExpected type: "
            f"{question_type}"
        )

        result = rag.answer(
            question=question,
            retrieval_k=10,
            final_k=3
        )

        answer = result["answer"]

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES:")

        for source in result["sources"]:

            print(
                f"- {source['source']} "
                f"(Page {source['page']})"
            )

        # ------------------------------------------------
        # Statistics
        # ------------------------------------------------

        if question_type == "answerable":

            answerable_total += 1

        else:

            unanswerable_total += 1

        # ------------------------------------------------
        # Manual evaluation
        # ------------------------------------------------

        print("\nMANUAL EVALUATION")

        if question_type == "answerable":

            print(
                "Was the answer useful and "
                "supported by the sources?"
            )

        else:

            print(
                "Did the system correctly refuse "
                "to invent information?"
            )

        print(
            "Enter Y if successful, "
            "N if unsuccessful:"
        )

        evaluation = input(
            "> "
        ).strip().lower()

        if evaluation == "y":

            if question_type == "answerable":

                answerable_success += 1

            else:

                unanswerable_success += 1

    # ----------------------------------------------------
    # Final results
    # ----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ANSWER QUALITY RESULTS")
    print("=" * 80)

    print(
        f"\nAnswerable questions: "
        f"{answerable_total}"
    )

    print(
        f"Successful answers: "
        f"{answerable_success}"
    )

    if answerable_total > 0:

        answerable_accuracy = (
            answerable_success
            / answerable_total
        )

        print(
            f"Answerable success rate: "
            f"{answerable_accuracy:.4f}"
        )

    print(
        f"\nUnanswerable questions: "
        f"{unanswerable_total}"
    )

    print(
        f"Correct abstentions: "
        f"{unanswerable_success}"
    )

    if unanswerable_total > 0:

        abstention_rate = (
            unanswerable_success
            / unanswerable_total
        )

        print(
            f"Abstention success rate: "
            f"{abstention_rate:.4f}"
        )


if __name__ == "__main__":
    main()