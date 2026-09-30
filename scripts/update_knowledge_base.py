import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_name):
    script_path = PROJECT_ROOT / "scripts" / script_name

    print(f"\nRunning: {script_name}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            f"{script_name} failed."
        )


def update_knowledge_base():

    print("=" * 60)
    print("UPDATING AGRICULTURAL KNOWLEDGE BASE")
    print("=" * 60)

    # Step 1:
    # Extract PDFs, clean text and create chunks

    run_script("ingest.py")

    # Step 2:
    # Generate embeddings and rebuild FAISS

    run_script("build_index.py")

    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE UPDATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    update_knowledge_base()