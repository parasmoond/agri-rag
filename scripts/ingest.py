import sys
import json
from pathlib import Path

# Allow importing modules from src/
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.pdf_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import chunk_text


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def process_pdf(pdf_path: Path):
    """
    Extract, clean, and chunk a single PDF.
    """

    print(f"\nProcessing: {pdf_path.name}")

    # Step 1: Extract text from PDF
    pages = extract_text_from_pdf(
        str(pdf_path)
    )

    print(f"Pages found: {len(pages)}")

    all_chunks = []

    # Step 2: Process each page
    for page in pages:

        page_number = page["page_number"]
        raw_text = page["text"]

        # Step 3: Clean extracted text
        cleaned_text = clean_text(
            raw_text
        )

        # Skip empty pages
        if not cleaned_text:
            continue

        # Step 4: Split text into chunks
        chunks = chunk_text(
            cleaned_text,
            chunk_size=1000,
            chunk_overlap=200
        )

        # Step 5: Add metadata to every chunk
        for chunk_id, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "source": pdf_path.name,
                "page": page_number,
                "chunk_id": chunk_id
            })

    print(
        f"Chunks created: {len(all_chunks)}"
    )

    return all_chunks


def save_chunks(chunks, output_path):
    """
    Save all processed chunks to a JSON file.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved chunks to: {output_path}"
    )


def main():

    # Create processed directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find all PDFs in data/raw/
    pdf_files = list(
        RAW_DATA_DIR.glob("*.pdf")
    )

    # Check if PDFs exist
    if not pdf_files:

        print(
            "No PDF files found in data/raw/"
        )

        return

    # Store chunks from all PDFs
    all_chunks = []

    # Process every PDF
    for pdf_path in pdf_files:

        chunks = process_pdf(
            pdf_path
        )

        all_chunks.extend(chunks)

    # Output file
    output_path = (
        PROCESSED_DATA_DIR
        / "chunks.json"
    )

    # Save all chunks
    save_chunks(
        all_chunks,
        output_path
    )


if __name__ == "__main__":
    main()