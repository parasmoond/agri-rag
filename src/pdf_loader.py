import fitz
from pathlib import Path


def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from every page of a PDF.

    Returns:
        list of dictionaries containing page number and text.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })

    document.close()

    return pages