import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving useful content.
    """

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove spaces at the beginning/end of lines
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return text.strip()