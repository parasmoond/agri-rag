import re


def split_into_sentences(text: str):
    """
    Split text approximately into sentences.
    """

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Create chunks while trying to preserve
    sentence boundaries.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    sentences = split_into_sentences(text)

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        # Check whether adding this sentence
        # exceeds our target size.
        proposed_chunk = (
            current_chunk + " " + sentence
        ).strip()

        if (
            len(proposed_chunk)
            <= chunk_size
        ):

            current_chunk = proposed_chunk

        else:

            if current_chunk:

                chunks.append(
                    current_chunk
                )

            # Keep some overlap
            if chunks:

                previous_chunk = chunks[-1]

                overlap_text = (
                    previous_chunk[
                        -chunk_overlap:
                    ]
                )

                current_chunk = (
                    overlap_text
                    + " "
                    + sentence
                ).strip()

            else:

                current_chunk = sentence

    if current_chunk:

        chunks.append(
            current_chunk
        )

    return chunks