def build_rag_prompt(
    question: str,
    retrieved_documents: list
) -> str:

    context_parts = []

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Document: {document['source']}
Page: {document['page']}

{document['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are an agricultural knowledge assistant.

Your task is to answer the user's question
using ONLY the information provided in the
context below.

If the context does not contain enough
information to answer the question, say:

"I don't have enough information in the
provided agricultural documents to answer
this question."

Do not invent facts.

Give a clear and practical answer.

Always mention the relevant source document
and page number when possible.

================ CONTEXT ================

{context}

============== END CONTEXT ==============

USER QUESTION:

{question}

================ ANSWER ================
"""

    return prompt