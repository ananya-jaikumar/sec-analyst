"""
models/analyst.py
RAG-based Q&A analyst using Gemini 2.5 Flash.

Flow:
  1. Embed user question
  2. Retrieve top-k relevant chunks from FAISS
  3. Build grounded prompt with retrieved passages
  4. Gemini generates answer with inline citations
"""

import os
import google.generativeai as genai

_model = None

SYSTEM_PROMPT = """You are a senior financial analyst specializing in SEC filings.
You answer questions about companies based ONLY on the provided excerpts from their SEC filings.

Rules:
- Ground every claim in the provided excerpts
- Cite your sources inline using [Excerpt N] notation
- If the excerpts don't contain enough information, say so clearly
- Be precise with numbers, dates, and percentages — never approximate
- Structure your answer clearly: lead with the direct answer, then supporting detail
- Do NOT speculate beyond what the filings state
- Write in a professional analyst tone
"""


def _get_model():
    global _model
    if _model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
    return _model


PRESET_QUESTIONS = [
    "What are the main risk factors the company has identified?",
    "How did revenue and net income change compared to the prior year?",
    "What is the company's strategy for growth?",
    "What are the key competitive threats mentioned?",
    "How does the company describe its liquidity and capital resources?",
    "What legal proceedings or regulatory risks are disclosed?",
    "What does management say about macroeconomic uncertainty?",
    "How is the company investing in AI or new technology?",
]


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """Build a grounded prompt from retrieved chunks."""
    excerpts = ""
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section", "General")
        text    = chunk["text"][:800]   # cap each excerpt
        excerpts += f"\n[Excerpt {i}] (Section: {section})\n{text}\n"

    return f"""Based on the following excerpts from the SEC filing, answer this question:

QUESTION: {question}

FILING EXCERPTS:
{excerpts}

Please provide a thorough answer with inline citations [Excerpt N] for every claim."""


def answer_question(
    question: str,
    vector_store,
    top_k: int = 6,
    stream: bool = False,
):
    """
    Full RAG pipeline: retrieve → prompt → generate.

    Args:
        question: User's natural language question
        vector_store: FilingVectorStore instance
        top_k: Number of chunks to retrieve
        stream: If True, returns streaming generator

    Returns:
        dict with keys: answer, chunks (retrieved), question
    """
    # Retrieve
    chunks = vector_store.search(question, top_k=top_k)
    prompt = build_rag_prompt(question, chunks)
    model  = _get_model()

    if stream:
        response = model.generate_content(prompt, stream=True)
        return {"chunks": chunks, "question": question, "stream": response}
    else:
        response = model.generate_content(prompt)
        return {
            "answer":   response.text,
            "chunks":   chunks,
            "question": question,
        }


def summarize_filing(vector_store, stream: bool = False):
    """Generate an executive summary of the filing."""

    # Sample key sections
    key_queries = [
        "business overview revenue products",
        "risk factors regulatory competition",
        "financial performance revenue income",
        "future outlook strategy growth",
    ]

    all_chunks = []
    for q in key_queries:
        chunks = vector_store.search(q, top_k=2)
        all_chunks.extend(chunks)

    # Deduplicate by chunk_id
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique_chunks.append(c)

    excerpts = "\n\n".join(
        f"[Section: {c.get('section','General')}]\n{c['text'][:600]}"
        for c in unique_chunks[:8]
    )

    prompt = f"""Based on these excerpts from the SEC 10-K filing, write a concise executive summary covering:
1. Business overview (what the company does, key products/services)
2. Financial highlights (revenue, profitability, key metrics)
3. Key risks (top 3 material risks)
4. Strategic outlook (growth plans, investments)

Keep it to 4 short paragraphs. Be specific — include actual numbers and figures where present.

EXCERPTS:
{excerpts}"""

    model = _get_model()
    if stream:
        return model.generate_content(prompt, stream=True)
    else:
        return model.generate_content(prompt).text
