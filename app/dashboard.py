"""
app/dashboard.py
SEC Filing Analyst — RAG-powered Q&A dashboard.
Run: streamlit run app/dashboard.py
"""

import os
import sys
import time
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from pipeline.edgar import KNOWN_COMPANIES, get_recent_filings, fetch_filing_text
from pipeline.vectorstore import FilingVectorStore, get_store_path, store_exists
from models.analyst import answer_question, summarize_filing, PRESET_QUESTIONS

st.set_page_config(
    page_title="SEC Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Kill the white header bar ── */
header[data-testid="stHeader"] {
  background: #0b0f1a !important;
  border-bottom: none !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── Main background ── */
.stApp, .stApp > div, .block-container {
  background: #0b0f1a !important;
}
.block-container {
  padding-top: 2.5rem !important;
  max-width: 1200px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #0f1420 !important;
  border-right: 1px solid #1a2236 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
  color: #8892a4 !important;
}
section[data-testid="stSidebar"] .stRadio > div > label > div > p {
  color: #c4cdd8 !important;
  font-size: 13px !important;
}

/* ── All text defaults ── */
p, span, li, td, th { color: #c4cdd8 !important; }
h1, h2, h3, h4 { color: #e8edf3 !important; }
label { color: #8892a4 !important; }
hr { border-color: #1a2236 !important; }

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
  background: #131927 !important;
  border: 1px solid #1e2d45 !important;
  border-radius: 7px !important;
  color: #c4cdd8 !important;
  font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input::placeholder { color: #3d4f6a !important; }
.stTextInput > div > div > input:focus {
  border-color: #3b82f6 !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* ── Primary button (amber → now blue) ── */
.stButton > button {
  background: #1d3461 !important;
  color: #93c5fd !important;
  border: 1px solid #2a4a8a !important;
  border-radius: 7px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  background: #243d77 !important;
  border-color: #3b82f6 !important;
  color: #bfdbfe !important;
}

/* ── Selected question button ── */
.stButton > button[data-selected="true"],
button.selected-q {
  background: #1e40af !important;
  border-color: #3b82f6 !important;
  color: #ffffff !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div { background: #3b82f6 !important; }

/* ── Expander ── */
details > summary {
  background: #131927 !important;
  border: 1px solid #1e2d45 !important;
  border-radius: 7px !important;
  color: #8892a4 !important;
  font-size: 12px !important;
}

/* ── Custom components ── */
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(59,130,246,0.08);
  border: 1px solid rgba(59,130,246,0.25);
  border-radius: 20px; padding: 4px 14px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
  color: #60a5fa !important; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 14px; display: inline-block;
}
.page-title {
  font-size: 2.1rem; font-weight: 600;
  color: #e8edf3 !important; margin: 0 0 8px;
  letter-spacing: -0.03em; line-height: 1.15;
}
.page-sub {
  font-size: 0.9rem; color: #5a6a80 !important;
  margin: 0 0 2.2rem; line-height: 1.6;
}
.section-label {
  font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
  color: #3b82f6 !important; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 10px; display: block;
}
.stat-card {
  background: #0f1420; border: 1px solid #1a2236;
  border-radius: 8px; padding: 1rem 1.2rem; text-align: center;
}
.stat-label {
  font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
  color: #3d4f6a !important; font-family: 'JetBrains Mono', monospace;
  margin-bottom: 6px;
}
.stat-value {
  font-size: 1.6rem; font-weight: 600;
  color: #60a5fa !important; font-family: 'JetBrains Mono', monospace;
}
.answer-wrap {
  border-radius: 8px; overflow: hidden; margin-bottom: 1.5rem;
  border: 1px solid #1a2236;
}
.question-header {
  background: #0f1420; padding: 0.7rem 1.4rem;
  font-size: 12px; font-weight: 600;
  color: #60a5fa !important; font-family: 'JetBrains Mono', monospace;
  border-bottom: 1px solid #1a2236; letter-spacing: 0.02em;
}
.answer-body {
  background: #131927; padding: 1.4rem 1.6rem;
  font-size: 0.93rem; line-height: 1.85; color: #c4cdd8 !important;
  border-top: 2px solid #3b82f6;
}
.excerpt-card {
  background: #0b0f1a; border: 1px solid #1a2236;
  border-left: 3px solid #22c55e;
  border-radius: 0 6px 6px 0;
  padding: 0.85rem 1.1rem; margin: 8px 0;
  font-size: 0.82rem; color: #6b7a90 !important; line-height: 1.7;
}
.tag {
  display: inline-block; font-size: 10px; padding: 2px 8px;
  border-radius: 10px; margin-right: 5px; margin-bottom: 6px;
  font-family: 'JetBrains Mono', monospace; font-weight: 500;
}
.tag-section { background: rgba(34,197,94,0.1); color: #4ade80 !important; border: 1px solid rgba(34,197,94,0.2); }
.tag-score   { background: rgba(59,130,246,0.1); color: #60a5fa !important; border: 1px solid rgba(59,130,246,0.2); }
.step-card {
  background: #0f1420; border: 1px solid #1a2236;
  border-radius: 8px; padding: 1.2rem 1.1rem;
}
.step-num-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
  color: #3b82f6 !important; font-family: 'JetBrains Mono', monospace;
  margin-bottom: 8px;
}
.step-title { font-size: 13px; font-weight: 600; color: #e8edf3 !important; margin-bottom: 6px; }
.step-desc  { font-size: 12px; color: #5a6a80 !important; line-height: 1.6; }
.sidebar-stat {
  background: #131927; border: 1px solid #1a2236;
  border-radius: 6px; padding: 0.65rem 0.9rem; margin-bottom: 6px;
}
.sidebar-label { font-size: 10px; color: #3d4f6a !important; letter-spacing: 0.08em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }
.sidebar-value { font-size: 13px; font-weight: 500; color: #c4cdd8 !important; margin-top: 2px; }
.loaded-pill {
  background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2);
  border-radius: 6px; padding: 0.6rem 0.9rem;
  font-size: 11px; color: #4ade80 !important;
  font-family: 'JetBrains Mono', monospace; margin-bottom: 10px;
}
.no-filing {
  background: #0f1420; border: 1px solid #1a2236;
  border-radius: 6px; padding: 0.9rem;
  font-size: 12px; color: #3d4f6a !important; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("vector_store", None), ("filing_meta", None),
    ("qa_history", []), ("summary", None),
    ("selected_q", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.75rem 0 1rem'>
      <div style='font-size:17px;font-weight:700;color:#e8edf3;letter-spacing:-0.02em'>
        📊 SEC Analyst
      </div>
      <div style='font-size:10px;color:#3d4f6a;font-family:JetBrains Mono,monospace;
                  letter-spacing:0.08em;text-transform:uppercase;margin-top:5px'>
        RAG · FAISS · GEMINI 2.5
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:0 0 1rem'>", unsafe_allow_html=True)

    page = st.radio(
        "nav", ["🏠 Load Filing", "💬 Q&A", "📄 Summary", "ℹ️ How it works"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)

    if st.session_state.filing_meta:
        m      = st.session_state.filing_meta
        chunks = len(st.session_state.vector_store.chunks) if st.session_state.vector_store else 0
        st.markdown(f"""
        <div class='loaded-pill'>● FILING LOADED</div>
        <div class='sidebar-stat'>
          <div class='sidebar-label'>Company</div>
          <div class='sidebar-value'>{m.get("company","?")}</div>
        </div>
        <div class='sidebar-stat'>
          <div class='sidebar-label'>Form · Filed</div>
          <div class='sidebar-value'>{m.get("form","?")} · {m.get("date","?")}</div>
        </div>
        <div class='sidebar-stat'>
          <div class='sidebar-label'>Chunks indexed</div>
          <div class='sidebar-value' style='color:#60a5fa !important;font-family:JetBrains Mono,monospace'>{chunks:,}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='no-filing'>
          No filing loaded.<br>
          Go to <strong style='color:#60a5fa'>Load Filing</strong> to get started.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · LOAD FILING
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Load Filing":
    st.markdown('<span class="hero-badge">📄 SEC EDGAR · FREE · NO AUTH REQUIRED</span>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">SEC Filing Analyst</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Load any 10-K or 10-Q, build a local FAISS index, and ask questions grounded in the actual filing — every answer cited to the source.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<span class="section-label">01 · Select company</span>', unsafe_allow_html=True)
        company_name = st.selectbox("Company", list(KNOWN_COMPANIES.keys()), label_visibility="collapsed")
        form_type    = st.selectbox("Form type", ["10-K", "10-Q"])

    with col2:
        st.markdown('<span class="section-label">02 · Select filing</span>', unsafe_allow_html=True)
        cik = KNOWN_COMPANIES[company_name]
        with st.spinner(""):
            try:
                filings = get_recent_filings(cik, form_type=form_type, count=5)
            except Exception as e:
                st.error(f"Error: {e}"); filings = []

        if filings:
            opts           = {f"{f['form']} · {f['date']}": f for f in filings}
            selected_label = st.selectbox("Filing", list(opts.keys()), label_visibility="collapsed")
            selected       = opts[selected_label]
            st.markdown(
                f"<div style='font-size:11px;color:#3d4f6a;font-family:JetBrains Mono,monospace;margin-top:6px'>"
                f"CIK {cik} · <a href='{selected.get('url','#')}' target='_blank' "
                f"style='color:#60a5fa;text-decoration:none'>View on SEC.gov ↗</a></div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No filings found."); selected = None

    st.markdown("<br>", unsafe_allow_html=True)

    if selected:
        store_path = get_store_path(company_name, form_type, selected["date"])
        cached     = store_exists(company_name, form_type, selected["date"])

        if cached:
            st.markdown("""
            <div style='background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.18);
                        border-radius:6px;padding:0.6rem 1rem;font-size:12px;
                        color:#4ade80;font-family:JetBrains Mono,monospace;margin-bottom:12px'>
              ✓ Already indexed — will load from cache instantly.
            </div>""", unsafe_allow_html=True)

        if st.button("⚡  Load & Index Filing", use_container_width=True):
            if cached:
                with st.spinner("Loading cached index..."):
                    st.session_state.vector_store = FilingVectorStore.load(store_path)
                    st.session_state.filing_meta  = selected
                    st.session_state.qa_history   = []
                    st.session_state.summary      = None
                st.success(f"Loaded {len(st.session_state.vector_store.chunks):,} chunks from cache.")
            else:
                bar = st.progress(0, text="⬇  Downloading from SEC EDGAR...")
                try:
                    text = fetch_filing_text(selected)
                    bar.progress(35, text="✂  Chunking into passages...")
                    store = FilingVectorStore()
                    store.build(text, selected)
                    bar.progress(80, text="💾  Saving FAISS index...")
                    store.save(store_path)
                    st.session_state.vector_store = store
                    st.session_state.filing_meta  = selected
                    st.session_state.qa_history   = []
                    st.session_state.summary      = None
                    bar.progress(100, text="Done!")
                    time.sleep(0.4); bar.empty()
                    st.success(f"✓  Indexed {len(store.chunks):,} chunks — go to Q&A to start asking questions →")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">How it works</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, n, title, desc in [
        (c1, "01", "Fetch",  "Downloads 10-K/10-Q HTML from SEC EDGAR's free public API. No auth needed."),
        (c2, "02", "Chunk",  "Splits the document into 500-word overlapping passages to preserve context."),
        (c3, "03", "Embed",  "Encodes passages with all-MiniLM-L6-v2 (local, 22MB) → FAISS flat index."),
        (c4, "04", "Answer", "Gemini 2.5 Flash answers using only the top-6 retrieved chunks, with [Excerpt N] citations."),
    ]:
        col.markdown(
            f'<div class="step-card"><div class="step-num-label">{n}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · Q&A
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Q&A":
    if not st.session_state.vector_store:
        st.warning("No filing loaded. Go to **Load Filing** first.")
        st.stop()

    m = st.session_state.filing_meta
    st.markdown('<span class="hero-badge">💬 GROUNDED Q&A · CITED ANSWERS</span>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-title">{m["company"]}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="page-sub">{m["form"]} · Filed {m["date"]} · '
        f'{len(st.session_state.vector_store.chunks):,} chunks indexed</p>',
        unsafe_allow_html=True,
    )

    # ── Preset questions with selected state ──
    st.markdown('<span class="section-label">Quick questions</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(2, gap="small")
    for i, q in enumerate(PRESET_QUESTIONS[:6]):
        is_selected = st.session_state.selected_q == q
        label       = f"✓  {q}" if is_selected else q
        btn_style   = "background:#1e40af!important;border-color:#3b82f6!important;color:#fff!important;" if is_selected else ""
        if cols[i % 2].button(label, key=f"preset_{i}", use_container_width=True):
            st.session_state.selected_q       = q
            st.session_state._pending_question = q
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Or ask your own</span>', unsafe_allow_html=True)

    user_q   = st.text_input("question", placeholder="e.g. What were the biggest revenue drivers this year?",
                              key="user_question", label_visibility="collapsed")
    question = getattr(st.session_state, "_pending_question", None) or user_q

    btn_col, _ = st.columns([1, 5])
    if question and btn_col.button("Ask →"):
        if not os.getenv("GEMINI_API_KEY"):
            st.error("Set GEMINI_API_KEY in your .env file.")
        else:
            with st.spinner("Searching filing and generating answer..."):
                try:
                    result = answer_question(question, st.session_state.vector_store, top_k=6)
                    st.session_state.qa_history.insert(0, result)
                    if hasattr(st.session_state, "_pending_question"):
                        del st.session_state._pending_question
                    st.session_state.selected_q = None
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Answer history ──
    if st.session_state.qa_history:
        st.markdown("<br>", unsafe_allow_html=True)
        for result in st.session_state.qa_history:
            st.markdown(
                f'<div class="answer-wrap">'
                f'<div class="question-header">Q · {result["question"]}</div>'
                f'<div class="answer-body">{result["answer"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"📎  View {len(result['chunks'])} source excerpts"):
                for i, chunk in enumerate(result["chunks"], 1):
                    score   = chunk.get("score", 0)
                    section = chunk.get("section", "General")
                    st.markdown(
                        f'<span class="tag tag-section">{section}</span>'
                        f'<span class="tag tag-score">score {score:.3f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="excerpt-card">'
                        f'<span style="color:#60a5fa!important;font-weight:600">[Excerpt {i}]</span> '
                        f'{chunk["text"][:420]}…</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Summary":
    if not st.session_state.vector_store:
        st.warning("No filing loaded. Go to **Load Filing** first.")
        st.stop()

    m  = st.session_state.filing_meta
    vs = st.session_state.vector_store

    st.markdown('<span class="hero-badge">📄 AI EXECUTIVE SUMMARY</span>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">Executive Summary</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-sub">{m["company"]} · {m["form"]} · Filed {m["date"]}</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    for col, label, val in [
        (c1, "Chunks indexed", f"{len(vs.chunks):,}"),
        (c2, "Filing type",    m["form"]),
        (c3, "Filed",          m["date"]),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.summary:
        st.markdown(
            f'<div class="answer-wrap">'
            f'<div class="question-header">EXECUTIVE SUMMARY · {m["company"].upper()} · {m["form"]} {m["date"]}</div>'
            f'<div class="answer-body">{st.session_state.summary}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("↺  Regenerate"):
            st.session_state.summary = None
            st.rerun()
    else:
        if st.button("⚡  Generate Executive Summary", use_container_width=True):
            if not os.getenv("GEMINI_API_KEY"):
                st.error("Set GEMINI_API_KEY in your .env file.")
            else:
                with st.spinner("Analysing filing..."):
                    try:
                        st.session_state.summary = summarize_filing(st.session_state.vector_store)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ How it works":
    st.markdown('<span class="hero-badge">ℹ️ ARCHITECTURE</span>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">How it works</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">RAG pipeline for grounded, cited financial document analysis</p>', unsafe_allow_html=True)

    st.markdown("""
### Retrieval-Augmented Generation (RAG)

Standard LLMs hallucinate when asked about specific documents. RAG solves this by anchoring every answer in retrieved passages from the actual filing:

1. **Chunk** — filing split into 500-word passages with 100-word overlap
2. **Embed** — each passage encoded with `all-MiniLM-L6-v2` → 384-dim vector
3. **Index** — vectors stored in a FAISS flat inner-product index (exact cosine similarity)
4. **Retrieve** — question embedded and top-6 nearest chunks returned in <5ms
5. **Generate** — Gemini 2.5 Flash answers using only those chunks, with `[Excerpt N]` citations

---

### Key design decisions

| Component | Choice | Why |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | 22MB, local CPU, no API cost |
| Vector DB | FAISS flat index | Exact search, zero approximation errors |
| Chunk size | 500 words / 100 overlap | Balances context vs. retrieval precision |
| LLM | Gemini 2.5 Flash | Free tier, strong instruction following |
| Filing source | SEC EDGAR API | Free, official, no auth, all US public companies |
| Citations | `[Excerpt N]` inline | Every claim auditable back to source |
    """)