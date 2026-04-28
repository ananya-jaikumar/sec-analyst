<div align="center">

# 📊 SEC Filing Analyst

### Ask anything about any public company. Get cited, grounded answers in seconds.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00ADD8?style=flat-square)](https://faiss.ai)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Free_API-4285F4?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.5x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

> A RAG pipeline that fetches 10-K and 10-Q filings directly from SEC EDGAR, builds a local FAISS vector index, and answers natural-language questions with inline `[Excerpt N]` citations — every claim traceable to the source filing.

<br/>

![Dashboard Preview](assets/screenshot.png)

</div>

---

## 🚨 The Problem

Financial analysts spend **4–6 hours** manually reading 80-page 10-K filings to extract risk signals, revenue drivers, and strategic outlook. Critical information is buried in dense legal prose across dozens of sections.

This project makes any 10-K instantly queryable in plain English — with every answer cited back to the exact passage it came from.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📥 **Live filing fetch** | Pulls 10-K and 10-Q filings directly from SEC EDGAR's free public API — no auth required |
| 🔍 **Semantic search** | Local FAISS vector index over document chunks, <5ms retrieval |
| 💬 **Cited Q&A** | Gemini 2.5 Flash answers grounded in retrieved excerpts with `[Excerpt N]` citations |
| 📄 **Executive summary** | One-click AI summary covering business, financials, risks, and outlook |
| ⚡ **Disk caching** | Indexed filings saved to disk — reload in under 1 second |
| 🎨 **Terminal UI** | Dark navy dashboard built with Streamlit + custom CSS |

---

## 🏗️ RAG Architecture

```
SEC EDGAR API (free, no auth required)
        │
        ▼
pipeline/edgar.py         →   Fetch 10-K HTML → navigate filing index → clean text (~300K chars)
        │
        ▼
pipeline/vectorstore.py   →   500-word chunks → all-MiniLM-L6-v2 embeddings → FAISS flat index
        │
        ▼
models/analyst.py         →   Embed query → top-6 retrieval → Gemini 2.5 Flash → cited answer
        │
        ▼
app/dashboard.py          →   Streamlit: Load Filing / Q&A / Summary / Architecture
```

---

## 💬 Example Q&A

**Q: What are the main risk factors Apple has identified?**

> Apple's 10-K identifies several material risks. **Data security and account management** risks include the potential for freezing accounts under suspicious circumstances, which can delay customer orders — the company's insurance coverage for such losses may be insufficient [Excerpt 2]. **Product quality and liability** exposure is flagged explicitly, noting that vulnerabilities exploited by third parties could compromise device safety, leading to recalls, write-offs, and regulatory fines [Excerpt 3]. **Business interruptions** due to reliance on single-source suppliers for critical components represent a further material risk [Excerpt 6]...

---

## 🚀 Quickstart

**Prerequisites:** Python 3.10+, free [Gemini API key](https://aistudio.google.com/apikey)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sec-analyst.git
cd sec-analyst

# 2. Virtual environment
python3 -m venv venv && source venv/bin/activate  # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Add your free Gemini API key
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key_here

# 5. Launch
python3 -m streamlit run app/dashboard.py
```

No dataset download needed — filings are fetched live from SEC EDGAR on demand.

---

## 📁 Project Structure

```
sec-analyst/
├── 📂 pipeline/
│   ├── edgar.py          # SEC EDGAR API client — fetch & parse filings
│   └── vectorstore.py    # Chunking + sentence-transformer embeddings + FAISS index
├── 📂 models/
│   └── analyst.py        # RAG core: retrieve → prompt → Gemini 2.5 Flash
├── 📂 app/
│   └── dashboard.py      # Streamlit dashboard — 4 pages
├── 📂 data/              # Auto-created on first load (gitignored)
├── .env.example          # API key template
├── requirements.txt
└── README.md
```

---

## 🧠 Key Technical Decisions

**Local embeddings (all-MiniLM-L6-v2)** — 22MB model runs entirely on CPU with zero API cost. Produces 384-dimensional vectors trained specifically for semantic similarity, capturing meaning far beyond keyword matching.

**FAISS flat index** — Exact inner-product search on normalized vectors (cosine similarity). No approximation errors. For ~800 chunks, search completes in under 5ms — no need for ANN methods.

**Chunk overlap** — 100-word overlap between 500-word chunks prevents information loss at section boundaries, which is critical for questions spanning multiple parts of a filing.

**Grounded prompting** — The system prompt strictly constrains Gemini to answer only from retrieved excerpts. Inline `[Excerpt N]` citations make every claim auditable back to the exact passage in the document.

**EDGAR filing index navigation** — Rather than using the primary document URL (often just the cover page), the pipeline fetches the filing's index JSON, identifies the largest `.htm` file by byte size, and downloads the full report body.

**Disk caching** — Indexed filings saved as FAISS binary + joblib chunks. Reloading takes <1 second vs. 2–3 minutes for full re-indexing.

---

## 🏢 Supported Companies

Apple · Microsoft · Tesla · Amazon · Google · Meta · Netflix · Nvidia

Easily extended — add any US public company's CIK number to `KNOWN_COMPANIES` in `pipeline/edgar.py`.

---

## 🛠️ Skills Demonstrated

`RAG` `FAISS` `Vector Embeddings` `Sentence Transformers` `LLM Integration` `Gemini API` `SEC EDGAR API` `Information Retrieval` `Grounded Generation` `Streamlit` `NLP` `Python`

---

## 📦 Data Source

**SEC EDGAR** — The SEC's free public API provides access to all filings from publicly traded US companies. No API key, no rate limits for reasonable use.

Documentation: [SEC EDGAR Developer Resources](https://www.sec.gov/developer)

---

## 📄 License

MIT © 2025
