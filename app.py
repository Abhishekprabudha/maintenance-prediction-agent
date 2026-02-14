import os, re, io, textwrap
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
from docx import Document

# ---------- Config ----------
st.set_page_config(page_title="Financial Q&A + Live Excel Calc", page_icon="📊", layout="wide")
st.title("📊 Financial Q&A + 🔢 Live Excel Calculator")
st.caption("Upload a financial document or a spreadsheet, then ask questions in chat. "
           "For semantic answers, set OPENAI_API_KEY in your environment (optional).")

# ---------- State ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Helpers ----------
def load_pdf(file) -> str:
    txt = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            txt.append(t)
    return "\n".join(txt)

def load_docx(file) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text, chunk_size=800):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def find_best_passages(query, text, topk=3):
    # Cheap but effective: keyword overlap on chunks
    q = set(re.findall(r"\w+", query.lower()))
    scored = []
    for ch in chunk_text(text, 160):
        t = set(re.findall(r"\w+", ch.lower()))
        score = len(q & t)
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:topk]]

def safe_number(x):
    try:
        return float(x)
    except:
        return None

def parse_excel_query(q, df):
    """
    Very small domain-specific parser for queries like:
      - total revenue
      - average margin
      - sum of Amount where Region = APAC
      - average of Cost where Month=Jan and Department=Ops
      - max of EBITDA by Segment
    Returns: result_df or scalar, and a human-readable "inferred formula".
    """
    ql = q.lower()

    # Identify op
    op = None
    for k in ["sum","total","avg","average","mean","min","max","count"]:
        if re.search(rf"\b{k}\b", ql):
            op = {"sum":"sum","total":"sum","avg":"mean","average":"mean","mean":"mean",
                  "min":"min","max":"max","count":"count"}[k]
            break
    if op is None:  # default to sum if "revenue" or "amount" etc. present
        op = "sum" if any(c in ql for c in ["revenue","amount","sales","value"]) else "mean"

    # Column candidates
    cols = {c.lower(): c for c in df.columns}
    # Try to find a measure column by fuzzy token presence
    measure = None
    for token in re.findall(r"[A-Za-z_]+", ql):
        if token in cols:
            # Prefer numeric
            if pd.api.types.is_numeric_dtype(df[cols[token]]):
                measure = cols[token]
                break
    if measure is None:
        # fallback: first numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        measure = num_cols[0] if num_cols else df.columns[0]

    # Filters: patterns like "where Region = APAC" or "Region: APAC"
    filt = {}
    patterns = [
        r"where\s+([A-Za-z0-9_ ]+)\s*=\s*([A-Za-z0-9_\-./ ]+)",
        r"([A-Za-z0-9_ ]+)\s*:\s*([A-Za-z0-9_\-./ ]+)"
    ]
    for pat in patterns:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            left, right = m.group(1).strip(), m.group(2).strip()
            # map to actual column
            key = cols.get(left.lower())
            if key and key in df.columns:
                filt[key] = right

    # Group-by: "by Segment" or "group by Month"
    gby = None
    gbym = re.search(r"\bby\s+([A-Za-z0-9_ ]+)", ql)
    if gbym:
        gcol_name = gbym.group(1).strip()
        gby = cols.get(gcol_name.lower())

    # Apply filters
    work = df.copy()
    for k, v in filt.items():
        # Try numeric match; otherwise string
        vn = safe_number(v)
        if vn is not None and pd.api.types.is_numeric_dtype(work[k]):
            work = work[work[k] == vn]
        else:
            work = work[work[k].astype(str).str.lower() == str(v).lower()]

    # Compute
    infer = []
    if gby and gby in work.columns:
        if op == "count":
            out = work.groupby(gby)[measure].count().reset_index(name="count")
            infer.append(f"COUNT({measure}) by {gby}")
        else:
            out = getattr(work.groupby(gby)[measure], op)().reset_index(name=f"{op.upper()}({measure})")
            infer.append(f"{op.upper()}({measure}) by {gby}")
        return out, "; ".join(infer)

    if op == "count":
        val = work[measure].count()
        infer.append(f"COUNT({measure})")
        return val, "; ".join(infer)

    val = getattr(work[measure], op)()
    infer.append(f"{op.upper()}({measure})")
    return val, "; ".join(infer)

def call_llm(query, context_text):
    api = os.getenv("OPENAI_API_KEY", "")
    if not api:
        # Fallback: extractive answer from best passages
        passages = find_best_passages(query, context_text, topk=2)
        if not passages:
            return "I couldn’t find a relevant passage. Try rephrasing or upload a different document."
        joined = "\n\n".join(passages)
        return f"Closest match from the document:\n\n{textwrap.shorten(joined, width=1200)}"
    # Optional: semantic answer using OpenAI (if key provided)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api)
        prompt = (f"You are a finance analyst. Use the context to answer the user. "
                  f"If unknown, say so briefly.\n\nUser: {query}\n\nContext:\n{context_text[:12000]}")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM disabled) {e}"

# ---------- UI: uploads ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Upload Financial Document (PDF/DOCX)")
    doc_file = st.file_uploader("Upload PDF or DOCX", type=["pdf","docx"], key="docxpdf")

with col2:
    st.subheader("📈 Upload Spreadsheet (Excel/CSV)")
    data_file = st.file_uploader("Upload XLSX/CSV", type=["xlsx","csv"], key="csvxlsx")

doc_text = ""
if doc_file:
    if doc_file.name.lower().endswith(".pdf"):
        doc_text = load_pdf(doc_file)
    else:
        doc_text = load_docx(doc_file)

df = None
if data_file:
    if data_file.name.lower().endswith(".csv"):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_excel(data_file)

# ---------- Chat Input ----------
st.divider()
query = st.chat_input("Ask a question about the document or the spreadsheet…")

# ---------- Chat Logic ----------
def add_msg(role, content):
    st.session_state.history.append({"role": role, "content": content})

chat = st.container()
with chat:
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        add_msg("user", query)

        answer_parts = []

        if doc_text:
            ans = call_llm(query, doc_text)
            answer_parts.append(f"**Document insight**\n\n{ans}")

        if df is not None:
            try:
                val, formula = parse_excel_query(query, df)
                if isinstance(val, pd.DataFrame):
                    st.dataframe(val)
                    answer_parts.append(f"**Spreadsheet calc** — inferred: `{formula}`")
                else:
                    answer_parts.append(f"**Spreadsheet calc** — inferred: `{formula}` → **{val:,.2f}**")
            except Exception as e:
                answer_parts.append(f"Spreadsheet calc error: {e}")

        if not answer_parts:
            answer_parts.append("Please upload a document or a spreadsheet first.")

        with st.chat_message("assistant"):
            st.markdown("\n\n".join(answer_parts))
        add_msg("assistant", "\n\n".join(answer_parts))

# ---------- Sidebar help ----------
with st.sidebar:
    st.header("Demo Tips")
    st.markdown("""
**Great prompts to try**
- *“Summarize the company’s revenue drivers from this PDF.”*
- *“What did management say about margin headwinds?”*
- *“Sum of `Revenue` where `Region = APAC`.”*
- *“Average of `Cost_per_Unit` by `Plant`.”*
- *“Min of `DSO` this quarter.”*
- *“Forecast accuracy by month.”*

**Notes**
- For semantic answers, set `OPENAI_API_KEY` in your environment before running.
- The spreadsheet Q&A works fully offline.
""")
