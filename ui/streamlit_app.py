import os, requests, streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Minimal RAG - GUI", layout="wide")
st.title("Minimal RAG - GUI")

def safe_request(method: str, path: str, **kwargs):
    url = f"{API_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=kwargs.pop("timeout", 10), **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API is not available yet. Please wait a moment and try again."
    except requests.exceptions.Timeout:
        return None, "Request timed out. The API might be starting up-try again shortly."
    except requests.exceptions.HTTPError as e:
        return None, f"API returned an error ({e.response.status_code}). Please retry soon."
    except Exception:
        return None, "Unexpected error talking to the API. Try again shortly."

with st.sidebar:
    st.header("Status")
    h, err = safe_request("GET", "/health")
    if h:
        st.success(f"API OK · index_size={h.get('index_size')} · llm_ready={h.get('llm_ready')}")
    else:
        st.info(err or "API not reachable yet. Waiting…")

st.subheader("Ingest")
tab1, tab2 = st.tabs(["Upload files", "Paste text"])

with tab1:
    files = st.file_uploader("Add documents to your knowledge base", type=None, accept_multiple_files=True)
    if st.button("Ingest uploaded files", type="primary", disabled=not files):
        files_payload = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in files]
        data, err = safe_request("POST", "/ingest/files", files=files_payload, timeout=120)
        if data: st.success(f"Ingested {data['ingested_chunks']} chunks. Index size: {data['index_size']}.")
        else:    st.info(err)

with tab2:
    doc_id = st.text_input("Document ID (optional)", "inline")
    text_input = st.text_area("Paste text here", height=180, placeholder="My name is Nauman...")
    if st.button("Ingest pasted text", disabled=not text_input.strip()):
        payload = {"texts": [text_input], "doc_id": doc_id or None, "source": "inline"}
        data, err = safe_request("POST", "/ingest/text", json=payload, timeout=60)
        if data: st.success(f"Ingested {data['ingested_chunks']} chunks. Index size: {data['index_size']}.")
        else:    st.info(err)

st.subheader("Query")
query = st.text_input("Ask a question", "")
top_k = st.slider("Top-K", 1, 10, 5)
strict_mode = st.toggle("Strict (context-only) answers", value=True, help="When on, answers must be supported by your indexed documents. When off, small-talk is allowed and answers fall back to best-effort.")

if st.button("Search & Answer", type="primary", disabled=not query.strip()):
    payload = {"query": query, "top_k": top_k, "strict": strict_mode}  # <-- send strict flag
    data, err = safe_request("POST", "/query", json=payload, timeout=60)
    if data:
        st.markdown("### Answer")
        st.code((data.get("answer") or "").strip(), language="markdown")
        st.caption(f"Mode: {'Strict (context-only)' if data.get('strict') else 'Chatty (non-strict)'}")
        st.markdown("### Sources")
        for i, s in enumerate(data.get("sources", []), 1):
            doc = s.get("doc_id") or s.get("source") or "doc"
            st.write(f"[{i}] {doc}: {s.get('snippet','')}")
    else:
        st.info(err)
