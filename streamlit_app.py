import streamlit as st
import requests
import os

API_BASE = os.environ.get("BACKEND_URL", "http://localhost:8000")



st.set_page_config(page_title="Axiom-Agent", page_icon="🔎", layout="centered")

st.title("🔎 Axiom-Agent")
st.caption("Self-verifying AI for claim and document fact-checking")

tab1, tab2 = st.tabs(["💬 Ask Axiom-Agent", "⚔️ Adversarial Test"])

with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_doc" not in st.session_state:
        st.session_state.uploaded_doc = None

    # Conversation history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["query"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            conf = entry.get("confidence", 0)
            if conf >= 0.7:
                st.caption("🟢 High confidence")
            elif conf >= 0.4:
                st.caption("🟡 Medium confidence")
            else:
                st.caption("🔴 Low confidence")
            if entry.get("sources"):
                st.caption(f"📄 Sources: {', '.join(entry['sources'])}")

    # File attach (collapsed, like a "+" attach button)
    with st.expander("📎 Attach a document (optional — grounds answers in its content)"):
        uploaded = st.file_uploader("Upload PDF/text", type=["pdf", "txt", "md"], label_visibility="collapsed")
        if uploaded and st.button("Attach & Ingest"):
            with st.spinner("Reading document..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=60)
                    data = resp.json()
                    if data.get("status") == "success":
                        st.session_state.uploaded_doc = uploaded.name
                        st.success(f"Attached: {uploaded.name} — you can now ask about it below.")
                    else:
                        st.error(data.get("message", "Ingestion failed."))
                except Exception as e:
                    st.error(f"Could not reach the backend: {e}")

    if st.session_state.uploaded_doc:
        st.caption(f"📎 Active document: {st.session_state.uploaded_doc}")

    # Chat input
    query = st.chat_input("Ask a question or verify a claim...")

    if query:
        context = ""
        if st.session_state.chat_history:
            last = st.session_state.chat_history[-1]
            context = f"Previous question: {last['query']}\nPrevious answer: {last['answer']}\n\nNew question: "
        full_query = context + query

        with st.spinner("Verifying... (sampling responses + multi-agent check)"):
            try:
                resp = requests.post(f"{API_BASE}/verify", json={"query": full_query}, timeout=120)
                data = resp.json()
            except Exception as e:
                st.error(f"Could not reach the backend: {e}")
                data = None

        if data:
            st.session_state.chat_history.append({
                "query": query,
                "answer": data.get("answer", ""),
                "confidence": data.get("confidence", 0),
                "sources": data.get("sources", []),
            })
            st.session_state.last_trace = data.get("reasoning_trace", {})
            st.rerun()

    if st.session_state.chat_history:
        with st.expander("🔍 Technical Details (last response)"):
            last = st.session_state.chat_history[-1]
            st.json({
                "confidence": last.get("confidence"),
                "sources": last.get("sources"),
            })
            if "last_trace" in st.session_state:
                st.write("Full reasoning trace:")
                st.json(st.session_state.last_trace)

with tab2:
    st.write("Test the system's guardrails against a jailbreak or harmful prompt.")
    with st.form(key="adv_form", clear_on_submit=False):
        adv_prompt = st.text_input("Enter a jailbreak or adversarial prompt")
        adv_submitted = st.form_submit_button("Run Adversarial Test")

    if adv_submitted:
        if not adv_prompt.strip():
            st.warning("Please enter a prompt to test.")
        else:
            with st.spinner("Checking..."):
                try:
                    resp = requests.post(f"{API_BASE}/adversarial-test", json={"prompt": adv_prompt}, timeout=30)
                    data = resp.json()
                    if data.get("blocked"):
                        st.error(f"🚫 BLOCKED — {data.get('reason')}")
                    else:
                        st.success(f"✅ PASSED — {data.get('reason')}")
                except Exception as e:
                    st.error(f"Could not reach the backend: {e}")

with st.sidebar:
    with st.expander("ℹ️ About this project"):
        st.write(
            "Axiom-Agent verifies claims using multi-agent cross-checking "
            "(fact-checker, skeptic, judge) and semantic-entropy-based uncertainty "
            "estimation, grounded in retrieved evidence, with AI-based safety guardrails."
        )