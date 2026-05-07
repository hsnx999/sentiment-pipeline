# Run with: streamlit run app.py

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Sentiment Analyser", page_icon="💬")
st.title("💬 Sentiment Analyser")
st.caption("Fine-tuned BERT model — binary sentiment classification")


# ── Check API is running ───────────────────────────────────────────────────
@st.cache_data(ttl=30)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


if not check_api():
    st.error("API is not running. Start it with: uvicorn api:app --reload")
    st.stop()


# ── Single prediction ──────────────────────────────────────────────────────
st.header("Single text")
text_input = st.text_area(
    "Enter any text to analyse:",
    placeholder="This movie was absolutely wonderful...",
    height=100,
)

if st.button("Analyse", type="primary"):
    if text_input.strip():
        with st.spinner("Analysing..."):
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": text_input}
            )
            result = response.json()

        # Display result
        col1, col2 = st.columns(2)
        label = result["label"]
        confidence = result["confidence"]

        with col1:
            color = "🟢" if label == "POSITIVE" else "🔴"
            st.metric("Sentiment", f"{color} {label}")
        with col2:
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        # Confidence bar chart
        scores = result["scores"]
        fig = go.Figure(go.Bar(
            x=list(scores.values()),
            y=list(scores.keys()),
            orientation="h",
            marker_color=["#EF4444", "#22C55E"],
        ))
        fig.update_layout(
            title="Confidence scores",
            xaxis_range=[0, 1],
            height=200,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter some text first.")

st.divider()

# ── Batch prediction ───────────────────────────────────────────────────────
st.header("Batch analysis")
st.caption("Enter one sentence per line.")

batch_input = st.text_area(
    "Texts to analyse (one per line):",
    placeholder="The acting was superb\nComplete waste of time\nI loved every minute",
    height=150,
)

if st.button("Analyse batch", type="primary"):
    texts = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
    if texts:
        with st.spinner(f"Analysing {len(texts)} texts..."):
            response = requests.post(
                f"{API_URL}/predict/batch",
                json={"texts": texts}
            )
            results = response.json()

        df = pd.DataFrame([{
            "Text":       r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"],
            "Sentiment":  r["label"],
            "Confidence": f"{r['confidence'] * 100:.1f}%",
        } for r in results])

        st.dataframe(df, use_container_width=True)

        # Summary pie chart
        counts = df["Sentiment"].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map={"POSITIVE": "#22C55E", "NEGATIVE": "#EF4444"},
            title="Sentiment distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter at least one line of text.")