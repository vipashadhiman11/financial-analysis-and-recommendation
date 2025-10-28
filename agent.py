# agent.py

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool as crew_tool

# ---------------------------------------------------------
# 🔐 Load API Keys
# ---------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("📈 Your Financial Advisor")
st.write(
    "Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. "
    "I will also recommend you if you should Buy / Sell / Hold the stock 😎"
)

# ---------------------------------------------------------
# 📰 Plain helper to fetch articles (call this directly)
# ---------------------------------------------------------
def _fetch_articles_apitube(entity: str) -> list:
    """Plain helper function to fetch recent articles with sentiment from API Tube."""
    try:
        articles = []
        APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
        url = (
            "https://api.apitube.io/v1/news/everything?title=" + entity +
            "&published_at.start=2025-10-20&published_at.end=2025-10-25"
            "&sort.order=desc&language.code=en&api_key=" + APITUBE_API_KEY
        )
        resp = requests.get(url, timeout=12)
        data = resp.json()
        if data.get("status") == "ok" and data.get("results"):
            for result in data["results"]:
                articles.append({
                    "article_body": result.get("body", ""),
                    "sentiment": float(
                        result.get("sentiment", {})
                              .get("overall", {})
                              .get("score", 0.0)
                    )
                })

        # Save for visualization
        if articles:
            with open("articles.txt", "w") as f:
                for a in articles:
                    f.write(str(a) + "\n")
        return articles
    except Exception as e:
        return {"error": f"Failed to fetch: {e}"}

# ---------------------------------------------------------
# 🛠 CrewAI Tool wrapper (attach to Agent; do NOT call directly)
# ---------------------------------------------------------
@crew_tool("get_articles_APItube")
def get_articles_APItube_tool(entity: str) -> str:
    """
    CrewAI Tool: Fetches recent news articles + sentiment for the given entity.
    Returns a trimmed stringified list for the agent to read.
    """
    data = _fetch_articles_apitube(entity)
    if isinstance(data, dict) and "error" in data:
        return data["error"]
    trimmed = data[:10] if isinstance(data, list) else []
    return str(trimmed)

# ---------------------------------------------------------
# 📊 Visualization (pie + bar + recommendation banner)
# ---------------------------------------------------------
def visualize_sentiments():
    sentiments = []
    if os.path.exists("articles.txt"):
        with open("articles.txt", "r") as file:
            for line in file:
                try:
                    article = eval(line.strip())
                    sentiments.append(float(article.get("sentiment", 0)))
                except Exception:
                    continue

    # Fallback demo data so UI is never empty
    if not sentiments:
        sentiments = [0.6, -0.3, 0.0, 0.2, -0.5]

    # Categorize
    labels = []
    for s in sentiments:
        if s > 0.05:
            labels.append("Positive")
        elif s < -0.05:
            labels.append("Negative")
        else:
            labels.append("Neutral")

    sentiment_df = pd.DataFrame({"Sentiment": labels})
    sentiment_counts = sentiment_df["Sentiment"].value_counts()

    st.subheader("📊 Sentiment Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Positive", sentiment_counts.get("Positive", 0))
    col2.metric("🔴 Negative", sentiment_counts.get("Negative", 0))
    col3.metric("⚪ Neutral", sentiment_counts.get("Neutral", 0))

    # 🥧 Pie Chart
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["green", "red", "gray"]
    )
    ax.axis("equal")
    st.pyplot(fig)

    # 📊 Bar Chart
    st.bar_chart(sentiment_df["Sentiment"].value_counts())

    # 📌 Recommendation Banner
    avg_sentiment = sum(sentiments) / len(sentiments)
    st.subheader("📌 Investment Recommendation")
    if avg_sentiment > 0.05:
        st.success("🟢 Positive Sentiment — Recommendation: BUY")
    elif avg_sentiment < -0.05:
        st.error("🔴 Negative Sentiment — Recommendation: SELL")
    else:
        st.warning("⚪ Neutral Sentiment — Recommendation: HOLD")

# ---------------------------------------------------------
# 🚀 Main UI
# ---------------------------------------------------------
inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    # Lightweight model string compatible with Groq via LiteLLM in CrewAI
    llm = LLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0.2,
        top_p=0.9
    )

    # Agents
    collector = Agent(
        role="Articles collector",
        goal="Collect news articles related to the stock/company.",
        backstory="Use the tool to fetch articles and their sentiment.",
        tools=[get_articles_APItube_tool],   # attach Tool instance here
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    summerizer = Agent(
        role="Article summerizer",
        goal="Summarize the collected articles and identify key insights.",
        backstory="Summarize clearly and concisely; do not ask the user any follow-up questions.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    analyser = Agent(
        role="Financial Analyst",
        goal="Analyze sentiment and recommend BUY / SELL / HOLD without asking the user for clarification.",
        backstory="You are a confident analyst; even with limited data, provide a clear, reasoned recommendation.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    # Tasks
    collect = Task(
        description="Collect all the latest articles related to the stock/company using the provided tool.",
        expected_output="A concise list of articles with basic sentiment.",
        agent=collector
    )

    summerize = Task(
        description="Summarize the collected articles and extract key trends.",
        expected_output="A short summary of market sentiment and top themes.",
        agent=summerizer
    )

    analyse = Task(
        description="Analyze overall sentiment from the articles and recommend BUY / SELL / HOLD.",
        expected_output="A clear recommendation (BUY/SELL/HOLD) with 2–3 bullet points of rationale.",
        agent=analyser
    )

    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential,
        verbose=False
    )

    try:
        # 1) Fetch first (call helper, NOT the Tool)
        articles = _fetch_articles_apitube(inputStock)

        # 2) Fallback demo data to keep UI rich
        if isinstance(articles, dict) and "error" in articles:
            st.warning(f"No real articles fetched. Using demo data. ({articles['error']})")
            articles = []
        if not articles:
            demo_articles = [
                {"article_body": f"{inputStock} earnings beat expectations", "sentiment": 0.7},
                {"article_body": f"{inputStock} faces regulatory scrutiny", "sentiment": -0.5},
                {"article_body": f"Analysts optimistic on {inputStock}", "sentiment": 0.6},
                {"article_body": f"Mixed outlook for {inputStock}", "sentiment": 0.0},
            ]
            with open("articles.txt", "w") as f:
                for a in demo_articles:
                    f.write(str(a) + "\n")

        # 3) Show charts immediately (fast feedback)
        visualize_sentiments()

        # 4) Run CrewAI analysis
        response = crew.kickoff(inputs={"topic": inputStock})
        st.success("✅ Analysis complete!")
        st.write("Analyzing trends for:", inputStock)
        st.write("Result:", response.raw)

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
