import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date, timedelta
from dotenv import load_dotenv

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
# 📰 Tool to fetch articles
# ---------------------------------------------------------
@tool("get_articles_APItube")
def get_articles_APItube(entity: str) -> list[list]:
    """
    Fetch sentiment articles for the given stock/company using API Tube.
    If fetching fails, return an error.
    """
    try:
        articles = []
        APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
        url = (
            f"https://api.apitube.io/v1/news/everything?title={entity}"
            f"&published_at.start=2025-10-20&published_at.end=2025-10-25"
            f"&sort.order=desc&language.code=en&api_key={APITUBE_API_KEY}"
        )
        response = requests.get(url).json()
        if response.get("status") == "ok" and response.get("results"):
            for result in response["results"]:
                articles.append({
                    "article_body": result["body"],
                    "sentiment": result["sentiment"]["overall"]["score"]
                })
        # Save fetched articles
        if articles:
            with open("articles.txt", "w") as file:
                for article in articles:
                    file.write(str(article) + "\n")
        return articles
    except Exception as e:
        return {"error": f"Failed to read URL {e}"}

# ---------------------------------------------------------
# 📊 Visualization Function
# ---------------------------------------------------------
def visualize_sentiments():
    sentiments = []
    if os.path.exists("articles.txt"):
        with open("articles.txt", "r") as file:
            for line in file:
                try:
                    article = eval(line.strip())
                    sentiments.append(float(article.get("sentiment", 0)))
                except:
                    continue

    # Fallback if no sentiments found
    if not sentiments:
        sentiments = [0.6, -0.3, 0.0, 0.2, -0.5]

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

    # Pie Chart
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

    # Bar Chart
    st.bar_chart(sentiment_df["Sentiment"].value_counts())

    # Final Recommendation
    st.subheader("📌 Investment Recommendation")
    avg_sentiment = sum(sentiments) / len(sentiments)
    if avg_sentiment > 0.05:
        st.success("🟢 Positive Sentiment — Recommendation: BUY")
    elif avg_sentiment < -0.05:
        st.error("🔴 Negative Sentiment — Recommendation: SELL")
    else:
        st.warning("⚪ Neutral Sentiment — Recommendation: HOLD")

# ---------------------------------------------------------
# 🚀 Main Execution
# ---------------------------------------------------------
inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    llm = LLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0.2,
        top_p=0.9
    )

    collector = Agent(
        role="Articles collector",
        goal="Collect news articles related to the stock/company.",
        backstory="Use API to fetch articles and their sentiment.",
        tools=[get_articles_APItube],
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    summerizer = Agent(
        role="Article summerizer",
        goal="Summarize collected articles to identify key insights.",
        backstory="Summarize precisely. Do not ask questions.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    analyser = Agent(
        role="Financial Analyst",
        goal="Analyze sentiment and recommend BUY / SELL / HOLD without asking the user for clarification.",
        backstory="You are a confident analyst. Even with little data, make a clear judgment.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    collect = Task(
        description="Collect all the latest articles related to the stock/company.",
        expected_output="List of articles with sentiment.",
        agent=collector
    )

    summerize = Task(
        description="Summarize the collected articles and extract key trends.",
        expected_output="A clear summary of market sentiment and trends.",
        agent=summerizer
    )

    analyse = Task(
        description="Analyze the overall sentiment and recommend BUY / SELL / HOLD.",
        expected_output="Final recommendation based on aggregated sentiment.",
        agent=analyser
    )

    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential,
        verbose=False
    )

    try:
        # Try fetching articles first
        articles = get_articles_APItube(inputStock)
        if not articles or isinstance(articles, dict):
            st.warning(f"No real articles found for {inputStock}. Using sample data for demo.")
            demo_articles = [
                {"article_body": f"{inputStock} reports record quarterly earnings", "sentiment": 0.8},
                {"article_body": f"{inputStock} faces regulatory challenges", "sentiment": -0.6},
                {"article_body": f"{inputStock} launches new AI product", "sentiment": 0.7},
                {"article_body": f"Mixed reaction to {inputStock} news", "sentiment": 0.0},
            ]
            with open("articles.txt", "w") as file:
                for article in demo_articles:
                    file.write(str(article) + "\n")

        # 🧭 Visualize results before LLM
        visualize_sentiments()

        # 🧠 Run analysis
        response = crew.kickoff(inputs={"topic": inputStock})
        st.success("✅ Analysis complete!")
        st.write("Analyzing trends for:", inputStock)
        st.write("Result:", response.raw)

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
