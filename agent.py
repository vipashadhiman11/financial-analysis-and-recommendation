# ================================================================
# 📊 AI Financial Advisor — NO transformers dependency (FAST FIX)
# ================================================================

from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date, timedelta
from dotenv import load_dotenv
import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

# Load Environment
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write(
    'Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. '
    'I will also recommend you if you should Buy/ Sell or Hold the stock 😎'
)

inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        temperature=0.2,
        top_p=0.9
    )

    # ------------------------------------------------
    # 📡 TOOL: Fetch articles with sentiment from APITube
    # ------------------------------------------------
    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> list:
        """Fetch news articles for the given entity using APITube API."""
        try:
            articles = []
            APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
            url = (
                f"https://api.apitube.io/v1/news/everything?title={entity}"
                f"&published_at.start=2025-10-20&published_at.end=2025-10-25"
                f"&sort.order=desc&language.code=en&api_key={APITUBE_API_KEY}"
            )
            response = requests.get(url).json()

            if response["status"] == "ok":
                for result in response["results"]:
                    articles.append({
                        "article_body": result["body"],
                        "sentiment": result["sentiment"]["overall"]["score"],
                        "published_at": result.get("published_at", "")
                    })

            # 💾 Save locally for charting
            with open("articles.txt", "w") as file:
                for article in articles:
                    file.write(str(article) + "\n")

            return articles
        except Exception as e:
            return {"error": f"Failed to fetch articles: {e}"}

    # ------------------------------------------------
    # 🧠 Crew Agents
    # ------------------------------------------------
    collector = Agent(
        role="Articles collector",
        goal="Collects articles related to the topic using the tool.",
        backstory="Fetch latest articles for the company or stock.",
        tools=[get_articles_APItube],
        llm=llm
    )

    summerizer = Agent(
        role="Article summarizer",
        goal="Summarizes all the articles to extract market sentiment.",
        backstory="Analyzes key trends and sentiments.",
        llm=llm
    )

    analyser = Agent(
        role="Financial Analyst",
        goal="Recommends Buy/Sell/Hold based on sentiment.",
        backstory="Uses sentiment to suggest investment action.",
        llm=llm
    )

    # ------------------------------------------------
    # 📝 Tasks
    # ------------------------------------------------
    collect = Task(description="Collect news articles", agent=collector)
    summerize = Task(description="Summarize articles", agent=summerizer)
    analyse = Task(description="Recommend Buy/Sell/Hold", agent=analyser)

    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential
    )

    # ------------------------------------------------
    # 🧭 Manual Article Fetch (Stable)
    # ------------------------------------------------
    articles = get_articles_APItube(inputStock)
    if isinstance(articles, dict) and "error" in articles:
        st.error(f"❌ Error fetching articles: {articles['error']}")
    elif not articles:
        st.warning("⚠️ No articles found for this stock.")
    else:
        st.success(f"✅ {len(articles)} articles fetched for {inputStock}")

        response = crew.kickoff(inputs={"topic": inputStock, "articles": articles})
        st.write("Analyzing trends for:", inputStock)
        st.write("Result:", response.raw)

        # ------------------------------------------------
        # 📊 Visualization Section
        # ------------------------------------------------
        sentiments = []
        try:
            with open("articles.txt", "r") as file:
                for line in file:
                    article = eval(line.strip())
                    sentiments.append(float(article.get('sentiment', 0)))
        except FileNotFoundError:
            st.error("❌ No articles.txt file found.")
            sentiments = []

        if sentiments:
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
            sentiment_counts = sentiment_df['Sentiment'].value_counts()

            # 📊 Overview Metrics
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
                autopct='%1.1f%%',
                startangle=90,
                colors=['green', 'red', 'gray']
            )
            ax.axis('equal')
            st.pyplot(fig)

            # 📊 Bar Chart
            st.bar_chart(sentiment_df['Sentiment'].value_counts())

            # 📌 Final Recommendation
            st.subheader("📌 Investment Recommendation")
            overall_sentiment_score = sum(sentiments) / len(sentiments)
            if overall_sentiment_score > 0.05:
                st.success("🟢 BUY — Market sentiment is positive.")
            elif overall_sentiment_score < -0.05:
                st.error("🔴 SELL — Market sentiment is negative.")
            else:
                st.warning("⚪ HOLD — Market sentiment is neutral.")
        else:
            st.warning("No sentiment data available to display charts.")
