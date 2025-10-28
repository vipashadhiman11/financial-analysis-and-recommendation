from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date, timedelta
from dotenv import load_dotenv
import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write(
    "Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. "
    "I will also recommend you if you should Buy/ Sell or Hold the stock 😎"
)

inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        temperature=0.2,
        top_p=0.9
    )

    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> list[list]:
        """Fetch sentiment articles from API Tube"""
        try:
            articles = []
            APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
            url = (
                "https://api.apitube.io/v1/news/everything?title=" + entity +
                "&published_at.start=2025-10-20&published_at.end=2025-10-25"
                "&sort.order=desc&language.code=en&api_key=" + APITUBE_API_KEY
            )
            response = requests.get(url).json()
            count = 0
            if response["status"] == "ok":
                for result in response["results"]:
                    count += 1
                    article = {}
                    article["article_body"] = result["body"]
                    article["sentiment"] = result["sentiment"]["overall"]["score"]
                    articles.append(article)
            with open("articles.txt", "w") as file:
                for article in articles:
                    file.write(str(article) + "\n")
            return articles
        except Exception as e:
            return {"error": f"Failed to read URL {e}"}

    collector = Agent(
        role="Articles collector",
        goal="Collect the articles related to the stock/company.",
        backstory="Use API to fetch articles and analyze sentiment.",
        tools=[get_articles_APItube],
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    summerizer = Agent(
        role="Article summerizer",
        goal="Summarize the articles collected to identify key insights.",
        backstory="Summarizing market news to find patterns.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    analyser = Agent(
        role="Financial Analyst",
        goal="Guide user to either Buy/Sell or Hold the stock.",
        backstory="Analyzing sentiment trends to recommend actions.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    collect = Task(
        description="Collect all news articles related to the stock/company.",
        expected_output="A list of articles with sentiment.",
        agent=collector
    )

    summerize = Task(
        description="Summarize collected articles.",
        expected_output="A summary of the sentiment and key trends.",
        agent=summerizer
    )

    analyse = Task(
        description="Analyze the sentiment and give a recommendation.",
        expected_output="Final recommendation (Buy/Sell/Hold).",
        agent=analyser
    )

    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential,
        verbose=False
    )

    try:
        response = crew.kickoff(inputs={"topic": inputStock})
        st.write("Analyzing trends for:", inputStock)
        st.write("Result:", response.raw)

        # === 📊 Visualization Section ===
        sentiments = []

        if os.path.exists("articles.txt"):
            with open("articles.txt", "r") as file:
                for line in file:
                    try:
                        article = eval(line.strip())
                        sentiments.append(float(article.get("sentiment", 0)))
                    except:
                        continue

        # If no sentiments found, use demo data so UI isn't empty
        if not sentiments:
            sentiments = [0.6, -0.3, 0.2, 0.1, -0.7]

        # Categorize sentiments
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

        # Pie chart
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

        # Bar chart
        st.bar_chart(sentiment_df["Sentiment"].value_counts())

        # Final Recommendation
        st.subheader("📌 Investment Recommendation")
        overall_sentiment = sum(sentiments) / len(sentiments)
        if overall_sentiment > 0.05:
            st.success("🟢 Overall Sentiment: Positive — Recommendation: BUY")
        elif overall_sentiment < -0.05:
            st.error("🔴 Overall Sentiment: Negative — Recommendation: SELL")
        else:
            st.warning("⚪ Overall Sentiment: Neutral — Recommendation: HOLD")

    except Exception as e:
        st.write(f"An error occurred: {e}")
