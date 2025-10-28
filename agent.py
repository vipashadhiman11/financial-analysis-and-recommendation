from crewai import Agent, Task, Crew, LLM, Process
from datetime import date, timedelta
from dotenv import load_dotenv
import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

# 🔐 Load API key
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write(
    "Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. "
    "I will also recommend you if you should Buy/ Sell or Hold the stock 😎"
)

inputStock = st.text_input("Enter stock name or company name:")

# ✅ Normal function (NO @tool)
def get_articles_APItube(entity: str) -> list:
    """
    Fetch articles from APITube, and if that fails or returns no articles,
    fallback to NewsAPI to fetch real news headlines for the given entity.
    """
    articles = []
    try:
        print("🛰️ Trying APITube first...")
        APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
        url = (
            "https://api.apitube.io/v1/news/everything?title=" + entity +
            "&published_at.start=2025-10-01&published_at.end=2025-10-28"
            "&sort.order=desc&language.code=en&api_key=" + APITUBE_API_KEY
        )
        response = requests.get(url).json()

        if response.get("status") == "ok" and response.get("results"):
            for result in response["results"]:
                articles.append({
                    "article_body": result["body"],
                    "sentiment": result["sentiment"]["overall"]["score"],
                    "published_at": result.get("published_at", "")
                })

    except Exception as e:
        print(f"⚠️ APITube error: {e}")

    # 🛑 If APITube gave nothing → try NewsAPI
    if not articles:
        try:
            print("📰 Falling back to NewsAPI...")
            NEWS_API_KEY = os.getenv("4850da14d83f4ddd92e0bf64caad7d96")
            news_url = f"https://newsapi.org/v2/everything?q={entity}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
            news_response = requests.get(news_url).json()

            if news_response.get("status") == "ok" and news_response.get("articles"):
                for item in news_response["articles"]:
                    # No sentiment from NewsAPI → set default neutral or 0
                    articles.append({
                        "article_body": item["title"] + " " + (item.get("description") or ""),
                        "sentiment": 0.0,
                        "published_at": item.get("publishedAt", "")
                    })
        except Exception as e:
            print(f"⚠️ NewsAPI fallback error: {e}")

    # Save fetched articles to file
    if articles:
        with open("articles.txt", "w") as file:
            for article in articles:
                file.write(str(article) + "\n")

    return articles

if st.button("Submit", type="primary"):
    # 🧠 Initialize model
    llm = LLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0.2,
        top_p=0.9
    )

    # 🧑 Agents
    collector = Agent(
        role="Articles collector",
        goal="Collect articles related to the given stock or company.",
        backstory="You collect news articles for sentiment analysis.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    summerizer = Agent(
        role="Article summerizer",
        goal="Summarize all collected articles precisely.",
        backstory="You make sure only important insights are retained.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    analyser = Agent(
        role="Financial Analyst",
        goal="Guide the user to Buy/Sell/Hold based on overall sentiment.",
        backstory="You analyze sentiment and market trends.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    # 📝 Tasks
    collect = Task(
        description="Collect news articles for the given stock.",
        expected_output="A list of articles with sentiment data.",
        agent=collector
    )

    summerize = Task(
        description="Summarize collected articles.",
        expected_output="A summary of the articles.",
        agent=summerizer
    )

    analyse = Task(
        description="Analyze and recommend Buy/Sell/Hold.",
        expected_output="A clear recommendation based on sentiment.",
        agent=analyser
    )

    # 🤝 Crew
    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential,
        verbose=False
    )

    # 🚀 Run
    try:
        articles = get_articles_APItube(inputStock)

        # ⚡ Fallback for demo if no articles found
        if not articles or len(articles) == 0 or (isinstance(articles, dict) and "error" in articles):
            st.warning(f"No real articles found for {inputStock}. Using sample data for demo.")
            articles = [
                {"article_body": "Strong earnings report", "sentiment": 0.8},
                {"article_body": "Product recall concerns", "sentiment": -0.6},
                {"article_body": "New product launch", "sentiment": 0.7},
                {"article_body": "Regulatory challenges", "sentiment": -0.8},
                {"article_body": "Market optimism", "sentiment": 0.9}
            ]
        else:
            st.success(f"✅ {len(articles)} articles fetched for {inputStock}")

        # 🔁 Run Crew
        response = crew.kickoff(inputs={"topic": inputStock})
        st.write("Analyzing trends for:", inputStock)
        st.write("Result:", response.raw)

        # 🥧 Pie Chart (basic sentiment visualization)
        sentiments = []
        for a in articles:
            try:
                sentiments.append(float(a.get("sentiment", 0)))
            except:
                pass

        if sentiments:
            positive = len([s for s in sentiments if s > 0.05])
            negative = len([s for s in sentiments if s < -0.05])
            neutral = len(sentiments) - positive - negative

            counts = [positive, negative, neutral]
            labels = ["Positive", "Negative", "Neutral"]
            colors = ["green", "red", "gray"]

            st.subheader("📊 Sentiment Overview")
            fig, ax = plt.subplots()
            ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.warning("No sentiment data available to display charts.")

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
