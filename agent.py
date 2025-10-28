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
    Fetch news articles from API Tube for the given entity (company or stock).
    Returns a list of articles with sentiment scores and publication date.
    """
    try:
        print("Running API")
        articles = []
        APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
        url = (
            "https://api.apitube.io/v1/news/everything?title=" + entity +
            "&published_at.start=2025-10-01&published_at.end=2025-10-28"
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
                article["published_at"] = result.get("published_at", "")
                articles.append(article)
            while response["has_next_pages"]:
                if count < 20:
                    next_page_url = response["next_page"]
                    next_page_response = requests.get(next_page_url).json()
                    if next_page_response["status"] == "ok":
                        for result in next_page_response["results"]:
                            count += 1
                            article = {}
                            article["article_body"] = result["body"]
                            article["sentiment"] = result["sentiment"]["overall"]["score"]
                            article["published_at"] = result.get("published_at", "")
                            articles.append(article)
                else:
                    break

        with open("articles.txt", "w") as file:
            for article in articles:
                file.write(str(article) + "\n")

        return articles

    except Exception as e:
        return {"error": f"Failed to fetch articles: {e}"}

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
