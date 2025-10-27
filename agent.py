from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date, timedelta
from dotenv import load_dotenv
import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import json

# ================== 🌿 Setup ===================
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

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

    # ================== 🧰 Tools ===================
    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> list[list]:
        try:
            articles = []
            APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
            url = (
                "https://api.apitube.io/v1/news/everything?title=" + entity +
                "&published_at.start=2025-10-20&published_at.end=2025-10-25"
                "&sort.order=desc&language.code=en&api_key=" + APITUBE_API_KEY
            )
            response = requests.get(url, timeout=20).json()
            count = 0
            if response.get("status") == "ok":
                for result in response.get("results", []):
                    count += 1
                    article = {
                        "article_body": result.get("body", ""),
                        "sentiment": result.get("sentiment", {}).get("overall", {}).get("score", 0),
                        "published_at": result.get("published_at", "")
                    }
                    articles.append(article)
                while response.get("has_next_pages"):
                    if count < 20:
                        next_page_url = response.get("next_page")
                        if not next_page_url:
                            break
                        response = requests.get(next_page_url, timeout=20).json()
                        if response.get("status") == "ok":
                            for result in response.get("results", []):
                                count += 1
                                article = {
                                    "article_body": result.get("body", ""),
                                    "sentiment": result.get("sentiment", {}).get("overall", {}).get("score", 0),
                                    "published_at": result.get("published_at", "")
                                }
                                articles.append(article)
                    else:
                        break

            # Persist for plotting layer
            with open("articles.txt", "w") as file:
                for article in articles:
                    file.write(str(article) + "\n")

            return articles
        except Exception as e:
            return {"error": f"Failed to fetch articles: {e}"}

    @tool("sentiment_analysis")
    def sentiment_analysis(articles: list[str]) -> str:
        # Optional tool (not used for charts); keeping for completeness
        from transformers import pipeline
        model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        sentiments = []
        for text in articles:
            if not text:
                sentiments.append({"label": "neutral", "score": 0.0})
                continue
            result = model(text[:512])[0]
            sentiments.append(result)
        return sentiments

    # ================== 🤖 Agents ===================
    collector = Agent(
        role="Articles collector",
        goal="Collect articles related to the topic using tools.",
        backstory="The {topic} will be an organisation or stock name.",
        tools=[get_articles_APItube],
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    summerizer = Agent(
        role="Article summerizer",
        goal="Summarize the collected articles to fetch the crux of it",
        backstory="You are summarizing all the articles into one with utmost precision.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    analyser = Agent(
        role="Financial Analyst",
        goal="Guide user to either Buy/Sell or Hold the stock based on sentiment.",
        backstory="You will observe the sentiment of all the articles and make a recommendation.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    # ================== 🧠 Tasks ===================
    collect = Task(
        description="Collect latest articles for {topic} using the tool.",
        expected_output="Articles related to the organisation or stock",
        agent=collector
    )

    summerize = Task(
        description="Summarize the collected articles in 500 words.",
        expected_output="Summary of articles",
        agent=summerizer
    )

    analyse = Task(
        description="Analyze sentiment and recommend Buy/Sell/Hold.",
        expected_output="Final investment recommendation",
        agent=analyser
    )

    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential,
        verbose=False
    )

    # ================== 🚀 Execution ===================
    try:
        # ✅ Manually call the tool to avoid model tool-call issues
        articles = get_articles_APItube(inputStock)

        if isinstance(articles, dict) and "error" in articles:
            st.error(f"❌ Error fetching articles: {articles['error']}")
        else:
            st.success(f"✅ {len(articles)} articles fetched for {inputStock}")

            # 🎯 Kick off Crew without relying on model tool invocation
            response = crew.kickoff(inputs={"topic": inputStock, "articles": articles})
            st.write("Analysing trends for: ", inputStock)
            st.write("Result:", response.raw)

            # ================== 📊 Sentiment Visualization ===================
            sentiments = []
            try:
                with open("articles.txt", "r") as file:
                    for line in file:
                        try:
                            article = eval(line.strip())  # quick parsing for demo
                            sentiments.append(float(article.get('sentiment', 0)))
                        except Exception:
                            continue
            except FileNotFoundError:
                sentiments = []

            if sentiments:
                # Map continuous sentiment to labels
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

                st.subheader("📊 Sentiment Overview")
                col1, col2, col3 = st.columns(3)
                col1.metric("🟢 Positive", int(sentiment_counts.get("Positive", 0)))
                col2.metric("🔴 Negative", int(sentiment_counts.get("Negative", 0)))
                col3.metric("⚪ Neutral", int(sentiment_counts.get("Neutral", 0)))

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

                # 📊 Bar chart
                st.bar_chart(sentiment_df['Sentiment'].value_counts())

                # 📅 Sentiment Trend Line Chart
                trend_data = []
                try:
                    with open("articles.txt", "r") as file:
                        for line in file:
                            try:
                                article = eval(line.strip())
                                s = float(article.get('sentiment', 0))
                                d = article.get('published_at', '')
                                if d:
                                    trend_data.append({"date": d.split("T")[0], "sentiment": s})
                            except Exception:
                                continue
                except FileNotFoundError:
                    trend_data = []

                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    trend_df['date'] = pd.to_datetime(trend_df['date'], errors='coerce')
                    trend_df = trend_df.dropna(subset=['date'])
                    trend_df = trend_df.groupby('date')['sentiment'].mean().reset_index()
                    st.subheader("📈 Sentiment Trend Over Time")
                    st.line_chart(trend_df.set_index('date')['sentiment'])
                else:
                    st.warning("No trend data available.")

                # 📢 Final Recommendation Banner
                st.subheader("📌 Investment Recommendation")
                overall_sentiment_score = sum(sentiments) / len(sentiments)

                if overall_sentiment_score > 0.05:
                    st.markdown(
                        "<div style='background-color:#d4edda;padding:15px;border-radius:10px;'>"
                        "<h3 style='color:#155724;'>🟢 Strong sentiment detected — Recommendation: <b>BUY</b></h3>"
                        "<p>The overall market mood for this stock appears positive. Consider buying or holding for upside potential.</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                elif overall_sentiment_score < -0.05:
                    st.markdown(
                        "<div style='background-color:#f8d7da;padding:15px;border-radius:10px;'>"
                        "<h3 style='color:#721c24;'>🔴 Negative sentiment detected — Recommendation: <b>SELL</b></h3>"
                        "<p>Market sentiment is weak. You may consider selling or avoiding this stock currently.</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div style='background-color:#e2e3e5;padding:15px;border-radius:10px;'>"
                        "<h3 style='color:#383d41;'>⚪ Neutral sentiment detected — Recommendation: <b>HOLD</b></h3>"
                        "<p>Sentiment is mixed. It may be wise to wait and watch before making major decisions.</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No sentiment data available to display charts.")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
