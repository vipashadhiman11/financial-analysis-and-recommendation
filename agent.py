import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process

# -------------------------------------------------------------------
# 🔐 Load environment variables
# -------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NEWS_API_KEY = "4850da14d83f4ddd92e0bf64caad7d96"

# -------------------------------------------------------------------
# 🖥️ Streamlit UI Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("📈 Your Financial Advisor")
st.write(
    "Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. "
    "I will also recommend you if you should **Buy / Sell / Hold** the stock 😎"
)

# -------------------------------------------------------------------
# 🧰 Article Fetch Function
# -------------------------------------------------------------------
def get_articles_APItube(entity: str) -> list:
    articles = []
    try:
        APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
        url = (
            f"https://api.apitube.io/v1/news/everything?title={entity}"
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

    # 📰 Fallback to NewsAPI
    if not articles and NEWS_API_KEY:
        try:
            news_url = f"https://newsapi.org/v2/everything?q={entity}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
            news_response = requests.get(news_url).json()
            if news_response.get("status") == "ok" and news_response.get("articles"):
                for item in news_response["articles"]:
                    articles.append({
                        "article_body": item["title"] + " " + (item.get("description") or ""),
                        "sentiment": 0.0,
                        "published_at": item.get("publishedAt", "")
                    })
        except Exception as e:
            print(f"⚠️ NewsAPI fallback error: {e}")

    # 💾 Save to file
    if articles:
        with open("articles.txt", "w") as file:
            for article in articles:
                file.write(str(article) + "\n")

    return articles

# -------------------------------------------------------------------
# 📊 Visualization Function
# -------------------------------------------------------------------
def visualize_sentiments():
    sentiments = []
    if not os.path.exists("articles.txt"):
        st.warning("❌ No articles.txt file found.")
        return

    with open("articles.txt", "r") as file:
        for line in file:
            try:
                article = eval(line.strip())
                sentiments.append(float(article.get("sentiment", 0)))
            except:
                continue

    if not sentiments:
        st.warning("No sentiment data available to display charts.")
        return

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

    # 🥧 Pie chart
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

    # 📊 Bar chart
    st.bar_chart(sentiment_df["Sentiment"].value_counts())

    # 📈 Trend Line Chart
    trend_data = []
    with open("articles.txt", "r") as file:
        for line in file:
            try:
                article = eval(line.strip())
                s = float(article.get("sentiment", 0))
                d = article.get("published_at", "")
                if d:
                    trend_data.append({"date": d.split("T")[0], "sentiment": s})
            except:
                continue

    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
        trend_df = trend_df.dropna(subset=["date"])
        trend_df = trend_df.groupby("date")["sentiment"].mean().reset_index()
        st.subheader("📈 Sentiment Trend Over Time")
        st.line_chart(trend_df.set_index("date")["sentiment"])

    # 📢 Recommendation Banner
    overall_sentiment_score = sum(sentiments) / len(sentiments)
    st.subheader("📌 Investment Recommendation")

    if overall_sentiment_score > 0.05:
        st.markdown(
            "<div style='background-color:#d4edda;padding:15px;border-radius:10px;'>"
            "<h3 style='color:#155724;'>🟢 Positive sentiment — Recommendation: <b>BUY</b></h3>"
            "<p>Overall market sentiment for this stock is positive. Consider buying or holding for upside potential.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    elif overall_sentiment_score < -0.05:
        st.markdown(
            "<div style='background-color:#f8d7da;padding:15px;border-radius:10px;'>"
            "<h3 style='color:#721c24;'>🔴 Negative sentiment — Recommendation: <b>SELL</b></h3>"
            "<p>Market sentiment is weak. Selling or avoiding this stock may be wise.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#e2e3e5;padding:15px;border-radius:10px;'>"
            "<h3 style='color:#383d41;'>⚪ Neutral sentiment — Recommendation: <b>HOLD</b></h3>"
            "<p>Sentiment is mixed. Wait and watch before making major decisions.</p>"
            "</div>",
            unsafe_allow_html=True
        )

# -------------------------------------------------------------------
# 🚀 Main Execution
# -------------------------------------------------------------------
inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    with st.spinner("⏳ Fetching articles..."):
        articles = get_articles_APItube(inputStock)

    if not articles:
        st.warning(f"No real articles found for {inputStock}. Using sample data for demo.")
        company = inputStock
        articles = [
            {"article_body": f"{company} reports record quarterly earnings", "sentiment": 0.8},
            {"article_body": f"{company} faces product recall concerns", "sentiment": -0.6},
            {"article_body": f"{company} launches new AI product", "sentiment": 0.7},
            {"article_body": f"{company} faces regulatory challenges", "sentiment": -0.8},
            {"article_body": f"Investors optimistic about {company}'s growth", "sentiment": 0.9}
        ]
        with open("articles.txt", "w") as file:
            for article in articles:
                file.write(str(article) + "\n")

    # 🧭 Visualize sentiments immediately (no LLM delay)
    visualize_sentiments()

    # 🧠 Now load model AFTER visualization
    with st.spinner("🧠 Running analysis..."):
        llm = LLM(model="groq/llama-3.1-8b-instant", temperature=0.2, top_p=0.9)

        collector = Agent(
            role="Articles collector",
            goal="Collect news articles related to the stock/company.",
            backstory="Use API to fetch articles and their sentiment.",
            llm=llm,
            allow_delegation=False,
            verbose=False
        )

        summerizer = Agent(
            role="Article summerizer",
            goal="Summarize collected articles to identify key insights.",
            backstory="Summarizing articles precisely and accurately.",
            llm=llm,
            allow_delegation=False,
            verbose=False
        )

        analyser = Agent(
            role="Financial Analyst",
            goal="Analyze sentiments and recommend Buy/Sell/Hold.",
            backstory="Based on sentiment trends, guide the investor appropriately.",
            llm=llm,
            allow_delegation=False,
            verbose=False
        )

        collect = Task(
            description="Collect all the latest articles related to the stock/company.",
            expected_output="A list of relevant articles with basic sentiment scores.",
            agent=collector
        )

        summerize = Task(
            description="Summarize the collected articles and extract key trends.",
            expected_output="A clean summary of market sentiment and trending topics.",
            agent=summerizer
        )

        analyse = Task(
            description="Analyze the overall sentiment and recommend BUY/SELL/HOLD.",
            expected_output="A final recommendation based on the aggregated sentiment of the articles.",
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
            st.success("✅ Analysis complete!")
            st.write("Analyzing trends for:", inputStock)
            st.write("Result:", response.raw)
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
