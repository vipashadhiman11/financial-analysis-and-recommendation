import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool as crew_tool

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["LITELLM_DEFAULT_MODEL"] = "openai/gpt-oss-120b"
os.environ["LITELLM_FALLBACKS"] = ""

st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("📈 Your Financial Advisor")
st.write(
    "Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. "
    "I will also recommend you if you should Buy / Sell / Hold the stock 😎"
)

FMP_API_KEY = "ES9nZy86YlYSNkohutKivy2xDUfEq"

def get_gainers(number):
    url = f"https://financialmodelingprep.com/stable/biggest-gainers?apikey={FMP_API_KEY}"
    try:
        response_gainers = requests.get(url, timeout=10).json()
    except Exception as e:
        st.error(f"Error fetching gainers: {e}")
        return [{"name": "API Error", "percentage": 0}]
    if not isinstance(response_gainers, list):
        return [{"name": "Invalid API Key", "percentage": 0}]
    gainers = []
    for response in response_gainers[:number]:
        if isinstance(response, dict) and "name" in response and "changesPercentage" in response:
            gainers.append({
                "name": response["name"],
                "percentage": response["changesPercentage"]
            })
    return gainers

def get_losers(number):
    url = f"https://financialmodelingprep.com/stable/biggest-losers?apikey={FMP_API_KEY}"
    try:
        response_losers = requests.get(url, timeout=10).json()
    except Exception as e:
        st.error(f"Error fetching losers: {e}")
        return [{"name": "API Error", "percentage": 0}]
    if not isinstance(response_losers, list):
        return [{"name": "Invalid API Key", "percentage": 0}]
    losers = []
    for response in response_losers[:number]:
        if isinstance(response, dict) and "name" in response and "changesPercentage" in response:
            losers.append({
                "name": response["name"],
                "percentage": response["changesPercentage"]
            })
    return losers

with st.sidebar:
    st.title("Top 5 Gainers")
    for g in get_gainers(5):
        color = "green" if g["percentage"] > 0 else "blue"
        st.badge(f"{g['name']} ({g['percentage']}%)", color=color)
    st.title("Top 5 Losers")
    for l in get_losers(5):
        color = "red" if l["percentage"] < 0 else "blue"
        st.badge(f"{l['name']} ({l['percentage']}%)", color=color)

def _fetch_articles_apitube(entity: str) -> list:
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
        if articles:
            with open("articles.txt", "w") as f:
                for a in articles:
                    f.write(str(a) + "\n")
        return articles
    except Exception as e:
        return {"error": f"Failed to fetch: {e}"}

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
    if not sentiments:
        st.warning("⚠️ Using fallback demo data.")
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
    st.bar_chart(sentiment_df["Sentiment"].value_counts())
    avg_sentiment = sum(sentiments) / len(sentiments)
    st.subheader("📌 Investment Recommendation")
    if avg_sentiment > 0.05:
        st.success("🟢 Positive sentiment — Recommendation: BUY")
    elif avg_sentiment < -0.05:
        st.error("🔴 Negative sentiment — Recommendation: SELL")
    else:
        st.warning("⚪ Neutral sentiment — Recommendation: HOLD")

inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    try:
        llm = LLM(
            model="openai/gpt-oss-120b",
            temperature=0.2,
            top_p=0.9
        )
    except Exception as e:
        st.error(f"LLM initialization failed. Check GROQ_API_KEY. Error: {e}")
        llm = None
    if llm:
        collector = Agent(
            role="Data Processor & Article Counter",
            goal="Read the articles provided in 'article_content' input and prepare for sentiment analysis.",
            tools=[],
            llm=llm,
            allow_delegation=False,
            verbose=False
        )
        summerizer = Agent(
            role="Article summerizer",
            goal="Summarize the collected articles and extract key trends.",
            llm=llm,
            allow_delegation=False,
            verbose=False
        )
        analyser = Agent(
            role="Financial Analyst",
            goal="Analyze sentiment and provide BUY/SELL/HOLD recommendation.",
            llm=llm,
            allow_delegation=False,
            verbose=False
        )
        collect = Task(
            description="Process the collected articles.",
            input_variables=["topic", "article_content"],
            expected_output="List of article bodies and sentiments",
            agent=collector
        )
        summerize = Task(
            description="Summarize the collected articles and key sentiment points.",
            expected_output="A concise summary of the market view.",
            agent=summerizer
        )
        analyse = Task(
            description="Analyze the sentiment and recommend BUY, SELL, or HOLD.",
            expected_output="Final investment recommendation.",
            agent=analyser
        )
        crew = Crew(
            agents=[collector, summerizer, analyser],
            tasks=[collect, summerize, analyse],
            process=Process.sequential,
            verbose=False
        )
        try:
            articles = _fetch_articles_apitube(inputStock)
            if isinstance(articles, dict) and "error" in articles:
                st.warning(f"No real articles fetched. Using demo data. ({articles['error']})")
                demo_articles = [
                    {"article_body": f"{inputStock} earnings beat expectations", "sentiment": 0.7},
                    {"article_body": f"{inputStock} faces regulatory scrutiny", "sentiment": -0.5},
                    {"article_body": f"Analysts optimistic on {inputStock}", "sentiment": 0.6},
                    {"article_body": f"Mixed outlook for {inputStock}", "sentiment": 0.0},
                ]
                with open("articles.txt", "w") as f:
                    for a in demo_articles:
                        f.write(str(a) + "\n")
            visualize_sentiments()
            article_content = ""
            if os.path.exists("articles.txt"):
                with open("articles.txt", "r") as f:
                    article_content = f.read()
            response = crew.kickoff(
                inputs={
                    "topic": inputStock,
                    "article_content": article_content
                }
            )
            st.success("✅ Analysis complete!")
            st.write("Analyzing trends for:", inputStock)
            st.markdown(f"**Result:**\n{response}")
        except Exception as e:
            st.error(f"❌ Error during CrewAI processing: {e}")
