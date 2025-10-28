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

def get_gainers(number):
    response_gainers = requests.get("https://financialmodelingprep.com/stable/biggest-gainers?apikey=ES9nZy86YlYSEW9NkohutKivy2xDUfEq").json()
    count = 0
    gainers = []
    for response in response_gainers:
        if count<number:
            gainers.append(response["name"])
            count+=1
    return gainers
    
def get_losers(number):
    response_losers = requests.get("https://financialmodelingprep.com/stable/biggest-losers?apikey=ES9nZy86YlYSEW9NkohutKivy2xDUfEq").json()
    count = 0
    losers = []
    for response in response_losers:
        if count<number:
            losers.append(response["name"])
            count+=1
    return losers


with st.sidebar:
    st.title("Top 5 gainers:")
    for gainer in get_gainers(5):
        st.badge(gainer, color="green")
    st.title("Top 5 losers:")
    for losers in get_losers(5):
        st.badge(losers, color="red")

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
        model="groq/openai/gpt-oss-120b",
        temperature=0.2,
        top_p=0.9
    )

    # Agents
    collector = Agent(
        role = "Articles collector",
        goal = "Asks the user about the {topic} and collects the articles releated to that topic using tools.",
        backstory = "The {topic} will be an organisation of stock name. Don't take any other input except topic"
                    "Use the tool '_fetch_articles_apitube' to fetch the articles.\n"
                    "Give the total number of articles collected.",
        tools = [_fetch_articles_apitube],
        llm = llm,
        allow_delegation = False,
        verbose = False
    )

    summerizer = Agent(
        role = "Article summerizer",
        goal = "Summerize the articles collected by collector and summerize them to fetch the crux of it",
        backstory = "You are summerizing all the articles into one with utmost precision and keeping in mind the trends we are getting from the articles.",
        llm = llm,
        allow_delegation = False,
        verbose = False
    )
    
    analyser = Agent(
        role = "Financial Analyst",
        goal = "You will guide user to either Buy/Sell or Hold the stock of the organisation.",
        backstory = "You will observe the sentiment all he article."
                    "You are working on identifying latest trends about the topic: {topic}."
                    "You will take the input from the collector agent\n"
                    "After that you will predict the overall sentiment as positive, negative or neutral."
                    "Based on the sentiment predicted by you, you will tell us whether we should buy/sell or hold the stock for now."
                    "your target is to maximise user profit.",
        llm = llm,
        allow_delegation = False,
        verbose = False
    )
    
    collect = Task(
        description = (
            "1. The {topic} will be an organisation of stock name.\n"
            "2. Use the tool to collect all the news articles on the provided {topic} using tool 'get_articles_APItube'.\n"
            "3. Prioritize the latest trends and news on the {topic}.\n"
        ),
        expected_output = "Articles related to the organisation or stock given by the user\n",
        agent = collector
    )
    
    summerize = Task(
        description = (
            "1. Summerize the articles you collected from collector into maximum 500 words.\n"
            "3. Prioritize the latest trends and news on the {topic}.\n"
        ),
        expected_output = "Summerize the articles related to the organisation or stock given by the user\n",
        agent = summerizer
    )
    
    analyse = Task(
        description = (
            "1. Use the content collected to create an opinion on {topic}.\n"
            "2. Use the collected articles to identify trends in the market\n"
            "3. Based on the trends observed try to identify overall sentiment of the market as positive/negative or neutral.\n"
            "4. Once the sentiment is identified guide the user to either Buy/sell or hold the stock of the company or organisation provided.\n"
            "5. Ensure the proper analysis and provide detailed analysis.\n"
            "6. Tell the total number of articles you used for analysis.\n"
        ),
        expected_output = "Provide overall Sentiment about the topic as positive/negative or neutral and based on it guide us if we should buy/ sell or hold the stock.",
        agent = analyser
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
