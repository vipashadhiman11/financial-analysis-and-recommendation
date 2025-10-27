import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, timedelta
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from transformers import pipeline


# ==============================
# 🔑 Load Environment Variables
# ==============================
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ==============================
# 🧰 Define Tools Globally
# ==============================

@tool("get_articles_APItube")
def get_articles_APItube(entity: str) -> list:
    """
    Fetch articles for a given entity using APITube API.
    Returns a list of dicts with article body, sentiment, and published date.
    """
    try:
        print(f"🔍 Fetching articles for: {entity}")
        articles = []
        APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"

        # ✅ Broader date range to increase hit rate
        today = date.today()
        start_date = today - timedelta(days=30)
        url = (
            f"https://api.apitube.io/v1/news/everything?title={entity}"
            f"&published_at.start={start_date}&published_at.end={today}"
            f"&sort.order=desc&language.code=en&api_key={APITUBE_API_KEY}"
        )

        response = requests.get(url).json()
        if response.get("status") == "ok":
            for result in response.get("results", []):
                article = {
                    "article_body": result.get("body", ""),
                    "sentiment": result.get("sentiment", {}).get("overall", {}).get("score", 0),
                    "published_at": result.get("published_at", "")
                }
                articles.append(article)

        # 💾 Save to file for visualization
        with open("articles.txt", "w") as file:
            for article in articles:
                file.write(str(article) + "\n")

        return articles

    except Exception as e:
        return {"error": f"Failed to fetch articles: {e}"}


@tool("sentiment_analysis")
def sentiment_analysis(articles: list[str]) -> str:
    """
    Perform sentiment analysis using FinBERT model.
    """
    try:
        print("🧠 Loading FinBERT model...")
        model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        sentiments = []
        for text in articles:
            if not text:
                sentiments.append({"label": "neutral", "score": 0.0})
                continue
            result = model(text[:512])[0]
            sentiments.append(result)
        return sentiments
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {e}"}


# ==============================
# 🖥️ Streamlit UI Setup
# ==============================
st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write(
    "Hello, I am your financial advisor. "
    "I will give you a complete analysis of your stock or organisation. "
    "I will also recommend you if you should Buy/ Sell or Hold the stock 😎"
)
