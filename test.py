"""

Market Sentiment Agent (Terminal Version)

-----------------------------------------

This script fetches recent financial news for a given company,

analyzes sentiment using a financial-specific model (FinBERT),

and prints a summary report in the console.
 
Requirements:

 pip install requests pandas transformers torch tabulate
 
Usage:

 python market_sentiment_agent.py

"""
 
import os

import requests

import pandas as pd

from datetime import datetime, timedelta

from transformers import pipeline

from tabulate import tabulate
 
 
# ----------------------------

# CONFIGURATION

# ----------------------------

NEWS_API_KEY = "d1916bcfb48543f9bab7a290319401b6" hashtag#os.getenv("NEWSAPI_KEY") # Set your NewsAPI key here or via environment variable

NEWS_API_URL = "https://newsapi.org/v2/everything"
 
# ----------------------------

# FETCH NEWS ARTICLES

# ----------------------------

def fetch_news(company: str, days_back: int = 2, page_size: int = 50):

 """

 Fetch recent news articles for a company from NewsAPI.

 """

 if not NEWS_API_KEY:

 raise ValueError("⚠️ Missing NEWSAPI_KEY. Set it as an environment variable.")
 
 from_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"
 
 params = {

 "q": company,

 "language": "en",

 "sortBy": "publishedAt",

 "from": from_date,

 "pageSize": page_size,

 "apiKey": NEWS_API_KEY,

 }
 
 response = requests.get(NEWS_API_URL, params=params, timeout=15)

 response.raise_for_status()

 data = response.json()
 
 articles = []

 for a in data.get("articles", []):

 articles.append({

 "publishedAt": a.get("publishedAt"),

 "source": a.get("source", {}).get("name", ""),

 "title": a.get("title", ""),

 "description": a.get("description", ""),

 "url": a.get("url", ""),

 })

 print("=====articles", articles)

 return pd.DataFrame(articles)
 
# ----------------------------

# SENTIMENT ANALYSIS

# ----------------------------

def analyze_sentiment(text_list):

 """

 Analyze sentiment for each text using FinBERT model.

 """

 print("🧠 Loading FinBERT model...")

 model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
 
 sentiments = []

 print("======text_list", text_list)

 for text in text_list:

 if not text:

 sentiments.append({"label": "neutral", "score": 0.0})

 continue

 result = model(text[:512])[0]

 sentiments.append(result)

 print("=====sentiments", sentiments)

 return sentiments
 
# ----------------------------

# MAIN AGENT LOGIC

# ----------------------------

def sentiment_agent(company: str):

 """

 Fetches news, analyzes sentiment, and prints results.

 """

 print(f"\nFetching news for: {company} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
 
 # Step 1: Fetch news

 df = fetch_news(company)

 if df.empty:

 print(f"❌ No news found for {company}")

 return
 
 # Step 2: Analyze sentiment

 texts = df["title"].fillna('') + ". " + df["description"].fillna('')

 df["sentiment"] = analyze_sentiment(texts.apply(lambda x: x.strip()).tolist())

 df["label"] = df["sentiment"].apply(lambda x: x["label"].lower())

 df["score"] = df["sentiment"].apply(lambda x: round(float(x["score"]), 3))
 
 # Step 3: Aggregate results

 total = len(df)

 positive = len(df[df["label"] == "positive"])

 negative = len(df[df["label"] == "negative"])

 neutral = len(df[df["label"] == "neutral"])

 avg_score = df["score"].mean()
 
 # Step 4: Display Results

 print("\n Market Sentiment Summary:")

 summary_table = [

 [company, total, round(avg_score, 3), positive, negative, neutral]

 ]

 headers = ["company", "total_articles", "avg_sentiment", "positive", "negative", "neutral"]

 print(tabulate(summary_table, headers=headers, tablefmt="fancy_grid"))

 data = (tabulate(summary_table, headers=headers, tablefmt="fancy_grid"))

 return summary_table

 print("\n Detailed Articles:\n")

 for i, row in df.iterrows():

 print(f" {row['title']}")

 print(f" Sentiment: {row['label'].capitalize()} ({row['score']})")

 print(f" Published: {row['publishedAt']} | Source: {row['source']}")

 print(f" {row['url']}\n")
 
# ----------------------------

# ENTRY POINT

# ----------------------------

if __name__ == "__main__":

 company_name = input("Enter company name (e.g., Tesla, Apple, Microsoft): ").strip()

 import uvicorn

 uvicorn.run(app, host = "127.0.0.1", port = 1200)

 if company_name:

 sentiment_agent(company_name)

 else:

 print("❌ Company name cannot be empty.")