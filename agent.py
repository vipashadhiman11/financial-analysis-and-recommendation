from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date, timedelta
from dotenv import load_dotenv
import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

# ✅ Load API keys
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ✅ Streamlit UI setup
st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write(
    'Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. '
    'I will also recommend you if you should Buy/ Sell or Hold the stock 😎'
)

# ✅ User input
inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        temperature=0.2,
        top_p=0.9
    )

    # 🧰 Tool for fetching articles
    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> list[list]:
        """
        Fetch articles related to a company or stock and save them locally.
        Returns: list of articles with sentiment scores.
        """
        try:
            print("Running API")
            articles = []
            APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"

            url = (
                f"https://api.apitube.io/v1/news/everything?"
                f"title={entity}&published_at.start=2025-09-01&published_at.end=2025-10-28"
                f"&sort.order=desc&language.code=en&api_key={APITUBE_API_KEY}"
            )

            response = requests.get(url).json()

            # No results case
            if response.get("status") != "ok" or not response.get("results"):
                return {"error": f"No articles found for '{entity}'"}

            for result in response.get("results", []):
                article = {
                    "article_body": result.get("body", ""),
                    "sentiment": result.get("sentiment", {}).get("overall", {}).get("score", 0),
                    "published_at": result.get("published_at", "")
                }
                articles.append(article)

            # Save to file
            with open("articles.txt", "w") as file:
                for article in articles:
                    file.write(str(article) + "\n")

            print(f"✅ Saved {len(articles)} articles")
            return articles

        except Exception as e:
            return {"error": f"Failed to fetch articles: {e}"}

    # 🧠 Crew AI setup
    collector = Agent(
    role="Articles collector",
    goal="Collect articles for a given topic using tools.",
    backstory="The topic will be an organisation or stock name.",
    tools=[get_articles_APItube],
    llm=llm,
    allow_delegation=False,
    verbose=False
)
