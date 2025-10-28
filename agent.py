from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date, timedelta
from dotenv import load_dotenv
import os
import time
import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# 🧭 Streamlit Page Setup
st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write(
    'Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. '
    'I will also recommend you if you should Buy/ Sell or Hold the stock 😎'
)

# 📥 Input box
inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    # ✅ Lightweight Groq model to prevent rate-limit issues
    llm = LLM(
        model="groq/gemini-2.5-flash-lite",
        temperature=0.2,
        top_p=0.9,
        max_tokens=512
    )

    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> list[list]:
        """
        Fetch recent news articles and sentiment data for a given company or stock.

        Args:
            entity (str): The company or stock name to search for.

        Returns:
            list[list]: A list of articles with sentiment scores.
        """
        try:
            print("Running API")
            articles = []
            APITUBE_API_KEY = "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
            url = (
                "https://api.apitube.io/v1/news/everything?title=" + entity +
                "&published_at.start=2025-10-20&published_at.end=2025-10-25"
                "&sort.order=desc&language.code=en&api_key=" + APITUBE_API_KEY
            )
            response = requests.get(url).json()
            count = 0
            if response.get("status") == "ok":
                for result in response["results"]:
                    count += 1
                    article = {}
                    article["article_body"] = result["body"]
                    article["sentiment"] = result["sentiment"]["overall"]["score"]
                    articles.append(article)
                while response.get("has_next_pages"):
                    if count < 20:
                        next_page_url = response["next_page"]
                        next_page_response = requests.get(next_page_url).json()
                        if next_page_response.get("status") == "ok":
                            for result in next_page_response["results"]:
                                count += 1
                                article = {}
                                article["article_body"] = result["body"]
                                article['sentiment'] = result["sentiment"]["overall"]["score"]
                                articles.append(article)
                    else:
                        break
            # Save for sentiment chart
            with open("articles.txt", "w") as file:
                for article in articles:
                    file.write(str(article) + "\n")
            return articles
        except Exception as e:
            return {"error": f"Failed to fetch articles: {e}"}

    # 🧠 Define CrewAI Agents
    collector = Agent(
        role="Articles collector",
        goal="Collect articles for the given topic using the APItube tool.",
        backstory="You will use the APItube tool to fetch relevant articles for the company or stock.",
        tools=[get_articles_APItube],
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    summerizer = Agent(
        role="Article summerizer",
        goal="Summarize all collected articles into one clear, insightful summary.",
        backstory="You will analyze and summarize the articles with precision, focusing on trends.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    analyser = Agent(
        role="Financial Analyst",
        goal="Guide user to either Buy/Sell or Hold the stock based on article sentiment.",
        backstory="Observe the sentiment of all articles, identify trends, and make a recommendation.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    # 🧭 Tasks
    collect = Task(
        description="Use APItube to collect articles about the topic.",
        expected_output="A list of articles with sentiment scores.",
        agent=collector
    )

    summerize = Task(
        description="Summarize the collected articles.",
        expected_output="A summarized text of articles and trends.",
        agent=summerizer
    )

    analyse = Task(
        description="Analyze sentiment of articles and give investment recommendation.",
        expected_output="Positive/Negative/Neutral sentiment and Buy/Sell/Hold recommendation.",
        agent=analyser
    )

    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential,
        verbose=False
    )

    # 🚀 Run Crew and Show Results
    try:
        response = crew.kickoff(inputs={"topic": inputStock})
        time.sleep(2)  # small delay to prevent burst rate limits
        st.write("Analyzing trends for:", inputStock)
        st.write("Result:", response.raw)

        # 📊 Pie Chart Visualization
        sentiments = []
        try:
            with open("articles.txt", "r") as file:
                for line in file:
                    if "sentiment" in line:
                        try:
                            score = float(line.split("'sentiment': ")[1].split("}")[0])
                            sentiments.append(score)
                        except:
                            continue
        except FileNotFoundError:
            st.warning("❌ No articles file found for sentiment chart.")

        if sentiments:
            labels = []
            for s in sentiments:
                if s > 0.05:
                    labels.append("Positive")
                elif s < -0.05:
                    labels.append("Negative")
                else:
                    labels.append("Neutral")

            df = pd.DataFrame({"Sentiment": labels})
            sentiment_counts = df["Sentiment"].value_counts()

            st.subheader("📊 Sentiment Overview")
            st.write(sentiment_counts)

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
        else:
            st.warning("No sentiment data available to display charts.")

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
