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

st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write(
    'Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. '
    'I will also recommend you if you should Buy/ Sell or Hold the stock 😎'
)

inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    # ✅ Lightweight model with token limit to prevent rate-limit errors
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
            if response["status"] == "ok":
                for result in response["results"]:
                    count += 1
                    article = {}
                    article["article_body"] = result["body"]
                    article["sentiment"] = result["sentiment"]["overall"]["score"]
                    articles.append(article)
                while response["has_next_pages"]:
                    if count < 20:
                        next_page_url = response["next_page"]
                        next_page_response = requests.get(url).json()
                        if response["status"] == "ok":
                            for result in response["results"]:
                                count += 1
                                article = {}
                                article["article_body"] = result["body"]
                                article['sentiment'] = result["sentiment"]["overall"]["score"]
                                articles.append(article)
                    else:
                        break
            with open("articles.txt", "w") as file:
                for article in articles:
                    file.write(str(article) + "\n")
            return articles
        except Exception as e:
            return {"error": f"Failed to read URL {e}"}

    @tool("sentiment_analysis")
    def sentiment_analysis(articles: list[str]) -> str:
        """
        Run sentiment analysis on article text using FinBERT.

        Args:
            articles (list[str]): A list of article texts.

        Returns:
            str: Sentiment labels and scores.
        """
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

    collector = Agent(
        role="Articles collector",
        goal="Asks the user about the {topic} and collects the articles releated to that topic using tools.",
        backstory="The {topic} will be an organisation of stock name. Don't take any other input except topic. "
                  "Use the tool 'get_articles_APItube' to fetch the articles.\n"
                  "Give the total number of articles collected.",
        tools=[get_articles_APItube],
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    summerizer = Agent(
        role="Article summerizer",
        goal="Summerize the articles collected by collector and summerize them to fetch the crux of it",
        backstory="You are summerizing all the articles into one with utmost precision and keeping in mind the trends we are getting from the articles.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    analyser = Agent(
        role="Financial Analyst",
        goal="You will guide user to either Buy/Sell or Hold the stock of the organisation.",
        backstory="You will observe the sentiment all the article. "
                  "You are working on identifying latest trends about the topic: {topic}. "
                  "You will take the input from the collector agent. "
                  "After that you will predict the overall sentiment as positive, negative or neutral. "
                  "Based on the sentiment predicted by you, you will tell us whether we should buy/sell or hold the stock for now. "
                  "Your target is to maximise user profit.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    collect = Task(
        description=(
            "1. The {topic} will be an organisation of stock name.\n"
            "2. Use the tool to collect all the news articles on the provided {topic} using tool 'get_articles_APItube'.\n"
            "3. Prioritize the latest trends and news on the {topic}.\n"
        ),
        expected_output="Articles related to the organisation or stock given by the user\n",
        agent=collector
    )

    summerize = Task(
        description=(
            "1. Summerize the articles you collected from collector into maximum 500 words.\n"
            "3. Prioritize the latest trends and news on the {topic}.\n"
        ),
        expected_output="Summerize the articles related to the organisation or stock given by the user\n",
        agent=summerizer
    )

    analyse = Task(
        description=(
            "1. Use the content collected to create an opinion on {topic}.\n"
            "2. Use the collected articles to identify trends in the market\n"
            "3. Based on the trends observed try to identify overall sentiment of the market as positive/negative or neutral.\n"
            "4. Once the sentiment is identified guide the user to either Buy/sell or hold the stock of the company or organisation provided.\n"
            "5. Ensure the proper analysis and provide detailed analysis.\n"
            "6. Tell the total number of articles you used for analysis.\n"
        ),
        expected_output="Provide overall Sentiment about the topic as positive/negative or neutral and based on it guide us if we should buy/ sell or hold the stock.",
        agent=analyser
    )

    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential,
        verbose=False
    )

    try:
        # 🚀 Run Crew
        response = crew.kickoff(inputs={"topic": inputStock})
        time.sleep(2)  # 🧊 Slow down slightly to avoid burst rate-limits
        st.write("Analysing trends for: ", inputStock)
        st.write("Result:", response.raw)

        # 📊 Sentiment Visualization Section
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
