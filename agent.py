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
    st.write("✅ Submit button clicked")

    try:
        # 🎯 Step 1: Initialize LLM
        llm = LLM(
            model="groq/openai/gpt-oss-120b",
            temperature=0.2,
            top_p=0.9
        )

        # 🧾 Step 2: Fetch Articles
        st.write(f"🔍 Fetching articles for: {inputStock}")
        articles = get_articles_APItube.func(inputStock)  # .func to call the tool directly
        st.write("🧾 Raw articles response:", articles)

        # 🧭 Step 3: Handle no data or error
        if not articles or (isinstance(articles, dict) and "error" in articles):
            st.warning("⚠️ No articles found or API failed. Try another stock name.")
            st.stop()

        st.success(f"✅ {len(articles)} articles fetched for {inputStock}")

        # 🧠 Step 4: Run Crew analysis
        response = crew.kickoff(inputs={"topic": inputStock, "articles": articles})
        st.subheader(f"📊 Analysing trends for: {inputStock}")
        st.write("📝 Result:", response.raw)

        # 🧮 Step 5: Sentiment Analysis Visualization
        sentiments = []
        with open("articles.txt", "r") as file:
            for line in file:
                try:
                    article = eval(line.strip())
                    sentiments.append(float(article.get('sentiment', 0)))
                except:
                    continue

        if sentiments:
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
            col1.metric("🟢 Positive", sentiment_counts.get("Positive", 0))
            col2.metric("🔴 Negative", sentiment_counts.get("Negative", 0))
            col3.metric("⚪ Neutral", sentiment_counts.get("Neutral", 0))

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

            # 📊 Bar Chart
            st.bar_chart(sentiment_df['Sentiment'].value_counts())

            # 📈 Sentiment Trend Line Chart
            trend_data = []
            with open("articles.txt", "r") as file:
                for line in file:
                    try:
                        article = eval(line.strip())
                        s = float(article.get('sentiment', 0))
                        d = article.get('published_at', '')
                        if d:
                            trend_data.append({"date": d.split("T")[0], "sentiment": s})
                    except:
                        continue

            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                trend_df['date'] = pd.to_datetime(trend_df['date'], errors='coerce')
                trend_df = trend_df.dropna(subset=['date'])
                trend_df = trend_df.groupby('date')['sentiment'].mean().reset_index()
                st.subheader("📈 Sentiment Trend Over Time")
                st.line_chart(trend_df.set_index('date')['sentiment'])
            else:
                st.warning("No trend data available.")

            # 📌 Final Recommendation
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
        st.error(f"❌ Error during processing: {str(e)}")

       

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
