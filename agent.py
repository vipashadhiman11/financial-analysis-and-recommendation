

from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date,timedelta
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title = "Market Trends Analyst", layout = "centered")
st.title("Your Financial Advisor")
st.write('Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. I will also recomment you if you should Buy/ Sell or Hold the stock :sunglasses:')

inputStock = st.text_input("Enter stock name or company name:")
# if user_name:

if st.button("Submit", type="primary"):
    llm = LLM(model = "groq/openai/gpt-oss-120b",
            temperature = 0.2,
            # max_completion_tokens = 256,
            top_p = 0.9
        )
    
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

        # 👉 Extend date range to ensure we get results
        url = (
            f"https://api.apitube.io/v1/news/everything?"
            f"title={entity}&published_at.start=2025-09-01&published_at.end=2025-10-28"
            f"&sort.order=desc&language.code=en&api_key={APITUBE_API_KEY}"
        )

        response = requests.get(url).json()
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
                if count < 30:  # ✅ allow more results
                    next_page_url = response.get("next_page")
                    if not next_page_url:
                        break
                    response = requests.get(next_page_url).json()
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

        # 📝 Save articles for charting
        with open("articles.txt", "w") as file:
            for article in articles:
                file.write(str(article) + "\n")

        print(f"✅ Saved {len(articles)} articles")
        return articles

    except Exception as e:
        return {"error": f"Failed to fetch articles: {e}"}

    @tool("sentiment_analysis")
    def sentiment_analysis(articles: list[str]) -> str:
        """
        Identify the sentiment of the article as positive, negative or neutral
        Args:
            article: List of string input that accepts a list of articles
        Returns:
            gives the sentiment of the article as:
                - Positive
                - Negative
                - Neutral
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
    
    collector = Agent(
        role = "Articles collector",
        goal = "Asks the user about the {topic} and collects the articles releated to that topic using tools.",
        backstory = "The {topic} will be an organisation of stock name. Don't take any other input except topic"
                    "Use the tool 'get_articles_APItube' to fetch the articles.\n"
                    "Give the total number of articles collected.",
        tools = [get_articles_APItube],
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
        agents = [collector, summerizer, analyser],
        tasks = [collect, summerize, analyse],
        process=Process.sequential,
        verbose = False
    )
    
    try:
        response = crew.kickoff(inputs = {"topic": inputStock})
        st.write("Analysing trends for: ", inputStock)
        st.write("Result:", response.raw)

            # ================== 📊 Sentiment Visualization ===================
        sentiments = []
        try:
            with open("articles.txt", "r") as file:
                for line in file:
                    try:
                        article = eval(line.strip())
                        sentiments.append(float(article.get('sentiment', 0)))
                    except:
                        continue
        except FileNotFoundError:
            sentiments = []

        if sentiments:
            # Convert sentiment scores to labels
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

            # --- 🥧 Pie Chart ---
            import matplotlib.pyplot as plt
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

            # --- 📊 Bar chart ---
            st.bar_chart(sentiment_df['Sentiment'].value_counts())

            # --- 📅 Sentiment Trend Line Chart ---
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
                        except:
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
                st.warning("No trend data available for trend chart.")

            # --- 📢 Final Recommendation Banner ---
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
        st.write(f"An error occured: {e}")
