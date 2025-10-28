# agent.py

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
# NOTE: Removed LLM import from crewai.tools as it is no longer used for data collection
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool as crew_tool

# ---------------------------------------------------------
# 🔐 Load API Keys
# ---------------------------------------------------------
load_dotenv()
# NOTE: Groq API Key is not strictly needed here if set as ENV variable, but kept for reference
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("📈 Your Financial Advisor")
st.write(
    "Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. "
    "I will also recommend you if you should Buy / Sell / Hold the stock 😎"
)

def get_gainers(number: int) -> list[dict]:
    response_gainers = requests.get("https://financialmodelingprep.com/stable/biggest-gainers?apikey=l0YbTN9l3jYhJnfsNiiuU9khcDLa9tia").json()
    count = 0
    gainers = []
    for response in response_gainers:
        if count<number:
            gainers.append({"name":response["name"],
                            "percentage":response["changesPercentage"]})
            count+=1
    return gainers
    
def get_losers(number: int) -> list[dict]:
    response_losers = requests.get("https://financialmodelingprep.com/stable/biggest-losers?apikey=l0YbTN9l3jYhJnfsNiiuU9khcDLa9tia").json()
    count = 0
    losers = []
    for response in response_losers:
        if count<number:
            losers.append({"name":response["name"],
                            "percentage":response["changesPercentage"]})
            count+=1
    return losers

get_gainers(5)
with st.sidebar:
    st.title("Top 5 gainers:")
    for gainer in get_gainers(5):
        st.markdown(
    ":green-badge["+gainer['name']+"] :blue-badge[+"+str(gainer['percentage'])+"%]"
        )
    st.title("Top 5 losers:")
    for losers in get_losers(5):
        st.markdown(
    ":red-badge["+losers['name']+"] :blue-badge["+str(losers['percentage'])+"%]"
        )

# ---------------------------------------------------------
# 📰 Plain helper to fetch articles (call this directly)
# ---------------------------------------------------------
# NOTE: This function is called directly before crew.kickoff() and saves to articles.txt
def _fetch_articles_apitube(entity: str) -> list:
    """Plain helper function to fetch recent articles with sentiment from API Tube."""
    try:
        articles = []
        # NOTE: Using a placeholder API key for demonstration; replace with secure key later
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

        # Save for visualization (Used by visualize_sentiments and read for CrewAI)
        if articles:
            with open("articles.txt", "w") as f:
                for a in articles:
                    f.write(str(a) + "\n")
        return articles
    except Exception as e:
        # Return a dictionary with an error message on failure
        return {"error": f"Failed to fetch: {e}"}

# ---------------------------------------------------------
# 🛠 CrewAI Tool wrapper (REMOVED: Tool is no longer needed)
# ---------------------------------------------------------
# @crew_tool("get_articles_APItube")
# def get_articles_APItube_tool(entity: str) -> str:
#     # ... Tool is removed to force the agent to use the pre-fetched file content
#     # ...
pass

# ---------------------------------------------------------
# 📊 Visualization (pie + bar + recommendation banner) - Remains unchanged
# ---------------------------------------------------------
def visualize_sentiments():
    sentiments = []
    if os.path.exists("articles.txt"):
        with open("articles.txt", "r") as file:
            for line in file:
                try:
                    article = eval(line.strip())
                    # Ensure sentiment is a float (API returns 0 if score is missing)
                    sentiments.append(float(article.get("sentiment", 0))) 
                except Exception:
                    continue

    # Fallback demo data ONLY if articles.txt is truly empty/unreadable
    if not sentiments:
        # If the fetch failed AND the file is empty, use a consistent demo set for the chart
        # NOTE: This only runs if the API fetch fails and the fallback writes fail/is empty
        st.warning("Using hardcoded demo data for visualization due to fetch failure.")
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
        # Ensure colors match the labels, handling cases where a category might be missing
        colors=[
            "green" if label == "Positive" else "red" if label == "Negative" else "gray"
            for label in sentiment_counts.index
        ]
    )
    ax.axis("equal")
    st.pyplot(fig)

    # 📊 Bar Chart
    st.bar_chart(sentiment_df["Sentiment"].value_counts())

    # 📌 Recommendation Banner
    # Check if we have any sentiments before calculating average
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
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
    # 1. LLM Initialization (Fixed model name to a standard Groq Llama 3)
    # Using 'groq/openai/gpt-oss-120b' is non-standard. Using a valid Llama 3 8B is better.
    # NOTE: If you must use a LiteLLM model string, verify the exact format, but Llama3 is standard.
    try:
        llm = LLM(
            model="llama3-8b-8192", 
            temperature=0.2,
            top_p=0.9
        )
    except Exception as e:
        st.error(f"LLM initialization failed. Ensure GROQ_API_KEY is set and model is valid. Error: {e}")
        llm = None # Set to None to prevent agents from being created

    if llm:
        # Agents (Roles updated to reflect data consumption, not collection)
        # ------------------------------------------------------------------
        collector = Agent(
            role = "Data Processor & Article Counter",
            # Goal is now to consume the content provided in the task input
            goal = "Read the articles provided in the 'article_content' input, count them, and prepare a list of article sentiments and bodies for the next agent.",
            backstory = "The {topic} is an organisation or stock name. Your job is to process the collected news articles provided in the task input. **Do NOT use any tools.**",
            tools = [], # <--- CRITICAL FIX: REMOVED TOOL
            llm = llm,
            allow_delegation = False,
            verbose = False
        )

        summerizer = Agent(
            role = "Article summerizer",
            goal = "Summerize the articles received from the collector agent to fetch the crux of it",
            backstory = "You are summerizing all the articles into one with utmost precision and keeping in mind the trends we are getting from the articles.",
            llm = llm,
            allow_delegation = False,
            verbose = False
        )
        
        analyser = Agent(
            role = "Financial Analyst",
            goal = "You will guide user to either Buy/Sell or Hold the stock of the organisation.",
            backstory = (
                "You will observe the sentiment all the article."
                "You are working on identifying latest trends about the topic: {topic}."
                "You will take the input from the collector and summerizer agents\n"
                "After that you will predict the overall sentiment as positive, negative or neutral."
                "Based on the sentiment predicted by you, you will tell us whether we should buy/sell or hold the stock for now."
                "your target is to maximise user profit."
            ),
            llm = llm,
            allow_delegation = False,
            verbose = False
        )
        
        # Tasks (Updated to use the new input variable)
        # ------------------------------------------------------------------
        collect = Task(
            description = (
                "1. The {topic} will be an organisation of stock name.\n"
                # CRITICAL FIX: Tell the agent where the content is coming from
                "2. Read the full content of the articles provided in the 'article_content' input:\n\n"
                "{article_content}\n\n"
                "3. Count the total number of articles you processed.\n"
                "4. Prioritize the latest trends and news on the {topic} from the content provided.\n"
            ),
            # CRITICAL FIX: Define the required inputs for this task
            input_variables=["topic", "article_content"], 
            expected_output = "The total count of articles and a structured, comprehensive list of the article bodies and their associated sentiment scores (e.g., [{'body': '...', 'sentiment': 0.8}, ...])",
            agent = collector
        )
        
        summerize = Task(
            description = (
                "1. Summerize the articles you collected from collector into maximum 500 words.\n"
                "2. Prioritize the latest trends and news on the {topic}.\n" # Fixed step number
            ),
            expected_output = "A concise summary of the articles related to the organisation or stock given by the user, highlighting key sentiment drivers.",
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
            expected_output = "Provide overall Sentiment about the topic as positive/negative or neutral and based on it guide us if we should buy/ sell or hold the stock. Ensure the total article count is included.",
            agent = analyser
        )

        crew = Crew(
            agents=[collector, summerizer, analyser],
            tasks=[collect, summerize, analyse],
            process=Process.sequential,
            verbose=False
        )

        try:
            # 1) Fetch articles (API call)
            articles = _fetch_articles_apitube(inputStock)

            # 2) Fallback to demo data if the API fetch failed (only if articles is empty AND is a dict with error)
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
            
            # 3) Show charts immediately (reads articles.txt)
            visualize_sentiments()

            # 4) Read the content of the file to pass to the Agents (Unifying the data source)
            article_content = ""
            if os.path.exists("articles.txt"):
                with open("articles.txt", "r") as f:
                    article_content = f.read()
            else:
                st.error("articles.txt file is missing after fetch/fallback.")


            # 5) Run CrewAI analysis, passing the file content as input
            response = crew.kickoff(
                inputs={
                    "topic": inputStock,
                    "article_content": article_content # <--- PASSES FILE CONTENT TO TASK
                }
            )
            st.success("✅ Analysis complete!")
            st.write("Analyzing trends for:", inputStock)
            st.markdown(f"**Result:**\n{response}")

        except Exception as e:
            st.error(f"❌ Error during CrewAI processing: {e}")
