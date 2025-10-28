import os
import json
import requests
from datetime import date, timedelta
from typing import Dict, Any

# Data Visualization and Streamlit
import streamlit as st
import matplotlib.pyplot as plt

# CrewAI & LLM
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

# CRITICAL FIX: Import the ChatGroq model explicitly
# Note: You MUST have 'langchain-groq' installed in your environment.
from langchain_groq import ChatGroq

# --- Configuration & Setup ---

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# NOTE: Using a placeholder for APITUBE_API_KEY for the tool to work
APITUBE_API_KEY = os.environ.get("APITUBE_API_KEY", "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write('Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. I will also recommend you if you should **Buy/Sell or Hold** the stock :sunglasses:')

inputStock = st.text_input("Enter stock name or company name (e.g., Apple, TSLA):")

# Define the Groq model we will use
GROQ_MODEL = "mixtral-8x7b-32768" 

if st.button("Submit", type="primary"):
    if not GROQ_API_KEY:
        st.error("❌ **Error:** GROQ_API_KEY not found in your environment variables. Please check your `.env` file.")
        st.stop()
        
    if not inputStock:
        st.error("Please enter a stock or company name to analyze.")
        st.stop()
        
    # --- LLM Initialization (CRITICAL FIX) ---
    st.info(f"Connecting to Groq using model: **{GROQ_MODEL}**")
    
    try:
        # 1. Instantiate the Groq client directly using the explicit class.
        # This object is a direct LangChain runnable.
        groq_llm_instance = ChatGroq(
            temperature=0.2, 
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY # Explicitly pass key for robustness
        )
    except Exception as e:
        st.error(f"❌ **LLM Initialization Error:** Failed to create ChatGroq instance. Ensure `langchain-groq` is installed and the model name is correct. Error: {e}")
        st.stop()
    
    # --- Tools Definition ---
    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> str:
        """
        Fetches news articles, extracts sentiment labels, and returns a string 
        with the article summaries and the final sentiment counts as JSON.
        
        args:
            entity: name of any organisation or stock 
        """
        
        # Calculate start and end dates dynamically (e.g., last 5 days)
        end_date = date.today()
        start_date = end_date - timedelta(days=5)
        
        url = (
            f"https://api.apitube.io/v1/news/everything?title={entity}"
            f"&published_at.start={start_date}&published_at.end={end_date}"
            f"&sort.order=desc&language.code=en&api_key={APITUBE_API_KEY}"
        )
        
        try:
            response = requests.get(url).json()
            articles = []
            pos, neg, neu = 0, 0, 0
            
            if response.get("status") == "ok":
                for result in response.get("results", []):
                    # Limit articles to a manageable number for the LLM
                    if len(articles) >= 15: 
                        break

                    score = result["sentiment"]["overall"]["score"]
                    if score >= 0.3:
                        label, pos = "Positive", pos + 1
                    elif score <= -0.3:
                        label, neg = "Negative", neg + 1
                    else:
                        label, neu = "Neutral", neu + 1
                            
                    title = result.get('title', 'No Title')
                    body_snippet = result.get('body', 'No Body')[:200]
                    
                    article_summary = f"Sentiment: {label}. Title: {title}. Body: {body_snippet}..."
                    articles.append(article_summary)
                
                # Generate the JSON string for the next agent
                sentiment_counts_json = json.dumps({
                    "Positive": pos,
                    "Negative": neg,
                    "Neutral": neu
                })
                
                final_output = (
                    f"Total articles collected: {len(articles)}\n"
                    f"Summary of articles (max 15):\n"
                    f"{'---'.join(articles)}\n\n"
                    f"SENTIMENT_COUNTS_JSON: {sentiment_counts_json}"
                )
                return final_output
            else:
                return f"Failed to fetch articles. API returned status: {response.get('status', 'N/A')} with message: {response.get('message', 'No message.')}"
                
        except requests.exceptions.RequestException as e:
            return f"Failed to fetch articles due to connection error: {e}"
        except Exception as e:
            return f"Failed to fetch articles due to an unexpected exception: {e}"

    # --- Agents Definition (LLM assigned later to bypass validation) ---

    collector = Agent(
        role="Articles collector",
        goal="Collect news articles and extract sentiment data for {topic} using the available tool.",
        backstory="Your task is to use the 'get_articles_APItube' tool to fetch news. You MUST ensure the final output contains the sentiment counts in the exact 'SENTIMENT_COUNTS_JSON' format for the next agent.",
        tools=[get_articles_APItube],
        llm=None, # Pass None initially
        allow_delegation=False,
        verbose=False
    )
    # Manual LLM assignment
    collector.llm = groq_llm_instance
    
    summerizer = Agent(
        role="Article summerizer",
        goal="Summarize the key findings from the collected articles and ensure sentiment counts are preserved.",
        backstory="You are summarizing all article findings into one report, focusing on market trends. You MUST copy the 'SENTIMENT_COUNTS_JSON' from the input to the end of your output.",
        llm=None, # Pass None initially
        allow_delegation=False,
        verbose=False
    )
    # Manual LLM assignment
    summerizer.llm = groq_llm_instance
    
    analyser = Agent(
        role="Financial Analyst",
        goal="Provide a clear recommendation (Buy/Sell/Hold) based on the overall market sentiment for {topic}.",
        backstory=(
            "You are a Senior Financial Analyst. You must use the sentiment summary and counts from the previous tasks "
            "to determine the overall market sentiment and provide a clear, justified **Buy, Sell, or Hold** recommendation. "
            "Your final output MUST also contain the `SENTIMENT_COUNTS_JSON` at the end."
        ),
        llm=None, # Pass None initially
        allow_delegation=False,
        verbose=False
    )
    # Manual LLM assignment
    analyser.llm = groq_llm_instance
    
    # --- Tasks Definition ---
    
    # Expected output is crucial for forcing the format transfer between agents
    json_output_format = "`SENTIMENT_COUNTS_JSON: {\"Positive\": <count>, \"Negative\": <count>, \"Neutral\": <count>}`"
    
    collect = Task(
        description=(
            "1. Use the tool 'get_articles_APItube' to collect all the news articles on the provided {topic}.\n"
            "2. Ensure the output is structured to pass the article summaries and the sentiment counts in the required JSON format."
        ),
        expected_output=f"A summary of articles and the final sentiment counts in the exact JSON format: {json_output_format}",
        agent=collector
    )
    
    summerize = Task(
        description=(
            "1. Summarize the articles collected from the previous task into a maximum 500-word report.\n"
            "2. **Crucially**, copy the entire string starting with `SENTIMENT_COUNTS_JSON:` from the previous task's result to the end of your summary."
        ),
        expected_output=f"A concise summary of the articles followed by the exact sentiment counts in the JSON format: {json_output_format}",
        agent=summerizer
    )
    
    analyse = Task(
        description=(
            "1. Use the summary and sentiment counts to perform a detailed financial analysis on {topic}.\n"
            "2. Identify the overall market sentiment (Positive, Negative, or Neutral) based on the counts.\n"
            "3. Provide a clear recommendation: **Buy, Sell, or Hold**.\n"
            "4. Your final output MUST end by repeating the entire string starting with `SENTIMENT_COUNTS_JSON:` from the previous task."
        ),
        expected_output=f"Provide a detailed analysis, a Buy/Sell/Hold recommendation, and the sentiment counts in the required JSON format at the end: {json_output_format}",
        agent=analyser
    )
    
    # --- Crew Setup & Execution ---
    
    crew = Crew(
        agents=[collector, summerizer, analyser],
        tasks=[collect, summerize, analyse],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        with st.spinner(f"🚀 Analysing trends for: **{inputStock}** using Groq's {GROQ_MODEL}..."):
            full_result = crew.kickoff(inputs={"topic": inputStock})
        
        st.subheader(f"📈 Financial Analysis for **{inputStock}**")
        st.divider()
        
        # --- Streamlit Output & Pie Chart Generation ---
        json_start_tag = "SENTIMENT_COUNTS_JSON:"
        
        if json_start_tag in full_result:
            parts = full_result.split(json_start_tag, 1)
            analysis_text = parts[0].strip()
            json_str = parts[-1].strip()
            
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            
            try:
                sentiment_counts = json.loads(json_str)
                
                st.subheader("📝 Analyst Report")
                st.markdown(analysis_text)
                st.divider()
                
                st.subheader("📊 Sentiment Distribution of News Articles")
                
                labels = list(sentiment_counts.keys())
                sizes = list(sentiment_counts.values())
                
                if sum(sizes) > 0:
                    color_map = {"Positive": '#4CAF50', "Negative": '#F44336', "Neutral": '#FFEB3B'}
                    colors = [color_map.get(label, '#CCCCCC') for label in labels]
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
                           wedgeprops={'edgecolor': 'black'})
                    ax.axis('equal')
                    
                    st.pyplot(fig)
                else:
                    st.warning("No articles were found to generate the sentiment chart.")
                
            except json.JSONDecodeError as e:
                st.error(f"❌ **Error:** Failed to parse sentiment counts for chart. The LLM output might be malformed JSON. Error: {e}")
                st.caption(f"Raw JSON string received: `{json_str}`") 
            
        else:
            st.warning("⚠️ **Warning:** The final analysis did not contain the required sentiment counts for charting. Please check the `verbose` output above for intermediate errors.")
            st.write("Result (Raw):", full_result)
            
    except Exception as e:
        st.error(f"❌ **CrewAI Kickoff Error:** An error occurred during the analysis process. Check your API keys and configuration. Error: {e}")
