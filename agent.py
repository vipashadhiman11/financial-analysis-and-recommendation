import os
import json
import requests
from datetime import date, timedelta
from typing import Dict, Any

# Data Visualization and Streamlit
import streamlit as st
import matplotlib.pyplot as plt

# CrewAI & LLM
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from dotenv import load_dotenv

# --- Configuration & Setup ---

load_dotenv()
# NOTE: GROQ_API_KEY is read automatically by the LLM adapter when setting the model.
# os.environ.get("GROQ_API_KEY") is checked implicitly.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Market Trends Analyst", layout="centered")
st.title("Your Financial Advisor")
st.write('Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. I will also recommend you if you should **Buy/Sell or Hold** the stock :sunglasses:')

inputStock = st.text_input("Enter stock name or company name (e.g., Apple, TSLA):")

# Define the Groq model we will use (use a valid, fast Groq model)
GROQ_MODEL = "mixtral-8x7b-32768" 
# ...
llm = LLM(
    model=GROQ_MODEL,
    temperature=0.2,
    top_p=0.9
)

if st.button("Submit", type="primary"):
    if not GROQ_API_KEY:
        st.error("❌ **Error:** GROQ_API_KEY not found in your environment variables. Please check your `.env` file.")
        st.stop()
        
    if not inputStock:
        st.error("Please enter a stock or company name to analyze.")
        st.stop()
        
    # --- LLM Initialization (FIXED) ---
    # Use the Litellm format for Groq
    llm = LLM(
        model=GROQ_MODEL, 
        temperature=0.2, 
        top_p=0.9
    )
    
    # --- Tools Definition ---
    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> str:
        """
        Fetches news articles, extracts sentiment labels, and returns a string 
        with the article summaries and the final sentiment counts as JSON.
        
        args:
            entity: name of any organisation or stock 
        """
        # NOTE: Using placeholder API key and dates. 
        # For production, consider using os.environ for APITUBE_API_KEY as well.
        APITUBE_API_KEY = os.environ.get("APITUBE_API_KEY", "api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg")
        
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
                            
                    # Ensure title and body are available
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
                
                # This string is the output the LLM sees
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

    # --- Agents Definition ---

    collector = Agent(
        role="Articles collector",
        goal="Collect news articles and extract sentiment data for {topic} using the available tool.",
        backstory="Your task is to use the 'get_articles_APItube' tool to fetch news. You MUST ensure the final output contains the sentiment counts in the exact 'SENTIMENT_COUNTS_JSON' format for the next agent.",
        tools=[get_articles_APITube],
        llm=llm,
        allow_delegation=False,
        verbose=False
    )
    
    summerizer = Agent(
        role="Article summerizer",
        goal="Summarize the key findings from the collected articles and ensure sentiment counts are preserved.",
        backstory="You are summarizing all article findings into one report, focusing on market trends. You MUST copy the 'SENTIMENT_COUNTS_JSON' from the input to the end of your output.",
        llm=llm,
        allow_delegation=False,
        verbose=False
    )
    
    analyser = Agent(
        role="Financial Analyst",
        goal="Provide a clear recommendation (Buy/Sell/Hold) based on the overall market sentiment for {topic}.",
        backstory=(
            "You are a Senior Financial Analyst. You must use the sentiment summary and counts from the previous tasks "
            "to determine the overall market sentiment and provide a clear, justified **Buy, Sell, or Hold** recommendation. "
            "Your final output MUST also contain the `SENTIMENT_COUNTS_JSON` at the end."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False
    )
    
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
        verbose=True # Keep True for Streamlit visibility
    )
    
    try:
        with st.spinner(f"🚀 Analysing trends for: **{inputStock}** using Groq's {GROQ_MODEL}..."):
            full_result = crew.kickoff(inputs={"topic": inputStock})
        
        st.subheader(f"📈 Financial Analysis for **{inputStock}**")
        st.divider()
        
        # --- Streamlit Output & Pie Chart Generation (CLEANED UP) ---
        json_start_tag = "SENTIMENT_COUNTS_JSON:"
        
        if json_start_tag in full_result:
            # 1. Extract the Analysis Text and JSON string
            parts = full_result.split(json_start_tag, 1) # Split only once
            analysis_text = parts[0].strip()
            json_str = parts[-1].strip()
            
            # Clean up the JSON string (LLMs can sometimes add extra formatting like triple backticks)
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            
            try:
                sentiment_counts = json.loads(json_str)
                
                # 2. Display the Analysis Text first
                st.subheader("📝 Analyst Report")
                st.markdown(analysis_text)
                st.divider()
                
                # 3. Display the Pie Chart
                st.subheader("📊 Sentiment Distribution of News Articles")
                
                labels = list(sentiment_counts.keys())
                sizes = list(sentiment_counts.values())
                
                # Check if there is data to plot
                if sum(sizes) > 0:
                    # Consistent colors for Positive, Negative, Neutral
                    color_map = {"Positive": '#4CAF50', "Negative": '#F44336', "Neutral": '#FFEB3B'}
                    colors = [color_map.get(label, '#CCCCCC') for label in labels] # Use a default grey for safety
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
                           wedgeprops={'edgecolor': 'black'})
                    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                    
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
