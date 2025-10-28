import os
import json
import requests
from datetime import date, timedelta

# Data Visualization and Streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# CrewAI & LLM
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from dotenv import load_dotenv

# --- Configuration & Setup ---

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title = "Market Trends Analyst", layout = "centered")
st.title("Your Financial Advisor")
st.write('Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. I will also recommend you if you should Buy/Sell or Hold the stock :sunglasses:')

inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    # Note: When using CrewAI with Groq, you might need a custom LLM adapter 
    # if using their specific model names. For this example, we assume 'llm'
    # is correctly set up to use the API key from the environment.
    # If using Groq via Litellm (which crewai uses), a model like 'mixtral-8x7b-instruct-v0.1' 
    # or 'llama2-70b-4096' is typical, but we'll use your placeholder.
    llm = LLM(model = "groq/openai/gpt-oss-120b",
              temperature = 0.2,
              top_p = 0.9)
    
    # --- Tools Definition ---
    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> str:
      """
      This function takes the entity as an input and returns a string containing 
      a list of all articles collected along with their sentiment labels and the overall sentiment counts.
      
      args:
        entity: name of any organisation or stock 
      """
      try:
        articles = []
        # NOTE: Using a placeholder API key and dates from your original code.
        APITUBE_API_KEY="api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
        url = "https://api.apitube.io/v1/news/everything?title="+entity+"&published_at.start=2025-10-20&published_at.end=2025-10-25&sort.order=desc&language.code=en&api_key="+APITUBE_API_KEY
        response = requests.get(url).json()
        
        pos = 0
        neg = 0
        neu = 0
        
        if response["status"] == "ok":
            for result in response["results"]:
                # Limit articles to a manageable number for the LLM
                if len(articles) >= 20: 
                    break

                # Extract score and determine label
                score = result["sentiment"]["overall"]["score"]
                if score >= 0.3:
                    label = "Positive"
                    pos += 1
                elif score <= -0.3:
                    label = "Negative"
                    neg += 1
                else:
                    label = "Neutral"
                    neu += 1
                    
                article_summary = f"Sentiment: {label}. Body: {result['body'][:200]}..."
                articles.append(article_summary)
            
            # --- CRITICAL: Generate the JSON string for the next agent ---
            sentiment_counts_json = json.dumps({
                "Positive": pos,
                "Negative": neg,
                "Neutral": neu
            })
            
            # This string is the output the LLM sees
            final_output = (
                f"Total articles collected: {len(articles)}\n"
                f"Summary of articles:\n"
                f"{'---'.join(articles)}\n\n"
                f"SENTIMENT_COUNTS_JSON: {sentiment_counts_json}"
            )
            return final_output
        else:
             return "Failed to fetch articles. API returned non-ok status."
             
      except Exception as e:
        return f"Failed to fetch articles due to an exception: {e}"

    # --- Agents Definition ---

    collector = Agent(
        role = "Articles collector",
        goal = "Collects news articles and extracts sentiment data for {topic}.",
        backstory = "Your task is to use the 'get_articles_APItube' tool to fetch news. You MUST ensure the final output contains the sentiment counts in the exact 'SENTIMENT_COUNTS_JSON' format.",
        tools = [get_articles_APItube],
        llm = llm,
        allow_delegation = False,
        verbose = False
    )
    
    summerizer = Agent(
        role = "Article summerizer",
        goal = "Summarize the key findings from the collected articles.",
        backstory = "You are summarizing all article findings into one report with utmost precision, focusing on market trends.",
        llm = llm,
        allow_delegation = False,
        verbose = False
    )
    
    analyser = Agent(
        role = "Financial Analyst",
        goal = "Guide the user to either Buy/Sell or Hold the stock of the organisation.",
        backstory = (
            "You are working on identifying the latest trends about the topic: {topic}. "
            "You must use the sentiment counts provided by the previous tasks to determine the overall market sentiment and provide a recommendation."
        ),
        llm = llm,
        # IMPORTANT: Allow delegation is False and no tools are listed, preventing the Groq tool error.
        allow_delegation = False,
        verbose = False
    )
    
    # --- Tasks Definition ---
    
    collect = Task(
        description = (
            "1. Use the tool 'get_articles_APItube' to collect all the news articles on the provided {topic}.\n"
            "2. Ensure the output is structured to easily pass the article summaries and sentiment counts to the next agent."
        ),
        expected_output = (
            "A summary of articles and the final sentiment counts in the exact JSON format: "
            "`SENTIMENT_COUNTS_JSON: {\"Positive\": <count>, \"Negative\": <count>, \"Neutral\": <count>}`"
        ),
        agent = collector
    )
    
    summerize = Task(
        description = (
            "1. Summarize the articles collected from the previous task into a maximum 500-word report.\n"
            "2. **Crucially**, include the `SENTIMENT_COUNTS_JSON` from the previous task at the end of your summary to ensure it is passed to the final analyser."
        ),
        expected_output = (
            "A concise summary of the articles followed by the exact sentiment counts in the JSON format: "
            "`SENTIMENT_COUNTS_JSON: {\"Positive\": <count>, \"Negative\": <count>, \"Neutral\": <count>}`"
        ),
        agent = summerizer
    )
    
    analyse = Task(
        description = (
            "1. Use the summary and the sentiment counts from the previous task's output to perform a detailed financial analysis on {topic}.\n"
            "2. Identify the overall market sentiment (Positive, Negative, or Neutral) based on the counts.\n"
            "3. Based on the overall sentiment and market trends, provide a clear recommendation: **Buy, Sell, or Hold**.\n"
            "4. Your final output MUST end with the sentiment counts extracted from the previous task in the required JSON format: "
            "`SENTIMENT_COUNTS_JSON: {\"Positive\": <count>, \"Negative\": <count>, \"Neutral\": <count>}`"
        ),
        expected_output = "Provide a detailed analysis, a Buy/Sell/Hold recommendation, and the sentiment counts in the required JSON format at the end.",
        agent = analyser
    )
    
    # --- Crew Setup & Execution ---
    
    crew = Crew(
        agents = [collector, summerizer, analyser],
        tasks = [collect, summerize, analyse],
        process=Process.sequential,
        verbose = True # Set to True for better debugging in Streamlit
    )
    
    try:
        with st.spinner(f"Analysing trends for: {inputStock}..."):
            full_result = crew.kickoff(inputs = {"topic": inputStock})
        
        st.subheader(f"📈 Financial Analysis for **{inputStock}**")
        
        # --- Streamlit Output & Pie Chart Generation ---
        json_start_tag = "SENTIMENT_COUNTS_JSON:"
        
        if json_start_tag in full_result:
            # 1. Extract the Analysis Text and JSON string
            parts = full_result.split(json_start_tag)
            analysis_text = parts[0].strip()
            json_str = parts[-1].strip()
            
            # Clean up the JSON string (LLMs can sometimes add extra formatting)
            json_str = json_str.replace('```json', '').replace('```', '').strip()
                
            try:
                sentiment_counts = json.loads(json_str)
                
                # 2. Display the Analysis Text first
                st.subheader("📝 Analyst Report")
                st.markdown(analysis_text)
                st.markdown("---")
                
                # 3. Display the Pie Chart
                st.subheader("📊 Sentiment Distribution of News Articles")
                
                labels = list(sentiment_counts.keys())
                sizes = list(sentiment_counts.values())
                
                # Check if there is data to plot
                if sum(sizes) > 0:
                    # Define colors for better representation (Positive=Green, Negative=Red, Neutral=Yellow)
                    colors = ['#4CAF50', '#F44336', '#FFEB3B']
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    # Create the pie chart
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
                           wedgeprops={'edgecolor': 'black'})
                    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                    
                    st.pyplot(fig)
                else:
                    st.warning("No articles were found to generate the sentiment chart.")
                
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse sentiment counts for chart. Error: {e}")
                st.write("Result (Raw):", full_result) # Display raw result if parsing fails
        else:
            st.warning("The final analysis did not contain the required sentiment counts for charting.")
            st.write("Result (Raw):", full_result)
        
    except Exception as e:
        st.error(f"An error occurred during the CrewAI kickoff: {e}")
