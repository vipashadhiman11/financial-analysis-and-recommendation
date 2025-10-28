from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date,timedelta
from dotenv import load_dotenv
import os
import streamlit as st
import requests # <--- **IMPORT REQUESTS**
import json # <--- **IMPORT JSON**
import pandas as pd # <--- **IMPORT PANDAS for charting**
import matplotlib.pyplot as plt # <--- **IMPORT MATPLOTLIB for pie chart**

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title = "Market Trends Analyst", layout = "centered")
st.title("Your Financial Advisor")
st.write('Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. I will also recomment you if you should Buy/ Sell or Hold the stock :sunglasses:')

inputStock = st.text_input("Enter stock name or company name:")

if st.button("Submit", type="primary"):
    llm = LLM(model = "groq/openai/gpt-oss-120b",
                temperature = 0.2,
                top_p = 0.9
        )
    
    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> list[dict]:
      """
      This function will take the entity as an input and returns the list of all articles collected with their sentiment.
      
      args:
        entity: name of any organisation or stock 
      """
      try:
        articles = []
        APITUBE_API_KEY="api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
        # Today's date is 2025-10-28, keeping the date range as in the original code for consistency
        url = "https://api.apitube.io/v1/news/everything?title="+entity+"&published_at.start=2025-10-20&published_at.end=2025-10-25&sort.order=desc&language.code=en&api_key="+APITUBE_API_KEY
        response = requests.get(url).json()
        
        # --- Simplified Article Collection Loop ---
        if response["status"] == "ok":
            for result in response["results"]:
                article = {}
                article["article_body"] = result["body"]
                # Convert score (-1 to 1) to a simple label for the LLM's use
                score = result["sentiment"]["overall"]["score"]
                if score >= 0.3:
                    label = "Positive"
                elif score <= -0.3:
                    label = "Negative"
                else:
                    label = "Neutral"
                    
                article["sentiment_label"] = label 
                articles.append(article)
                if len(articles) >= 20: # Limit to 20 articles
                    break
            
            # Write a summary of sentiment counts to a file for the analyser agent to reference
            pos = sum(1 for a in articles if a["sentiment_label"] == "Positive")
            neg = sum(1 for a in articles if a["sentiment_label"] == "Negative")
            neu = sum(1 for a in articles if a["sentiment_label"] == "Neutral")
            sentiment_summary = f'{{"positive": {pos}, "negative": {neg}, "neutral": {neu}}}'
            
            with open("sentiment_summary.txt", "w") as file:
                file.write(sentiment_summary)
            
            return articles
        else:
             return {"error": "API returned non-ok status."}
             
      except Exception as e:
        return {"error": f"Failed to fetch articles: {e}"}

    # Removed the redundant sentiment_analysis tool

    collector = Agent(
        role = "Articles collector",
        goal = "Collects articles and their sentiments for {topic} using the available tool.",
        backstory = "The {topic} will be an organisation or stock name. Use the tool 'get_articles_APItube' to fetch articles and ensure the sentiment counts (positive, negative, neutral) are available for the next agent.",
        tools = [get_articles_APItube],
        llm = llm,
        allow_delegation = False,
        verbose = False
    )
    
    summerizer = Agent(
        role = "Article summerizer",
        goal = "Summerize the articles collected by collector to fetch the crux of it",
        backstory = "You are summerizing all the articles into one with utmost precision and keeping in mind the trends we are getting from the articles.",
        llm = llm,
        allow_delegation = False,
        verbose = False
    )
    
    analyser = Agent(
        role = "Financial Analyst",
        goal = "You will guide user to either Buy/Sell or Hold the stock of the organisation.",
        backstory = (
            "You are working on identifying latest trends about the topic: {topic}."
            "You will observe the sentiment of all articles. You MUST look up the sentiment_summary.txt file to get the final positive, negative, and neutral article counts and include them in your final output in a **JSON** format."
            "Based on the sentiment, you will predict the overall market trend and recommend to buy/sell or hold the stock."
            "Your target is to maximise user profit."
        ),
        llm = llm,
        allow_delegation = False,
        verbose = False
    )
    
    collect = Task(
        description = (
            "1. Collect all the news articles on the provided {topic} using tool 'get_articles_APItube'.\n"
            "2. Prioritize the latest trends and news on the {topic}.\n"
        ),
        expected_output = "A list of articles and their respective Positive, Negative, or Neutral sentiment labels.",
        agent = collector
    )
    
    summerize = Task(
        description = (
            "1. Summarize the articles you collected from collector into a concise report (maximum 500 words).\n"
            "2. Focus on the latest trends and news on the {topic}.\n"
        ),
        expected_output = "A concise summary of the key findings from the collected articles.",
        agent = summerizer
    )
    
    analyse = Task(
        description = (
            "1. Use the collected information and summary to create a detailed financial analysis on {topic}.\n"
            "2. Based on the overall market sentiment, guide the user to either **Buy/Sell or Hold** the stock.\n"
            "3. After your analysis and recommendation, you **MUST** include the final sentiment counts in the following JSON format at the very end of your output, replacing the placeholders with the actual numbers: "
            "`SENTIMENT_COUNTS_JSON: {\"Positive\": <count>, \"Negative\": <count>, \"Neutral\": <count>}`"
        ),
        expected_output = "A detailed analysis, a Buy/Sell/Hold recommendation, and the sentiment counts in the required JSON format at the end.",
        agent = analyser
    )
    
    crew = Crew(
        agents = [collector, summerizer, analyser],
        tasks = [collect, summerize, analyse],
        process=Process.sequential,
        verbose = True # Set to True to debug in Streamlit if needed
    )
    
    try:
        with st.spinner(f"Analysing trends for: {inputStock}..."):
            response = crew.kickoff(inputs = {"topic": inputStock})
        
        st.subheader(f"📈 Financial Analysis for **{inputStock}**")
        
        # --- Extract and Display Analysis ---
        full_result = response
        
        # 1. Extract the JSON from the end of the response
        json_start_tag = "SENTIMENT_COUNTS_JSON: "
        if json_start_tag in full_result:
            json_str = full_result.split(json_start_tag)[-1].strip()
            # Clean up the JSON string (LLMs can sometimes add extra backticks or characters)
            if json_str.startswith('```json'):
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                
            try:
                sentiment_counts = json.loads(json_str)
                
                # 2. Display the Pie Chart
                st.markdown("---")
                st.subheader("📊 Sentiment Distribution of News Articles")
                
                labels = sentiment_counts.keys()
                sizes = sentiment_counts.values()
                
                # Define colors for better representation
                colors = ['#4CAF50', '#F44336', '#FFEB3B'] # Green, Red, Yellow
                
                # Create a DataFrame for Matplotlib
                df = pd.DataFrame(sizes, index=labels, columns=['Count'])
                
                fig, ax = plt.subplots()
                # Create the pie chart
                ax.pie(df['Count'], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
                       wedgeprops={'edgecolor': 'black'})
                ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                
                st.pyplot(fig)
                
                # 3. Display the main analysis text (without the final JSON tag)
                analysis_text = full_result.split(json_start_tag)[0].strip()
                st.markdown("---")
                st.subheader("📝 Analyst Report")
                st.write(analysis_text)
                
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse sentiment counts for chart. Error: {e}")
                st.write("Result (Raw):", full_result) # Display raw result if parsing fails
        else:
            st.warning("Could not find the sentiment counts in the expected format. Displaying raw result.")
            st.write("Result (Raw):", full_result)
        
    except Exception as e:
        st.error(f"An error occurred during the CrewAI kickoff: {e}")
