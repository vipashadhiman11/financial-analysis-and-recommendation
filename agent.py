from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from datetime import date,timedelta
from dotenv import load_dotenv
import os
import streamlit as st
import requests

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# GROQ_API_KEY="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"

st.set_page_config(page_title = "Market Trends Analyst", layout = "centered")
st.title("Your Financial Advisor")
st.write('Hello, I am your financial advisor. I will give you a complete analysis of your stock or organisation. I will also recomment you if you should Buy/ Sell or Hold the stock :sunglasses:')

def get_gainers(number):
    response_gainers = requests.get("https://financialmodelingprep.com/stable/biggest-gainers?apikey=iifNjIBLSqHJ0q8wO57yE87LhNZ762yf").json()
    print(response_gainers)
    count = 0
    gainers = []
    for response in response_gainers:
        if count<number:
            gainers.append({"name":response["name"],
                            "percentage":response["changesPercentage"]})
            count+=1
    return gainers
    
def get_losers(number):
    response_losers = requests.get("https://financialmodelingprep.com/stable/biggest-losers?apikey=iifNjIBLSqHJ0q8wO57yE87LhNZ762yf").json()
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

inputStock = st.text_input("Enter your Organisation:")



if st.button("Submit", type="primary"):
    llm = LLM(model = "groq/openai/gpt-oss-120b",
            temperature = 0.2,
            # max_completion_tokens = 256,
            top_p = 0.9
        )

     
    @tool("get_articles_APItube")
    def get_articles_APItube(entity: str) -> list[list]:
      """
      This function will take the entity as an input and returns the list of all articles collected with their sentiment.
    
      args:
        entity: name of any organisation or stock 
      """
      try:
        print("Running API")
        articles = []
        APITUBE_API_KEY="api_live_auBHrOWRNh2UGkBZaczSeeOM5GNDnHd3ZqJNFbTT3gHUvg"
        # today = date.today()
        # last_week_same_day = today - timedelta(weeks=1)
        url = "https://api.apitube.io/v1/news/everything?title="+entity+"&published_at.start=2025-10-20&published_at.end=2025-10-25&sort.order=desc&language.code=en&api_key="+APITUBE_API_KEY
        response = requests.get(url).json()
        count = 0
        if response["status"] == "ok":
          for result in response["results"]:
            count+=1
            article = {}
            article["article_body"] = result["body"]
            article["sentiment"] = result["sentiment"]["overall"]["score"]
            articles.append(article)
          while response["has_next_pages"]:
            if count<20:
              next_page_url = response["next_page"]
              next_page_response = requests.get(url).json()
              if response["status"] == "ok":
                for result in response["results"]:
                  count+=1
                  article = {}
                  article["article_body"] = result["body"]
                  article['sentiment'] = result["sentiment"]["overall"]["score"]
                  articles.append(article)
            else:
              break
        print(articles)
        with open("articles.txt", "w") as file:
          for article in articles:
            file.write(str(article)+"\n")
        print("articles"+articles)
        return articles
      except Exception as e:
          return {"error": f"Failed to read URL {e}"}

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
        goal = "Asks the user about the {topic} and collects the articles releated to that topic using tool.",
        backstory = "The {topic} will be an organisation of stock name. Don't take any other input except topic"
                    "fetch the articles.\n",
        tools=[get_articles_APItube],
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
            "2. collect all the news articles on the provided {topic}.\n"
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
        st.write("You entered: ", inputStock)
        st.write("Result:", response.raw)
        
    except Exception as e:
        st.write(f"An error occured: {e}")
