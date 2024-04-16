from llama_index.indices.managed.vectara import VectaraIndex
from dotenv import load_dotenv
import os
from PIL import Image
import requests
from Bio import Entrez
import together
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage, MessageRole
from pprint import pprint
from llama_index.indices.managed.vectara import query
from llama_index.core.schema import Document
import io
import datetime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import streamlit as st
from googleapiclient.discovery import build
from typing import List, Optional

load_dotenv()

os.environ["VECTARA_INDEX_API_KEY"] = os.getenv("VECTARA_INDEX_API_KEY", "zwt_ni_bLu6MRQXzWKPIU__Uubvy_0Xz_FEr-2sfUg")
os.environ["VECTARA_QUERY_API_KEY"] = os.getenv("VECTARA_QUERY_API_KEY", "zwt_ni_bLu6MRQXzWKPIU__Uubvy_0Xz_FEr-2sfUg")
os.environ["VECTARA_API_KEY"] = os.getenv("VECTARA_API_KEY", "zut_ni_bLoa0I3AeNSjxeZ-UfECnm_9Xv5d4RVBAqw")
os.environ["VECTARA_CORPUS_ID"] = os.getenv("VECTARA_CORPUS_ID", "2")
os.environ["VECTARA_CUSTOMER_ID"] = os.getenv("VECTARA_CUSTOMER_ID", "2653936430")
os.environ["TOGETHER_API"] = os.getenv("TOGETHER_API", "7e6c200b7b36924bc1b4a5973859a20d2efa7180e9b5c977301173a6c099136b")
os.environ["GOOGLE_SEARCH_API_KEY"] = os.getenv("GOOGLE_SEARCH_API_KEY", "AIzaSyBnQwS5kPZGKuWj6sH1aBx5F5bZq0Q5jJk")

endpoint = 'https://api.together.xyz/inference'

index = VectaraIndex()
retriever = index.as_retriever(similarity_top_k=7)

# Load the hallucination evaluation model
model_name = "vectara/hallucination_evaluation_model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def search_pubmed(query: str) -> Optional[List[str]]:
    """
    Searches PubMed for a given query and returns a list of formatted results 
    (or None if no results are found).
    """
    Entrez.email = "jayashbhardwaj3@gmail.com"  # Replace with your email

    try:
        # Use ESearch to retrieve UIDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
        record = Entrez.read(handle)
        id_list = record["IdList"]

        if not id_list:  # Check for empty results
            return None

        # Fetch details for each UID using EFetch
        handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
        articles = Entrez.read(handle)

        results = []
        for article in articles['PubmedArticle']:
            # Safely access article data, handling potential KeyError
            try:
                medline_citation = article['MedlineCitation']
                article_data = medline_citation['Article']
                title = article_data['ArticleTitle']
                abstract = article_data.get('Abstract', {}).get('AbstractText', [""])[0]

                # Create result string
                result = f"**Title:** {title}\n**Abstract:** {abstract}\n"
                result += f"**Link:** https://pubmed.ncbi.nlm.nih.gov/{medline_citation['PMID']}\n\n"
                results.append(result)
            except KeyError:
                print(f"Error parsing article: {article}")  # Log error for debugging

        return results

    except IOError as e:
        print(f"Error accessing PubMed: {e}")
        return None


def chat_with_pubmed(article_text, article_link):
    """
    Engages in a chat-like interaction with a PubMed article using TogetherLLM.
    """
    llm = TogetherLLM(model="QWEN/QWEN1.5-14B-CHAT", api_key=os.environ['TOGETHER_API'])
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant summarizing and answering questions about the following medical research article: " + article_link),
        ChatMessage(role=MessageRole.USER, content=article_text)
    ]
    response = llm.chat(messages)
    if response:
        return str(response)
    else:
        return "I'm sorry, I couldn't generate a summary for this article."

def search_web(query: str, num_results: int = 3) -> Optional[List[str]]:
    """
    Searches the web using the Google Search API and returns a list of formatted results
    (or None if no results are found).
    """
    try:
        # Set up the Google Search API client
        service = build("customsearch", "v1", developerKey=os.environ["GOOGLE_SEARCH_API_KEY"])

        # Execute the search request
        res = service.cse().list(q=query, cx="877170db56f5c4629", num=num_results).execute()

        if "items" not in res:
            return None

        results = []
        for item in res["items"]:
            title = item["title"]
            link = item["link"]
            snippet = item["snippet"]
            result = f"**Title:** {title}\n**Link:** {link}\n**Snippet:** {snippet}\n\n"
            results.append(result)

        return results

    except Exception as e:
        print(f"Error performing web search: {e}")
        return None

def medmind_chatbot(user_input, chat_history=None):
    if chat_history is None:
        chat_history = []

    # 1. Query Vectara for medical knowledge base context
    query_str = user_input
    response = index.as_query_engine().query(query_str)
    vectara_response = f"**MedMind Vectara Knowledge Base Response:**\n{response.response}"

    # 2. Search PubMed
    pubmed_results = search_pubmed(user_input)

    # 3. Process PubMed results for chat interaction
    pubmed_response = "**PubMed Articles (Chat & Summarize):**\n\n"
    if pubmed_results:
        for article_text in pubmed_results:
            title, abstract, link = article_text.split("\n")[:3]
            chat_summary = chat_with_pubmed(abstract, link)
            pubmed_response += f"**{title}**\n{chat_summary}\n{link}\n\n"
    else:
        pubmed_response += "No relevant PubMed articles found.\n\n"

    # 4. Search the web
    web_results = search_web(user_input)

    # 5. Process web search results
    web_response = "**Web Search Results:**\n\n"
    if web_results:
        web_response += "\n".join(web_results)
    else:
        web_response += "No relevant web search results found.\n\n"

    # 6. Combine all responses
    response_text = vectara_response + "\n\n" + pubmed_response + "\n\n" + web_response

    # Hallucination Evaluation
    def vectara_hallucination_evaluation_model(text):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        hallucination_probability = outputs.logits[0][0].item()  
        return hallucination_probability

    hallucination_score = vectara_hallucination_evaluation_model(response_text)

    # Response Filtering/Modification
    HIGH_HALLUCINATION_THRESHOLD = 0.8  
    if hallucination_score > HIGH_HALLUCINATION_THRESHOLD:
        response_text = "I'm still under development and learning. I cannot confidently answer this question yet."  

    chat_history.append((user_input, response_text))
    return response_text, chat_history

def show_info_popup():
    with st.expander("How to use MedMind"):
        st.write("""
        **MedMind is an AI-powered chatbot designed to assist with medical information.**

        **Capabilities:**

        *   **Answers general medical questions:** MedMind utilizes a curated medical knowledge base to provide answers to a wide range of health-related inquiries.
        *   **Summarizes relevant research articles from PubMed:** The chatbot can retrieve and summarize research articles from the PubMed database, making complex scientific information more accessible.
        *   **Provides insights from a curated medical knowledge base:** Beyond simple answers, MedMind offers additional insights and context from its knowledge base to enhance understanding. 
        *   **Perform safe web searches related to your query:** The chatbot can perform web searches using the Google Search API, ensuring the safety and relevance of the results.

        **Limitations:**

        *   **Not a substitute for professional medical advice:** MedMind is not intended to replace professional medical diagnosis and treatment. Always consult a qualified healthcare provider for personalized medical advice.
        *   **General knowledge and educational purposes:** The information provided by MedMind is for general knowledge and educational purposes only and may not be exhaustive or specific to individual situations.
        *   **Under development:** MedMind is still under development and may occasionally provide inaccurate or incomplete information. It's important to critically evaluate responses and cross-reference with reliable sources.
        *   **Hallucination potential:** While MedMind employs a hallucination evaluation model to minimize the risk of generating fabricated information, there remains a possibility of encountering inaccurate responses, especially for complex or niche queries.

        **How to use:**

        1.  **Type your medical question in the text box.**
        2.  **MedMind will provide a comprehensive response combining information from various sources.** This may include insights from its knowledge base, summaries of relevant research articles, and safe web search results.
        3.  **You can continue the conversation by asking follow-up questions or providing additional context.** This helps MedMind refine its search and offer more tailored information.
        4.  **in case the Medmind doesn't show the output please check your internet connection or rerun the same command**
        """)
# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Define function to display chat history
def display_chat_history():
    for user_msg, bot_msg in st.session_state.chat_history:
        st.write(f"**You:** {user_msg}")
        st.write(f"**MedMind:** {bot_msg}")

# Define function to start a new chat
def start_new_chat():
    st.session_state.chat_history = []

# Define main function
def main():
    st.set_page_config(page_title="MedMind Chatbot", layout="wide")
    st.markdown(
        """
        <style>
        .css-18e3th9 {
            padding-top: 2rem;
            padding-right: 1rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        body {
            background-color: #F0FDF4;
            color: #333333;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #388E3C;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("MedMind Chatbot")
    st.write("Ask your medical questions and get reliable information!")

    # Example questions
    example_questions = [
        "What are the symptoms of COVID-19?",
        "How can I manage my diabetes?",
        "What are the potential side effects of ibuprofen?",
        "What lifestyle changes can help prevent heart disease?"
    ]
    st.sidebar.header("Example Questions")
    for question in example_questions:
        st.sidebar.write(question)

    # Output container
    output_container = st.container()

    # User input
    input_container = st.container()
    with input_container:
        user_input = st.text_input("You: ", key="input_placeholder", placeholder="Type your medical question here...")

        # Start new chat button
        new_chat_button = st.button("Start New Chat")
        if new_chat_button:
            start_new_chat()

    if user_input:
        # Get chatbot response
        response, st.session_state.chat_history = medmind_chatbot(user_input, st.session_state.chat_history)

        with output_container:
            display_chat_history()

    # Show info popup
    show_info_popup()

if __name__ == "__main__":
    main()