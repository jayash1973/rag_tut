from llama_index.indices.managed.vectara import VectaraIndex
from dotenv import load_dotenv
import os
import gradio as gr
import random
import requests
from Bio import Entrez
import together
from pprint import pprint
from llama_index.indices.managed.vectara import query
from llama_index.core.schema import Document
from PIL import Image
import io
import datetime

load_dotenv()

os.environ["VECTARA_INDEX_API_KEY"] = os.getenv("VECTARA_INDEX_API_KEY", "zwt_ni_bLu6MRQXzWKPIU__Uubvy_0Xz_FEr-2sfUg")
os.environ["VECTARA_QUERY_API_KEY"] = os.getenv("VECTARA_QUERY_API_KEY", "zwt_ni_bLu6MRQXzWKPIU__Uubvy_0Xz_FEr-2sfUg")
os.environ["VECTARA_API_KEY"] = os.getenv("VECTARA_API_KEY", "zut_ni_bLoa0I3AeNSjxeZ-UfECnm_9Xv5d4RVBAqw")
os.environ["VECTARA_CORPUS_ID"] = os.getenv("VECTARA_CORPUS_ID", "2")
os.environ["VECTARA_CUSTOMER_ID"] = os.getenv("VECTARA_CUSTOMER_ID", "2653936430")
os.environ["TOGETHER_API"] = os.getenv("TOGETHER_API", "7e6c200b7b36924bc1b4a5973859a20d2efa7180e9b5c977301173a6c099136b")

endpoint = 'https://api.together.xyz/inference'

index = VectaraIndex()
retriever = index.as_retriever(similarity_top_k=7)

def process_image(image_data):
    """
    Process an image input and extract relevant information.
    """
    image = Image.open(io.BytesIO(image_data))
    # Add image processing logic here, e.g., using computer vision models
    extracted_info = "This is an image of..."
    return extracted_info

def get_proactive_suggestions(user_context):
    """
    Analyze the user's context and provide relevant proactive suggestions.
    """
    # Example: Suggest drinking water if it's a hot day
    current_time = datetime.datetime.now()
    if current_time.hour > 12 and current_time.hour < 18:
        return "Don't forget to stay hydrated by drinking water regularly."
    return None

def analyze_sentiment(user_input):
    """
    Use the Together.ai API to analyze the sentiment of the user's input.
    """
    headers = {"Authorization": f"Bearer {os.environ['TOGETHER_API']}"}
    data = {"text": user_input}
    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code == 200:
        sentiment_result = response.json()
        sentiment = sentiment_result.get("sentiment")
    else:
        sentiment = "neutral"
    return sentiment

def search_pubmed(query):
    Entrez.email = "jayashbhardwaj36@gmail.com"  # Replace with your email
    handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
    record = Entrez.read(handle)
    ids = record["IdList"]
    if ids:
        return f"https://pubmed.ncbi.nlm.nih.gov/?term={query}&filter=simsearch1.fha"
    else:
        return None 

def detect_emotional_keywords(text):
    keywords = ["sad", "happy", "angry", "anxious", "depressed", "stressed", "frustrated", "excited", "content", "worried"]
    return any(word in text.lower() for word in keywords)

def generate_reflective_response(text):
    responses = [
        "It sounds like you're feeling {}.",
        "I understand that you're {}.",
        "So, you feel {} about that."
    ]
    sentiment = analyze_sentiment(text) 
    return random.choice(responses).format(sentiment)

def generate_open_ended_question(text):
    questions = [
        "Can you tell me more about what led to feeling this way?",
        "How long have you been feeling like this?",
        "What have you tried to cope with these feelings?", 
        "Is there anything specific that's contributing to this feeling?"
    ]
    return random.choice(questions)

def safe_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        results = "\n".join([f"- {result['Abstract']}" for result in data['RelatedTopics'][:3]])
        return results
    else:
        return None



def medmind_chatbot(user_input, chat_history=None, image_data=None):
    if chat_history is None:
        chat_history = []

    # Process image input, if provided
    if image_data:
        extracted_info = process_image(image_data)
        user_input = f"{user_input} {extracted_info}"

    # Retrieve relevant information from Vector Index
    query_str = user_input
    chat = index.as_chat_engine(chat_mode='condense_plus_context')
    response = index.as_query_engine().query(query_str)
    retrieved_text = response.response
    # Therapist-like Interaction
    if detect_emotional_keywords(user_input):
        response_text = generate_reflective_response(user_input)
        response_text += "\n" + generate_open_ended_question(user_input) 
    else:
        # Information Retrieval
        query_str = user_input
        chat = index.as_chat_engine(chat_mode='condense_plus_context')
        response = index.as_query_engine().query(query_str)
        retrieved_text = response.response

        pubmed_results = search_pubmed(user_input) 
        if pubmed_results:
            retrieved_text += f"\n\n**Relevant PubMed Articles:**\n{pubmed_results}"

        if user_input.startswith("search:"):
            search_query = user_input[7:]  
            search_results = safe_search(search_query) 
            if search_results:
                retrieved_text += f"\n\n**Search Results:**\n{search_results}"

        response_text = retrieved_text

        # Proactive Suggestions and Sentiment
        proactive_suggestion = get_proactive_suggestions(chat_history)
        if proactive_suggestion:
            response_text = f"{proactive_suggestion} {response_text}"

        sentiment = analyze_sentiment(user_input)
        if sentiment == "positive":
            response_text = f"I'm glad you're feeling positive! {response_text}"
        elif sentiment == "negative":
            response_text = f"I'm sorry you're feeling down. Is there anything I can do to help? {response_text}"

    # Additional Options 
    if user_input.startswith("feeling:"):
        feeling = user_input[8:]
        if feeling.lower() in ["down", "depressed", "hopeless"]: 
            response_text += "\nConsider trying some activities you used to enjoy, even if you don't feel like it. Engaging in activities can sometimes help improve your mood." 
        elif feeling.lower() in ["anxious", "stressed", "overwhelmed"]:
            response_text += "\nDeep breathing exercises can be helpful for managing anxiety. Would you like me to guide you through a simple breathing exercise?"
        # ... Add more cases for different feelings ... 
    elif user_input.startswith("need help"):
        response_text += "\nHere are some resources that might be helpful:\n"
        response_text += "- National Suicide Prevention Lifeline: 988 (US)\n"
        response_text += "- Crisis Text Line: Text HOME to 741741 (US)\n" 
        response_text += "- You can also search online for mental health organizations or hotlines in your area."

    chat_history.append((user_input, response_text))
    return response_text, chat_history

iface = gr.Interface(
    fn=medmind_chatbot,
    inputs=["text", "state", gr.Image()],
    outputs=["text", "state"],
    title="MedMind Chatbot"
)

iface.launch()

def medmind_chatbot(user_input, chat_history=None):
    if chat_history is None:
        chat_history = []

    if user_input.startswith("search:"):
        user_intent = "search"
    elif detect_emotional_keywords(user_input):
        user_intent = "emotional"
    elif "pubmed" in user_input.lower():  # Check if the user explicitly mentions PubMed
        user_intent = "pubmed"
    
    # Retrieve Information Based on Intent
    response_text = ""

    if user_intent == "search":
        search_query = user_input[7:]
        search_results = safe_search(search_query)
        if search_results:
            response_text = f"**Search Results:**\n{search_results}"
        else:
            response_text = "I couldn't find any relevant search results."
    elif user_intent == "pubmed":
        pubmed_results = search_pubmed(user_input)
        if pubmed_results:
            response_text = f"**Relevant PubMed Articles:**\n{pubmed_results}"
        else:
            response_text = "I couldn't find any relevant PubMed articles."
    else:  # General or emotional intent - use Vectara
        query_str = user_input
        chat = index.as_chat_engine(chat_mode='condense_plus_context')
        response = index.as_query_engine().query(query_str)
        retrieved_text = response.response
        response_text = retrieved_text

    # Retrieve relevant information from Vector Index
    query_str = user_input
    chat = index.as_chat_engine(chat_mode='condense_plus_context')
    response = index.as_query_engine().query(query_str)
    retrieved_text = response.response

    # Therapist-like Interaction
    if detect_emotional_keywords(user_input):
        response_text = generate_reflective_response(user_input)
        response_text += "\n" + generate_open_ended_question(user_input) 
    else:
        # Information Retrieval
        query_str = user_input
        chat = index.as_chat_engine(chat_mode='condense_plus_context')
        response = index.as_query_engine().query(query_str)
        retrieved_text = response.response

        pubmed_results = search_pubmed(user_input) 
        if pubmed_results:
            retrieved_text += f"\n\n**Relevant PubMed Articles:**\n{pubmed_results}"

        if user_input.startswith("search:"):
            search_query = user_input[7:]  
            search_results = safe_search(search_query) 
            if search_results:
                retrieved_text += f"\n\n**Search Results:**\n{search_results}"

        response_text = retrieved_text

    # Hallucination Evaluation 
    def vectara_hallucination_evaluation_model(text):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        # Get the probability for the "hallucination" class (assuming it's the first class)
        hallucination_probability = outputs.logits[0][0].item()  
        return hallucination_probability

    hallucination_score = vectara_hallucination_evaluation_model(response_text)
    
    # Response Filtering/Modification
    HIGH_HALLUCINATION_THRESHOLD = 0.8  # Adjust as needed 
    if hallucination_score > HIGH_HALLUCINATION_THRESHOLD:
        response_text = "I'm still under development and learning. I cannot confidently answer this question yet."  # Or another appropriate response 
    else:
        # Proactive Suggestions and Sentiment
        proactive_suggestion = get_proactive_suggestions(chat_history)
        if proactive_suggestion:
            response_text = f"{proactive_suggestion} {response_text}"

        sentiment = analyze_sentiment(user_input)
        if sentiment == "positive":
            response_text = f"I'm glad you're feeling positive! {response_text}"
        elif sentiment == "negative":
            response_text = f"I'm sorry you're feeling down. Is there anything I can do to help? {response_text}"

    # Additional Options 
    if user_input.startswith("feeling:"):
        feeling = user_input[8:]
        if feeling.lower() in ["down", "depressed", "hopeless"]: 
            response_text += "\nConsider trying some activities you used to enjoy, even if you don't feel like it. Engaging in activities can sometimes help improve your mood." 
        elif feeling.lower() in ["anxious", "stressed", "overwhelmed"]:
            response_text += "\nDeep breathing exercises can be helpful for managing anxiety. Would you like me to guide you through a simple breathing exercise?"
        # ... Add more cases for different feelings ... 
    elif user_input.startswith("need help"):
        response_text += "\nHere are some resources that might be helpful:\n"
        response_text += "- National Suicide Prevention Lifeline: 988 (US)\n"
        response_text += "- Crisis Text Line: Text HOME to 741741 (US)\n" 
        response_text += "- You can also search online for mental health organizations or hotlines in your area."

    chat_history.append((user_input, response_text))
    return response_text, chat_history