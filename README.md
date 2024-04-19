
# MedMind: AI-Powered Medical Chatbot

MedMind is an advanced Retrieval Augmented Generation (RAG) chatbot designed to provide reliable and informative responses to your medical inquiries. 

## Demo
https://huggingface.co/spaces/jayash391/RAG_MedMind

## Features

* **Answers General Medical Questions:** Utilizes a curated medical knowledge base and advanced language models to answer a wide range of health-related questions.
* **Summarizes Research Articles:** Retrieves and summarizes relevant research articles from PubMed, making complex scientific information more accessible.
* **Offers Additional Insights:** Provides context and insights from its knowledge base to enhance understanding.
* **Safe Web Searches:** Performs safe web searches using the Google Search API, ensuring the safety and relevance of the results.
* **Hallucination Evaluation:** Employs a model to minimize the risk of generating fabricated information.

## Technology Stack

* **Streamlit:** For building the user interface.
* **LlamaIndex:** For indexing and querying data.
* **Vectara:** As a vector database for storing and retrieving medical knowledge. 
* **TogetherLLM:** Large language model for generating responses and summaries.
* **LangChain:** For building question-answering chains.
* **Chroma:** For indexing and querying uploaded documents.
* **Hugging Face Transformers:** For embeddings and hallucination evaluation. 
* **PubMed API:** For searching and retrieving research articles.
* **Google Search API:** For performing safe web searches.

## Limitations

* Not a substitute for professional medical advice.
* Information is for general knowledge and educational purposes only.
* Under development and may occasionally provide inaccurate or incomplete information.
* Potential for hallucinations, especially for complex queries. 

## Installation and Running 

1. **Clone the repository:** `git clone https://github.com/your-username/MedMind.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Set up environment variables:** Refer to the `.env.example` file and create a `.env` file with your API keys.
4. **Run the Streamlit app:** `streamlit run medmind.py`

![medmind_preview jpg](https://github.com/jayash1973/rag_tut/assets/139613263/61f203ee-0a69-4eb0-a6a7-c65de5f0a548)


## Contributing

* [@stomanin](https://github.com/stomanin)
* [@jayash1973](https://github.com/jayash1973)

Contributions are welcome! 

## License

This project is licensed under the MIT License.
