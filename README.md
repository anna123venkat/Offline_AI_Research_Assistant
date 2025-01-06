---

## Offline AI Research Assistant

**Offline AI Research Assistant** is an intelligent tool designed to assist researchers by providing access to a wide range of information and tools even in offline environments. This project was developed as part of a mini-project for the **B.Tech in Artificial Intelligence and Data Science** program at **Mepco Schlenk Engineering College**.

---

## Project Overview

The project leverages artificial intelligence technologies to help researchers:
- Retrieve relevant information from research documents.
- Analyze datasets, identify patterns, and generate insights.
- Adapt to individual research needs and preferences.

The assistant operates without requiring an internet connection, making it ideal for remote locations or areas with limited connectivity.

---

## Features

1. **Offline Functionality**: Access research documents and tools without internet connectivity.
2. **Document Processing**:
   - Load and split research documents (PDFs) into smaller chunks for efficient processing.
3. **Conversational Capabilities**:
   - Query-based search for relevant information.
   - Natural Language Processing (NLP) for understanding complex queries.
4. **Embeddings and Vector Stores**:
   - Use `OpenAIEmbeddings` for text processing.
   - Build vector stores with `FAISS` for quick information retrieval.
5. **Web-Based Interface**:
   - Built using `Streamlit` for a user-friendly interface.
6. **Support for Large Language Models (LLMs)**:
   - Includes `Mistral 7b` for generating intelligent responses.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - [Streamlit](https://streamlit.io/) for the web interface.
  - [Langchain](https://langchain.com/) for NLP and AI integration.
  - [FAISS](https://faiss.ai/) for vector storage and retrieval.
  - [dotenv](https://pypi.org/project/python-dotenv/) for environment variable management.
  - [PyPDFLoader](https://pypi.org/project/langchain/) for document loading.
- **Models**:
  - Mistral 7b
  - Ada-embeddings-002

---

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Offline_AI_Research_Assistant.git
   cd Offline_AI_Research_Assistant
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add a `.env` file for configuration:
   ```bash
   OPENAI_DEPLOYMENT_NAME=your_openai_api_key
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Load your research documents into the `data/` directory (supports PDFs).
2. Start the assistant and interact via the web interface.
3. Query the assistant for relevant information, and it will respond intelligently.

---

## System Approach

Below is the flow of the system:

1. Load research documents using `PyPDFLoader`.
2. Split documents into chunks using `RecursiveCharacterTextSplitter`.
3. Create embeddings and a vector store using `OpenAIEmbeddings` and `FAISS`.
4. Process user queries via `ConversationalRetrievalChain` and return intelligent responses.

---

## Contributors

- **Dinesh Kumar S** (9517202109016)
- **Naboth Demitrius R** (9517202109036)
- **Prasanna Venkatesh S** (9517202109040)

---

## Acknowledgements

We thank the management, faculty, and staff of **Mepco Schlenk Engineering College**, particularly:
- **Dr. J. Angela Jennifa Sujana**, Associate Professor and Head, Department of AI & Data Science.
- **Ms. L. Prasika**, Assistant Professor (SG), for her guidance and support.

---
