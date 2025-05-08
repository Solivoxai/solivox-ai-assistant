# 🤖 Solivox AI Assistant

Solivox AI Assistant is an interactive Streamlit web application that leverages **Google's Gemini language model**, **LangChain**, and **FAISS vector store** to process PDF documents and answer user queries based on their content.

## 🚀 Features

- 📄 **Upload any PDF** to analyze its content
- 💬 **Ask questions** and get accurate, AI-generated answers
- 🧠 Powered by **Google Gemini API** for conversational responses
- 🔍 Uses **Google Generative AI Embeddings** for document understanding
- ⚡ Integrated with **FAISS** for efficient vector-based search
- 🎨 Clean UI with helpful instructions, tips, and real-time status

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – Web interface
- [LangChain](https://www.langchain.com/) – LLM orchestration
- [Google Generative AI](https://ai.google.dev/) – Gemini LLM + Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) – PDF parsing and chunking

## 📦 Installation

```bash
git clone https://github.com/yourusername/solivox-ai-assistant.git
cd solivox-ai-assistant
pip install -r requirements.txt
