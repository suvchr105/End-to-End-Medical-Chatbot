# 🤖 MediBot AI Assistant – End-to-End Medical GenAI Chatbot

MediBot AI is an intelligent healthcare chatbot built using **LangChain**, **Flask**, and **Pinecone** that answers medical-related questions by retrieving context from a domain-specific PDF knowledge base. The system integrates advanced **LLM retrieval-augmented generation (RAG)** workflows, embeddings, and a futuristic UI.

# [🌐 Live Demo(click)](https://end-to-end-medical-chatbot-ufzt.onrender.com)

<img width="833" alt="Screenshot 2025-05-05 at 12 37 41 AM" src="https://github.com/user-attachments/assets/13f3c76d-40bb-4328-a8c6-0fb6f619226b" />
---

# 📌 Features

- 🧠 LLM-based intelligent medical answering system (GPT 3.5 via OpenRouter)
- 📘 PDF ingestion, chunking, and vector storage using Pinecone
- 💬 Beautiful full-screen webchat UI (HTML + CSS)
- ⚡ RAG pipeline with LangChain, HuggingFace & Pinecone
- 🧠 Fast semantic search via MiniLM + cosine similarity
- 🌍 Deployable via Render, Railway, or locally with Gunicorn
---


# 🛠️ Tech Stack

| Category       | Technology                        |
|----------------|-----------------------------------|
| Backend        | Python, Flask, Gunicorn           |
| Gen AI         | LangChain, ChatOpenAI (OpenRouter)|
| Embeddings     | HuggingFace Transformers          |
| Vector DB      | Pinecone                          |
| Frontend UI    | HTML, CSS, JavaScript             |
| Deployment     | Render.com                        |

---

![Agile development](https://github.com/user-attachments/assets/d3d9a031-fb53-4404-b152-0fe7e2de56dd)




# 🗂️ Project Structure

```bash
End-to-End-Medical-Chatbot/
├── app.py                    # Flask entrypoint
├── src/
│   ├── helper.py             # PDF loader + splitter
│   ├── prompt.py             # System prompts
│   └── __init__.py
├── store_index.py            # Stores chunks in Pinecone index
├── template.py               # RAG chain builder
├── static/
│   └── style.css             # UI styling
├── templates/
│   └── index.html            # Chat UI page
├── Data/
│   └── Medical_book.pdf      # Medical knowledge base
├── .env                      # API keys
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```
---
# 🔐 .env Setup

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key

```

# 📦 Installation
## 1. Clone Repo

```bash
git clone https://github.com/yourusername/End-to-End-Medical-Chatbot.git
cd End-to-End-Medical-Chatbot

```
## 2. Create Environment

```bash
conda create -n medibot python=3.10
conda activate medibot

```
## 3. Install Dependencies

```bash
pip install -r requirements.txt

```

# ⚙️ Build Vector Index
```bash
python store_index.py
```
This script will:

📘 Load and parse Medical_book.pdf

🧩 Split it into chunks using LangChain

🧠 Generate embeddings using MiniLM

📊 Push the embeddings to Pinecone vector DB


# 🚀 Run the App
```bash
python app.py
```
Then open http://localhost:8080


# 🌐 Deploy on Render
1. Push your code to GitHub.

2. Go to https://render.com and select New Web Service.

3. Set the following options:

Build Command:

```bash
pip install -r requirements.txt
```
Start Command:
```bash
gunicorn app:app
```
Add the following environment variables:
```bash
PINECONE_API_KEY

OPENROUTER_API_KEY
```
Click Deploy ✅

# 🔥 Sample User Queries

"What is Acne?"

"How do pimples form?"

"Explain diabetes symptoms"

"What are the causes of high blood pressure?"


# 📘 Research Notes
You can find exploratory notebooks and trials under:

```bash
research/trails.ipynb
Generative_AI_Projects/
```

# 📝 License

This project is licensed under the MIT License.

# 🙌 Credits
LangChain

OpenRouter

Pinecone

HuggingFace Transformers

Sentence Transformers
