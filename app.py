from flask import Flask,render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from src.prompt import *
app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)

retriever = docsearch.as_retriever(search_type= "similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    temperature=0.4,
    max_tokens=500,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
from langchain_core.prompts import ChatPromptTemplate


system_prompt = (
    "You are an assistant for medical question-answering tasks.\n"
    "Use the retrieved context to answer the question.\n"
    "If you don't know the answer, say you don't know.\n"
    "Use maximum three concise sentences.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get("msg", "")
    print("User:", msg)
    
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    
    return jsonify({"answer": response["answer"]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug=True)
