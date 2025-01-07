from bs4 import BeautifulSoup
import requests
import os
import json
import re
from PyPDF2 import PdfReader
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain.load import dumps, loads

@st.cache_resource
def get_doc_vectorstore():
    """
    Returns a Chroma vector store as output
    """
    return Chroma(collection_name="langchain_store", embedding_function=OllamaEmbeddings(model="llama3.2", base_url=f'http://{os.environ.get("LLM_HOST")}:{os.environ.get("LLM_PORT")}'))

def add_documents_from_text(text, vectorstore):
    """
    Add documents to the vector store from a string.
    
    Args:
    text (str): the text content of the document to add to the vector store.
    vectorstore (Chroma): The Chroma vector store.
    
    Returns:
    None
    """
    # Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    documents = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    # Add the documents to the vector store
    vectorstore.add_documents(documents)

def get_unique_docs(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def generate_step_back_query(original_query):
    """
    Generate a step-back query to retrieve broader context.
    """
    model = ChatOllama(model="llama3.2", base_url=f'http://{os.environ.get("LLM_HOST")}:{os.environ.get("LLM_PORT")}', temperature=0)

    step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
    Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.
    
    Original query: {original_query}
    
    Step-back query:"""

    step_back_prompt = PromptTemplate.from_template(step_back_template)

    # Create an LLMChain for step-back prompting
    # step_back_chain = step_back_prompt | model | StrOutputParser
    prompt = step_back_prompt.invoke(original_query)
    resp = model.invoke(prompt)
    out = StrOutputParser.invoke(self=StrOutputParser(), input=resp)
    return out

def decompose_query_with_step_back(original_query: str):
    """
    Decompose the original query into simpler sub-queries.
    
    Args:
    original_query (str): The original complex query
    
    Returns:
    List[str]: A list of simpler sub-queries
    """
    original_query = generate_step_back_query(original_query)
    model = ChatOllama(model="llama3.2", base_url=f'http://{os.environ.get("LLM_HOST")}:{os.environ.get("LLM_PORT")}', temperature=0, max_tokens=4000)

    # Create a prompt template for sub-query decomposition
    subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
    Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.
    
    Original query: {original_query}
    
    example: What are the impacts of climate change on the environment?
    
    Sub-queries:
    1. What are the impacts of climate change on biodiversity?
    2. How does climate change affect the oceans?
    3. What are the effects of climate change on agriculture?
    4. What are the impacts of climate change on human health?"""

    subquery_decomposition_prompt = PromptTemplate.from_template(subquery_decomposition_template)
    
    # Create an LLMChain for sub-query decomposition
    subquery_decomposer_chain = subquery_decomposition_prompt | model

    response = subquery_decomposer_chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    return sub_queries

def retrieve_documents(query, retriever):
    """
    Takes a query as input
    Returns the top 5 most relevant document chunks as outputs
    """
    retrieval_chain = decompose_query_with_step_back | retriever.map() | get_unique_docs
    documents = retrieval_chain.invoke(query)
    vectorstore = Chroma.from_documents(documents=documents, embedding=OllamaEmbeddings(model="llama3.2", base_url=f'http://{os.environ.get("LLM_HOST")}:{os.environ.get("LLM_PORT")}'))
    return vectorstore.similarity_search(query, 5)

class Question:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

@st.cache_resource
class LastQuestions:
    def __init__(self, n=3, max_length=5000, questions: list[Question]=[]):
        self.n = n
        self.max_length = max_length - 30
        self.questions = questions
    
    def add(self, question:Question):
        self.questions.append(question)
        if len(self.questions) > self.n:
            self.questions.pop(0)

    def get_prompt(self):
        if len(self.questions) == 0:
            return ""
        else:
            out = "Last questions:\n\n --- \n"
            for i in range(len(self.questions)):
                temp = f"Question {i}:\n{self.questions[i].question}\n\nAnswer {i}:\n{self.questions[i].answer}"
                if len(temp) > (self.max_length / len(self.questions)) - 5:
                    temp = temp[:(self.max_length / len(self.questions)) - 5]
                    temp += "...\n\n"
                out += temp
            out += " --- \n"
        return out

def response_generation(question, last_questions, retriever):
    """
    Cette fonction prend une requête en entrée et retourne la réponse finale.
    """
    try:
        st.warning("ChatOllama base_url: " + f'http://{os.environ.get("LLM_HOST")}:{os.environ.get("LLM_PORT")}')
        # Document Context
        context = retrieve_documents(question, retriever)
        
        # Prompt
        template = """Here is the question you need to answer:
        \n --- \n {question} \n --- \n
        Here is additional context relevant to the question: 
        \n --- \n {context} \n --- \n
        {last_questions}
        Use the above context to answer the question: \n {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOllama(model="llama3.2", base_url=f'http://{os.environ.get("LLM_HOST")}:{os.environ.get("LLM_PORT")}', temperature=0, stream_usage=True)
        prompt = prompt.invoke({"question": question, "context": context, "last_questions": last_questions})

        for chunk in model.stream(prompt):
            yield chunk.content
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        yield "I don't know what to answer to that ! Would you want to know something else ?"

footer="""<style>
a, a:link, a:visited, a:hover{
color: #555;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: #555;
text-align: center;
z-index: 100;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: inline-block; text-align: center;' href="https://github.com/Baltoch" target="_blank">Balthazar LEBRETON</a></p>
</div>
"""

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text