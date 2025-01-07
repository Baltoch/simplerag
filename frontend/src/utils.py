import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.load import dumps, loads
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

@st.cache_resource
def get_chroma_collection():
    st.session_state.documents = []
    return  chromadb.Client().create_collection(name="simplerag", embedding_function=embedding_functions.OllamaEmbeddingFunction(model_name="llama3.2", url=f'http://{os.environ.get("LLM_HOST")}:{os.environ.get("LLM_PORT")}/api/embeddings'))

def add_chroma_document(name, text, collection):
    """
    Add document separated by chunks to the Chroma vector store.
    
    Args:
    name (str): the name of the document to add to the vector store.
    text (str): the text content of the document to add to the vector store.
    collection (Chroma): The Chroma vector store.
    
    Returns:
    None
    """
    # Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    documents = [chunk for chunk in splitter.split_text(text)]

    # Add the documents to the vector store
    collection.add(documents=documents, ids=[f"{name}-{i}" for i in range(len(documents))], metadatas=[{"document": name} for _ in documents])

    # Update the list of uploaded documents
    st.session_state.documents.append(name)

def remove_chroma_document(name, collection):
    """
    Remove document from the Chroma vector store.
    
    Args:
    name (str): the name of the document to remove from the vector store.
    collection (Chroma): The Chroma vector store.
    
    Returns:
    None
    """
    # Remove the document from the vector store
    collection.delete(where={"document": name})

    # Update the list of uploaded documents
    st.session_state.documents.remove(name)

def query_chroma(query, collection, n_results=5):
    """
    Query the Chroma vector store.
    
    Args:
    query (str or list[str]): the query to search for in the vector store.
    collection (Chroma): The Chroma vector store.
    n_results (int): the number of results to return.
    
    Returns:
    List[str]: A list of document names that match the query.
    """
    st.warning(query)
    if type(query) == str:
        query = [query]
    out = collection.query(query_texts=query, n_results=n_results, include=["documents"])
    st.warning(out)
    return out

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

def response_generation(question, last_questions, collection):
    """
    Cette fonction prend une requête en entrée et retourne la réponse finale.
    """
    try:
        # Document Context
        context = query_chroma(query=question, collection=collection)
        
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