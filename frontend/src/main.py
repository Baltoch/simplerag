import streamlit as st
from dotenv import load_dotenv
from utils import LastQuestions, Question, add_chroma_document, extract_text_from_pdf, remove_chroma_document, footer, get_chroma_collection, response_generation, upload_file

st.set_page_config(
    page_title='Simple RAG',
    page_icon="💬"                  
)

load_dotenv()

# Loading cached data
chroma_collection = get_chroma_collection()
last_questions = LastQuestions()

# Interface
st.title("💬 Simple RAG")
st.markdown(footer,unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if "documents" not in st.session_state:
    st.session_state["documents"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        message = st.write_stream(response_generation(prompt, last_questions.get_prompt(), chroma_collection))
    st.session_state.messages.append({"role": "assistant", "content": message})
    last_questions.add(Question(prompt, message))

with st.sidebar:
    for doc in st.session_state.documents:
        with st.container(key=doc, border=True):
            st.write(doc)
            if st.button("Remove"):
                remove_chroma_document(name=doc, collection=chroma_collection)
                st.rerun()
    if st.button("Upload a document"):
        upload_file(chroma_collection)