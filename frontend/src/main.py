import streamlit as st
import requests
import os
from dotenv import load_dotenv
from utils import LastQuestions, Question, add_chroma_document, extract_text_from_pdf, remove_chroma_document, footer, get_chroma_collection, response_generation

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
        with st.container(key=doc):
            st.write(doc)
            if st.button("Remove"):
                remove_chroma_document(name=doc, collection=chroma_collection)
    uploaded_file = st.file_uploader("file upload", type=["jpg", "png", "pdf", "txt", "md"], label_visibility="hidden")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload"):
            if uploaded_file is None:
                st.warning("No file selected.")
            else:
                # Handle different file types
                if uploaded_file.type in ["image/jpeg", "image/png"]:
                    try:
                        # Send the image to the Express.js server
                        response = requests.post(
                            url=os.environ.get("OCR_URL"),
                            headers={"Content-Type": uploaded_file.type},
                            data=uploaded_file.read(),
                        )

                        # Handle the server response
                        if response.status_code == 200:
                            add_chroma_document(name=uploaded_file.name, text=response.text, collection=chroma_collection)
                        else:
                            st.error(f"File upload failed: {response.status_code} - {response.text}")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

                elif uploaded_file.type == "application/pdf":
                    # Extract text from the uploaded PDF
                    add_chroma_document(name=uploaded_file.name, text=extract_text_from_pdf(uploaded_file), collection=chroma_collection)

                elif (uploaded_file.type == "text/plain") or (uploaded_file.type == "text/markdown"):
                    # Extract text from the uploaded TXT or MD file
                    add_chroma_document(name=uploaded_file.name, text=uploaded_file.getvalue().decode(), collection=chroma_collection)

                else:
                    st.warning("Unsupported file type. Please upload JPG, PNG, PDF, TXT, or MD.")
                st.rerun()
    with col2:
        if st.button("Cancel"):
            st.rerun()