import streamlit as st
import requests
import os
from dotenv import load_dotenv
from utils import LastQuestions, Question, extract_text_from_pdf, footer, get_doc_vectorstore, response_generation, add_documents_from_text

st.set_page_config(
    page_title='GenAI Chat',
    page_icon="💬"                  
)

load_dotenv()

# Loading cached data
vectorstore = get_doc_vectorstore()
last_questions = LastQuestions()

# Interface
st.title("💬 GenAI Chat")
st.markdown(footer,unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        message = st.write_stream(response_generation(prompt, last_questions.get_prompt(), vectorstore.as_retriever()))
    st.session_state.messages.append({"role": "assistant", "content": message})
    last_questions.add(Question(prompt, message))

with st.popover("Upload Documents"):
    st.write("Upload your documents here")
    if uploaded_file := st.file_uploader("Upload your documents", type=["jpg", "png", "pdf", "txt", "md"]):
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
                    add_documents_from_text(response.text, vectorstore)
                else:
                    st.error(f"File upload failed: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        elif uploaded_file.type == "application/pdf":
            # Extract text from the uploaded PDF
            add_documents_from_text(extract_text_from_pdf(uploaded_file), vectorstore)

        elif (uploaded_file.type == "text/plain") or (uploaded_file.type == "text/markdown"):
            # Extract text from the uploaded TXT or MD file
            add_documents_from_text(uploaded_file.getvalue().decode(), vectorstore)

        else:
            st.warning("Unsupported file type. Please upload JPG, PNG, PDF, TXT, or MD.")