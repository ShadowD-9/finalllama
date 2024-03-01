import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
from langchain.llms import CTransformers

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    data = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm=CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5),
        retriever=vectors.as_retriever())


def conversational_chat(query):
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    result = chain({"question": query, "chat_history": st.session_state['history']})
    if "answer" in result:
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    else:
        return "I'm sorry, I couldn't understand your query."


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form(key='my_form'):
    user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    output = conversational_chat(user_input)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

if st.session_state['generated']:
    for i, (user_msg, bot_msg) in enumerate(st.session_state['history']):
        message(user_msg, is_user=True, key=str(i) + '_user', avatar_style="big-smile")
        message(bot_msg, key=str(i), avatar_style="thumbs")
