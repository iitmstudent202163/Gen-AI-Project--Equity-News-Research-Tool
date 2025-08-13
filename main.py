import os
import streamlit as st
import pickle
import time
import langchain
import tiktoken
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv() # take environment variables from .env.

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_file_path = "faiss_index"
main_placeholder = st.empty()
llm = OpenAI(temperature=0.5, max_tokens = 500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Storing vector index create to local
    vectorstore_openai.save_local(index_file_path)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_file_path):
        vectorstore = FAISS.load_local(index_file_path, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result=chain({"question": query}, return_only_outputs=True)
        # {'answer':"", "sources": []}
        st.header("Answer")
        st.subheader(result["answer"])

        # Display sources, if available
        sources = result.get("sources","")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n") # Split the sources by newline
            for source in sources_list:
                st.write(source)

