import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenLLM
from langchain.chains import create_retrieval_chain
from langchain.memory import LongTermMemory

# Streamlit interface to upload documents
st.title("Document Upload for Vector Database with LLM Memory")
uploaded_files = st.file_uploader("Choose your unstructured documents", accept_multiple_files=True)

# Assuming the user has an API key for the OpenLLM
open_llm_api_key = st.text_input("Enter your OpenLLM API key")

# Initialize long term memory
ltm = LongTermMemory(file_path="long_term_memory.json")

# Process the uploaded files and create a vector database
if uploaded_files and open_llm_api_key:
    # Load the documents using UnstructuredFileLoader
    unstructured_docs = []
    for uploaded_file in uploaded_files:
        loader = UnstructuredFileLoader(file_path=uploaded_file)
        docs = loader.load()
        unstructured_docs.extend(docs)
    
    # Initialize the embeddings model
    embeddings = OllamaEmbeddings()

    # Create a FAISS vector store from the unstructured documents
    vector_db = FAISS.from_documents(unstructured_docs, embeddings)

    # Initialize an OpenLLM with the provided API key and long term memory
    open_llm = OpenLLM(api_key=open_llm_api_key, long_term_memory=ltm)

    # Create a retrieval chain using the vector database and the OpenLLM
    retrieval_chain = create_retrieval_chain(vector_db.as_retriever(), open_llm)

    # Streamlit interface to input a query
    query = st.text_input("Input your query here")

    # Function to process a query and retrieve documents
    def process_query(query):
        # Invoke the retrieval chain with the query
        response = retrieval_chain.invoke({"input": query})
        
        # Extract the answer and the context from the response
        answer = response.get("answer")
        context = response.get("context")
        
        # Generate conclusions based on the retrieved documents and the answer
        conclusions = {
            "answer": answer,
            "documents": [doc.metadata for doc in context]
        }
        
        return conclusions

    # If the user enters a query and hits enter
    if query:
        # Process the query and display the conclusions
        conclusions = process_query(query)
        st.write("Answer:", conclusions["answer"])
        st.write("Relevant Documents:")
        for doc in conclusions["documents"]:
            st.write(doc)

        # Update long term memory with the new context
        ltm.update(context)


