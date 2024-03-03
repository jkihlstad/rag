from langchain_community.document_loaders import UnstructuredFileLoader


from langchain_community.document_loaders import PDFLoader

# Assuming the PDFLoader is initialized similarly to the UnstructuredFileLoader
pdf_loader = PDFLoader(
    file_path="example_data/sample.pdf",
    api_key="FAKE_API_KEY",
)

docs = pdf_loader.load()
# Assuming we need to store the loaded PDF documents in a vector database using FAISS and OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the embeddings model
embeddings = OllamaEmbeddings()

# Create a FAISS vector store from the documents
vector_db = FAISS.from_documents(docs, embeddings)



filenames = ["example_data/fake.docx", "example_data/fake-email.eml"]


loader = UnstructuredAPIFileLoader(
    file_path=filenames[0],
    api_key="FAKE_API_KEY",
)           