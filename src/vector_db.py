#vector-db-create.py
# create a vector database from a pdf file

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loaders = [PyPDFLoader('../knowledge/pdf/Life_Events_as_Stressors_ A_Methodological_Inquiry.pdf')]

docs = []
for file in loaders:
    docs.extend(file.load())
#split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
#print(len(docs))

vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="../embeddings/chroma_db_nccn")

print(vectorstore._collection.count())