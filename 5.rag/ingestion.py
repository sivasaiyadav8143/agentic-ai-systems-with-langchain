import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting data...")
    loader = TextLoader("mediumblog1.txt")
    document = loader.load()

    # print(f"Loaded {len(document)} document(s)")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Split into {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    print("Creating Pinecone vector store...")

    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.getenv("INDEX_NAME")
        )
    print("Data ingested successfully!")
