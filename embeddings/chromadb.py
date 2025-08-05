import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChromaDB():
    def __init__(self, client, collection):
        self.client = client
        self.collection = collection
    
    def chunk_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[". ", "? ", "! "],
            chunk_size=2000,
            chunk_overlap=300,
        )
        return text_splitter.create_documents([text])
    
    def add_documents(self, documents, metadatas):
        if not metadatas["title"] or not metadatas["id"]:
            raise ValueError("Metadata must contain 'title' and 'id'")
            return
        chunks = self.chunk_text(documents)
        for idx, chunk in enumerate(chunks):
            doc_text = chunk.page_content
            self.collection.add(
                documents=[doc_text],
                ids=[f"{metadatas.get('title')}_{metadatas.get('id')}"],
                metadatas=[metadatas]
            )
    
    def delete_document(self, metadatas):
        self.collection.delete(ids=[f"{metadatas.get('title')}_{metadatas.get('id')}"])