import chromadb
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

doc_metadata =  {
        "title": "The Mind and Its Education", 
        "author" : "George Herbert Betts",
        "source_url": "https://www.gutenberg.org/ebooks/20220",
        "filename": "themind.txt"
}

client_openai = OpenAI()

def get_completion(prompt):
    response = client_openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant connected to a database for document search." },
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

text_splitter = RecursiveCharacterTextSplitter(
    separators=[". ", "? ", "! "],
    chunk_size=2000,
    chunk_overlap=300,
)

with open("themind.txt", "r") as file:
    content = file.read()
chunks = text_splitter.create_documents([content])
print(chunks)

# Đường dẫn tới thư mục chroma database
db_path = "/chromaDatabase"
# Kiểm tra và tạo thư mục nếu chưa tồn tại
if not os.path.exists(db_path):
    os.makedirs(db_path)
# Khởi tạo client với đường dẫn
client_chroma = chromadb.PersistentClient(path=db_path)
collection = client_chroma.get_or_create_collection(name=db_path, metadata={"hnsw:space": "cosine"})

for idx, chunk in enumerate(chunks):
    print(idx)
    print(chunk)
    doc_text = chunk.page_content
    book_metadata["chunk_idx"] = idx
    collection.add(
        documents=[doc_text],
        ids=[f"{book_metadata['title']}_{idx}"],
        metadatas=[book_metadata]
    )

def populate_rag_query(query, n_results=1):
    search_results = collection.query(query_texts=[query], n_results=n_results)
    result_str = ""
    for idx, result in enumerate(search_results["documents"][0]):
        metadata = search_results["metadatas"][0][idx]
        formatted_result = f"""<SEARCH RESULT>
        <DOCUMENT>{result}</DOCUMENT>
        <METADATA>
        <TITLE>{metadata['title']}</TITLE>
        <AUTHOR>{metadata['author']}</AUTHOR>
        <CHUNK_IDX>{metadata['chunk_idx']}</CHUNK_IDX>
        <URL>{metadata['source_url']}</URL>
        </METADATA>
        </SEARCH RESULT>"""
        result_str += formatted_result
    return result_str

def make_rag_prompt(query, results):
    return f"""<INSTRUCTIONS>
    Your task is to answer the following user question. The search results of a document search have been included to give you more context. Use the information in Search Results to help you answer the question accurately.
    Not all information in the search results will be useful. However, if you find any information that's useful for answering the user's question, draw from it in your answer.
   <EXAMPLE CITATION>
   Answer to the user query in your own words, drawn from the search results.
   - "Direct quote from source material backing up the claim" - [Source: Title, Author, Chunk: chunk index, Link: url]
   </EXAMPLE CITATION>
   </INSTRUCTIONS>

    <USER QUERY>
    {query}
    </USER QUERY>

    <SEARCH RESULTS>
    {results}
    </SEARCH RESULTS>
    
    Your answer:"""

