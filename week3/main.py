import os
import sys
import ollama
import chromadb
import fitz
from pydantic import ValidationError
from dotenv import load_dotenv
from google import genai

import pipeline
from pipeline import RAGResponse

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap 
    return chunks

def vector_db(chunks, embedding_model, pdf_collection_name):
    db_path = f"./chromadb/{embedding_model}"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(pdf_collection_name)

    if collection.count() > 0:
        return collection
    
    for i, chunk in enumerate(chunks):
        if embedding_model == "gemini":
            response = gemini_client.models.embed_content(model='gemini-embedding-001', contents=chunk)
            embedding = response.embeddings[0].values
        elif embedding_model == "ollama":
            response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
            embedding = response["embedding"]
        
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{pdf_collection_name}_chunk_{i}"]
        )

    return collection

def ask_question_json(query, collection, embedding_model):
    if embedding_model == "gemini":
        response = gemini_client.models.embed_content(model='gemini-embedding-001', contents=query)
        query_embedding = response.embeddings[0].values
    elif embedding_model == "ollama":
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_embedding = response["embedding"]

    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])

    prompt = f"Use ONLY the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
    
    llm_response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}],
        format=RAGResponse.model_json_schema(),
    )

    raw_json = llm_response['message']['content']

    try:
        validated = pipeline.parse_and_validate(raw_json)
    except ValidationError as e:
        print(f"LLM output failed schema validation, not saving.\nRaw output: {raw_json}\nError: {e}")
        return None

    pipeline.save_response(query, embedding_model, validated)
    return validated


if __name__ == "__main__":
    pdf_path1 = "pdf_collection/lora.pdf"
    pdf_path2 = "pdf_collection/attentionisallyouneed.pdf"
    
    raw_text1 = extract_text_from_pdf(pdf_path1)
    raw_text2 = extract_text_from_pdf(pdf_path2)

    chunks1 = chunk_text(raw_text1)
    chunks2 = chunk_text(raw_text2)

    gemini_col = vector_db(chunks1 + chunks2, "gemini", "ai_papers")
    ollama_col = vector_db(chunks1 + chunks2, "ollama", "ai_papers")

    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        print("\n--- Using Gemini Embeddings ---")
        gemini_answer = ask_question_json(query, gemini_col, "gemini")
        print(gemini_answer)

        print("\n--- Using Ollama Embeddings ---")
        ollama_answer = ask_question_json(query, ollama_col, "ollama")
        print(ollama_answer)