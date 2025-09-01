import os
import json
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Path setup
PDF_FOLDER = r"C:\Users\djeev\rag_medingen_chatbot\medingen_pdfs"
INDEX_FOLDER = r"C:\Users\djeev\rag_medingen_chatbot\faiss_index"
PRODUCT_LIST_JSON = "product_list.json"


def load_and_split_pdfs(pdf_folder):
    documents = []
    product_names = []

    for filename in tqdm(os.listdir(pdf_folder), desc=" Loading PDFs"):
        if filename.endswith(".pdf"):
            try:
                product_name = filename.split()[0].lower()  
                product_names.append(product_name)

                loader = PyPDFLoader(os.path.join(pdf_folder, filename))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["product"] = product_name

                documents.extend(docs)
                print(f"Loaded and tagged: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return documents, sorted(set(product_names))

def embed_and_store(documents, index_folder):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        print("No chunks created from documents. Check PDF content.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    vectorstore.save_local(index_folder)
    print(f"FAISS index saved to: {index_folder}")

def save_product_list(product_names, output_file):
    with open(output_file, "w") as f:
        json.dump(product_names, f)
    print(f"Product list saved to: {output_file}")

if __name__ == "__main__":
    docs, product_names = load_and_split_pdfs(PDF_FOLDER)
    embed_and_store(docs, INDEX_FOLDER)
    save_product_list(product_names, PRODUCT_LIST_JSON)
