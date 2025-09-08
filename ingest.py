import os
import json
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Path setup
PDF_FOLDER = r"C:\Users\djeev\Rag_medingen-product-details-_chatbot\medingen_pdfs"
INDEX_FOLDER = r"C:\Users\djeev\rag_medingen_chatbot\faiss_index"
PRODUCT_LIST_JSON = "product_list.json"


def load_and_split_pdfs(pdf_folder):
    documents = []
    catmap = {}

    for filename in tqdm(os.listdir(pdf_folder), desc=" Loading PDFs"):
        if filename.endswith(".pdf"):
            try:
                # remove extension and normalize
                name = os.path.splitext(filename)[0].lower()

                # split filename into words
                words = name.split()

                # assume last word (or last two words) is category
                if len(words) > 1:
                    category = words[-1]
                    # special case: handle two-word categories like "eye drop"
                    if words[-2] == "eye":
                        category = " ".join(words[-2:])
                else:
                    category = "general"

                # product name = everything except category
                product = name.replace(category, "").strip()

                # build catmap
                catmap.setdefault(category, [])
                if product not in catmap[category]:
                    catmap[category].append(product)

                # load pdf
                loader = PyPDFLoader(os.path.join(pdf_folder, filename))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["product"] = product
                    doc.metadata["category"] = category

                documents.extend(docs)
                print(f"Loaded and tagged: {filename} → product='{product}', category='{category}'")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return documents, catmap


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

def save_product_list(catmap, output_file):
    with open(output_file, "w") as f:
        json.dump(catmap, f, indent=2)
    print(f"Product list saved to: {output_file}")


if __name__ == "__main__":
    docs, catmap = load_and_split_pdfs(PDF_FOLDER)
    embed_and_store(docs, INDEX_FOLDER)
    save_product_list(catmap, PRODUCT_LIST_JSON)
