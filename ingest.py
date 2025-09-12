
import os
import json
import re
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Use relative paths so the app can run both locally and on Streamlit Cloud
PDF_FOLDER = "medingen_pdfs"
INDEX_DIR = "faiss_index"           
PRODUCT_LIST_JSON = "product_list.json"

SECTION_PATTERNS = [
    r"^\s*(side\s*effects|adverse\s*reactions|adverse\s*events)\b[:\-]?\s*$",
    r"^\s*(benefits|indications|uses|what\s+is\s+it\s+used\s+for)\b[:\-]?\s*$",
    r"^\s*(dosage|dose|administration|when\s+to\s+take|when\s+should\s+i)\b[:\-]?\s*$",
    r"^\s*(contraindications|warnings|precautions)\b[:\-]?\s*$",
    r"^\s*(composition|ingredients)\b[:\-]?\s*$",
]

CHUNK_SIZE = 900
STRIDE = 450


class SectionSplitter:
    def __init__(self, section_patterns=None, chunk_size=CHUNK_SIZE, stride=STRIDE):
        self.section_regexes = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in (section_patterns or SECTION_PATTERNS)]
        self.chunk_size = chunk_size
        self.stride = stride

    def _find_headings(self, text):
        headings = []
        for m in re.finditer(r"(?m)^[^\n]+$", text):
            line = m.group(0).strip()
            for rx in self.section_regexes:
                mo = rx.match(line)
                if mo:
                    heading_name = mo.group(1).strip().lower()
                    headings.append((m.start(), heading_name))
                    break
        return headings

    def split_documents(self, documents):
        chunks = []
        for doc in documents:
            text = doc.page_content or ""
            page_no = doc.metadata.get("page_number") or doc.metadata.get("page", None)
            headings = self._find_headings(text)
            if headings:
                for i, (pos, heading_name) in enumerate(headings):
                    start = pos
                    end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
                    span_text = text[start:end].strip()
                    if span_text:
                        meta = dict(doc.metadata or {})
                        meta.update({"section": heading_name, "page": page_no})
                        chunks.append(Document(page_content=span_text, metadata=meta))
            else:
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        meta = dict(doc.metadata or {})
                        meta.update({"section": "general", "page": page_no})
                        chunks.append(Document(page_content=chunk_text, metadata=meta))
                    start += self.stride
        return chunks


def load_and_prepare_pdfs(pdf_folder):
    documents = []
    product_names = []
    for filename in tqdm(os.listdir(pdf_folder), desc="Loading PDFs"):
        if not filename.lower().endswith(".pdf"):
            continue
        try:
            product_name = filename.split()[0].lower()
            product_names.append(product_name)
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            pages = loader.load()
            for i, page in enumerate(pages, start=1):
                page.metadata = page.metadata or {}
                page.metadata["product"] = product_name
                page.metadata["page_number"] = i
            documents.extend(pages)
            print(f"Loaded: {filename} -> {product_name}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return documents, sorted(set(product_names))


def embed_and_store(documents, index_dir):
    splitter = SectionSplitter()
    chunks = splitter.split_documents(documents)
    if not chunks:
        print("No chunks created.")
        return
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)   # writes index.faiss and index.pkl inside index_dir
    print(f"Index saved to {index_dir}")


def save_product_list(product_names, output_file):
    with open(output_file, "w") as f:
        json.dump(product_names, f, indent=2)
    print(f"Saved products to {output_file}")


if __name__ == "__main__":
    docs, product_names = load_and_prepare_pdfs(PDF_FOLDER)
    embed_and_store(docs, INDEX_DIR)
    save_product_list(product_names, PRODUCT_LIST_JSON)