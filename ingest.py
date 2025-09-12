import os
import json
import re
from tqdm import tqdm
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_FOLDER = "medingen_pdfs"
INDEX_DIR = "faiss_index"
PRODUCT_LIST_JSON = "product_list.json"

# Broader heading patterns (capture common variants and trailing words)
SECTION_PATTERNS = [
    r"^\s*(side(?:\s+effects)?|adverse(?:\s+reactions?|s)?|adverse\s+events)\b.*$",
    r"^\s*(benefits|indications?|uses?|what\s+is\s+it\s+used\s+for|how\s+it\s+works)\b.*$",
    r"^\s*(dosage|dose|administration|when\s+to\s+take|when\s+should\s+i|posology)\b.*$",
    r"^\s*(contraindications|warnings|precautions|interactions)\b.*$",
    r"^\s*(composition|ingredients|active\s+ingredient|formulation)\b.*$",
    r"^\s*(overdose|storage|shelf\s+life)\b.*$",
]

CHUNK_SIZE = 900
STRIDE = 450

# normalized mapping - keep canonical labels for easier downstream filtering
HEADING_NORMALIZATION = {
    "side": "side effects",
    "side effects": "side effects",
    "adverse reactions": "side effects",
    "adverse reaction": "side effects",
    "adverse events": "side effects",
    "benefits": "benefits",
    "indication": "benefits",
    "indications": "benefits",
    "uses": "benefits",
    "use": "benefits",
    "how it works": "benefits",
    "dosage": "dosage",
    "dose": "dosage",
    "administration": "dosage",
    "contraindications": "contraindications",
    "warnings": "contraindications",
    "precautions": "contraindications",
    "composition": "composition",
    "ingredients": "composition",
    "overdose": "other",
    "storage": "other",
}


class SectionSplitter:
    def __init__(self, section_patterns=None, chunk_size=CHUNK_SIZE, stride=STRIDE):
        patterns = section_patterns or SECTION_PATTERNS
        self.section_regexes = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
        self.chunk_size = chunk_size
        self.stride = stride

    def _normalize_heading(self, raw_heading: str) -> str:
        if not raw_heading:
            return "general"
        key = raw_heading.strip().lower()
        # take first few words only (most headings are short)
        key_short = " ".join(key.split()[:3])
        for k, v in HEADING_NORMALIZATION.items():
            if re.search(r"\b" + re.escape(k) + r"\b", key_short):
                return v
        return "general"

    def _find_headings(self, text: str):
        headings = []
        for m in re.finditer(r"(?m)^[^\n]{1,120}$", text):
            line = m.group(0).strip()
            if not line:
                continue
            for rx in self.section_regexes:
                mo = rx.match(line)
                if mo:
                    # take the matched group 1 if present or the matched text
                    heading_name = (mo.group(1) if mo.groups() else line).strip().lower()
                    norm = self._normalize_heading(heading_name)
                    headings.append((m.start(), norm))
                    break
        return headings

    def split_documents(self, documents):
        chunks = []
        for doc in documents:
            text = doc.page_content or ""
            if not text.strip():
                continue
            page_no = doc.metadata.get("page_number") or doc.metadata.get("page", None)
            headings = self._find_headings(text)
            if headings:
                # ensure headings are unique and sorted
                headings = sorted(headings, key=lambda x: x[0])
                for i, (pos, heading_name) in enumerate(headings):
                    start = pos
                    end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
                    span_text = text[start:end].strip()
                    if span_text:
                        meta = dict(doc.metadata or {})
                        meta.update({"section": heading_name, "page": page_no})
                        chunks.append(Document(page_content=span_text, metadata=meta))
            else:
                # fallback to sliding window chunking
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
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        print(f"PDF folder '{pdf_folder}' does not exist.")
        return documents, []
    for filename in tqdm(os.listdir(pdf_folder), desc="Loading PDFs"):
        if not filename.lower().endswith(".pdf"):
            continue
        try:
            stem = Path(filename).stem
            # normalize product name: remove extra punctuation, collapse spaces
            product_name = re.sub(r"[^\w\s\-]", "", stem).strip().lower()
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
    vectorstore.save_local(index_dir)
    print(f"Index saved to {index_dir}")


def save_product_list(product_names, output_file):
    with open(output_file, "w") as f:
        json.dump(sorted(product_names), f, indent=2)
    print(f"Saved products to {output_file}")


if __name__ == "__main__":
    docs, product_names = load_and_prepare_pdfs(PDF_FOLDER)
    embed_and_store(docs, INDEX_DIR)
    save_product_list(product_names, PRODUCT_LIST_JSON)