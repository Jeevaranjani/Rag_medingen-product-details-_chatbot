import os
import json
import re
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_FOLDER = "medingen_pdfs"
INDEX_DIR = "faiss_index"
PRODUCT_LIST_JSON = "product_list.json"

# Enhanced section patterns with more comprehensive matching
SECTION_PATTERNS = [
    r"^\s*(?:side\s*effects?|adverse\s*reactions?|adverse\s*events?)\b[:\-]?\s*$",
    r"^\s*(?:benefits?|indications?|uses?|what\s+is\s+(?:it\s+)?used\s+for|therapeutic\s+uses?)\b[:\-]?\s*$",
    r"^\s*(?:dosage|dose|administration|when\s+to\s+take|when\s+should\s+i|how\s+to\s+take)\b[:\-]?\s*$",
    r"^\s*(?:contraindications?|warnings?|precautions?|cautions?)\b[:\-]?\s*$",
    r"^\s*(?:composition|ingredients?|active\s+ingredients?)\b[:\-]?\s*$",
    r"^\s*(?:how\s+it\s+works?|mechanism\s+of\s+action|pharmacology)\b[:\-]?\s*$",
    r"^\s*(?:expert'?s?\s+advice|medical\s+advice|doctor'?s?\s+advice)\b[:\-]?\s*$",
    r"^\s*(?:frequently\s+asked\s+questions?|faqs?)\b[:\-]?\s*$",
]

CHUNK_SIZE = 900
STRIDE = 450

class EnhancedSectionSplitter:
    def __init__(self, section_patterns=None, chunk_size=CHUNK_SIZE, stride=STRIDE):
        self.section_regexes = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in (section_patterns or SECTION_PATTERNS)]
        self.chunk_size = chunk_size
        self.stride = stride
    
    def _normalize_section_name(self, section_text):
        """Normalize section names to consistent format"""
        section_lower = section_text.lower().strip()
        
        # Map variations to standard names
        if any(term in section_lower for term in ['benefit', 'indication', 'use', 'therapeutic']):
            return 'benefits'
        elif any(term in section_lower for term in ['side effect', 'adverse']):
            return 'side_effects'
        elif any(term in section_lower for term in ['dose', 'dosage', 'administration', 'how to take']):
            return 'dosage'
        elif any(term in section_lower for term in ['contraindication', 'warning', 'precaution']):
            return 'contraindications'
        elif any(term in section_lower for term in ['composition', 'ingredient']):
            return 'composition'
        elif any(term in section_lower for term in ['how it work', 'mechanism']):
            return 'mechanism'
        elif any(term in section_lower for term in ['expert', 'advice', 'medical advice']):
            return 'expert_advice'
        elif any(term in section_lower for term in ['question', 'faq']):
            return 'faq'
        else:
            return section_lower.replace(' ', '_')
    
    def _find_headings(self, text):
        """Find section headings in text with improved pattern matching"""
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if line matches any section pattern
            for regex in self.section_regexes:
                match = regex.match(line_stripped)
                if match:
                    # Calculate position in original text
                    pos = text.find(line_stripped)
                    if pos != -1:
                        section_name = self._normalize_section_name(line_stripped)
                        headings.append((pos, section_name))
                        break
        
        return sorted(headings)
    
    def split_documents(self, documents):
        """Split documents into sections and chunks"""
        chunks = []
        
        for doc in documents:
            text = doc.page_content or ""
            page_no = doc.metadata.get("page_number") or doc.metadata.get("page", None)
            product = doc.metadata.get("product", "unknown")
            
            headings = self._find_headings(text)
            
            if headings:
                # Split by sections
                for i, (pos, heading_name) in enumerate(headings):
                    start = pos
                    end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
                    section_text = text[start:end].strip()
                    
                    if section_text:
                        # Clean section text (remove the header line)
                        lines = section_text.split('\n')
                        if lines and any(regex.match(lines[0].strip()) for regex in self.section_regexes):
                            section_content = '\n'.join(lines[1:]).strip()
                        else:
                            section_content = section_text
                        
                        if section_content:
                            meta = dict(doc.metadata or {})
                            meta.update({
                                "section": heading_name,
                                "page": page_no,
                                "product": product
                            })
                            chunks.append(Document(page_content=section_content, metadata=meta))
            else:
                # No sections found, use sliding window
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunk_text = text[start:end].strip()
                    
                    if chunk_text:
                        meta = dict(doc.metadata or {})
                        meta.update({
                            "section": "general",
                            "page": page_no,
                            "product": product
                        })
                        chunks.append(Document(page_content=chunk_text, metadata=meta))
                    
                    start += self.stride
        
        return chunks

def load_and_prepare_pdfs(pdf_folder):
    """Load PDFs and extract product names"""
    documents = []
    product_names = []
    
    for filename in tqdm(os.listdir(pdf_folder), desc="Loading PDFs"):
        if not filename.lower().endswith(".pdf"):
            continue
        
        try:
            # Extract product name (first word before any delimiter)
            product_name = re.split(r'[-_\s]', filename)[0].lower()
            product_names.append(product_name)
            
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            pages = loader.load()
            
            for i, page in enumerate(pages, start=1):
                page.metadata = page.metadata or {}
                page.metadata["product"] = product_name
                page.metadata["page_number"] = i
                page.metadata["source_file"] = filename
            
            documents.extend(pages)
            print(f"Loaded: {filename} -> {product_name}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return documents, sorted(set(product_names))

def embed_and_store(documents, index_dir):
    """Create embeddings and store in FAISS index"""
    splitter = EnhancedSectionSplitter()
    chunks = splitter.split_documents(documents)
    
    if not chunks:
        print("No chunks created.")
        return
    
    print(f"Created {len(chunks)} chunks")
    
    # Print sample chunks for debugging
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nSample chunk {i+1}:")
        print(f"Section: {chunk.metadata.get('section', 'unknown')}")
        print(f"Product: {chunk.metadata.get('product', 'unknown')}")
        print(f"Content preview: {chunk.page_content[:100]}...")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"Index saved to {index_dir}")

def save_product_list(product_names, output_file):
    """Save product list to JSON file"""
    with open(output_file, "w") as f:
        json.dump(product_names, f, indent=2)
    print(f"Saved {len(product_names)} products to {output_file}")

if __name__ == "__main__":
    docs, product_names = load_and_prepare_pdfs(PDF_FOLDER)
    embed_and_store(docs, INDEX_DIR)
    save_product_list(product_names, PRODUCT_LIST_JSON)
