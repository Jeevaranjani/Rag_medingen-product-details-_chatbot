import streamlit as st
import json
import re
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pathlib import Path

INDEX_FOLDER = Path(os.environ.get("FAISS_INDEX_PATH", "faiss_index"))
PRODUCT_LIST_JSON = Path("product_list.json")
SIMILARITY_THRESHOLD = 0.3  # Lowered threshold for better recall
SEARCH_K = 15  # Increased search results
RETURN_TOP_K = 8  # More documents to analyze

# Enhanced keyword mappings
SECTION_KEYWORDS = {
    "benefits": [
        "benefit", "benefits", "use", "used for", "indication", "indications",
        "how it works", "relieves", "treats", "helps", "pain relief", "fever",
        "analgesic", "antipyretic", "therapeutic", "treatment", "medicine for"
    ],
    "side_effects": [
        "side effect", "side effects", "side-effect", "side-effects",
        "adverse reaction", "adverse reactions", "adverse event", "adverse events",
        "nausea", "vomit", "vomiting", "rash", "itch", "itching", "dizziness",
        "headache", "constipation", "insomnia", "fatigue", "allergic", "allergy", "swelling"
    ],
    "dosage": [
        "dose", "dosage", "take", "when to take", "how to take", "intake",
        "before", "after", "daily", "every", "hour", "hr", "tablet", "tablets"
    ],
}

GREETINGS = {"hi", "hello", "hey", "hii", "hiya"}

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Load the FAISS vectorstore"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

def normalize_product_name(name: str) -> str:
    """Normalize product name for comparison"""
    return name.strip().lower() if name else name

def sanitize(q: str) -> str:
    """Clean and sanitize query text"""
    return re.sub(r"[^\w\s]", " ", q.lower()).strip()

def is_gibberish(q_clean: str) -> bool:
    """Check if query appears to be gibberish"""
    tokens = re.findall(r"\w+", q_clean)
    if not tokens:
        return True
    
    short_tokens = [t for t in tokens if len(t) <= 2]
    if len(short_tokens) / max(1, len(tokens)) > 0.7:
        return True
    
    return False

def contains_greeting_short(q_clean: str) -> bool:
    """Check if query is just a greeting"""
    tokens = re.findall(r"\w+", q_clean)
    return bool(tokens and tokens[0] in GREETINGS and len(tokens) <= 3)

def detect_section_intent(q_clean: str) -> str:
    """Detect the intended section based on query keywords"""
    query_words = set(q_clean.lower().split())
    
    for section, keywords in SECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q_clean.lower():
                return section
    
    return ""

def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def extract_benefits_content(docs: List, top_n=8) -> List[str]:
    """Extract benefit-related content from documents"""
    benefit_keywords = set(SECTION_KEYWORDS["benefits"])
    
    # Prioritize documents from benefits section
    benefits_docs = []
    other_docs = []
    
    for doc in docs:
        section = (doc.metadata or {}).get("section", "").lower()
        if "benefit" in section or section == "benefits":
            benefits_docs.append(doc)
        else:
            other_docs.append(doc)
    
    search_docs = benefits_docs if benefits_docs else docs
    
    extracted_content = []
    
    for doc in search_docs:
        content = (doc.page_content or "").strip()
        section = (doc.metadata or {}).get("section", "")
        
        # For benefits section, extract the entire content
        if section == "benefits" or "benefit" in section.lower():
            sentences = split_sentences(content)
            for sentence in sentences:
                if len(sentence) > 20:  # Filter out very short sentences
                    extracted_content.append(sentence.strip())
        else:
            # For other sections, look for benefit-related sentences
            sentences = split_sentences(content)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in benefit_keywords):
                    if len(sentence) > 20:
                        extracted_content.append(sentence.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_content = []
    for item in extracted_content:
        item_clean = re.sub(r'\s+', ' ', item.lower())
        if item_clean not in seen:
            unique_content.append(item)
            seen.add(item_clean)
        if len(unique_content) >= top_n:
            break
    
    return unique_content

def extract_side_effects_content(docs: List, top_n=8) -> List[str]:
    """Extract side effects content from documents"""
    side_effect_keywords = set(SECTION_KEYWORDS["side_effects"])
    
    # Prioritize documents from side effects section
    side_effects_docs = []
    other_docs = []
    
    for doc in docs:
        section = (doc.metadata or {}).get("section", "").lower()
        if "side" in section or "effect" in section or "adverse" in section:
            side_effects_docs.append(doc)
        else:
            other_docs.append(doc)
    
    search_docs = side_effects_docs if side_effects_docs else docs
    
    extracted_content = []
    
    for doc in search_docs:
        content = (doc.page_content or "").strip()
        section = (doc.metadata or {}).get("section", "")
        
        # For side effects section, extract the content carefully
        if section == "side_effects" or "side" in section.lower():
            # If content looks like a list of side effects
            if any(effect in content.lower() for effect in ["nausea", "vomiting", "headache", "dizziness"]):
                # Split by common delimiters and clean
                effects = re.split(r'[,;]\s*|\n', content)
                for effect in effects:
                    effect_clean = effect.strip()
                    if effect_clean and len(effect_clean) > 3:
                        extracted_content.append(effect_clean)
            else:
                # Extract sentences mentioning side effects
                sentences = split_sentences(content)
                for sentence in sentences:
                    if len(sentence) > 10:
                        extracted_content.append(sentence.strip())
        else:
            # For other sections, look for side effect mentions
            sentences = split_sentences(content)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in side_effect_keywords):
                    if len(sentence) > 10:
                        extracted_content.append(sentence.strip())
    
    # Remove duplicates and clean up
    seen = set()
    unique_content = []
    for item in extracted_content:
        # Clean up the content
        item_clean = re.sub(r'\s+', ' ', item).strip()
        item_lower = item_clean.lower()
        
        if item_lower not in seen and len(item_clean) > 3:
            unique_content.append(item_clean)
            seen.add(item_lower)
        
        if len(unique_content) >= top_n:
            break
    
    return unique_content

def extract_general_content(docs: List, query: str, section_intent: str, top_n=6) -> List[str]:
    """Extract general content relevant to the query"""
    candidates = []
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    for doc in docs:
        content = (doc.page_content or "").strip()
        sentences = split_sentences(content)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            # Score based on section intent
            if section_intent and any(kw in sentence_lower for kw in SECTION_KEYWORDS.get(section_intent, [])):
                score += 3.0
            
            # Score based on query word matches
            sentence_words = set(sentence_lower.split())
            common_words = query_words.intersection(sentence_words)
            score += len(common_words) * 0.5
            
            # Score based on content relevance
            if any(word in sentence_lower for word in ["take", "dose", "tablet", "mg", "hour"]):
                score += 1.0
            
            if score > 0:
                candidates.append((score, sentence.strip()))
    
    # Sort by score and remove duplicates
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    seen = set()
    result = []
    for _, sentence in candidates:
        sentence_clean = re.sub(r'\s+', ' ', sentence).strip()
        sentence_lower = sentence_clean.lower()
        
        if sentence_lower not in seen and len(sentence_clean) > 15:
            result.append(sentence_clean)
            seen.add(sentence_lower)
        
        if len(result) >= top_n:
            break
    
    return result

# Streamlit UI
st.set_page_config(page_title="💊 Medingen RAG Chatbot", layout="centered")
st.title("💬 Medingen RAG Chatbot")
st.markdown("Ask questions based on Medingen's product details")

# Load vectorstore
try:
    vectorstore = load_vectorstore()
    st.success("✅ Knowledge base loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading knowledge base: {e}")
    st.stop()

# Load products
try:
    with open(PRODUCT_LIST_JSON, "r") as f:
        products = json.load(f)
    products = ["none"] + sorted(products)
except Exception as e:
    st.error(f"❌ Error loading products: {e}")
    st.stop()

# Product selection
product_filter = st.selectbox("🧪 Filter by product:", products)
selected_product = normalize_product_name(product_filter)

if selected_product == "none":
    st.info("ℹ️ Select a product to enable querying.")
    st.stop()

# Query input
query = st.text_input("🔍 Ask your question below:", placeholder="e.g., What are the benefits? What are the side effects?")

if query:
    q_original = query.strip()
    q_clean = sanitize(q_original)
    
    # Input validation
    if contains_greeting_short(q_clean):
        st.info("👋 Hello! Please ask a specific question about the selected product (e.g., 'What are the benefits?' or 'What are the side effects?').")
        st.stop()
    
    if is_gibberish(q_clean):
        st.warning("⚠️ I couldn't understand that input. Please type a clear question about the selected product.")
        st.stop()
    
    # Detect section intent
    section_intent = detect_section_intent(q_clean)
    
    try:
        # Search for relevant documents
        retrieved = vectorstore.similarity_search_with_score(q_original, k=SEARCH_K)
    except Exception:
        # Fallback if scoring not available
        docs_only = vectorstore.similarity_search(q_original, k=SEARCH_K)
        retrieved = [(d, None) for d in docs_only]
    
    # Filter by selected product
    filtered = []
    for doc, score in retrieved:
        meta = doc.metadata or {}
        doc_product = normalize_product_name(meta.get("product"))
        if doc_product == selected_product:
            filtered.append((doc, score))
    
    if not filtered:
        st.warning("⚠️ No relevant information found for that product. Please try a different question or check if the product data is available.")
        st.stop()
    
    # Extract top documents
    top_docs = [doc for doc, _ in filtered[:RETURN_TOP_K]]
    
    # Process based on detected intent
    if section_intent == "benefits":
        st.markdown("### 💊 Benefits")
        benefit_content = extract_benefits_content(top_docs, top_n=8)
        
        if benefit_content:
            for i, content in enumerate(benefit_content, 1):
                st.write(f"• {content}")
        else:
            st.info("ℹ️ No specific benefits information found in the documents for this product.")
    
    elif section_intent == "side_effects":
        st.markdown("### ⚠️ Side Effects")
        side_effect_content = extract_side_effects_content(top_docs, top_n=8)
        
        if side_effect_content:
            for i, content in enumerate(side_effect_content, 1):
                st.write(f"• {content}")
        else:
            st.info("ℹ️ No specific side effects information found in the documents for this product.")
    
    else:
        # General query handling
        st.markdown("### 📋 Answer")
        general_content = extract_general_content(top_docs, q_original, section_intent, top_n=6)
        
        if general_content:
            for content in general_content:
                st.write(f"• {content}")
        else:
            st.info("ℹ️ No specific information found for your question. Please try rephrasing or ask about benefits, side effects, or dosage.")

    # Debug information (optional - can be removed in production)
    with st.expander("🔧 Debug Information", expanded=False):
        st.write(f"**Query:** {q_original}")
        st.write(f"**Detected Intent:** {section_intent}")
        st.write(f"**Product:** {selected_product}")
        st.write(f"**Documents Retrieved:** {len(filtered)}")
        
        if filtered:
            st.write("**Top Document Sections:**")
            for i, (doc, score) in enumerate(filtered[:3]):
                section = doc.metadata.get("section", "unknown")
                st.write(f"  {i+1}. Section: {section}, Score: {score}")
