import streamlit as st
import json
import re
from typing import List, Tuple
from pathlib import Path
import os

from langchain.document_loaders import PyPDFLoader  # (not used directly here, kept for reference)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

INDEX_FOLDER = Path(os.environ.get("FAISS_INDEX_PATH", "faiss_index"))
PRODUCT_LIST_JSON = Path("product_list.json")

SIMILARITY_THRESHOLD = 0.45
SEARCH_K = 16        # slightly larger pool
RETURN_TOP_K = 3

# Extended keywords for intent detection and sentence extraction
SECTION_KEYWORDS = {
    "side effects": [
        "side effect", "side-effects", "adverse reaction", "adverse reactions", "adverse event", "adverse events",
        "nausea", "vomit", "vomiting", "rash", "itch", "dizziness", "headache", "constipation",
        "insomnia", "fatigue", "allergic", "allergy", "swelling", "bleeding", "anaphylaxis"
    ],
    "dosage": [
        "dose", "dosage", "administration", "when to take", "when should", "posology",
        "intake", "before", "after", "daily", "every", "hour", "hr", "tablet", "tablets", "capsule", "capsules"
    ],
    "benefits": [
        "benefit", "benefits", "use", "uses", "indication", "indications", "how it works",
        "used for", "relieve", "relieves", "treats", "helps", "pain relief", "fever", "symptom"
    ],
}

GREETINGS = {"hi", "hello", "hey", "hii", "hiya"}

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

def normalize_product_name(name: str) -> str:
    return name.strip().lower() if name else name

def sanitize(q: str) -> str:
    return re.sub(r"[^\w\s]", " ", q.lower()).strip()

def is_gibberish(q_clean: str) -> bool:
    tokens = re.findall(r"\w+", q_clean)
    if not tokens:
        return True
    short_tokens = [t for t in tokens if len(t) <= 2]
    if len(short_tokens) / max(1, len(tokens)) > 0.7:
        return True
    return False

def contains_greeting_short(q_clean: str) -> bool:
    tokens = re.findall(r"\w+", q_clean)
    return bool(tokens and tokens[0] in GREETINGS and len(tokens) <= 3)

def detect_section_intent(q_clean: str) -> str:
    # stronger detection using word boundaries and synonyms
    if not q_clean:
        return ""
    for section, keys in SECTION_KEYWORDS.items():
        for k in keys:
            if re.search(r"\b" + re.escape(k) + r"\b", q_clean):
                return section
    # some common question forms
    if re.search(r"\b(benefit|benefits|used for|indicated for|what is it used)\b", q_clean):
        return "benefits"
    if re.search(r"\b(side effect|side effects|adverse|adverse reaction|adverse event)\b", q_clean):
        return "side effects"
    if re.search(r"\b(dose|dosage|how many|how much|when to take)\b", q_clean):
        return "dosage"
    return ""

def split_sentences(text: str) -> List[str]:
    # simple sentence splitter: treat line breaks as well as punctuation
    text = text.replace("\r", " ")
    sentences = re.split(r'(?<=[\.\?\!\n])\s+', text.strip())
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned

def _looks_like_heading_noise(s: str) -> bool:
    # Be conservative: only treat very short uppercase/label-like lines as noise
    if not s:
        return True
    # If very short (<=3 tokens) and ends with ':' or is all-uppercase tokens, treat as heading
    tokens = re.findall(r"\w+", s)
    if len(tokens) <= 3:
        if s.strip().endswith(":"):
            return True
        uppercase_tokens = sum(1 for t in tokens if t.isupper())
        if uppercase_tokens >= len(tokens) and len(tokens) >= 1:
            return True
    return False

def extract_side_effect_sentences_from_docs(docs: List, top_n=6) -> List[str]:
    symptom_keywords = set(kw.lower() for kw in SECTION_KEYWORDS["side effects"])
    prioritized_docs = []
    fallback_docs = []
    for d in docs:
        sec = (d.metadata or {}).get("section", "") or ""
        sec = sec.lower()
        if "side" in sec or "effect" in sec or "adverse" in sec:
            prioritized_docs.append(d)
        else:
            fallback_docs.append(d)
    search_docs = prioritized_docs if prioritized_docs else docs

    candidates = []
    for d in search_docs:
        text = (d.page_content or "").strip()
        for s in split_sentences(text):
            if _looks_like_heading_noise(s):
                continue
            sl = s.lower()
            # match explicit mentions of adverse events, symptom keywords, or phrasing indicating "may cause"
            if (any(kw in sl for kw in symptom_keywords)
                or re.search(r"\b(may cause|can cause|can lead to|might cause|side effect|adverse|adversely)\b", sl)):
                cleaned = re.sub(r'\s{2,}', ' ', s).strip()
                candidates.append(cleaned)

    # fallback: expand search to fallback_docs if earlier search produced nothing
    if not candidates and fallback_docs:
        for d in fallback_docs:
            text = (d.page_content or "").strip()
            for s in split_sentences(text):
                if _looks_like_heading_noise(s):
                    continue
                sl = s.lower()
                if (any(kw in sl for kw in symptom_keywords)
                    or re.search(r"\b(may cause|can cause|side effect|adverse)\b", sl)):
                    cleaned = re.sub(r'\s{2,}', ' ', s).strip()
                    candidates.append(cleaned)

    # Deduplicate and return top_n
    seen = set()
    out = []
    for s in candidates:
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if s_clean and s_clean.lower() not in seen:
            out.append(s_clean)
            seen.add(s_clean.lower())
        if len(out) >= top_n:
            break
    return out

def extract_benefit_sentences_from_docs(docs: List, top_n=6) -> List[str]:
    benefit_keywords = set(kw.lower() for kw in SECTION_KEYWORDS["benefits"])
    prioritized_docs = []
    fallback_docs = []
    for d in docs:
        sec = (d.metadata or {}).get("section", "") or ""
        sec = sec.lower()
        if "benefit" in sec or "indication" in sec or "use" in sec:
            prioritized_docs.append(d)
        else:
            fallback_docs.append(d)
    search_docs = prioritized_docs if prioritized_docs else docs

    candidates = []
    for d in search_docs:
        text = (d.page_content or "").strip()
        for s in split_sentences(text):
            if _looks_like_heading_noise(s):
                continue
            sl = s.lower()
            # prefer sentences containing usage/benefit patterns
            if (any(kw in sl for kw in benefit_keywords)
                or re.search(r"\b(is used for|used to|indicated for|helps|relieve|relieves|treats|for pain|for fever|pain relief)\b", sl)):
                cleaned = re.sub(r'\s{2,}', ' ', s).strip()
                candidates.append(cleaned)

    # fallback: broaden to any sentence in fallback docs that mentions 'used for' or 'indicated for'
    if not candidates and fallback_docs:
        for d in fallback_docs:
            text = (d.page_content or "").strip()
            for s in split_sentences(text):
                if _looks_like_heading_noise(s):
                    continue
                sl = s.lower()
                if re.search(r"\b(is used for|used to|indicated for|helps|relieve|treats)\b", sl):
                    cleaned = re.sub(r'\s{2,}', ' ', s).strip()
                    candidates.append(cleaned)

    seen = set()
    out = []
    for s in candidates:
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if s_clean and s_clean.lower() not in seen:
            out.append(s_clean)
            seen.add(s_clean.lower())
        if len(out) >= top_n:
            break
    return out

def extract_concise_sentences_for_question(docs: List, question: str, section_intent: str, top_n=3) -> List[str]:
    candidates = []
    ql = question.lower()
    q_words = set(re.findall(r"\w+", ql))
    for d in docs:
        text = (d.page_content or "").strip()
        for s in split_sentences(text):
            if _looks_like_heading_noise(s):
                continue
            sc = 0.0
            sl = s.lower()
            # strong boost if sentence contains section keywords
            if section_intent and any(kw in sl for kw in SECTION_KEYWORDS.get(section_intent, [])):
                sc += 3.0
            # boost for dosage words if looking for dosage
            if any(word in sl for word in ["take", "dose", "dosage", "tablet", "capsule", "administer", "hour", "hr", "daily"]):
                sc += 1.5
            # boost for numeric measurement
            if re.search(r"\b\d+\s*(mg|g|ml|tablets|capsules|hours|hrs|minutes)\b", sl):
                sc += 1.5
            # overlap with query words
            overlap = sum(1 for w in q_words if w in sl)
            sc += 0.3 * overlap
            if sc > 0:
                candidates.append((sc, s.strip()))
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out = []
    for _, s in candidates:
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if s_clean and s_clean.lower() not in seen:
            out.append(s_clean)
            seen.add(s_clean.lower())
        if len(out) >= top_n:
            break
    return out

def retrieve_and_prioritize_docs(vectorstore, product: str, query: str, section_intent: str, base_k=SEARCH_K) -> List:
    # 1) initial retrieval
    try:
        retrieved = vectorstore.similarity_search_with_score(query, k=base_k, filter=None)
    except TypeError:
        docs_only = vectorstore.similarity_search(query, k=base_k)
        retrieved = [(d, None) for d in docs_only]

    # strict product filtering
    filtered = []
    for doc, score in retrieved:
        meta = doc.metadata or {}
        doc_product = normalize_product_name(meta.get("product"))
        if doc_product == product:
            filtered.append((doc, score))

    # if section_intent asked, prioritize docs whose metadata.section matches intent
    if section_intent and filtered:
        preferred = []
        others = []
        for doc, score in filtered:
            sec = (doc.metadata or {}).get("section", "") or ""
            if section_intent in sec.lower():
                preferred.append((doc, score))
            else:
                others.append((doc, score))
        if preferred:
            # return preferred first (deduplicated) up to base_k
            merged = preferred + others
            docs = [d for (d, _) in merged][:base_k]
            return docs

    # If not enough or no section-matching docs, run an augmented search using the section intent as a prefix
    if section_intent:
        augmented_query = f"{section_intent} {query}"
        try:
            aug_retrieved = vectorstore.similarity_search_with_score(augmented_query, k=base_k, filter=None)
        except TypeError:
            docs_only = vectorstore.similarity_search(augmented_query, k=base_k)
            aug_retrieved = [(d, None) for d in docs_only]

        # merge augmented retrieval (preserve product filter)
        merged_by_id = {}
        for doc, score in (retrieved + aug_retrieved):
            meta = doc.metadata or {}
            doc_product = normalize_product_name(meta.get("product"))
            if doc_product != product:
                continue
            # de-dup by text+page
            key = (meta.get("product"), meta.get("page"), (doc.page_content or "")[:200])
            # prefer higher score if available
            if key not in merged_by_id:
                merged_by_id[key] = (doc, score)
            else:
                # replace if new score is better (when scores are numeric)
                existing_score = merged_by_id[key][1]
                try:
                    if score is not None and existing_score is not None and score > existing_score:
                        merged_by_id[key] = (doc, score)
                except Exception:
                    pass
        merged = list(merged_by_id.values())
        # sort by score when available
        merged_sorted = sorted(merged, key=lambda x: (x[1] is not None, x[1] or 0), reverse=True)
        docs = [d for (d, _) in merged_sorted][:base_k]
        return docs

    # default: return top filtered documents (docs only)
    docs = [d for (d, _) in filtered][:base_k]
    return docs

# UI
st.set_page_config(page_title="💊 Medingen RAG Chatbot", layout="centered")
st.title("💬 Medingen RAG Chatbot")
st.markdown("Ask questions based on Medingen's product details")

vectorstore = load_vectorstore()

with open(PRODUCT_LIST_JSON, "r") as f:
    products = json.load(f)

products = ["none"] + sorted(products)
product_filter = st.selectbox("🧪 Filter by product :", products)
selected_product = normalize_product_name(product_filter)
if selected_product == "none":
    st.info("Select a product to enable querying.")
    st.stop()

query = st.text_input("🔍 Ask your question below:", placeholder="e.g., What are the side effects?")

if query:
    q_original = query.strip()
    q_clean = sanitize(q_original)

    if contains_greeting_short(q_clean):
        st.info("Hello — please ask a specific question about the selected product (e.g., 'What are the side effects?').")
        st.stop()

    if is_gibberish(q_clean):
        st.warning("I couldn't understand that input. Please type a clear question about the selected product.")
        st.stop()

    section_intent = detect_section_intent(q_clean)

    # retrieve and prioritize docs (product + section-aware)
    docs_for_question = retrieve_and_prioritize_docs(vectorstore, selected_product, q_original, section_intent, base_k=SEARCH_K)

    if not docs_for_question:
        st.warning("No relevant information found for that product with your query.")
        st.stop()

    # Attempt specialized extractors when intent is clear
    if section_intent == "side effects":
        side_sentences = extract_side_effect_sentences_from_docs(docs_for_question, top_n=6)
        if not side_sentences:
            # fallback to concise extraction with a slight expansion
            concise = extract_concise_sentences_for_question(docs_for_question, q_original, section_intent, top_n=3)
            if not concise:
                st.info("No side effects information found in the documents for this product.")
                st.stop()
            else:
                st.markdown("### Answer (best-effort):")
                for s in concise:
                    st.write("- " + s)
                st.stop()
        st.markdown("### Answer:")
        for s in side_sentences:
            st.write("- " + s)
        st.stop()

    if section_intent == "benefits":
        benefit_sentences = extract_benefit_sentences_from_docs(docs_for_question, top_n=6)
        if not benefit_sentences:
            concise = extract_concise_sentences_for_question(docs_for_question, q_original, section_intent, top_n=3)
            if not concise:
                st.info("No benefits information found in the documents for this product.")
                st.stop()
            else:
                st.markdown("### Answer (best-effort):")
                for s in concise:
                    st.write("- " + s)
                st.stop()
        st.markdown("### Answer:")
        for s in benefit_sentences:
            st.write("- " + s)
        st.stop()

    # fallback: general concise extraction
    concise = extract_concise_sentences_for_question(docs_for_question, q_original, section_intent, top_n=3)
    if not concise:
        st.info("No concise information found in the documents for that question.")
        st.stop()

    st.markdown("### Answer:")
    for s in concise:
        st.write("- " + s)