import streamlit as st
import json
import re
from typing import List, Tuple
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from pathlib import Path

INDEX_FOLDER = Path(os.environ.get("FAISS_INDEX_PATH", "faiss_index"))
PRODUCT_LIST_JSON = Path("product_list.json")

SIMILARITY_THRESHOLD = 0.45
SEARCH_K = 12
RETURN_TOP_K = 3

SECTION_KEYWORDS = {
    "side effects": ["side effect", "side-effects", "adverse reaction", "adverse event", "nausea", "vomit", "vomiting", "rash", "itch", "dizziness", "headache", "constipation", "insomnia", "fatigue", "allergic", "allergy", "swelling"],
    "dosage": ["dose", "dosage", "take", "when to take", "intake", "before", "after", "daily", "every", "hour", "hr"],
    "benefits": ["benefit", "benefits", "use", "indication", "how it works", "used for", "relieve", "relieves", "treats", "helps"],
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
    for section, keys in SECTION_KEYWORDS.items():
        for k in keys:
            if k in q_clean:
                return section
    return ""


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _looks_like_heading_noise(s: str) -> bool:
    tokens = re.findall(r"\w+", s.lower())
    if len(tokens) <= 2:
        return True
    section_tokens = 0
    for label_keys in SECTION_KEYWORDS.values():
        for k in label_keys:
            if k in s.lower():
                section_tokens += 1
    if section_tokens >= 2:
        return True
    known_labels = ["benefits", "how", "works", "side", "effects", "expert", "advice", "frequently", "asked", "questions"]
    count_label_like = sum(1 for t in tokens if t in known_labels)
    if count_label_like / max(1, len(tokens)) > 0.4:
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
            if any(kw in sl for kw in symptom_keywords) or "side effect" in sl or "adverse" in sl:
                cleaned = re.sub(r'\b(benefits|how it works|side effects|expert\'s advice|frequently asked questions)\b', '', s, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
                candidates.append(cleaned)
    if not candidates and prioritized_docs and fallback_docs:
        for d in fallback_docs:
            text = (d.page_content or "").strip()
            for s in split_sentences(text):
                if _looks_like_heading_noise(s):
                    continue
                sl = s.lower()
                if any(kw in sl for kw in symptom_keywords) or "side effect" in sl or "adverse" in sl:
                    cleaned = re.sub(r'\b(benefits|how it works|side effects|expert\'s advice|frequently asked questions)\b', '', s, flags=re.IGNORECASE)
                    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
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


def extract_benefit_sentences_from_docs(docs: List, top_n=6) -> List[str]:
    # Keywords to identify benefit/usage sentences
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
                cleaned = re.sub(r'\b(side effects|how it works|expert\'s advice|frequently asked questions)\b', '', s, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
                candidates.append(cleaned)
    # fallback: broaden to any sentence in fallback docs that mentions 'used for' or 'is used'
    if not candidates and fallback_docs:
        for d in fallback_docs:
            text = (d.page_content or "").strip()
            for s in split_sentences(text):
                if _looks_like_heading_noise(s):
                    continue
                sl = s.lower()
                if re.search(r"\b(is used for|used to|indicated for|helps|relieve|relieves|treats|for pain|for fever|pain relief)\b", sl):
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
    for d in docs:
        text = (d.page_content or "").strip()
        for s in split_sentences(text):
            sc = 0
            sl = s.lower()
            if section_intent and any(kw in sl for kw in SECTION_KEYWORDS.get(section_intent, [])):
                sc += 3.0
            if any(word in sl for word in ["take", "dose", "dosage", "tablet", "capsule", "administer", "hour", "hr", "minute", "daily"]):
                sc += 1.5
            if any(num_kw in sl for num_kw in ["mg", "g", "ml", "tablets", "capsules", "hours", "hrs", "minutes"]):
                sc += 1.5
            if any(w in sl for w in ql.split()):
                sc += 0.5
            if sc > 0:
                candidates.append((sc, s.strip()))
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out = []
    for _, s in candidates:
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if s_clean not in seen:
            out.append(s_clean)
            seen.add(s_clean)
        if len(out) >= top_n:
            break
    return out


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

    try:
        retrieved = vectorstore.similarity_search_with_score(q_original, k=SEARCH_K, filter=None)
    except TypeError:
        docs_only = vectorstore.similarity_search(q_original, k=SEARCH_K)
        retrieved = [(d, None) for d in docs_only]

    # strict product filtering
    filtered = []
    for doc, score in retrieved:
        meta = doc.metadata or {}
        doc_product = normalize_product_name(meta.get("product"))
        if doc_product == selected_product:
            filtered.append((doc, score))

    if not filtered:
        st.warning("No relevant information found for that product with your query.")
        st.stop()

    # internal confidence check
    scores = [s for (_, s) in filtered if s is not None]
    if scores:
        s0 = scores[0]
        if isinstance(s0, (int, float)) and -1.0 <= s0 <= 1.0:
            if max(scores) < SIMILARITY_THRESHOLD:
                st.warning("I couldn't find good evidence. Please rephrase.")
                st.stop()
        else:
            if min(scores) > 1.9:
                st.warning("I couldn't find good evidence. Please rephrase.")
                st.stop()

    filtered_sorted = sorted(filtered, key=lambda x: (x[1] is not None, x[1] or 0), reverse=True) if any(s is not None for (_, s) in filtered) else filtered
    top_docs = [d for (d, _) in filtered_sorted][:RETURN_TOP_K]

    # If user intent is specifically side effects or benefits, use the dedicated extractors
    if section_intent == "side effects":
        side_sentences = extract_side_effect_sentences_from_docs(top_docs, top_n=6)
        if not side_sentences:
            st.info("No side effects information found in the documents for this product.")
            st.stop()
        st.markdown("### Answer:")
        for s in side_sentences:
            st.write("- " + s)
        st.stop()

    if section_intent == "benefits":
        benefit_sentences = extract_benefit_sentences_from_docs(top_docs, top_n=6)
        if not benefit_sentences:
            st.info("No benefits information found in the documents for this product.")
            st.stop()
        st.markdown("### Answer:")
        for s in benefit_sentences:
            st.write("- " + s)
        st.stop()

    # fallback: general concise extraction
    concise = extract_concise_sentences_for_question(top_docs, q_original, section_intent, top_n=3)
    if not concise:
        st.info("No concise information found in the documents for that question.")
        st.stop()

    st.markdown("### Answer:")
    for s in concise:
        st.write("- " + s)