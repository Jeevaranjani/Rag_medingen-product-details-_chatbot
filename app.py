
import streamlit as st
import json
import re
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_FOLDER = r"C:\Users\djeev\rag_medingen_chatbot\faiss_index"
PRODUCT_LIST_JSON = "product_list.json"

SIMILARITY_THRESHOLD = 0.45
SEARCH_K = 12
RETURN_TOP_K = 3

SECTION_KEYWORDS = {
    "side effects": ["side effect", "side-effects", "adverse reaction", "adverse event", "nausea", "vomit", "vomiting", "rash", "itch", "dizziness", "headache", "constipation"],
    "dosage": ["dose", "dosage", "take", "when to take", "intake", "before", "after", "daily", "every", "hour", "hr"],
    "benefits": ["benefit", "benefits", "use", "indication", "how it works"],
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


def extract_side_effect_sentences_from_docs(docs: List, top_n=4) -> List[str]:
    candidates = []
    for d in docs:
        text = (d.page_content or "").strip()
        for s in split_sentences(text):
            sl = s.lower()
            if any(kw in sl for kw in SECTION_KEYWORDS["side effects"]):
                candidates.append(s.strip())
    # deduplicate preserving order
    seen = set()
    out = []
    for s in candidates:
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if s_clean not in seen:
            out.append(s_clean)
            seen.add(s_clean)
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

    # simple score-based confidence check (internal only)
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

    # prepare top docs
    filtered_sorted = sorted(filtered, key=lambda x: (x[1] is not None, x[1] or 0), reverse=True) if any(s is not None for (_, s) in filtered) else filtered
    top_docs = [d for (d, _) in filtered_sorted][:RETURN_TOP_K]

    # If intent is side effects: return only side-effect sentences (concise), no supporting evidence shown
    if section_intent == "side effects":
        side_sentences = extract_side_effect_sentences_from_docs(top_docs, top_n=6)
        if not side_sentences:
            st.info("No side effects information found in the documents for this product.")
        else:
            # Combine into a short, readable block
            answer = " ".join(side_sentences)
            st.markdown("### Answer:")
            st.write(answer)
        st.stop()

    # For other intents: attempt concise extraction (no supporting evidence)
    concise = extract_concise_sentences_for_question(top_docs, q_original, section_intent, top_n=3)
    if not concise:
        st.info("No concise information found in the documents for that question.")
        st.stop()

    answer = " ".join(concise)
    st.markdown("### Answer:")
    st.write(answer)