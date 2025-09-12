import os
import re
import json
from pathlib import Path
from typing import List

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_FOLDER = Path(os.environ.get("FAISS_INDEX_PATH", "faiss_index"))
PRODUCT_LIST_JSON = Path("product_list.json")

SIMILARITY_THRESHOLD = 0.45
SEARCH_K = 12
RETURN_TOP_K = 4

SECTION_KEYWORDS = {
    "side effects": [
        "side effect", "side-effects", "adverse reaction", "adverse event", "nausea",
        "vomit", "vomiting", "rash", "itch", "dizziness", "headache", "constipation",
        "insomnia", "fatigue", "allergic", "allergy", "swelling"
    ],
    "benefits": [
        "benefit", "benefits", "use", "indication", "how it works",
        "used for", "used to", "relieve", "relieves", "treats", "helps", "indicated"
    ],
    "dosage": ["dose", "dosage", "take", "when to take", "intake", "before", "after"],
}

GREETINGS = {"hi", "hello", "hey", "hii", "hiya"}


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not INDEX_FOLDER.exists():
        return None
    try:
        vs = FAISS.load_local(str(INDEX_FOLDER), embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception as e:
        st.session_state["_load_error"] = str(e)
        return None


def normalize_product_name(name: str) -> str:
    return name.strip().lower() if name else name


def sanitize(q: str) -> str:
    return re.sub(r"[^\w\s]", " ", q.lower()).strip()


def detect_section_intent(q_clean: str) -> str:
    # look for clear intent words, prefer exact matches and synonyms
    for section, keys in SECTION_KEYWORDS.items():
        for k in keys:
            if re.search(r"\b" + re.escape(k) + r"\b", q_clean):
                return section
    # fallback: simple heuristics
    if "side" in q_clean and "effect" in q_clean:
        return "side effects"
    if "benefit" in q_clean or "used for" in q_clean or "indicat" in q_clean:
        return "benefits"
    return ""


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[\.\?\!\n])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def looks_like_heading_noise(s: str) -> bool:
    tokens = re.findall(r"\w+", s.lower())
    if len(tokens) <= 2:
        return True
    # if it contains multiple section labels concatenated, treat as noise
    labels = sum(1 for label_list in SECTION_KEYWORDS.values() for k in label_list if k in s.lower())
    if labels >= 2:
        return True
    # lines that are all-capital-like or short are often headings (best-effort)
    if len(s) < 35 and s == s.title() and any(w in s.lower() for w in ["benefits", "side", "effects", "indication"]):
        return True
    return False


def extract_sentences_by_keywords(docs: List, keywords: List[str], top_n=4, require_keyword=True):
    # collect sentences containing any of keywords; deduplicate preserve order
    candidates = []
    kws = [k.lower() for k in keywords]
    for d in docs:
        text = (d.page_content or "").strip()
        for s in split_sentences(text):
            sl = s.lower()
            if looks_like_heading_noise(s):
                continue
            matched = any(re.search(r"\b" + re.escape(k) + r"\b", sl) for k in kws)
            if matched or not require_keyword:
                cleaned = re.sub(r'\s{2,}', ' ', s).strip()
                candidates.append(cleaned)
    # dedupe
    seen = set()
    out = []
    for s in candidates:
        key = s.lower()
        if key not in seen:
            out.append(s)
            seen.add(key)
        if len(out) >= top_n:
            break
    return out


def extract_benefits(docs: List, top_n=4):
    # First prefer docs labeled 'benefit' or 'indication'
    prioritized = [d for d in docs if "benefit" in (d.metadata or {}).get("section", "").lower() or "indicat" in (d.metadata or {}).get("section", "").lower()]
    search_docs = prioritized if prioritized else docs
    # look for explicit benefit patterns
    benefit_patterns = [
        r"\bis used for\b", r"\bused to\b", r"\bindicated for\b", r"\bhelps (to )?\b", r"\b(relieve|relieves|treats|for pain|for fever)\b"
    ]
    candidates = []
    for d in search_docs:
        for s in split_sentences((d.page_content or "").strip()):
            if looks_like_heading_noise(s):
                continue
            sl = s.lower()
            if any(re.search(p, sl) for p in benefit_patterns) or any(k in sl for k in SECTION_KEYWORDS["benefits"]):
                cleaned = re.sub(r'\b(side effects|how it works|expert\'s advice|frequently asked questions)\b', '', s, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
                candidates.append(cleaned)
    # fallback: search all docs for sentences that include "used for" or "is used" etc.
    if not candidates:
        candidates = extract_sentences_by_keywords(docs, SECTION_KEYWORDS["benefits"], top_n=top_n, require_keyword=False)
    # dedupe and trim
    seen = set()
    out = []
    for s in candidates:
        k = s.lower()
        if k not in seen:
            out.append(s)
            seen.add(k)
        if len(out) >= top_n:
            break
    return out


def extract_side_effects(docs: List, top_n=4):
    prioritized = [d for d in docs if "side" in (d.metadata or {}).get("section", "").lower() or "adverse" in (d.metadata or {}).get("section", "").lower()]
    search_docs = prioritized if prioritized else docs
    candidates = extract_sentences_by_keywords(search_docs, SECTION_KEYWORDS["side effects"], top_n=top_n, require_keyword=True)
    # fallback: broaden search in all docs
    if not candidates:
        candidates = extract_sentences_by_keywords(docs, SECTION_KEYWORDS["side effects"], top_n=top_n, require_keyword=False)
    return candidates


# UI
st.set_page_config(page_title="💊 Medingen RAG Chatbot", layout="centered")
st.title("💬 Medingen RAG Chatbot")
st.markdown("Ask questions based on Medingen's product details")

vectorstore = load_vectorstore()
if vectorstore is None:
    load_err = st.session_state.get("_load_error", "")
    st.error("Vector index not available. Ensure faiss_index exists and is uploaded or FAISS_INDEX_PATH is set.")
    if load_err:
        st.info(load_err)
    st.stop()

if not PRODUCT_LIST_JSON.exists():
    st.error("Missing product_list.json in the app folder.")
    st.stop()

with open(PRODUCT_LIST_JSON, "r") as f:
    products = json.load(f)

products = ["none"] + sorted(products)
product_filter = st.selectbox("🧪 Filter by product :", products)
selected_product = normalize_product_name(product_filter)
if selected_product == "none":
    st.info("Select a product to enable querying.")
    st.stop()

query = st.text_input("🔍 Ask your question below:", placeholder="e.g., What are the side effects?")
if not query or query.strip() == "":
    st.stop()

q_original = query.strip()
q_clean = sanitize(q_original)

if len(q_clean.split()) < 2:
    st.info("Please enter a clear full question.")
    st.stop()

if any(g in q_clean.split() for g in GREETINGS):
    st.info("Hello — please ask a specific question about the selected product.")
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

# If user asked specifically for side effects or benefits, use targeted extractors
if section_intent == "side effects":
    side_answers = extract_side_effects(top_docs, top_n=6)
    if not side_answers:
        # broaden search across top_docs more aggressively
        side_answers = extract_side_effects([d for (d, _) in filtered], top_n=6)
    if not side_answers:
        st.info("No side effects information found in the documents for this product.")
        st.stop()
    st.markdown("### Answer:")
    for s in side_answers:
        st.write("- " + s)
    st.stop()

if section_intent == "benefits":
    ben_answers = extract_benefits(top_docs, top_n=6)
    if not ben_answers:
        ben_answers = extract_benefits([d for (d, _) in filtered], top_n=6)
    if not ben_answers:
        # last-resort: try concise extraction with benefit keyword boost
        concise = extract_sentences_by_keywords([d for (d, _) in filtered], SECTION_KEYWORDS["benefits"], top_n=4, require_keyword=False)
        if not concise:
            st.info("No benefits information found in the documents for this product.")
            st.stop()
        ben_answers = concise
    st.markdown("### Answer:")
    for s in ben_answers:
        st.write("- " + s)
    st.stop()

# fallback: general concise extraction
# boost sentences that contain intent words if any
candidates = []
for d in top_docs:
    text = (d.page_content or "").strip()
    for s in split_sentences(text):
        sl = s.lower()
        score = 0
        if section_intent and any(k in sl for k in SECTION_KEYWORDS.get(section_intent, [])):
            score += 3
        if any(w in sl for w in q_clean.split()):
            score += 1
        if score > 0:
            candidates.append((score, s))
candidates.sort(key=lambda x: x[0], reverse=True)
seen = set()
out = []
for _, s in candidates:
    key = s.strip().lower()
    if key not in seen and not looks_like_heading_noise(s):
        out.append(s.strip())
        seen.add(key)
    if len(out) >= 4:
        break

if not out:
    st.info("No concise information found in the documents for that question.")
    st.stop()

st.markdown("### Answer:")
for s in out:
    st.write("- " + s)