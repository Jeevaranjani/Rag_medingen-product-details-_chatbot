import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json

# Set FAISS index path
INDEX_FOLDER = r"C:\Users\djeev\rag_medingen_chatbot\faiss_index"

# Load FAISS index with embeddings
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

# Streamlit UI setup
st.set_page_config(page_title="💊 Medingen PDF Chatbot", layout="centered")
st.title("💬 Medingen RAG Chatbot")
st.markdown("Ask questions based on Medingen's product details")

# Load index
vectorstore = load_vectorstore()

with open("product_list.json", "r") as f:
    available_products = json.load(f)
available_products.insert(0, "none")


# Product selector UI
product_filter = st.selectbox("🧪 Filter by product :", available_products)

if product_filter == "none":
    st.warning("⚠️ Please select a valid product from the dropdown to proceed.")
    st.stop()

# User query input
query = st.text_input("🔍 Ask your question below:", placeholder="About medicine details")


# Search and display results
if query:
    selected_product = product_filter.lower() if product_filter else None
    query_lower = query.lower()

    # Block search when 'none' is selected
    if selected_product is None:
        st.warning("You selected 'none' as the product filter. Please choose a valid product to search.")
        st.stop()

    # Apply metadata filter
    metadata_filter = {"product": selected_product} if selected_product else None

    # Perform search
    results = vectorstore.similarity_search(query, k=3, filter=metadata_filter)
    

    if results:
        st.markdown("📄 Top Matches:")
        for i, doc in enumerate(results, 1):
            st.markdown(f"**Match {i} (Product: {doc.metadata.get('product', 'N/A')}):**")
            st.write(doc.page_content[:500] + "...")
    else:
        st.warning(f"No relevant information found for **'{query}'** in **'{product_filter}'**")



