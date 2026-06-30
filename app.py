import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import chromadb
import pypdf
import os

# ── Setup ──────────────────────────────────────────
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("hotel_rag")

# ── Helper Functions ───────────────────────────────
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        pdf = pypdf.PdfReader(uploaded_file)
        return " ".join(page.extract_text() for page in pdf.pages)
    return uploaded_file.read().decode("utf-8")

def split_into_chunks(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

def store_in_chromadb(chunks, filename):
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))]
    )

def search_chromadb(question, top_n=2):
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_n
    )
    return results["documents"][0], results["distances"][0]

def ask_gemini(question, context):
    prompt = f"""Use the following context to answer the question.
If the answer is not in the context say "I don't know".

Context:
{context}

Question: {question}

Answer:"""
    response = gemini_model.generate_content(prompt)
    return response.text

# ── Streamlit UI ───────────────────────────────────
st.title("🏨 Hotel RAG Application")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Document", "Ask Questions", "View ChromaDB"])

# ── Page 1: Upload Document ────────────────────────
if page == "Upload Document":
    st.header("📄 Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")

        with st.spinner("Processing document..."):
            # Extract text
            text = extract_text(uploaded_file)
            st.subheader("📝 Extracted Text Preview")
            st.text_area("First 500 characters:", text[:500], height=150)

            # Chunk the text
            chunks = split_into_chunks(text)
            st.subheader("✂️ Chunks Created")
            st.write(f"Total chunks: {len(chunks)}")

            for i, chunk in enumerate(chunks[:3]):
                st.info(f"**Chunk {i+1}:** {chunk[:200]}...")

            # Store in ChromaDB
            store_in_chromadb(chunks, uploaded_file.name)
            st.success(f"✅ {len(chunks)} chunks stored in ChromaDB!")

# ── Page 2: Ask Questions ──────────────────────────
elif page == "Ask Questions":
    st.header("💬 Ask Questions")
    question = st.text_input("Type your question here:")

    if question:
        with st.spinner("Searching ChromaDB..."):
            relevant_chunks, distances = search_chromadb(question)

        st.subheader("🔍 Retrieved Chunks")
        for i, (chunk, distance) in enumerate(zip(relevant_chunks, distances)):
            st.info(f"**Chunk {i+1}** (similarity: {1-distance:.2f}):\n{chunk}")

        with st.spinner("Getting answer from Gemini..."):
            context = "\n\n".join(relevant_chunks)
            answer = ask_gemini(question, context)

        st.subheader("💡 Answer")
        st.success(answer)

# ── Page 3: View ChromaDB ──────────────────────────
elif page == "View ChromaDB":
    st.header("🗄️ ChromaDB Contents")

    data = collection.get()

    if not data["documents"]:
        st.warning("No data stored yet! Upload a document first.")
    else:
        st.write(f"Total chunks stored: {len(data['documents'])}")
        st.subheader("Stored Chunks:")
        for i, (doc, id) in enumerate(zip(data["documents"], data["ids"])):
            with st.expander(f"Chunk {i+1} — {id}"):
                st.write(doc)