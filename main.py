import os
import json
import sqlite3
import atexit
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv
import glob
import PyPDF2
import re
import nltk
from typing import List, Tuple, Dict, Any

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# ------------------ CONFIGURATION ------------------
DOCS_PATH = "pmc_docs"
OUT_DIR = "embeddings"
EMBED_FILE = os.path.join(OUT_DIR, "embeddings.npz")
META_FILE = os.path.join(OUT_DIR, "metadata.json")
FAISS_INDEX_FILE = os.path.join(OUT_DIR, "faiss_index.index")
DB_PATH = "rag_store.db"

EMBED_MODEL = "BAAI/bge-small-en"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 2000  # characters per chunk
OVERLAP = 200      # overlap between chunks

load_dotenv()

# Predefined Gemini model - using Gemini 1.5 Flash for best performance
GEMINI_MODEL = "gemini-2.0-flash"  # Fixed model - no env var needed
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # must be set

if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in environment before running the server.")

# Enhanced System prompt for medical/scientific Q&A with JSON output requirement
SYSTEM_PROMPT = """You are a precise medical and scientific research assistant. Your role is to provide accurate, evidence-based answers using ONLY the provided source documents.

CRITICAL GUIDELINES:
1. Base your answer STRICTLY on the provided source paragraphs - do not use external knowledge
2. If the information is not in the sources, explicitly state "This information is not available in the provided sources"
3. Cite sources inline using the format: [Source: {filename}, page {page}]
4. Be concise and factual - avoid speculation or personal opinions
5. For medical information, emphasize consulting healthcare professionals
6. Maintain scientific accuracy and precision
7. Don't output answer that is unrelated to the query

IMPORTANT OUTPUT FORMAT:
You MUST return your response as a valid JSON object with the following structure:
{
  "response": "Your main answer text here",
  "pdfs": [
    {"name": "filename1.pdf", "page": 1},
    {"name": "filename2.pdf", "page": 3}
  ]
}

The "pdfs" array should contain ALL source documents used in your answer, with their exact filenames and page numbers.
"""

# ------------------ DOCUMENT PROCESSING FUNCTIONS ------------------


def extract_text_from_pdf_with_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Extract text from PDF file with page numbers.
    Args:
    - pdf_path : Path of pdf

    Returns:
    - List of tuples (page_number, text)
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages_text = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                pages_text.append((page_num, text))
            return pages_text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return []


def clean_text_remove_citations(text: str) -> str:
    """Remove references and citation-like lines from text.
    Args:
    - text : The whole text of a pdf

    Returns:
    - text : The clean text of a pdf with no citations
    """
    lines = text.split("\n")
    cleaned_lines = []
    inside_references = False

    for line in lines:
        # Detect start of references section
        if re.search(r'^(references|bibliography|works cited)', line.strip(), re.IGNORECASE):
            inside_references = True
            break  # stop at references section

        # Skip lines that look like standalone citations
        if re.match(r'^\s*(\[\d+\]|^\d+\.|\(\d+\))', line.strip()):
            continue
        if re.search(r'(et al\.|doi:|http|arxiv|vol\.|pp\.|www)', line, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def chunk_by_context_with_pages(pages_text: List[Tuple[int, str]], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[Dict[str, Any]]:
    """Split text into chunks with sentence-aware overlapping chunks, preserving page info.
    Args:
    - pages_text : List of (page_number, text) tuples
    - chunk_size : Maximum chunk size
    - overlap    : Number of characters for overlap

    Returns:
    - chunks : List of dictionaries with chunk text and page info
    """
    from nltk.tokenize import sent_tokenize
    
    chunks = []
    
    for page_num, text in pages_text:
        # Clean the page text
        cleaned_text = clean_text_remove_citations(text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if not cleaned_text:
            continue
            
        sentences = sent_tokenize(cleaned_text)
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_length = len(sent)
            
            # If can fit, then fit the sentence inside
            if current_length + sent_length <= chunk_size:
                current_chunk.append(sent)
                current_length += sent_length
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "filename": ""  # Will be set later
                    })
                
                # Handle overlap
                if overlap > 0:
                    overlap_chars = 0
                    overlap_sentences = []
                    for overlap_sent in reversed(current_chunk):
                        if overlap_chars + len(overlap_sent) <= overlap:
                            overlap_sentences.insert(0, overlap_sent)
                            overlap_chars += len(overlap_sent)
                        else:
                            break
                    current_chunk = overlap_sentences + [sent]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sent]
                    current_length = sent_length
        
        # Add the last chunk for this page
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "page": page_num,
                "filename": ""  # Will be set later
            })
    
    return chunks


def load_and_chunk_documents() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load all documents from DOCS_PATH, chunk them, and return chunks and doc_ids.
    Returns:
    - documents : List of chunk dictionaries with text, page, and filename
    - doc_ids   : All chunk id in the format of 0000.pdf_chunk1
    """
    documents = []
    doc_ids = []
    
    if not os.path.exists(DOCS_PATH):
        print(f"⚠️ Documents directory '{DOCS_PATH}' not found. Creating empty directory.")
        os.makedirs(DOCS_PATH, exist_ok=True)
        return documents, doc_ids
    
    # Get all PDF files
    pdf_files = glob.glob(os.path.join(DOCS_PATH, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in '{DOCS_PATH}'. Starting with empty documents.")
        return documents, doc_ids
    
    print(f"📚 Found {len(pdf_files)} PDF files. Processing...")
    
    for pdf_file in sorted(pdf_files):  # Sort for consistent ordering
        filename = os.path.basename(pdf_file)
        
        # Extract text from PDF with page numbers
        pages_text = extract_text_from_pdf_with_pages(pdf_file)
        
        if not pages_text:
            print(f"    ⚠️ No text extracted from {filename}")
            continue
        
        # Chunk text using sentence-aware chunking with overlap, preserving page info
        chunks = chunk_by_context_with_pages(pages_text, CHUNK_SIZE, OVERLAP)
        
        # Create doc_ids and add to collections
        for i, chunk in enumerate(chunks):
            chunk["filename"] = filename  # Set filename in chunk data
            doc_id = f"{filename}_chunk{i}_page{chunk['page']}"
            documents.append(chunk)
            doc_ids.append(doc_id)
        
    
    print(f"📊 Total: {len(documents)} chunks from {len(pdf_files)} files")
    return documents, doc_ids


def generate_embeddings(documents: List[Dict[str, Any]]) -> np.ndarray:
    """Generate embeddings for all documents.
    Args:
    - documents : List of chunk dictionaries

    Returns: 
    - embeddings : 2D vectors of total_chunks x embedding_dimension 
    """
    print(f"🔮 Generating embeddings using {EMBED_MODEL}...")
    
    # Extract just the text for embedding
    texts = [doc["text"] for doc in documents]
    
    # Initialize embedder
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    
    # Generate embeddings in batches
    embeddings = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, doc_ids: List[str], documents: List[Dict[str, Any]]):
    """Save embeddings and metadata to files."""
    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Save embeddings
    np.savez_compressed(EMBED_FILE, embeddings=embeddings)
    
    # Save metadata
    metadata = {
        "doc_ids": doc_ids,
        "docs": documents,  # Now contains full chunk objects with page info
        "embed_model": EMBED_MODEL,
        "total_chunks": len(documents),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP
    }
    
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved embeddings to {EMBED_FILE}")
    print(f"💾 Saved metadata to {META_FILE}")

# ------------------ INITIALIZATION ------------------

print(f"⚡ Using device: {DEVICE}")
print(f"🤖 Using Gemini model: {GEMINI_MODEL}")
print("🔁 Starting server — setting up documents and embeddings...")

# Remove existing DB on every startup
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# Create DB and table
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT,
        filename TEXT,
        chunk_index INTEGER,
        page INTEGER,
        chunk_text TEXT
    )
    """
)
conn.commit()

# Global mapping from original doc_id to new 4-digit coded doc_id
doc_id_mapping = {}

def parse_doc_id(doc_id):
    """Parse doc_id to extract filename, chunk index, and page.
    Args:
    - doc_id : Document id 

    Returns:
    - filename    : File name of chunk
    - chunk_index : Chunk index
    - page        : Page of chunk
    """

    if "_chunk" in doc_id and "_page" in doc_id:
        # Format: filename_chunkX_pageY
        base_part = doc_id.rsplit("_chunk", 1)[0]
        rest = doc_id.rsplit("_chunk", 1)[1]
        chunk_part = rest.split("_page")[0]
        page_part = rest.split("_page")[1]
        try:
            chunk_index = int(chunk_part)
            page = int(page_part)
        except ValueError:
            chunk_index = None
            page = None
        filename = base_part

    elif "_chunk" in doc_id:
        parts = doc_id.rsplit("_chunk", 1)
        filename = parts[0]
        try:
            chunk_index = int(parts[1])
        except ValueError:
            chunk_index = None
        page = None
    else:
        filename = doc_id
        chunk_index = None
        page = None
    
    return filename, chunk_index, page


def assign_4digit_codes(doc_ids):
    """Assign 4-digit codes to documents in alphabetical order.
    Args:
    - Doc ids : Doc id {filename}_chunk{chunk_index}_page{page}
    
    Return:
    - Mapping : Mapping of doc ids to coded doc ids {code}_chunk{chunk_index}_page{page}
    
    """
    filenames = set()
    for doc_id in doc_ids:
        filename, _, _ = parse_doc_id(doc_id)
        filenames.add(filename)
    
    sorted_filenames = sorted(list(filenames))
    filename_to_code = {}
    
    for i, filename in enumerate(sorted_filenames, 1):
        code = f"{i:04d}"
        filename_to_code[filename] = code
    
    mapping = {}
    for doc_id in doc_ids:
        filename, chunk_index, page = parse_doc_id(doc_id)
        code = filename_to_code[filename]
        
        if chunk_index is not None and page is not None:
            new_doc_id = f"{code}_chunk{chunk_index}_page{page}"
        elif chunk_index is not None:
            new_doc_id = f"{code}_chunk{chunk_index}"
        elif page is not None:
            new_doc_id = f"{code}_page{page}"
        else:
            new_doc_id = code
            
        mapping[doc_id] = new_doc_id
    
    return mapping


# Load or generate embeddings and metadata
documents = []  # Now list of dictionaries
doc_ids = []
embeddings = None
faiss_index = None

if os.path.exists(META_FILE) and os.path.exists(EMBED_FILE):
    print("✅ Found existing embeddings. Loading...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
        doc_ids = meta.get("doc_ids", [])
        documents = meta.get("docs", [])  # Now contains full chunk objects
    
    doc_id_mapping = assign_4digit_codes(doc_ids)
    
    npz = np.load(EMBED_FILE, allow_pickle=True)
    embeddings = npz["embeddings"]
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))
    print(f"✅ Loaded {len(documents)} chunks and embeddings (dim={dim}).")

else:
    print("🔄 No existing embeddings found. Generating new embeddings...")
    documents, doc_ids = load_and_chunk_documents()
    
    if documents:
        embeddings = generate_embeddings(documents)
        save_embeddings(embeddings, doc_ids, documents)
        doc_id_mapping = assign_4digit_codes(doc_ids)
        
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(np.array(embeddings))
        print(f"✅ Created FAISS index with {len(documents)} chunks (dim={dim}).")
    else:
        print("⚠️ No documents processed. Starting with empty index.")
        faiss_index = None

# Populate database with documents
if documents and doc_ids:
    print("💾 Populating database with documents...")
    for original_doc_id, chunk_data in zip(doc_ids, documents):
        new_doc_id = doc_id_mapping.get(original_doc_id, original_doc_id)
        filename, chunk_index, page = parse_doc_id(new_doc_id)
        # Use the actual page from chunk data if available
        actual_page = chunk_data.get("page", page)
        chunk_text = chunk_data.get("text", "")
        
        cur.execute(
            "INSERT INTO documents (doc_id, filename, chunk_index, page, chunk_text) VALUES (?, ?, ?, ?, ?)",
            (new_doc_id, filename, chunk_index, actual_page, chunk_text)
        )
    conn.commit()
    print(f"✅ Database populated with {len(documents)} documents.")

# Initialize embedder
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

# Initialize Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Flask app
app = Flask(__name__)
CORS(app)

# Cleanup on exit
@atexit.register
def close_db():
    try:
        conn.commit()
        conn.close()
    except Exception:
        pass

# ------------------ UTILITIES ------------------

def encode_query(query_text):
    """Return numpy embedding for the query."""
    emb = embedder.encode([query_text], batch_size=BATCH_SIZE, convert_to_numpy=True)
    return emb

def search_faiss(query_emb, k=5):
    """Return list of chunk data for top-k matches."""
    if faiss_index is None or faiss_index.ntotal == 0:
        return []
    D, I = faiss_index.search(np.array(query_emb), k)
    results = []
    for j, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(documents):
            continue
        score = float(D[0][j])
        original_doc_id = doc_ids[idx] if idx < len(doc_ids) else f"doc_{idx}"
        doc_id = doc_id_mapping.get(original_doc_id, original_doc_id)
        chunk_data = documents[idx]
        
        # Extract source information
        filename = chunk_data.get("filename", "")
        page = chunk_data.get("page", None)
        text = chunk_data.get("text", "")
        
        results.append({
            "doc_id": doc_id,
            "filename": filename,
            "page": page,
            "chunk_text": text,
            "score": score
        })
    return results

def summarize_query_with_history(query, history):
    """Use Gemini 1.5 Flash to summarize query and history with weightage on current query."""
    if not history:
        return query  # No history to summarize
    
    summary_prompt = f"""
    You are a query summarization assistant. Your task is to combine the current query with the conversation history to create a comprehensive, standalone query.
    
    CONVERSATION HISTORY:
    {history}
    
    CURRENT QUERY:
    {query}
    
    INSTRUCTIONS:
    1. Give more weight to the CURRENT QUERY - it should be the primary focus
    2. Incorporate relevant context from the conversation history
    3. Create a natural, coherent query that captures the full context
    4. Keep the summarized query concise but comprehensive
    5. Maintain the original intent and specificity of the current query
    
    Return ONLY the summarized query without any additional text or explanations.
    """
    
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=summary_prompt)])]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.8,
        max_output_tokens=500,
    )
    
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=generate_content_config
        )
        summarized_query = response.text.strip()
        print(f"🔍 Original query: {query}")
        print(f"📝 Summarized query: {summarized_query}")
        return summarized_query
    except Exception as e:
        print(f"⚠️ Summarization failed, using original query: {e}")
        return query

def build_gemini_prompt(query, retrieved, max_context_chars=5000):
    """Build prompt for Gemini with source information."""
    parts = []
    total = 0
    for r in retrieved:
        filename = r['filename']
        page = r.get('page')
        
        block = f"Source: {filename}"
        if page is not None:
            block += f" (page {page})"
        block += f"\n{r['chunk_text']}\n---\n"
        
        total += len(block)
        parts.append(block)
        if total > max_context_chars:
            break
    
    context = "\n".join(parts)
    
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"QUESTION: {query}\n\n"
        "RELEVANT SOURCE DOCUMENTS:\n"
        f"{context}\n\n"
        "Based on the above sources, provide your answer (at most 3 paragraph) following the guidelines and JSON format. "
    )
    return prompt

def call_gemini(prompt, timeout=30):
    """Call Gemini and return response."""
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.8,
        max_output_tokens=4000,
        response_modalities=["TEXT"],
    )

    out_text = []
    try:
        for chunk in genai_client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=generate_content_config,
            timeout=timeout,
        ):
            if not chunk or not getattr(chunk, "candidates", None):
                continue
            candidate = chunk.candidates[0]
            if getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
                for p in candidate.content.parts:
                    if p.text:
                        out_text.append(p.text)
            elif getattr(candidate, "text", None):
                out_text.append(candidate.text)
    except Exception as e:
        try:
            resp = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=generate_content_config
            )
            out_text.append(resp.text)
        except Exception as e2:
            raise RuntimeError(f"Gemini call failed: {e} | {e2}")
    return "".join(out_text)

def parse_gemini_response(response_text):
    """Parse Gemini response to extract JSON or handle fallback."""
    try:
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != 0:
            json_str = response_text[json_start:json_end]
            response_data = json.loads(json_str)
            
            # Validate required fields
            if "response" in response_data and "pdfs" in response_data:
                return response_data
        
        # Fallback: create structured response
        return {
            "response": response_text.strip(),
            "pdfs": []
        }
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "response": response_text.strip(),
            "pdfs": []
        }

# ------------------ ROUTES ------------------

@app.route("/query", methods=["POST"])
def query_route():
    """
    POST JSON: {
        "query": "current question", 
        "history": "previous conversation context"
    }
    """
    try:
        print("📥 Received request to /query")
        
        body = request.get_json(force=True)
        if not body:
            return jsonify({"error": "No JSON body provided"}), 400
            
        query = body.get("query") or body.get("q")
        history = body.get("history", "")
        
        print(f"📥 Query: {query}")
        # print(f"📚 History: {history}")

        if not query:
            return jsonify({"error": "please provide 'query' in JSON body"}), 400

        # Check if we have documents and embeddings
        if not documents or faiss_index is None or faiss_index.ntotal == 0:
            print("❌ No documents or FAISS index available")
            return jsonify({
                "response": "The knowledge base is currently empty. Please add documents first.",
                "pdfs": []
            }), 200

        # 1) Summarize query with history using Gemini 1.5 Flash
        # print("🔄 Summarizing query with conversation history...")
        summarized_query = summarize_query_with_history(query, history)
        # print(f"📝 Summarized query: {summarized_query}")
        
        # 2) Encode summarized query
        # print("🔮 Encoding query...")
        q_emb = encode_query(summarized_query)
        # print(f"✅ Query encoded, shape: {q_emb.shape}")

        # 3) Search FAISS with predefined k
        k = 5
        # print(f"🔍 Searching FAISS for top {k} matches...")
        retrieved = search_faiss(q_emb, k=k)
         #print(f"✅ Retrieved {len(retrieved)} documents")

        if not retrieved:
            print("❌ No documents retrieved from FAISS")
            return jsonify({
                "response": "No relevant information found in the knowledge base for your query.",
                "pdfs": []
            }), 200

        # 4) Build prompt and call Gemini
        # print("📝 Building prompt for Gemini...")
        prompt = build_gemini_prompt(summarized_query, retrieved, max_context_chars=8000)
        # print(f"📋 Prompt length: {len(prompt)} characters")
        # print(f"📋 First 500 chars of prompt: {prompt[:500]}...")
        
        # 5) Call Gemini
        # print("🤖 Calling Gemini API...")
        gemini_reply = call_gemini(prompt)
        # print(f"✅ Gemini response received: {len(gemini_reply)} characters")
        # print(f"📄 Gemini response preview: {gemini_reply[:200]}...")

        # 6) Parse the response into structured format
        # print("🔄 Parsing Gemini response...")
        structured_response = parse_gemini_response(gemini_reply)
        # print(f"✅ Parsed response: {structured_response}")

        # 7) Return the structured response
        return jsonify(structured_response)

    except Exception as e:
        print(f"❌ Error in /query route: {str(e)}")
        import traceback
        print("🔍 Full traceback:")
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "detail": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/reset", methods=["POST"])
def reset_route():
    """Reset the database and regenerate embeddings from documents."""
    global faiss_index, documents, doc_ids, embeddings, doc_id_mapping
    
    try:
        cur.execute("DROP TABLE IF EXISTS documents")
        cur.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                filename TEXT,
                chunk_index INTEGER,
                page INTEGER,
                chunk_text TEXT
            )
            """
        )
        conn.commit()
        
        documents = []
        doc_ids = []
        embeddings = None
        faiss_index = None
        doc_id_mapping = {}
        
        print("🔄 Regenerating embeddings from documents...")
        new_documents, new_doc_ids = load_and_chunk_documents()
        
        if new_documents:
            new_embeddings = generate_embeddings(new_documents)
            save_embeddings(new_embeddings, new_doc_ids, new_documents)
            doc_id_mapping = assign_4digit_codes(new_doc_ids)
            
            dim = new_embeddings.shape[1]
            new_index = faiss.IndexFlatL2(dim)
            new_index.add(np.array(new_embeddings))
            faiss_index = new_index
            documents = new_documents
            doc_ids = new_doc_ids
            
            for original_doc_id, chunk_data in zip(doc_ids, documents):
                new_doc_id = doc_id_mapping[original_doc_id]
                filename, chunk_index, page = parse_doc_id(new_doc_id)
                actual_page = chunk_data.get("page", page)
                chunk_text = chunk_data.get("text", "")
                
                cur.execute(
                    "INSERT INTO documents (doc_id, filename, chunk_index, page, chunk_text) VALUES (?, ?, ?, ?, ?)",
                    (new_doc_id, filename, chunk_index, actual_page, chunk_text)
                )
            conn.commit()
            message = f"✅ Regenerated embeddings and loaded {len(documents)} documents"
        else:
            message = "⚠️ No documents found to process"
            
        return jsonify({"status": "ok", "message": message}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    info = {
        "faiss_loaded": (faiss_index is None or getattr(faiss_index, "ntotal", 0) > 0),
        "num_chunks": len(documents),
        "embed_model": EMBED_MODEL,
        "gemini_model": GEMINI_MODEL,
        "device": DEVICE,
        "docs_directory": DOCS_PATH,
        "docs_exists": os.path.exists(DOCS_PATH),
        "temperature_setting": 0,
        "predefined_k": 5
    }
    return jsonify(info)

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)