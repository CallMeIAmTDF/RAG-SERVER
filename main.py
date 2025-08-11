import os
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles
import PyPDF2
from io import BytesIO
from fastapi.responses import FileResponse
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Cấu hình
UPLOAD_DIR = "uploads"
CHROMA_DB_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10
RERANK_TOP_K = 5
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")
# Đảm bảo thư mục tồn tại
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Khởi tạo FastAPI
app = FastAPI(title="RAG Legal Chatbot", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class ChatRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 1000


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]





# Khởi tạo các model
class RAGSystem:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.llm = genai.GenerativeModel('gemini-2.0-flash')
        self.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="legal_documents",
            metadata={"hnsw:space": "cosine"}
        )

        logger.info("RAG System initialized successfully")

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Trích xuất text từ PDF với thông tin trang"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            pages_content = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages_content.append({
                        "page": page_num,
                        "content": text.strip()
                    })

            return pages_content
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

    def chunk_text(self, text: str, page_num: int) -> List[str]:
        """Chia nhỏ text thành các chunk với overlap"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]

            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                cut_point = max(last_period, last_newline)

                if cut_point > start + CHUNK_SIZE // 2:
                    chunk = text[start:cut_point + 1]
                    end = cut_point + 1

            if chunk.strip():
                chunks.append(chunk.strip())

            start = end - CHUNK_OVERLAP
            if start >= len(text):
                break

        return chunks

    async def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Xử lý tài liệu: trích xuất, chunk, embedding"""
        try:
            pages_content = self.extract_text_from_pdf(file_content)

            file_hash = hashlib.md5(file_content).hexdigest()

            existing_docs = self.collection.get(
                where={"file_hash": file_hash}
            )

            if existing_docs['ids']:
                return {
                    "status": "already_exists",
                    "message": f"File {filename} đã được xử lý trước đó",
                    "chunks_count": len(existing_docs['ids'])
                }

            all_chunks = []
            chunk_metadatas = []
            chunk_ids = []

            for page_data in pages_content:
                page_num = page_data["page"]
                page_text = page_data["content"]

                chunks = self.chunk_text(page_text, page_num)

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file_hash}_{page_num}_{i}"

                    metadata = {
                        "filename": filename,
                        "file_hash": file_hash,
                        "page": page_num,
                        "chunk_index": i,
                        "upload_time": datetime.now().isoformat()
                    }

                    all_chunks.append(chunk)
                    chunk_metadatas.append(metadata)
                    chunk_ids.append(chunk_id)

            embeddings = self.embedding_model.encode(all_chunks,
                                                     convert_to_numpy=True,
                                                     show_progress_bar=True)

            # Lưu vào ChromaDB
            self.collection.add(
                ids=chunk_ids,
                documents=all_chunks,
                metadatas=chunk_metadatas,
                embeddings=embeddings.tolist()
            )

            logger.info(f"Processed {filename}: {len(all_chunks)} chunks from {len(pages_content)} pages")

            return {
                "status": "success",
                "filename": filename,
                "pages_count": len(pages_content),
                "chunks_count": len(all_chunks)
            }

        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    async def search_similar_chunks(self, query: str, top_k: int = TOP_K, similarity_threshold: float = 0.3) -> List[
        Dict[str, Any]]:
        """Tìm kiếm chunks tương tự, lọc theo độ tương đồng"""
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            chunks = []
            for i in range(len(results['ids'][0])):
                similarity = 1 - results['distances'][0][i]

                if similarity >= similarity_threshold:
                    chunks.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity": round(similarity, 4)
                    })

            return chunks

        except Exception as e:
            logger.error(f"Error searching chunks: {str(e)}")
            return []

    def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = RERANK_TOP_K) -> List[
        Dict[str, Any]]:
        """Rerank các chunks sử dụng cross-encoder"""
        try:
            if not chunks:
                return []

            pairs = [(query, chunk["content"]) for chunk in chunks]

            scores = self.reranker.predict(pairs)

            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])

            reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

            return reranked_chunks

        except Exception as e:
            logger.error(f"Error reranking chunks: {str(e)}")
            return chunks[:top_k]

    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], max_tokens: int = 1000) -> str:
        """Tạo câu trả lời sử dụng Gemini"""
        try:
            context_parts = []
            for chunk in context_chunks:
                metadata = chunk["metadata"]
                source_info = f"[{metadata['filename']}, trang {metadata['page']}]"
                context_parts.append(f"{source_info}:\n{chunk['content']}")

            context = "\n\n".join(context_parts)

            prompt = f"""
            Bạn là một trợ lý AI chuyên về pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
            NGỮ CẢNH:
            {context}
            
            CÂU HỎI: {query}
            
            HƯỚNG DẪN:
            - Trả lời chính xác dựa trên thông tin trong ngữ cảnh
            - Nếu không có thông tin đủ, hãy nói rõ điều đó
            - Trả lời bằng tiếng Việt
            - Tham khảo các điều luật cụ thể nếu có
            - Giải thích rõ ràng và dễ hiểu
            
            TRẢ LỜI:"""

            # Gọi Gemini API
            response = await asyncio.to_thread(
                self.llm.generate_content,
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )

            return response.text.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Xin lỗi, tôi không thể tạo câu trả lời cho câu hỏi này do lỗi hệ thống."


rag_system = RAGSystem()


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        logger.info("Multipart request detected")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


@app.post("/upload")
async def upload_file(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF")

    contents = await file.read()
    file_size = len(contents)

    if file_size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File quá lớn (max 50MB)")

    file_hash = hashlib.md5(contents).hexdigest()

    file_path = os.path.join(UPLOAD_DIR, f"{file_hash}.pdf")
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(contents)

    background_tasks.add_task(rag_system.process_document, contents, f"{file_hash}.pdf")

    return {
        "filename": file.filename,
        "size": file_size,
        "hash": file_hash
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint với RAG"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query không được để trống")

        similar_chunks = await rag_system.search_similar_chunks(query, TOP_K)

        if not similar_chunks:
            return ChatResponse(
                answer="Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu.",
                sources=[]
            )

        reranked_chunks = rag_system.rerank_chunks(query, similar_chunks, RERANK_TOP_K)

        answer = await rag_system.generate_answer(query, reranked_chunks, request.max_tokens)
        sources = []
        if reranked_chunks:
            first_chunk = reranked_chunks[0]
            metadata = first_chunk["metadata"]

            sources.append({
                "source": metadata["filename"],
                "page": metadata["page"],
                "similarity": round(first_chunk.get("similarity", 0), 3),
                "rerank_score": round(first_chunk.get("rerank_score", 0), 3)
            })

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/stats")
async def get_stats():
    """Thống kê hệ thống"""
    try:
        collection_count = rag_system.collection.count()
        return {
            "total_chunks": collection_count,
            "embedding_model": "keepitreal/vietnamese-sbert",
            "llm_model": "gemini-pro",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-2-v2"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/files/{filename}")
def get_file(filename: str):
    file_path = f"{UPLOAD_DIR}/{filename}"
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(path=file_path, filename=filename, media_type="application/pdf")

