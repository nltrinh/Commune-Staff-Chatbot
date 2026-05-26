import json
import os
import uuid
import shutil
import logging
from typing import List, Optional
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.database import Database, get_users_col, get_depts_col, get_history_col, get_files_col
from app.services.auth_service import AuthService, oauth2_scheme
from app.services.admin_service import AdminService
from app.services.document_service import DocumentService
from app.services.history_service import HistoryService
from app.core.factory import get_ai_provider, get_vector_provider, get_document_processor
from app.services.rag_service import RAGService

# Logging
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Service Instances (Dependency Injection ready)
auth_service = AuthService()
admin_service = AdminService()
doc_service = DocumentService()
history_service = HistoryService()

# RAG Orchestrator
rag_service = RAGService(
    ai_provider=get_ai_provider(),
    vector_provider=get_vector_provider(),
    processor=get_document_processor()
)

# --- Models ---
class User(BaseModel):
    username: str
    full_name: str
    age: int
    department: str
    can_upload_file: bool = True
    restrict_chatbot_dept: bool = True
    disabled: Optional[bool] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# Models moved or simplified
class Token(BaseModel):
    access_token: str
    token_type: str

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

# --- Dependency ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực thông tin.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    username = auth_service.decode_token(token)
    if username is None:
        raise credentials_exception
    
    # Host Admin Check
    if username == settings.ADMIN_USERNAME:
        return User(username=settings.ADMIN_USERNAME, full_name="Quản trị viên Hệ thống", age=0, department="admin")
    
    user = get_users_col().find_one({"username": username})
    if user is None:
        raise credentials_exception
    return User(**user)

# --- Auth Routes ---
from fastapi.security import OAuth2PasswordRequestForm

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # 1. Host Admin
    if form_data.username == settings.ADMIN_USERNAME and form_data.password == settings.ADMIN_PASSWORD:
        token = auth_service.create_access_token(
            data={"sub": settings.ADMIN_USERNAME},
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return {"access_token": token, "token_type": "bearer"}

    # 2. DB Users
    user = get_users_col().find_one({"username": form_data.username})
    if not user or not auth_service.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không đúng.")
    
    token = auth_service.create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/users/change-password")
async def change_password(req: PasswordChange, current_user: User = Depends(get_current_user)):
    if current_user.username == settings.ADMIN_USERNAME:
        raise HTTPException(status_code=400, detail="Vui lòng đổi mật khẩu Host Admin trong file .env")
    
    user_db = get_users_col().find_one({"username": current_user.username})
    if not user_db or not auth_service.verify_password(req.old_password, user_db["hashed_password"]):
        raise HTTPException(status_code=400, detail="Mật khẩu cũ không đúng.")
    
    new_hashed = auth_service.get_password_hash(req.new_password)
    get_users_col().update_one({"username": current_user.username}, {"$set": {"hashed_password": new_hashed}})
    return {"message": "Thành công"}

# --- Chat Endpoints ---

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, current_user: User = Depends(get_current_user)):
    session_id = req.session_id or str(uuid.uuid4())
    history = history_service.get_history(session_id)
    
    # Logic for search scope
    is_unrestricted = current_user.username == settings.ADMIN_USERNAME

    if is_unrestricted:
        search_depts = ["tat_ca"]
    else:
        search_depts = [current_user.department, "tat_ca"]
        
    result = rag_service.chat(req.message, search_depts, history)
    
    # Save history
    new_history = history + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": result["answer"]}
    ]
    history_service.save_history(session_id, new_history, username=current_user.username)
    
    return result

@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, current_user: User = Depends(get_current_user)):
    session_id = req.session_id or str(uuid.uuid4())
    history = history_service.get_history(session_id)
    
    # Logic for search scope
    is_unrestricted = current_user.username == settings.ADMIN_USERNAME

    if is_unrestricted:
        search_depts = ["tat_ca"] # RAGService treats ["tat_ca"] as "all"
    else:
        # Restricted user: only their dept + shared
        search_depts = [current_user.department, "tat_ca"]

    async def event_generator():
        full_answer = ""
        # 1. Info
        yield json.dumps({"type": "info", "session_id": session_id}, ensure_ascii=False) + "\n"
        
        # 2. Metadata (Sources)
        sources = rag_service.search_context(req.message, search_depts)
        yield json.dumps({"type": "metadata", "sources": sources}, ensure_ascii=False) + "\n"
        
        # 3. Stream
        for chunk in rag_service.chat_stream(req.message, search_depts, history):
            full_answer += chunk
            yield json.dumps({"type": "text", "content": chunk}, ensure_ascii=False) + "\n"
        
        # 4. Save History
        new_history = history + [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": full_answer}
        ]
        history_service.save_history(session_id, new_history, username=current_user.username)
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/sessions")
async def get_sessions(current_user: User = Depends(get_current_user)):
    return {"sessions": history_service.get_user_sessions(current_user.username)}

@app.get("/admin/all-sessions")
async def get_all_sessions(current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    return {"sessions": history_service.get_all_sessions()}

@app.get("/history/{session_id}")
async def get_session_history(session_id: str, current_user: User = Depends(get_current_user)):
    history = history_service.get_history(session_id)
    return {"messages": history}

@app.delete("/history/{session_id}")
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    success = history_service.delete_session(session_id)
    if not success: raise HTTPException(404, "Hội thoại không tồn tại")
    return {"message": "Đã xóa"}

# --- Admin: User & Dept Management ---

class UserRegister(BaseModel):
    username: str
    password: Optional[str] = None
    full_name: str
    age: int
    department: str
    can_upload_file: Optional[bool] = True
    restrict_chatbot_dept: Optional[bool] = True

class DepartmentCreate(BaseModel):
    code: str
    name: str

@app.get("/admin/users")
async def list_users(current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    return {"users": admin_service.get_all_users()}

@app.post("/admin/create-user")
async def create_user(req: UserRegister, current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    user_data = req.dict()
    user_data["hashed_password"] = auth_service.get_password_hash(req.password or "123456")
    user_data["plain_password"] = req.password or "123456"
    user_data.pop("password")
    user_data["created_at"] = datetime.now(timezone.utc)
    if not admin_service.create_user(user_data): raise HTTPException(400, "Username exists")
    return {"message": "Success"}

@app.put("/admin/users/{username}")
async def update_user(username: str, req: UserRegister, current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    update_data = req.dict()
    if req.password:
        update_data["hashed_password"] = auth_service.get_password_hash(req.password)
        update_data["plain_password"] = req.password
    update_data.pop("password")
    if not admin_service.update_user(username, update_data): raise HTTPException(404)
    return {"message": "Success"}

@app.delete("/admin/users/{username}")
async def delete_user(username: str, current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    if not admin_service.delete_user(username): raise HTTPException(400)
    return {"message": "Deleted"}

@app.get("/admin/departments")
async def list_depts(current_user: User = Depends(get_current_user)):
    return {"departments": admin_service.get_all_departments()}

@app.post("/admin/departments")
async def create_dept(req: DepartmentCreate, current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    if not admin_service.create_department(req.dict()): raise HTTPException(400)
    return {"message": "Success"}

@app.put("/admin/departments/{code}")
async def update_dept(code: str, req: DepartmentCreate, current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    if not admin_service.update_department(code, req.dict()): raise HTTPException(404)
    return {"message": "Success"}

@app.delete("/admin/departments/{code}")
async def delete_dept(code: str, current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    if not admin_service.delete_department(code): raise HTTPException(400, "Cannot delete dept with users")
    return {"message": "Deleted"}

@app.get("/admin/stats")
async def get_stats(current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    return admin_service.get_system_stats()

# --- Admin: File Management ---

@app.post("/admin/upload")
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    department: Optional[str] = Form(None), 
    uploaded_by_user: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    query_params = request.query_params
    final_dept = department or query_params.get("department") or "tat_ca"
    final_uploader = uploaded_by_user or query_params.get("uploaded_by_user") or current_user.username
    
    # Check if restricted
    if current_user.username != settings.ADMIN_USERNAME:
        if not current_user.can_upload_file:
            raise HTTPException(403, "Bạn không có quyền upload tài liệu.")

    file_id = str(uuid.uuid4())
    uploader_name = current_user.full_name
    is_admin_proxy = False
    
    # Admin Proxy Logic
    if current_user.username == settings.ADMIN_USERNAME and final_uploader != current_user.username:
        target_user = get_users_col().find_one({"username": final_uploader})
        if target_user:
            uploader_name = target_user.get("full_name", final_uploader)
            is_admin_proxy = True

    # Save file to disk for re-indexing support
    os.makedirs("data/uploads", exist_ok=True)
    file_path = os.path.join("data/uploads", f"{file_id}_{file.filename}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    doc_service.save_file_metadata({
        "file_id": file_id, 
        "file_name": file.filename, 
        "file_path": file_path,
        "status": "processing",
        "department": final_dept, 
        "departments": final_dept.split(",") if "," in final_dept else [final_dept],
        "uploaded_by": final_uploader, 
        "uploader_name": uploader_name,
        "is_admin_proxy": is_admin_proxy,
        "created_at": datetime.now(timezone.utc)
    })
    
    file_type = os.path.splitext(file.filename)[1].lower()
    
    def process_task():
        try:
            result = rag_service.ingest_file(content, file.filename, file_type, file_id, final_dept)
            doc_service.update_file_status(file_id, "completed", chunks_count=result.get("chunks_saved"))
        except Exception as e:
            logger.error(f"Upload error: {e}")
            doc_service.update_file_status(file_id, "error", error=str(e))

    background_tasks.add_task(process_task)
    return {"file_id": file_id}

@app.post("/admin/files/{file_id}/reindex")
async def reindex_file(file_id: str, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    if current_user.username != settings.ADMIN_USERNAME: raise HTTPException(403)
    
    file_meta = doc_service.get_file_by_id(file_id)
    if not file_meta: raise HTTPException(404, "File not found")
    
    file_path = file_meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(400, "Original file not found on disk. Cannot re-index.")

    doc_service.update_file_status(file_id, "processing")
    
    # Delete old chunks first
    rag_service.vector_store.delete_file(file_id)
    
    def process_task():
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            file_type = os.path.splitext(file_meta["file_name"])[1].lower()
            result = rag_service.ingest_file(content, file_meta["file_name"], file_type, file_id, file_meta["department"])
            doc_service.update_file_status(file_id, "completed", chunks_count=result.get("chunks_saved"))
        except Exception as e:
            logger.error(f"Reindex error: {e}")
            doc_service.update_file_status(file_id, "error", error=str(e))

    background_tasks.add_task(process_task)
    return {"message": "Re-indexing started"}

@app.get("/admin/files")
async def list_files(current_user: User = Depends(get_current_user)):
    is_unrestricted = current_user.username == settings.ADMIN_USERNAME
    
    if is_unrestricted:
        return {"files": doc_service.get_all_files()}
    else:
        # Normalize: Always include user's departments and 'tat_ca'
        user_depts = current_user.department.split(",") if current_user.department else []
        search_depts = list(set(user_depts + ["tat_ca"]))
        
        # Filter files: shared OR user's department
        filter_dict = {"departments": {"$in": search_depts}}
        return {"files": doc_service.get_all_files(filter_dict)}

@app.delete("/admin/files/{file_id}")
async def delete_file(file_id: str, current_user: User = Depends(get_current_user)):
    # 1. Get metadata to find file path
    file_meta = doc_service.get_file_by_id(file_id)
    
    # 2. Delete from vector store
    rag_service.vector_store.delete_file(file_id)
    
    # 3. Delete from metadata store
    doc_service.delete_file_metadata(file_id)
    
    # 4. Delete from disk
    if file_meta and file_meta.get("file_path"):
        path = file_meta["file_path"]
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logger.error(f"Error deleting file from disk: {e}")

    return {"message": "Deleted"}

# --- Static UI Routes ---
@app.get("/")
async def read_index(): return FileResponse("static/dashboard.html")

@app.get("/login_ui")
async def read_login(): return FileResponse("static/login.html")

@app.on_event("startup")
async def startup_event():
    # Bootstrap departments — use upsert to avoid duplicate inserts
    # when running with multiple workers (e.g., --workers 4)
    depts_col = get_depts_col()
    default_depts = [
        {"code": "tu_phap", "name": "Phòng Tư pháp"},
        {"code": "dia_chinh", "name": "Phòng Địa chính"},
        {"code": "cong_an", "name": "Công an Xã"},
    ]
    for dept in default_depts:
        depts_col.update_one(
            {"code": dept["code"]},
            {"$setOnInsert": {"code": dept["code"], "name": dept["name"], "created_at": datetime.now(timezone.utc)}},
            upsert=True
        )
    logger.info("[STARTUP] Production Architecture Ready")

# --- Serve UI ---

@app.get("/")
async def read_index():
    return FileResponse("static/dashboard.html")

@app.get("/login_ui")
async def read_login():
    return FileResponse("static/login.html")

# REMOVED /register_ui

@app.get("/admin/ui")
async def read_admin():
    return FileResponse("static/dashboard.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
