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
from pymongo import MongoClient
from bson import ObjectId

# Auth imports
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.core.config import settings
from app.rag.pipeline import ingest_file, rag_chat, rag_chat_stream

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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

# Auth Configuration
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# MongoDB
client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]
users_col = db[settings.COLLECTION_USERS]
history_col = db[settings.COLLECTION_CHAT_HISTORY]
files_col = db[settings.COLLECTION_UPLOADED_FILES]

# --- Models ---

class User(BaseModel):
    username: str
    full_name: str
    age: int
    department: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class UserRegister(BaseModel):
    username: str
    password: str
    full_name: str
    age: int
    department: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# --- Auth Helpers ---

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = users_col.find_one({"username": token_data.username})
    if user is None:
        raise credentials_exception
    return User(**user)

# --- Routes ---

@app.post("/register", response_model=User)
async def register(user_in: UserRegister):
    if users_col.find_one({"username": user_in.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    user_db = {
        "username": user_in.username,
        "hashed_password": get_password_hash(user_in.password),
        "full_name": user_in.full_name,
        "age": user_in.age,
        "department": user_in.department,
        "disabled": False
    }
    users_col.insert_one(user_db)
    return User(**user_db)

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_col.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# --- Chat Endpoints (Updated) ---

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, current_user: User = Depends(get_current_user)):
    session_id = req.session_id or str(uuid.uuid4())
    history = get_history(session_id)
    
    # Pass current user's department to RAG
    result = rag_chat(req.message, history, department=current_user.department)
    return result

@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, current_user: User = Depends(get_current_user)):
    session_id = req.session_id or str(uuid.uuid4())
    history = get_history(session_id)
    
    # Pass current user's department to RAG Stream
    return StreamingResponse(
        rag_chat_stream(req.message, session_id, history, department=current_user.department),
        media_type="text/event-stream"
    )

# --- Admin Endpoints (Updated) ---

@app.post("/admin/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    file_id = str(uuid.uuid4())
    file_type = os.path.splitext(file.filename)[1].lower()
    
    if file_type not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(400, "Chỉ hỗ trợ PDF, DOCX, TXT")

    content = await file.read()
    
    # Store file record
    files_col.insert_one({
        "file_id": file_id,
        "file_name": file.filename,
        "file_type": file_type,
        "status": "queued",
        "department": current_user.department,
        "uploaded_by": current_user.username,
        "created_at": datetime.now(timezone.utc)
    })

    # Background ingestion with department tag
    background_tasks.add_task(
        process_ingestion, content, file.filename, file_type, file_id, current_user.department
    )

    return {"file_id": file_id, "file_name": file.filename, "status": "queued"}

def process_ingestion(content, filename, file_type, file_id, department):
    try:
        files_col.update_one({"file_id": file_id}, {"$set": {"status": "processing"}})
        ingest_file(content, filename, file_type, file_id, department=department)
        files_col.update_one({"file_id": file_id}, {"$set": {"status": "completed"}})
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        files_col.update_one({"file_id": file_id}, {"$set": {"status": "error", "error": str(e)}})

# --- Common Endpoints ---

@app.get("/sessions")
async def list_sessions():
    # In a real app, filter by current user
    sessions = history_col.find().sort("updated_at", -1).limit(20)
    return {"sessions": [{"id": s["session_id"], "title": s.get("title", s["session_id"])} for s in sessions]}

@app.get("/history/{session_id}")
def get_history_endpoint(session_id: str):
    history = get_history(session_id)
    return {"messages": history}

def get_history(session_id: str) -> list:
    doc = history_col.find_one({"session_id": session_id})
    return doc["messages"] if doc else []

def save_history(session_id: str, messages: list):
    title = messages[0]["content"][:30] + "..." if messages else "New Chat"
    history_col.update_one(
        {"session_id": session_id},
        {"$set": {"messages": messages, "title": title, "updated_at": datetime.now(timezone.utc)}},
        upsert=True
    )

@app.get("/admin/stats")
async def get_stats():
    count = db[settings.COLLECTION_DOCUMENTS].count_documents({})
    files = files_col.count_documents({})
    return {"total_chunks": count, "total_files": files}

@app.get("/admin/files")
async def list_files(current_user: User = Depends(get_current_user)):
    files = list(files_col.find({}, {"_id": 0}).sort("created_at", -1))
    return {"files": files}

# --- Serve UI ---

@app.get("/")
async def read_index():
    return FileResponse("static/chat.html")

@app.get("/login_ui")
async def read_login():
    return FileResponse("static/login.html")

@app.get("/register_ui")
async def read_register():
    return FileResponse("static/register.html")

@app.get("/admin/ui")
async def read_admin():
    return FileResponse("static/admin.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
