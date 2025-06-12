
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from deepface import DeepFace
from pinecone import Pinecone
from PIL import Image
import shutil
import os
import uuid

# === Configuration ===
MODEL_NAME = "Facenet"  # Model dimension: 4096
DB_DIM = 128  # Embedding vector dimension for Facenet
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "breadmaker"

# === Pinecone Setup (v3 SDK) ===
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=DB_DIM, metric="cosine")

index = pc.Index(INDEX_NAME)

# === FastAPI App Initialization ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Routes ===

@app.get("/")
def root():
    return {"message": "🧠 Face Recognition API is Running!"}

@app.get("/ui/enroll", response_class=HTMLResponse)
def enroll_ui():
    return """
    <html><body>
        <h2>Enroll Face</h2>
        <form action="/enroll/" enctype="multipart/form-data" method="post">
            <input name="name" type="text" placeholder="Your Name" required><br><br>
            <input name="file" type="file" accept="image/*" required><br><br>
            <input type="submit">
        </form>
    </body></html>
    """

@app.get("/ui/verify", response_class=HTMLResponse)
def verify_ui():
    return """
    <html><body>
        <h2>Verify Face</h2>
        <form action="/verify/" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/*" required><br><br>
            <input type="submit">
        </form>
    </body></html>
    """

@app.post("/enroll/")
async def enroll_face(name: str = Form(...), file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if os.path.getsize(temp_path) < 1024:
            raise HTTPException(status_code=400, detail="Uploaded file is too small or corrupted.")

        img = Image.open(temp_path).convert("RGB")
        img.save(temp_path, format="JPEG")

        embedding_objs = DeepFace.represent(img_path=temp_path, model_name=MODEL_NAME, detector_backend="opencv")
        if not embedding_objs:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        emb = embedding_objs[0]["embedding"]
        if len(emb) != DB_DIM:
            raise HTTPException(status_code=400, detail=f"Embedding dimension mismatch. Got {len(emb)}, expected {DB_DIM}")

        vector_id = f"{name.lower()}_{uuid.uuid4().hex[:8]}"
        index.upsert(vectors=[(vector_id, emb, {"name": name})])

        return {"message": f"{name} enrolled successfully", "id": vector_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/verify/")
async def verify_face(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if os.path.getsize(temp_path) < 1024:
            raise HTTPException(status_code=400, detail="Uploaded file is too small or corrupted.")

        img = Image.open(temp_path).convert("RGB")
        img.save(temp_path, format="JPEG")

        embedding_objs = DeepFace.represent(img_path=temp_path, model_name=MODEL_NAME, detector_backend="opencv")
        if not embedding_objs:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        emb = embedding_objs[0]["embedding"]
        if len(emb) != DB_DIM:
            raise HTTPException(status_code=400, detail=f"Embedding dimension mismatch. Got {len(emb)}, expected {DB_DIM}")

        query_result = index.query(vector=emb, top_k=1, include_metadata=True)
        if not query_result.matches:
            raise HTTPException(status_code=404, detail="No match found in the face database")

        best_match = query_result.matches[0]
        return {
            "match_found": True,
            "matched_user": best_match.metadata.get("name"),
            "id": best_match.id,
            "score": best_match.score,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
