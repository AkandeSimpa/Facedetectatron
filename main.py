from dotenv import load_dotenv      # Needed to load .env file
load_dotenv()
import os
import shutil
import uuid

import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pinecone import Pinecone

# === Config ===
MODEL_PATH = "models/openface.nn4.small2.v1.t7"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "breadmaker"
EMBEDDING_DIM = 128

# === Load OpenFace Model ===
face_embedder = cv2.dnn.readNetFromTorch(MODEL_PATH)

# === Pinecone Setup ===
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine")
index = pc.Index(INDEX_NAME)

# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Embedding Extraction ===
def get_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image")

    resized = cv2.resize(image, (96, 96))
    blob = cv2.dnn.blobFromImage(resized, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    face_embedder.setInput(blob)
    vec = face_embedder.forward()
    return vec.flatten().tolist()

# === Routes ===

@app.get("/")
def root():
    return {"message": "👤 OpenFace API running with OpenCV!"}

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

        emb = get_embedding(temp_path)

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

        emb = get_embedding(temp_path)

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
