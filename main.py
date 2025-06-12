from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from PIL import Image
from pinecone import Pinecone
import os
import uuid
import io
import numpy as np
from typing import List

# === Constants ===
MODEL_NAME = "Facenet"
DB_DIM = 128
MODEL_DIR = "/home/runner/.models"
PINECONE_API_KEY = "pcsk_5hzKQE_CdKim6Y9uxdjYGMMZjWMyKVrVXBX7h4c4zq3toRGryjGft3MLK17RSs2DXMRfsz"
INDEX_NAME = "breadmaker"

# === Supported image formats ===
SUPPORTED_FORMATS = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
    'image/bmp', 'image/tiff', 'image/tif', 'image/gif'
}

# === Ensure model dir exists ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Environment variable to override Keras model storage ===
os.environ["KERAS_HOME"] = MODEL_DIR

# === Pinecone setup ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# === FastAPI setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image_format(file: UploadFile) -> None:
    """Validate that the uploaded file is a supported image format."""
    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {file.content_type}. "
                   f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )


def process_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Process image from bytes and return as numpy array suitable for DeepFace.

    Args:
        file_bytes: Raw image bytes

    Returns:
        numpy array representing the image
    """
    try:
        # Create BytesIO object from bytes
        image_stream = io.BytesIO(file_bytes)

        # Open image with PIL
        img = Image.open(image_stream)

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert PIL image to numpy array
        img_array = np.array(img)

        return img_array

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )


def get_face_embedding(img_array: np.ndarray) -> List[float]:
    """
    Extract face embedding from image array.

    Args:
        img_array: numpy array representing the image

    Returns:
        Face embedding as list of floats
    """
    try:
        # DeepFace.represent can work directly with numpy arrays
        embeddings = DeepFace.represent(
            img_path=img_array,
            model_name=MODEL_NAME,
            detector_backend="opencv"
        )

        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        emb = embeddings[0]["embedding"]

        if len(emb) != DB_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension mismatch. Expected {DB_DIM}, got {len(emb)}"
            )

        return emb

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting face embedding: {str(e)}"
        )


@app.get("/")
def root():
    return {"message": "🧠 Face Recognition API is Running!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "supported_formats": list(SUPPORTED_FORMATS),
        "model": MODEL_NAME,
        "embedding_dimension": DB_DIM
    }


@app.get("/ui/enroll", response_class=HTMLResponse)
def enroll_ui():
    supported_formats_str = ", ".join([fmt.split("/")[1].upper() for fmt in SUPPORTED_FORMATS])
    return f"""
    <html>
    <head>
        <title>Face Enrollment</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            form {{ max-width: 400px; }}
            input {{ margin: 10px 0; padding: 8px; width: 100%; }}
            .info {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h2>Enroll Face</h2>
        <div class="info">
            Supported formats: {supported_formats_str}
        </div>
        <form action="/enroll/" enctype="multipart/form-data" method="post">
            <input name="name" type="text" placeholder="Your Name" required><br>
            <input name="file" type="file" accept="image/*" required><br>
            <input type="submit" value="Enroll Face">
        </form>
    </body>
    </html>
    """


@app.get("/ui/verify", response_class=HTMLResponse)
def verify_ui():
    supported_formats_str = ", ".join([fmt.split("/")[1].upper() for fmt in SUPPORTED_FORMATS])
    return f"""
    <html>
    <head>
        <title>Face Verification</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            form {{ max-width: 400px; }}
            input {{ margin: 10px 0; padding: 8px; width: 100%; }}
            .info {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h2>Verify Face</h2>
        <div class="info">
            Supported formats: {supported_formats_str}
        </div>
        <form action="/verify/" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/*" required><br>
            <input type="submit" value="Verify Face">
        </form>
    </body>
    </html>
    """


@app.post("/enroll/")
async def enroll_face(name: str = Form(...), file: UploadFile = File(...)):
    """
    Enroll a new face in the system.

    Args:
        name: Name to associate with the face
        file: Image file containing the face to enroll

    Returns:
        Success message with enrollment details
    """
    # Validate image format
    validate_image_format(file)

    # Validate name
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    try:
        # Read file bytes
        file_bytes = await file.read()

        # Process image from bytes
        img_array = process_image_from_bytes(file_bytes)

        # Get face embedding
        embedding = get_face_embedding(img_array)

        # Create unique vector ID
        vector_id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

        # Store in Pinecone
        index.upsert(vectors=[(vector_id, embedding, {"name": name.strip()})])

        return {
            "message": f"Successfully enrolled {name}",
            "id": vector_id,
            "name": name.strip(),
            "embedding_dimension": len(embedding)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Enrollment failed: {str(e)}"
        )


@app.post("/verify/")
async def verify_face(file: UploadFile = File(...), threshold: float = Form(0.7)):
    """
    Verify a face against enrolled faces.

    Args:
        file: Image file containing the face to verify
        threshold: Similarity threshold for matching (default: 0.7)

    Returns:
        Verification result with match details
    """
    # Validate image format
    validate_image_format(file)

    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Threshold must be between 0.0 and 1.0"
        )

    try:
        # Read file bytes
        file_bytes = await file.read()

        # Process image from bytes
        img_array = process_image_from_bytes(file_bytes)

        # Get face embedding
        embedding = get_face_embedding(img_array)

        # Query Pinecone for matches
        result = index.query(
            vector=embedding,
            top_k=5,  # Get top 5 matches for better analysis
            include_metadata=True
        )

        if not result.matches:
            return {
                "match_found": False,
                "message": "No enrolled faces found in database"
            }

        # Get best match
        best_match = result.matches[0]

        # Check if score meets threshold
        if best_match.score >= threshold:
            return {
                "match_found": True,
                "matched_user": best_match.metadata.get("name"),
                "id": best_match.id,
                "confidence_score": round(best_match.score, 4),
                "threshold_used": threshold,
                "all_matches": [
                    {
                        "name": match.metadata.get("name"),
                        "id": match.id,
                        "score": round(match.score, 4)
                    }
                    for match in result.matches[:3]  # Show top 3
                ]
            }
        else:
            return {
                "match_found": False,
                "message": f"No match found above threshold ({threshold})",
                "best_match_score": round(best_match.score, 4),
                "best_match_name": best_match.metadata.get("name"),
                "threshold_used": threshold
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


@app.delete("/remove/{vector_id}")
async def remove_face(vector_id: str):
    """
    Remove an enrolled face from the system.

    Args:
        vector_id: ID of the vector to remove

    Returns:
        Success message
    """
    try:
        index.delete(ids=[vector_id])
        return {"message": f"Successfully removed face with ID: {vector_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove face: {str(e)}"
        )


@app.get("/list/")
async def list_enrolled_faces():
    """
    List all enrolled faces (metadata only).
    Note: This is a basic implementation. For production, consider pagination.

    Returns:
        List of enrolled faces with their metadata
    """
    try:
        # Query with a dummy vector to get all stored vectors
        # This is not ideal for large datasets - consider using Pinecone's describe_index_stats
        dummy_vector = [0.0] * DB_DIM
        result = index.query(
            vector=dummy_vector,
            top_k=1000,  # Adjust based on your needs
            include_metadata=True
        )

        faces = [
            {
                "id": match.id,
                "name": match.metadata.get("name"),
                "score": match.score
            }
            for match in result.matches
        ]

        return {
            "total_faces": len(faces),
            "faces": faces
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list faces: {str(e)}"
        )
