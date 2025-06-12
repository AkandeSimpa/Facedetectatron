from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from PIL import Image
from pinecone import Pinecone
import os
import tempfile
import uuid
import io
import numpy as np
from typing import List

# === Constants ===
MODEL_NAME = "Facenet"
DB_DIM = 128
# Use temporary directory or current directory for model storage in Render
MODEL_DIR = os.path.join(tempfile.gettempdir(), ".models")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY",
                             "pcsk_5hzKQE_CdKim6Y9uxdjYGMMZjWMyKVrVXBX7h4c4zq3toRGryjGft3MLK17RSs2DXMRfsz")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "breadmaker")

# === Supported image formats ===
SUPPORTED_FORMATS = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
    'image/bmp', 'image/tiff', 'image/tif', 'image/gif'
}

# === Ensure model dir exists with proper error handling ===
try:
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Model directory created/verified: {MODEL_DIR}")
except PermissionError:
    # Fallback to current directory if temp dir fails
    MODEL_DIR = os.path.join(os.getcwd(), ".models")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Using fallback model directory: {MODEL_DIR}")
    except PermissionError:
        # Final fallback - use current directory
        MODEL_DIR = os.getcwd()
        print(f"Using current directory for models: {MODEL_DIR}")

# === Environment variable to override Keras model storage ===
os.environ["KERAS_HOME"] = MODEL_DIR

# === Additional environment variables for Render ===
# Set TensorFlow to use CPU only (reduces memory usage)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# === Pinecone setup with error handling ===
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"Warning: Failed to connect to Pinecone: {e}")
    index = None

# === FastAPI setup ===
app = FastAPI(
    title="Face Recognition API",
    description="Face enrollment and verification service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_pinecone_connection():
    """Validate that Pinecone is properly connected."""
    if index is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable. Please check configuration."
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
            detector_backend="opencv",
            enforce_detection=False  # More lenient for edge cases
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
    return {
        "message": "🧠 Face Recognition API is Running!",
        "status": "healthy",
        "environment": "render",
        "model_dir": MODEL_DIR
    }


@app.get("/health")
def health_check():
    pinecone_status = "connected" if index is not None else "disconnected"
    return {
        "status": "healthy",
        "supported_formats": list(SUPPORTED_FORMATS),
        "model": MODEL_NAME,
        "embedding_dimension": DB_DIM,
        "model_directory": MODEL_DIR,
        "pinecone_status": pinecone_status,
        "environment": "render"
    }


@app.get("/ui/enroll", response_class=HTMLResponse)
def enroll_ui():
    supported_formats_str = ", ".join([fmt.split("/")[1].upper() for fmt in SUPPORTED_FORMATS])
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Enrollment</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 40px auto; 
                max-width: 500px;
                background: #f5f5f5;
                padding: 20px;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            form {{ max-width: 400px; }}
            input {{ 
                margin: 10px 0; 
                padding: 12px; 
                width: 100%; 
                border: 1px solid #ddd;
                border-radius: 5px;
                box-sizing: border-box;
            }}
            input[type="submit"] {{
                background: #007bff;
                color: white;
                border: none;
                cursor: pointer;
                font-weight: bold;
            }}
            input[type="submit"]:hover {{
                background: #0056b3;
            }}
            .info {{ 
                color: #666; 
                font-size: 0.9em; 
                margin-bottom: 20px; 
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }}
            h2 {{ color: #333; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🧠 Enroll Your Face</h2>
            <div class="info">
                <strong>Supported formats:</strong> {supported_formats_str}<br>
                <strong>Tips:</strong> Use clear, well-lit photos with your face clearly visible.
            </div>
            <form action="/enroll/" enctype="multipart/form-data" method="post">
                <input name="name" type="text" placeholder="Enter your full name" required><br>
                <input name="file" type="file" accept="image/*" required><br>
                <input type="submit" value="Enroll Face">
            </form>
        </div>
    </body>
    </html>
    """


@app.get("/ui/verify", response_class=HTMLResponse)
def verify_ui():
    supported_formats_str = ", ".join([fmt.split("/")[1].upper() for fmt in SUPPORTED_FORMATS])
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Verification</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 40px auto; 
                max-width: 500px;
                background: #f5f5f5;
                padding: 20px;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            form {{ max-width: 400px; }}
            input {{ 
                margin: 10px 0; 
                padding: 12px; 
                width: 100%; 
                border: 1px solid #ddd;
                border-radius: 5px;
                box-sizing: border-box;
            }}
            input[type="submit"] {{
                background: #28a745;
                color: white;
                border: none;
                cursor: pointer;
                font-weight: bold;
            }}
            input[type="submit"]:hover {{
                background: #218838;
            }}
            .info {{ 
                color: #666; 
                font-size: 0.9em; 
                margin-bottom: 20px; 
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }}
            h2 {{ color: #333; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🔍 Verify Your Face</h2>
            <div class="info">
                <strong>Supported formats:</strong> {supported_formats_str}<br>
                <strong>Tips:</strong> Use a clear photo similar to your enrollment image.
            </div>
            <form action="/verify/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required><br>
                <input name="threshold" type="number" step="0.01" min="0" max="1" value="0.7" placeholder="Threshold (0.0-1.0)"><br>
                <input type="submit" value="Verify Face">
            </form>
        </div>
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
    # Validate Pinecone connection
    validate_pinecone_connection()

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
            "embedding_dimension": len(embedding),
            "status": "success"
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
    # Validate Pinecone connection
    validate_pinecone_connection()

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
                "message": "No enrolled faces found in database",
                "status": "no_matches"
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
                "status": "match_found",
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
                "threshold_used": threshold,
                "status": "below_threshold"
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
    validate_pinecone_connection()

    try:
        index.delete(ids=[vector_id])
        return {
            "message": f"Successfully removed face with ID: {vector_id}",
            "status": "success"
        }
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
    validate_pinecone_connection()

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
            "faces": faces,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list faces: {str(e)}"
        )


# Health check endpoint for Render
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Service is running"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)