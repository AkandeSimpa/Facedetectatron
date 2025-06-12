#!/bin/bash
mkdir -p /opt/render/.deepface/weights
wget -O /opt/render/.deepface/weights/facenet_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
uvicorn main:app --host 0.0.0.0 --port 10000
