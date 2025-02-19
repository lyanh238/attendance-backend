from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import os
from models.scrfd import SCRFD
from models.arcface import ArcFace
from utils.helpers import compute_similarity, build_targets

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo các model và tham số
base_dir = os.path.dirname(os.path.abspath(__file__))
det_path = os.path.join(base_dir, "weights", "det_10g.onnx")
rec_path = os.path.join(base_dir, "weights", "w600k_r50.onnx")

detector = SCRFD(det_path)
recognizer = ArcFace(rec_path)

class Params:
    def __init__(self):
        self.faces_dir = os.path.join(base_dir, "faces")
        self.similarity_thresh = 0.4

params = Params()

# Load embeddings của các khuôn mặt mẫu
targets = build_targets(detector, recognizer, params)

@app.post("/api/face-recognition")
async def face_recognition(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        bboxes, kpss = detector.detect(image)
        
        results = []
        for bbox, kps in zip(bboxes, kpss):
            embedding = recognizer(image, kps)
            
            max_similarity = 0
            best_match = "Unknown"
            
            for target, name in targets:
                similarity = compute_similarity(target, embedding)
                if similarity > max_similarity and similarity > params.similarity_thresh:
                    max_similarity = similarity
                    best_match = name

            results.append({
                "bbox": bbox.tolist(),
                "name": best_match,
                "confidence": float(max_similarity)
            })

        return {"success": True, "results": results}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.websocket("/ws/face-recognition")
async def websocket_face_recognition(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            bboxes, kpss = detector.detect(image)
            results = []

            for bbox, kps in zip(bboxes, kpss):
                embedding = recognizer(image, kps)
                
                max_similarity = 0
                best_match = "Unknown"
                
                for target, name in targets:
                    similarity = compute_similarity(target, embedding)
                    if similarity > max_similarity and similarity > params.similarity_thresh:
                        max_similarity = similarity
                        best_match = name

                results.append({
                    "bbox": bbox.tolist(),
                    "name": best_match,
                    "confidence": float(max_similarity)
                })

            await websocket.send_json({"success": True, "results": results})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
