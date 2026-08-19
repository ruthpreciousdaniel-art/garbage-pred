import io
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from model import GarbageCNN

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "model" / "best_model.pt"

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

app = FastAPI(title="GreenCycle Garbage Classifier")
_state = {}


@app.on_event("startup")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = GarbageCNN(num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _state.update(
        model=model,
        classes=ckpt["classes"],
        img_size=ckpt["img_size"],
        device=device,
    )
    print(f"Loaded model. Classes={ckpt['classes']} img_size={ckpt['img_size']}")


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_k: List[dict]


@app.get("/health")
def health():
    return {"status": "ok", "classes": _state.get("classes")}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    tf = transforms.Compose([
        transforms.Resize((_state["img_size"], _state["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    x = tf(image).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        probs = F.softmax(_state["model"](x), dim=1).squeeze(0).cpu()

    classes = _state["classes"]
    top_prob, top_idx = probs.max(dim=0)
    k = min(3, len(classes))
    top_vals, top_idxs = probs.topk(k)
    top_k = [{"class": classes[i.item()], "probability": float(v.item())} for v, i in zip(top_vals, top_idxs)]

    return PredictionResponse(
        predicted_class=classes[top_idx.item()],
        confidence=float(top_prob.item()),
        top_k=top_k,
    )


# Serve the simple web UI at "/"
app.mount("/", StaticFiles(directory=str(BASE_DIR / "app" / "static"), html=True), name="static")
