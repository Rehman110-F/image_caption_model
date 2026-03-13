
import io
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

_predictor = None   # loaded once at startup


def _get_predictor():
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return _predictor


class CaptionResponse(BaseModel):
    caption: str


app = FastAPI(
    title="Image Captioning API",
    description="Upload an image → get an AI-generated caption.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global _predictor
    try:
        from configs import CFG
        from src.inference.predictor import CaptionPredictor

        with open(CFG.paths.vocab_path) as f:
            vocab = json.load(f)

        _predictor = CaptionPredictor.from_config(CFG, vocab)
        print("Model loaded ✓")
    except Exception as e:
        print(f"[WARNING] Could not load model at startup: {e}")
        print("The /caption endpoint will return 503 until the model is available.")


@app.get("/")
async def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/caption", response_model=CaptionResponse)
async def caption(file: UploadFile = File(...)):
    """
    Upload a JPEG or PNG image.
    Returns the greedy-decoded caption (notebook cell 37 logic).
    """
    pred = _get_predictor()

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415,
                            detail=f"Unsupported type: {file.content_type}. Use JPEG or PNG.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    try:
        caption_text = pred.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return CaptionResponse(caption=caption_text)