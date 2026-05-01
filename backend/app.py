import torch
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
from captum.attr import IntegratedGradients
import os

app = FastAPI(title="AI Fake News Detector — Dual Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# 1. DEVICE
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Running on: {DEVICE}")

# ─────────────────────────────────────────────
# 2. LOAD MODEL A — DistilBERT
# ─────────────────────────────────────────────
DISTILBERT_PATH = "./model_output"
_has_distilbert = (
    os.path.exists(os.path.join(DISTILBERT_PATH, "pytorch_model.bin")) or
    os.path.exists(os.path.join(DISTILBERT_PATH, "model.safetensors"))
)
_distilbert_source = DISTILBERT_PATH if _has_distilbert else "distilbert-base-uncased"

print(f"📦 Loading DistilBERT from: {_distilbert_source}")
tokenizer_distilbert = DistilBertTokenizer.from_pretrained(_distilbert_source)
model_distilbert = DistilBertForSequenceClassification.from_pretrained(_distilbert_source).to(DEVICE)
model_distilbert.eval()
print("✅ DistilBERT loaded.")

# ─────────────────────────────────────────────
# 3. LOAD MODEL B — RoBERTa
#    Detects folder automatically; falls back to
#    the HuggingFace base if no local weights found.
# ─────────────────────────────────────────────
ROBERTA_PATH = "./model_output_roberta"  # adjust if your folder name differs

# Try to find the actual folder (handles truncated names like model_output_rober...)
if not os.path.isdir(ROBERTA_PATH):
    # scan backend dir for any folder that starts with "model_output_rob"
    for entry in os.listdir("."):
        if os.path.isdir(entry) and entry.startswith("model_output_rob"):
            ROBERTA_PATH = entry
            break

_has_roberta = (
    os.path.exists(os.path.join(ROBERTA_PATH, "pytorch_model.bin")) or
    os.path.exists(os.path.join(ROBERTA_PATH, "model.safetensors"))
)
_roberta_source = ROBERTA_PATH if _has_roberta else "roberta-base"

print(f"📦 Loading RoBERTa from: {_roberta_source}")
# Use Auto classes so it works with both fine-tuned and base checkpoints
tokenizer_roberta = AutoTokenizer.from_pretrained(_roberta_source)
model_roberta = AutoModelForSequenceClassification.from_pretrained(_roberta_source).to(DEVICE)
model_roberta.eval()
print("✅ RoBERTa loaded.")


# ─────────────────────────────────────────────
# 4. SCHEMAS
# ─────────────────────────────────────────────
class NewsRequest(BaseModel):
    text: str


# ─────────────────────────────────────────────
# 5. XAI HELPER — Integrated Gradients
# ─────────────────────────────────────────────
def get_red_flags(text: str, model, tokenizer, target_class: int) -> list[str]:
    """Return top-5 tokens driving the prediction using Integrated Gradients."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=128, padding="max_length"
    ).to(DEVICE)
    input_ids = inputs["input_ids"]

    def predict_func(inp):
        return model(inp)[0]

    ig = IntegratedGradients(predict_func)
    baseline = torch.zeros_like(input_ids).to(DEVICE)
    attributions = (
        ig.attribute(input_ids, baseline, target=target_class)
        .sum(dim=-1).squeeze(0).cpu().detach().numpy()
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    skip = {'[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'}
    token_importance = [
        (t, s) for t, s in zip(tokens, attributions)
        if t not in skip and not t.startswith('##') and not t.startswith('Ġ')
    ]
    # also expose Ġ-prefixed RoBERTa tokens after stripping the prefix
    token_importance += [
        (t.lstrip('Ġ'), s) for t, s in zip(tokens, attributions)
        if t.startswith('Ġ') and t.lstrip('Ġ') not in skip
    ]
    token_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    return [item[0] for item in token_importance[:5]]


# ─────────────────────────────────────────────
# 6. SINGLE-MODEL INFERENCE HELPER
# ─────────────────────────────────────────────
def run_inference(text: str, model, tokenizer, model_name: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = "Fake" if pred_idx.item() == 1 else "Real"
    confidence = round(float(conf.item()) * 100, 2)
    # Full probability breakdown
    prob_real = round(float(probs[0][0].item()) * 100, 2)
    prob_fake = round(float(probs[0][1].item()) * 100, 2)

    flags = []
    if label == "Fake":
        try:
            flags = get_red_flags(text, model, tokenizer, pred_idx.item())
        except Exception as xai_err:
            print(f"[{model_name}] XAI error: {xai_err}")

    return {
        "model": model_name,
        "prediction": label,
        "confidence": confidence,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "red_flags": flags,
        "explanation": (
            f"Flagged as {label}. Key indicators: {', '.join(flags)}"
            if flags else "No specific suspicious patterns detected."
        )
    }


# ─────────────────────────────────────────────
# 7. ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict_distilbert(request: NewsRequest):
    """Original single-model endpoint (DistilBERT) — keeps backward compatibility."""
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        return run_inference(request.text, model_distilbert, tokenizer_distilbert, "DistilBERT")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/predict/roberta")
async def predict_roberta(request: NewsRequest):
    """RoBERTa-only endpoint."""
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        return run_inference(request.text, model_roberta, tokenizer_roberta, "RoBERTa")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/predict/compare")
async def predict_compare(request: NewsRequest):
    """
    Runs BOTH models and returns a side-by-side comparison.
    Also provides an 'ensemble' verdict based on averaged probabilities.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        result_a = run_inference(request.text, model_distilbert, tokenizer_distilbert, "DistilBERT")
        result_b = run_inference(request.text, model_roberta, tokenizer_roberta, "RoBERTa")

        # Ensemble: average the fake probability
        avg_fake = (result_a["prob_fake"] + result_b["prob_fake"]) / 2
        avg_real = (result_a["prob_real"] + result_b["prob_real"]) / 2
        ensemble_label = "Fake" if avg_fake > avg_real else "Real"
        ensemble_conf = round(max(avg_fake, avg_real), 2)

        # Agreement flag
        models_agree = result_a["prediction"] == result_b["prediction"]

        return {
            "distilbert": result_a,
            "roberta": result_b,
            "ensemble": {
                "prediction": ensemble_label,
                "confidence": ensemble_conf,
                "prob_real": round(avg_real, 2),
                "prob_fake": round(avg_fake, 2),
                "models_agree": models_agree,
                "note": (
                    "Both models agree." if models_agree
                    else "⚠️ Models disagree — treat result with caution."
                )
            }
        }
    except Exception as e:
        print(f"Error in compare: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/predict/sentences")
async def predict_sentences(request: NewsRequest):
    """
    Splits the article into sentences and scores each one with the ensemble.
    Returns per-sentence fake probability for heatmap rendering in the UI.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        # Split into sentences — handles ., !, ? with basic regex
        raw_sentences = re.split(r'(?<=[.!?])\s+', request.text.strip())
        # Filter out empty or very short fragments (< 10 chars)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= 10]

        if not sentences:
            raise HTTPException(status_code=400, detail="No valid sentences found")

        results = []
        for sentence in sentences:
            db = run_inference(sentence, model_distilbert, tokenizer_distilbert, "DistilBERT")
            rb = run_inference(sentence, model_roberta, tokenizer_roberta, "RoBERTa")

            avg_fake = round((db["prob_fake"] + rb["prob_fake"]) / 2, 2)
            avg_real = round((db["prob_real"] + rb["prob_real"]) / 2, 2)
            label = "Fake" if avg_fake > avg_real else "Real"

            results.append({
                "sentence": sentence,
                "prob_fake": avg_fake,
                "prob_real": avg_real,
                "label": label,
            })

        return {"sentences": results, "total": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in sentences: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "distilbert_source": _distilbert_source,
        "roberta_source": _roberta_source,
        "device": str(DEVICE)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)