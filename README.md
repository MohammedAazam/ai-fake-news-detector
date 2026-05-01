# AI Fake News Detector

Fake news detection with a browser extension frontend and Python backend model support.

## Repository Structure

- `backend/` - Python backend service and trained model artifacts.
  - `app.py` - Backend application entry point.
  - `requirements.txt` - Python dependencies.
  - `model_output/` - Saved model files for one classifier.
  - `model_output_roberta/` - Saved RoBERTa model files.
- `extension/` - Browser extension that interacts with the backend.
  - `popup.html` - Extension popup UI.
  - `popup.js` - Frontend logic for the extension.
  - `background.js` - Background script.
  - `manifest.json` - Extension manifest.
- `frontend/` - Web frontend assets.
  - `index.html` - Simple frontend page.
- `research/` - Model training scripts.
  - `train_fake_news_model.py` - Training script for custom model.
  - `train_fake_news_model_RoBERTa.py` - Training script for RoBERTa-based model.
- `paper.tex` - Research paper source.
- `presentation.html` - Presentation file.

## Getting Started

### Backend

1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Run the backend:
   ```bash
   python backend/app.py
   ```

### Browser Extension

1. Open browser extension management in Chrome or Edge.
2. Enable developer mode.
3. Load the `extension/` folder as an unpacked extension.
4. Use the popup to send text for fake news prediction.

## Model Training

If you want to retrain or fine-tune the model, use the training scripts in `research/`:

```bash
python research/train_fake_news_model.py
python research/train_fake_news_model_RoBERTa.py
```

## Notes

- The backend includes saved models under `backend/model_output/` and `backend/model_output_roberta/`.
- Adjust `backend/app.py` and the extension code as needed for your deployment or API endpoint.