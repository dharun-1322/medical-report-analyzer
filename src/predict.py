"""
ReportAnalyzer — unified inference class.

Loads all three models (classifier, NER, summarizer) and runs them
together on a single transcription input.

Used by: app/streamlit_app.py, tests/test_predict.py
"""

import sys
import json
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))
from config import CLF_SAVED_DIR, LABEL_ENCODER_PATH
from src.preprocess import basic_clean


class ReportAnalyzer:
    """
    Three-in-one medical report analysis:
      1. classify()  — predict medical specialty (BioClinicalBERT)
      2. extract()   — extract diseases & medications (scispaCy NER)
      3. summarize() — patient-friendly plain-language summary (BART)
      4. analyze()   — runs all three in one call
    """

    def __init__(self, load_summarizer: bool = True, load_ner: bool = True):
        """
        Args:
            load_summarizer: Set False to skip BART (saves ~1.6GB RAM for quick tests)
            load_ner:        Set False to skip scispaCy (saves ~500MB)
        """
        self._clf        = None
        self._tokenizer  = None
        self._le         = None
        self._summarizer = None
        self._ner        = None

        self._load_classifier()

        if load_ner:
            try:
                from src.ner import extract_entities
                self._ner = extract_entities
            except OSError as e:
                print(f"[NER] Skipped — {e}")

        if load_summarizer:
            try:
                from src.summarize import summarize
                self._summarizer = summarize
            except Exception as e:
                print(f"[Summarizer] Skipped — {e}")

 def _load_classifier(self):
    """Load fine-tuned BioClinicalBERT + label encoder."""
    if not CLF_SAVED_DIR.exists():
        self._clf = None
        self._classes = []
        print(f"[Classifier] Skipped — model not found at {CLF_SAVED_DIR}")
        return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Load label config
        with open(CLF_SAVED_DIR / "label_config.json") as f:
            cfg = json.load(f)
        self._classes = cfg["classes"]

        self._tokenizer = AutoTokenizer.from_pretrained(str(CLF_SAVED_DIR))
        self._clf = AutoModelForSequenceClassification.from_pretrained(
            str(CLF_SAVED_DIR)
        )
        self._clf.eval()   # inference mode — disables dropout

        # Move to GPU if available
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clf.to(self._device)

        print(f"Classifier loaded ({self._device}) | {len(self._classes)} specialties")

    # ── Individual model calls ────────────────────────────────────────────────

    def classify(self, text: str) -> Dict:
    """
    Returns top-3 specialty predictions with confidence scores.

    DL concept: we run a forward pass through the fine-tuned BERT,
    get raw logits, apply softmax to get probabilities.
    """
    if self._clf is None:
        return {
            "top_specialty": "Model not available",
            "confidence": 0.0,
            "top_3": []
        }

    clean  = basic_clean(text)
    inputs = self._tokenizer(
        clean,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():                           # no gradients at inference
            logits = self._clf(**inputs).logits         # shape: (1, num_classes)

        probs   = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # → probabilities
        top_idx = probs.argsort()[::-1][:3]                       # top-3 indices

        return {
            "top_specialty": self._classes[top_idx[0]].replace("_", " ").title(),
            "confidence":    round(float(probs[top_idx[0]]) * 100, 1),
            "top_3": [
                {
                    "specialty":  self._classes[i].replace("_", " ").title(),
                    "confidence": round(float(probs[i]) * 100, 1),
                }
                for i in top_idx
            ],
        }

    def extract(self, text: str) -> Dict:
        """Run scispaCy NER. Returns empty dict if model not loaded."""
        if self._ner is None:
            return {"DISEASE": [], "CHEMICAL": [], "note": "NER model not loaded"}
        return self._ner(text)

    def summarize_report(self, text: str) -> Dict:
        """Run BART summarization. Returns placeholder if model not loaded."""
        if self._summarizer is None:
            return {
                "summary": "Summarizer not loaded.",
                "disclaimer": "This is non-diagnostic output.",
            }
        return self._summarizer(text)

    # ── Master method ─────────────────────────────────────────────────────────

    def analyze(self, raw_text: str) -> Dict[str, Any]:
        """
        Run all three models and return a unified result dict.

        Returns:
            {
              "classification": { top_specialty, confidence, top_3 },
              "entities":       { DISEASE: [...], CHEMICAL: [...] },
              "summary":        { summary, disclaimer },
              "disclaimer":     str
            }
        """
        return {
            "classification": self.classify(raw_text),
            "entities":       self.extract(raw_text),
            "summary":        self.summarize_report(raw_text),
            "disclaimer": (
                "Non-diagnostic. For informational/educational use only. "
                "Always consult a licensed physician."
            ),
        }


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = ReportAnalyzer(load_summarizer=False)   # skip BART for quick test

    sample = (
        "Patient presents with severe chest pain radiating to the left arm. "
        "History of hypertension and hyperlipidemia. "
        "On aspirin and atorvastatin. EKG shows ST elevation. "
        "Diagnosis: Acute myocardial infarction."
    )

    result = analyzer.analyze(sample)

    print("── Classification ──────────────────────────────────────")
    clf = result["classification"]
    print(f"  Specialty  : {clf['top_specialty']} ({clf['confidence']}%)")
    for item in clf["top_3"]:
        print(f"  {item['specialty']:<30} {item['confidence']}%")

    print("\n── Entities ────────────────────────────────────────────")
    for label, ents in result["entities"].items():
        if isinstance(ents, list):
            print(f"  {label}: {[e['text'] for e in ents]}")
