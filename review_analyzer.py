"""Reusable review-analysis module.

Packages the preprocessing, aspect/risk detection and both trained models from
`sentiment_analysis.ipynb` so they can be used outside the notebook (e.g. by the web demo in `app.py`).

    from review_analyzer import ReviewAnalyzer
    analyzer = ReviewAnalyzer()
    analyzer.analyze("Great fit but the zip broke after a week.")
"""
import html
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "models"
LABELS = ["negative", "neutral", "positive"]
MAX_LEN = 128

# ---------------------------------------------------------------------------
# Preprocessing (identical to Section 4 of the notebook)
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    r"won't": "will not", r"can't": "can not", r"n't": " not", r"'re": " are", r"'s": "",
    r"'d": " would", r"'ll": " will", r"'ve": " have", r"'m": " am",
}
NEGATIONS = r"\b(?:not|no|never|nothing|nobody|none|nor|cannot|without)\b"


def basic_clean(text: str) -> str:
    # Light cleaning safe for any model: decode HTML, drop tags/URLs, collapse whitespace.
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise(text: str) -> str:
    # Lowercase, expand contractions, shorten elongations, keep !/? as sentiment tokens.
    text = text.lower().replace("’", "'")
    for pattern, repl in CONTRACTIONS.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    text = re.sub(r"!+", " ! ", text)
    text = re.sub(r"\?+", " ? ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z!?.,;:_ ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mark_negation(text: str) -> str:
    # Prefix neg_ to words in the scope of a negation (scope ends at punctuation).
    out, negated = [], False
    for tok in re.findall(r"[a-z_]+|[!?.,;:]", text):
        if tok in "!?.,;:":
            negated = False
            out.append(tok)
        elif re.fullmatch(NEGATIONS, tok):
            negated = True
            out.append(tok)
        else:
            out.append(f"neg_{tok}" if negated else tok)
    return " ".join(out)


def clean_text(text: str) -> str:
    # Full pipeline used by the TF-IDF model.
    return mark_negation(normalise(basic_clean(text)))


# ---------------------------------------------------------------------------
# Aspect and risk lexicons (identical to Section 9 of the notebook)
# ---------------------------------------------------------------------------
ASPECTS = {
    "Quality / durability": r"quality|broke|broken|break|durable|flimsy|sturdy|cheaply made|fell apart|stopped working|defect\w*|lasted|last long",
    "Price / value": r"price|value|worth|expensive|overpriced|money|cheap(?!ly)|deal|cost",
    "Delivery / shipping": r"shipping|shipped|delivery|delivered|arrived|arrive|late|on time|took \w+ (?:days|weeks)",
    "Packaging": r"packag\w*|box|wrapped|seal\w*",
    "Size / fit": r"size|sizing|fit|fits|too small|too big|too large|tight|loose|runs small|runs large",
    "Ease of use / setup": r"easy to|install\w*|set ?up|instructions?|assembl\w*|user friendly|complicated",
    "Customer service / returns": r"customer service|seller|refund\w*|return\w*|replacement|warranty|customer support|tech support|contacted",
    "Battery / power": r"batter\w*|charg(?:e|er|ers|es|ing)|power",
    "Appearance / design": r"colou?rs?|looks|looked|design|pictur\w*|photo|style|pretty|beautiful|ugly",
    "Comfort / feel": r"comfortable|comfort|uncomfortable|soft|itchy|scratchy",
}
ASPECT_RE = {a: re.compile(rf"\b(?:{p})\b", re.I) for a, p in ASPECTS.items()}

RISK_TERMS = {
    "safety": r"fire|smoke|burn\w*|melt\w*|shock\w*|spark\w*|explod\w*|injur\w*|hurt|danger\w*|unsafe|choking|toxic|allergic|rash",
    "fraud / counterfeit": r"fake|counterfeit|scam\w*|knock ?off|not genuine|used item|stolen",
    "defect / missing": r"broken|defective|doa|dead on arrival|missing|never arrived|never received|wrong item",
}
RISK_RE = {k: re.compile(rf"\b(?:{p})\b", re.I) for k, p in RISK_TERMS.items()}
RISK_WEIGHT = {"safety": 1.0, "fraud / counterfeit": 0.7, "defect / missing": 0.4}


def split_sentences(text: str) -> list:
    return [s for s in re.split(r"(?<=[.!?;])\s+|\s+but\s+", basic_clean(text)) if len(s.split()) >= 3]


def detect_aspects(sentence: str) -> list:
    return [a for a, rx in ASPECT_RE.items() if rx.search(sentence)]


def detect_risks(text: str) -> list:
    return [k for k, rx in RISK_RE.items() if rx.search(text)]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class ReviewAnalyzer:
    # Loads both trained models and exposes prediction, explanation and full review analysis.

    MODELS = ("bert", "logreg")

    def __init__(self, model_dir: Path = MODEL_DIR, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        bert_dir = Path(model_dir) / "distilbert-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.bert = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(self.device).eval()
        lr = joblib.load(Path(model_dir) / "tfidf_logreg.joblib")
        self.tfidf, self.logreg = lr["tfidf"], lr["model"]

    @torch.no_grad()
    def predict_proba(self, texts, model: str = "bert", batch_size: int = 64) -> np.ndarray:
        # Class probabilities (columns follow LABELS) for a list of texts.
        texts = list(texts)
        if not texts:
            return np.zeros((0, len(LABELS)))
        if model == "logreg":
            return self.logreg.predict_proba(self.tfidf.transform([clean_text(t) for t in texts]))
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer([basic_clean(t) for t in texts[i:i + batch_size]], truncation=True,
                                 max_length=MAX_LEN, padding=True, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.bert(**enc).logits, -1).float().cpu().numpy())
        return np.vstack(out)

    def explain(self, text: str, model: str = "bert", max_words: int = 120) -> list:
        # Occlusion importance: how much p(predicted class) drops when each word is removed.
        words = text.split()[:max_words]
        if len(words) < 2:
            return [{"word": w, "importance": 0.0} for w in words]
        base = self.predict_proba([" ".join(words)], model)[0]
        pred = int(base.argmax())
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        drops = base[pred] - self.predict_proba(variants, model)[:, pred]
        return [{"word": w, "importance": round(float(d), 4)} for w, d in zip(words, drops)]

    def analyze(self, text: str, model: str = "bert", explain: bool = True) -> dict:
        # Full analysis of one review: sentiment, aspects with sentence-level sentiment, risks, explanation.
        proba = self.predict_proba([text], model)[0]
        label = LABELS[int(proba.argmax())]
        mentions = [(s, a) for s in split_sentences(text) for a in detect_aspects(s)]
        sent_proba = self.predict_proba([s for s, _ in mentions], model)
        aspects = [{"aspect": a, "sentence": s, "sentiment": LABELS[int(p.argmax())], "confidence": float(p.max())}
                   for (s, a), p in zip(mentions, sent_proba)]
        risks = detect_risks(text)
        return {
            "model": model,
            "sentiment": label,
            "confidence": float(proba.max()),
            "probabilities": dict(zip(LABELS, map(float, proba))),
            "aspects": aspects,
            "risk_flags": risks,
            "urgency": float(proba[0] * (1 + sum(RISK_WEIGHT[r] for r in risks))),
            "explanation": self.explain(text, model) if explain else None,
        }

    def analyze_batch(self, texts, model: str = "bert") -> list:
        # Batch analysis without explanations (one forward pass for reviews, one for aspect sentences).
        texts = [str(t) for t in texts]
        proba = self.predict_proba(texts, model)
        mentions = [(i, s, a) for i, t in enumerate(texts) for s in split_sentences(t) for a in detect_aspects(s)]
        sent_labels = self.predict_proba([m[1] for m in mentions], model).argmax(1) if mentions else []
        aspects = [dict() for _ in texts]
        for (i, _, a), lab in zip(mentions, sent_labels):
            aspects[i][a] = LABELS[lab]
        results = []
        for t, p, asp in zip(texts, proba, aspects):
            risks = detect_risks(t)
            results.append({
                "text": t,
                "sentiment": LABELS[int(p.argmax())],
                "confidence": round(float(p.max()), 3),
                "p_negative": round(float(p[0]), 3),
                "aspects": asp,
                "risk_flags": risks,
                "urgency": round(float(p[0] * (1 + sum(RISK_WEIGHT[r] for r in risks))), 3),
            })
        return results
