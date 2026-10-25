# Retail Customer Feedback Analysis: Review Sentiment Classification

BC3415 Individual Assignment, Option A (coding-based submission).

This project classifies Amazon product reviews as **positive**, **neutral** or **negative**. It then turns those predictions into product insights: a pain-point map showing sentiment per aspect, recurring complaint themes, reviews whose text contradicts the star rating, and a queue of urgent issues.

## Results (5,000 unseen test reviews)

| Model | Accuracy | Macro-F1 | Neutral F1 |
|---|---|---|---|
| TF-IDF + Complement Naive Bayes | 0.713 | 0.618 | 0.276 |
| TF-IDF + Linear SVM | **0.733** | 0.673 | 0.401 |
| TF-IDF + Logistic Regression | 0.723 | 0.687 | 0.462 |
| **DistilBERT (fine-tuned)** | 0.731 | **0.699** | **0.476** |

Macro-F1 is the primary metric because the classes are imbalanced (neutral is 20% of reviews).

## Project structure

```
sentiment_analysis.ipynb   main notebook (run top to bottom; this is also the demo)
requirements.txt           Python dependencies
models/
  tfidf_logreg.joblib      TF-IDF + Logistic Regression pipeline
  distilbert-sentiment/    fine-tuned DistilBERT (weights, tokenizer, training log)
review_analyzer.py         reusable module: preprocessing, aspects, risk flags, both models
app.py                     FastAPI backend for the web demo
static/                    web demo front-end (HTML/CSS/JS) + 300 sample reviews
outputs/
  new_reviews.csv          example batch of new reviews
  new_reviews_scored.csv   the same reviews scored by ReviewAnalyzer
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

The dataset (`SetFit/amazon_reviews_multi_en`) downloads automatically from HuggingFace.

By default the notebook reuses the saved DistilBERT model and runs in about 10 minutes. To fine-tune from scratch, set `CONFIG["retrain_bert"] = True`. That takes about 30 minutes on an Apple-silicon Mac and less on a CUDA GPU.

## Using the model on new reviews

After running the notebook up to Section 10:

```python
analyzer = ReviewAnalyzer()  # loads models/distilbert-sentiment
analyzer.analyze(["Great fit but the zip broke after a week."])
analyzer.analyze_csv("my_reviews.csv", text_col="review_text")  # writes my_reviews_scored.csv
```

Each review gets a sentiment label, a confidence score, the aspects it mentions (each with its own sentiment) and any risk flags.

## Web demo

```bash
.venv/bin/uvicorn app:app --port 8000
# then open http://localhost:8000
```

- **Analyze a review**: type a review or pick an example. You get the sentiment, the class probabilities, word-level highlighting showing why the model decided, the sentiment for each aspect mentioned, and an escalation alert for safety, counterfeit or defect issues.
- **Batch analysis**: upload a CSV, or click *Use 300 sample Amazon reviews*. You get KPI tiles, a pain-point map, an urgent-issue queue and a searchable, filterable table of all reviews. You can download the scored CSV.
- The switch at the top right toggles between the fine-tuned DistilBERT model and the fast TF-IDF + Logistic Regression model.

The API can also be called directly:

```bash
curl -X POST localhost:8000/api/analyze -H 'Content-Type: application/json' \
     -d '{"text": "Arrived broken and the seller ignored me.", "model": "bert"}'
```
