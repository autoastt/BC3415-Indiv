# Video walkthrough script (about 5 minutes)

**Before recording:** open `sentiment_analysis.ipynb` with all outputs already rendered; don't re-run anything on camera. Collapse the long training-log output in Section 7. Zoom the browser to about 110%. Times are approximate.

---

### 0:00–0:30 · Problem (title cell)
> "A large online retailer gets thousands of reviews every day, far too many to read. My solution automatically classifies each review as positive, neutral or negative, then goes one step further: it tells the business *what* customers are unhappy about and *which* reviews need urgent action. It's built in Python with scikit-learn and HuggingFace Transformers."

Scroll slowly over the contents table.

### 0:30–0:55 · Setup and data (Sections 1–2)
> "Everything is driven by one CONFIG dictionary, so the pipeline can be pointed at a new dataset or model without touching the code. I use about 200,000 real Amazon reviews. Since retailers rarely have hand-labelled sentiment, I use the star rating: 1–2 stars is negative, 3 is neutral, 4–5 is positive. I also remove duplicates and any training review that appears in the test set, to avoid leakage."

### 0:55–1:20 · EDA (Section 3)
Point at the class chart.
> "Neutral is only 20% of the data, so accuracy alone would be misleading. I use macro-F1 as my main metric and weight the classes during training. Negative reviews are also longer: unhappy customers explain what went wrong."

### 1:20–2:00 · Preprocessing (Section 4)
Point at the step-by-step example output.
> "My cleaning pipeline removes HTML and URLs, expands contractions and shortens 'sooooo'. Two choices are specific to sentiment. First, I *don't* remove stop-words, because they include 'not' and 'never'. Second, I mark negation: in 'not worth the money', 'worth' becomes 'neg_worth', so the model can tell 'worth it' from 'not worth it'."

Point at the ablation table.
> "I didn't just assume this helps; I tested it. General cleaning alone barely changes the score, but negation marking gives the best result, so that's the step that matters."

### 2:00–2:40 · Models and explainability (Sections 5–7)
> "I trained three classical baselines on TF-IDF features: Naive Bayes, Logistic Regression and a linear SVM. Then I fine-tuned DistilBERT, a compact transformer, using a class-weighted loss."

Show the top-features chart.
> "The linear model is fully transparent. You can see the words that drive each class, including learned negation features like neg_recommend."

Show the occlusion chart.
> "For BERT, I remove one word at a time and measure how the prediction changes, which shows which words the model relied on."

### 2:40–3:15 · Comparison and error analysis (Section 8)
Show the comparison chart.
> "DistilBERT wins on macro-F1, 0.70, and on every per-class F1, even though it saw only 15% of the training data. Logistic Regression is close behind at 0.69 and is about 90 times faster, so it's a good low-cost fallback. Accuracy is around 73%, but the error analysis shows the model almost never confuses positive with negative: only about 2.5% of the time. Most errors involve neutral, and many of those are actually label noise. Look at these '3-star' reviews that clearly read as positive."

### 3:15–4:15 · Business insights (Section 9). Spend the most time here.
Show the pain-point map.
> "This is where the text insights are combined. I split every review into sentences, detect which aspect each sentence is about (quality, delivery, price and so on) and classify each sentence separately. So 'love the colour, but delivery took three weeks' counts as positive for appearance and negative for delivery. Customer service and packaging are the most negative aspects, and quality has the most complaints in total, so that's where to act first."

Show the topics table.
> "Topic modelling on negative reviews finds complaint themes automatically, with no predefined list."

Show the mismatch and urgency tables.
> "I also flag reviews where the text contradicts the stars, such as a glowing review with 1 star, so they don't unfairly hurt a product's rating. And an urgency score ranks safety, counterfeit and defect complaints, like 'arrived completely melted', for same-day escalation."

### 4:15–5:00 · Reusable tool and live web demo
> "Finally, everything is packaged as a reusable module, review_analyzer.py, and served through a FastAPI backend with a web front end, so business users don't need to touch code."

Switch to the browser at http://localhost:8000:
1. Click the **Mixed** example. Point at the word highlighting and the aspect table: *colour* and *fit* are positive, *delivery* and *packaging* are negative.
2. Click **Safety issue** and show the escalation alert.
3. Flip the model switch to **TF-IDF + LogReg** to show the same review under the fast model.
4. Open **Batch analysis**, click **Use 300 sample Amazon reviews**, and walk through the KPI tiles, the pain-point map and the urgent queue. Click **Download scored CSV**.

### 5:00–5:15 · Close (Section 11)
> "To summarise: DistilBERT for daily scoring, Logistic Regression as a fast, explainable fallback, and the aspect and urgency analysis to turn reviews into actions. Next steps would be a human-labelled evaluation set and adding the customer images with a multimodal model. Thank you."
