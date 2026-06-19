import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -------------------- LOAD & MERGE DATASETS --------------------

# Original 2005-era UCI SMS Spam Collection
original = pd.read_csv("../dataset/spam.csv", encoding="latin-1")[["v1", "v2"]]
original.columns = ["label", "message"]

# Modern phishing/spam augmentation (covers 2020s patterns:
# fake prizes, phishing links, fake bank alerts, fake offers, etc.)
augmented = pd.read_csv("augmented_modern.csv", encoding="utf-8")
augmented.columns = ["label", "message"]

data = pd.concat([original, augmented], ignore_index=True)
data = data.drop_duplicates(subset="message").reset_index(drop=True)

print(f"Original dataset: {len(original)} rows")
print(f"Augmented (modern) dataset: {len(augmented)} rows")
print(f"Combined (deduped): {len(data)} rows")
print(data["label"].value_counts())
print()

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Train-test split (stratified so both classes are proportionally
# represented in train and test sets)
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"],
    test_size=0.2, random_state=42, stratify=data["label"]
)

# -------------------- VECTORIZE --------------------

# ngram_range=(1,2) lets the model learn short phrases ("click here",
# "verify now") in addition to single words -- single words like "click"
# or "verify" alone were too weak a signal on their own in testing.
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------- TRAIN --------------------

# Switched from MultinomialNB to LogisticRegression:
# - Naive Bayes assumes word independence and tends to produce
#   poorly-calibrated probabilities (e.g. stuck near 40-50% when
#   signal is weak/mixed) -- exactly the symptom we diagnosed.
# - Logistic Regression directly optimizes for calibrated probability
#   estimates and generally handles overlapping vocabulary better.
# class_weight="balanced" compensates for spam still being a minority
# class even after augmentation.
model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
model.fit(X_train_vec, y_train)

# -------------------- EVALUATE --------------------

y_pred = model.predict(X_test_vec)
print("=== Evaluation on held-out test set ===")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, y_pred))

# -------------------- SAVE --------------------

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model retrained (Logistic Regression + merged dataset) and saved successfully")
