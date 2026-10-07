import argparse
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from phish_utils import META_COLS, clean_text, iter_local_emails, meta_features, tidy

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="phishing_email.csv")
ap.add_argument("--out", default="model.joblib")
ap.add_argument("--extra-legit", help="folder of your OWN real emails (.eml / .mbox) to add as legitimate")
ap.add_argument("--feedback", help="feedback.csv saved by the app (rows with a true_label)")
ap.add_argument("--repeat", type=int, default=5, help="how many times to repeat extra emails (they are few vs 82k)")
args = ap.parse_args()

with open(args.data, "r", encoding="utf-8-sig") as f:
    first = f.readline().strip().lower()
if "text" in first and "label" in first:
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    df = df.rename(columns={"text_combined": "text"})
else:
    df = pd.read_csv(args.data, encoding="utf-8-sig", header=None, names=["text", "label"])
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)
print("Rows:", len(df))

print("Cleaning text (takes a few minutes on the full dataset)...")
df["clean_text"] = df["text"].apply(clean_text)
meta = pd.DataFrame([meta_features(t) for t in df["text"]], columns=META_COLS, index=df.index)

tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df["label"])

extra_texts, extra_labels = [], []
if args.extra_legit:
    for e in iter_local_emails(args.extra_legit):
        extra_texts.append(tidy(f'{e["subject"]}\n{e["body"]}'))
        extra_labels.append(0)
    print("Extra legitimate emails loaded:", len(extra_texts))
if args.feedback:
    fb = pd.read_csv(args.feedback)
    extra_texts += fb["text"].astype(str).tolist()
    extra_labels += fb["true_label"].astype(int).tolist()
    print("Feedback rows loaded:", len(fb))

train_clean = df.loc[tr_idx, "clean_text"].tolist()
train_meta = meta.loc[tr_idx].values.tolist()
ytr_list = df.loc[tr_idx, "label"].tolist()
for txt, lab in zip(extra_texts, extra_labels):
    for _ in range(args.repeat):
        train_clean.append(clean_text(txt))
        train_meta.append(meta_features(txt))
        ytr_list.append(lab)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
Xtr_tf = tfidf.fit_transform(train_clean)
Xte_tf = tfidf.transform(df.loc[te_idx, "clean_text"])

scaler = MaxAbsScaler()
Xtr = hstack([Xtr_tf, scaler.fit_transform(train_meta)]).tocsr()
Xte = hstack([Xte_tf, scaler.transform(meta.loc[te_idx].values)]).tocsr()
ytr, yte = pd.Series(ytr_list), df.loc[te_idx, "label"]

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)

for name, m in [("Random Forest", rf), ("Logistic Regression", lr)]:
    p = m.predict(Xte)
    print(f"{name:20s} acc={accuracy_score(yte, p):.4f} prec={precision_score(yte, p):.4f} "
          f"rec={recall_score(yte, p):.4f} f1={f1_score(yte, p):.4f}")

joblib.dump({"tfidf": tfidf, "scaler": scaler, "rf": rf, "lr": lr}, args.out, compress=3)
print("Saved ->", args.out)
