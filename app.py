# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# ──────── 1. LOAD & PREPROCESS ─────────

# Download & read cleaned CSV
path = kagglehub.dataset_download("khaleeel347/harga-rumah-seluruh-kecamatan-di-kota-bandung")
df = pd.read_csv(f"{path}/results_cleaned.csv")

# Drop/rename
df.drop(columns=['house_name'], errors='ignore', inplace=True)
df.rename(columns={'building_area (m2)': 'building_area'}, inplace=True)

# One-hot encode location
df = pd.get_dummies(df, columns=['location'], prefix='loc')
location_cols = [c for c in df.columns if c.startswith("loc_")]
locations     = [c.replace("loc_", "") for c in location_cols]

# Features & targets
X = df.drop(columns=['price'])
y_price = df['price']
# classification target: terciles
df['price_cat'] = pd.qcut(df['price'], 3, labels=[0,1,2])
y_cat = df['price_cat']

# Impute missing
imp = SimpleImputer(strategy='most_frequent')
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, y_price_train, y_price_test = train_test_split(
    X_imp, y_price, test_size=0.2, random_state=42)
_, _, y_cat_train, y_cat_test = train_test_split(
    X_imp, y_cat,   test_size=0.2, random_state=42)

# ──────── 2. TRAIN & EVALUATE ─────────

# 2.1 Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_price_train)
y_pred_reg = rf_reg.predict(X_test)
mse = mean_squared_error(y_price_test, y_pred_reg)
r2  = r2_score(y_price_test, y_pred_reg)

# 2.2 Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_cat_train)
y_pred_clf = rf_clf.predict(X_test)
acc = accuracy_score(y_cat_test, y_pred_clf)

# 2.3 Dynamic, rounded tercile thresholds
preds_all = rf_reg.predict(X_imp)
t1, t2   = np.percentile(preds_all, [33.33, 66.67])
t1_rnd   = int(np.round(t1, -6))
t2_rnd   = int(np.round(t2, -6))

price_labels = {0: "Murah", 1: "Menengah", 2: "Mahal"}
price_range_text = (
    f"Rentang Harga (RF tertiles):<br>"
    f"• Murah: < Rp {t1_rnd:,.0f}<br>"
    f"• Menengah: Rp {t1_rnd:,.0f} – Rp {t2_rnd:,.0f}<br>"
    f"• Mahal: > Rp {t2_rnd:,.0f}"
)

# ──────── 3. FLASK APP & ROUTES ─────────

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Gather inputs
        feat = {
            'bedroom_count': float(request.form['bedroom_count']),
            'bathroom_count': float(request.form['bathroom_count']),
            'carport_count': float(request.form['carport_count']),
            'land_area':     float(request.form['land_area']),
            'building_area': float(request.form['building_area'])
        }
        loc = request.form['location']
        for col in location_cols:
            feat[col] = 1 if col == f"loc_{loc}" else 0

        sample = pd.DataFrame([feat])[X_imp.columns]

        # Regression branch
        price_pred = rf_reg.predict(sample)[0]
        price_text = f"Prediksi Harga: Rp {price_pred:,.0f}"

        # Classification branch
        cls = rf_clf.predict(sample)[0]
        cls_text = f"Kategori Harga: {price_labels[int(cls)]}"
        cls_extra = price_range_text + f"<br>Accuracy: {acc:.3f}"

        return render_template("result.html",
                               price_text=price_text,
                               cls_text=cls_text,
                               cls_extra=cls_extra)

    # GET: show form + metrics
    metrics = {
        'mse': f"{mse:,.0f}",
        'r2':  f"{r2:.3f}",
        'acc': f"{acc:.3f}"
    }
    return render_template("index.html",
                           locations=locations,
                           metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True)
