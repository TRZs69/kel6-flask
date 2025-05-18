from flask import Flask, render_template, request
import pandas as pd
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# ──────── 1. LOAD & PREPROCESS ─────────

# 1.1 Download dataset
path = kagglehub.dataset_download("khaleeel347/harga-rumah-seluruh-kecamatan-di-kota-bandung")
df = pd.read_csv(f"{path}/results_cleaned.csv")

# 1.2 Clean / rename
df.drop(columns=['house_name'], errors='ignore', inplace=True)
df.rename(columns={'building_area (m2)': 'building_area'}, inplace=True)

# 1.3 One-hot encode locations
df = pd.get_dummies(df, columns=['location'], prefix='loc')
location_cols = [c for c in df.columns if c.startswith("loc_")]
locations      = [c.replace("loc_", "") for c in location_cols]

# 1.4 Prepare features & targets
X_full = df.drop(columns=['price'])
y_price = df['price']

# Classification target: price terciles
df['price_cat'] = pd.qcut(df['price'], 3, labels=[0,1,2])
y_cat = df['price_cat']

# 1.5 Impute
imp = SimpleImputer(strategy='most_frequent')
X_full_imputed = pd.DataFrame(imp.fit_transform(X_full), columns=X_full.columns)

# 1.6 Split into train/test
X_train, X_test, y_price_train, y_price_test = train_test_split(
    X_full_imputed, y_price, test_size=0.2, random_state=42)
_, _, y_cat_train, y_cat_test = train_test_split(
    X_full_imputed, y_cat,   test_size=0.2, random_state=42)

# ──────── 2. TRAIN & EVALUATE ─────────

# Regression
linreg = LinearRegression()
linreg.fit(X_train, y_price_train)
y_pred_lr = linreg.predict(X_test)
mse       = mean_squared_error(y_price_test, y_pred_lr)
r2        = r2_score(y_price_test, y_pred_lr)

# Classification
dt = DecisionTreeClassifier(max_depth=5,x min_samples_split=10, random_state=42)
dt.fit(X_train, y_cat_train)
y_pred_dt = dt.predict(X_test)
acc       = accuracy_score(y_cat_test, y_pred_dt)


# 1. Prediksi semua data (setelah fitting lin_reg_model)
preds = linreg.predict(X_full_imputed)  

# 2. Hitung 33.33% & 66.67% percentiles
t1, t2 = np.percentile(preds, [33.33, 66.67])

# 3. Bulatkan ke jutaan terdekat (10^6)
t1_round = int(np.round(t1, -6))
t2_round = int(np.round(t2, -6))

# 4. Buat HTML‐friendly range text
price_range_text = (
    f"Rentang Harga Berdasarkan Prediksi Linear Regression:<br>"
    f"• Murah: < Rp {t1_round:,.0f}<br>"
    f"• Menengah: Rp {t1_round:,.0f} – Rp {t2_round:,.0f}<br>"
    f"• Mahal: > Rp {t2_round:,.0f}"
)


# ──────── 3. FLASK SETUP ─────────

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Read form inputs
        features = {
            'bedroom_count': float(request.form['bedroom_count']),
            'bathroom_count': float(request.form['bathroom_count']),
            'carport_count': float(request.form['carport_count']),
            'land_area':     float(request.form['land_area']),
            'building_area': float(request.form['building_area']),
        }
        chosen_loc = request.form['location']
        # build full feature vector
        for col in location_cols:
            features[col] = 1 if col == f"loc_{chosen_loc}" else 0

        sample = pd.DataFrame([features])[X_full_imputed.columns]

        # Which model?
        if request.form['model_choice'] == "Regression":
            pred_price = linreg.predict(sample)[0]
            result = f"Prediksi Harga: Rp {pred_price:,.0f}"
            extra  = f"MSE: {mse:,.0f} &nbsp;|&nbsp; R²: {r2:.3f}"
        else:
            cls     = dt.predict(sample)[0]
            label   = price_labels[int(cls)]
            result  = f"Kategori Harga: {label}"
            extra   = range_text + f"<br>Accuracy pada test set: {acc:.3f}"

        return render_template("result.html",
                               result=result,
                               extra=extra)

    # GET: show form + model metrics
    metrics = {
        'mse': f"{mse:,.0f}",
        'r2': f"{r2:.3f}",
        'acc': f"{acc:.3f}"
    }
    return render_template("index.html",
                           locations=locations,
                           metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True)
