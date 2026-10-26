import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

BASE = Path(__file__).parent  # app.py wala folder

st.set_page_config(page_title="ML Dashboard", page_icon="📊", layout="wide")
st.title("📊 ML Dashboard")
st.caption("Functional API models: Customer Churn (classification) and KC House Price (regression)")


# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_churn():
    model = keras.models.load_model(BASE / "Customer_Transactions.keras")
    scaler = joblib.load(BASE / "churn_scaler.pkl")
    with open(BASE / "churn_meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta


@st.cache_resource
def load_house():
    model = keras.models.load_model(BASE / "house_price.keras")
    scaler = joblib.load(BASE / "house_scaler.pkl")
    with open(BASE / "house_meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta


def pretty(name: str) -> str:
    return name.replace("_", " ").title()



def missing_files(names):
    return [n for n in names if not (BASE / n).exists()]


tab_churn, tab_house = st.tabs(["🧑‍💼 Customer Churn", "🏠 House Price"])

# ---------------------------------------------------------------- churn tab
with tab_churn:
    st.subheader("Churn prediction")
    churn_ok = True
    miss = missing_files(["Customer_Transactions.keras", "churn_scaler.pkl", "churn_meta.json"])
    if miss:
        st.error(f"Yeh files is folder mein nahi mili: {', '.join(miss)}")
        st.caption(f"Folder: {BASE}")
        churn_ok = False
    else:
        try:
            c_model, c_scaler, c_meta = load_churn()
        except Exception as e:
            st.error(f"Churn files load nahi hui: {e}")
            churn_ok = False

    if churn_ok:
        columns = c_meta["columns"]
        num_cols = c_meta["num_cols"]
        country_cols = [c for c in columns if c.startswith("country_")]
        # scaler ne jo mean/std seekhi, usse sliders ki range banate hain
        stats = {c: (float(m), float(s)) for c, m, s in zip(num_cols, c_scaler.mean_, c_scaler.scale_)}

        left, right = st.columns(2)
        values = {}

        with left:
            st.markdown("**Customer details**")
            for col in num_cols:
                mean, std = stats[col]
                lo = float(max(0.0, mean - 4 * std))
                hi = float(mean + 4 * std)
                values[col] = st.slider(pretty(col), lo, hi, float(mean), key=f"churn_{col}")

        with right:
            st.markdown("**Profile**")
            if "gender" in columns:
                g = st.selectbox("Gender", ["Female", "Male"])
                values["gender"] = 0 if g == "Female" else 1

            if country_cols:
                # drop_first=True ki wajah se ek country (baseline) columns mein nahi hoti
                ALL_COUNTRIES = ["Brazil", "France", "Germany", "India", "Japan", "South Africa", "UK", "USA"]
                present = {c.replace("country_", "").lower() for c in country_cols}
                baseline = [n for n in ALL_COUNTRIES if n.lower() not in present]
                base_label = baseline[0] if len(baseline) == 1 else "Other (baseline country)"
                options = [base_label] + [c.replace("country_", "") for c in country_cols]
                chosen = st.selectbox("Country", options)
                for c in country_cols:
                    values[c] = 1 if chosen == c.replace("country_", "") else 0

            leftover = [c for c in columns if c not in values]
            for col in leftover:
                values[col] = st.number_input(pretty(col), value=0.0, key=f"churn_extra_{col}")

            threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                                  help="Zyada threshold = kam galat alarm, lekin kuch churn miss ho sakte hain.")

        if st.button("Predict churn", type="primary"):
            row = pd.DataFrame([values])[columns].astype(float)
            row[num_cols] = c_scaler.transform(row[num_cols])
            prob = float(c_model.predict(row.values.astype("float32"), verbose=0)[0][0])

            m1, m2 = st.columns(2)
            m1.metric("Churn probability", f"{prob * 100:.1f}%")
            if prob >= threshold:
                m2.error("⚠️ High churn risk")
            else:
                m2.success("✅ Low churn risk")
            st.progress(min(max(prob, 0.0), 1.0))

# ---------------------------------------------------------------- house tab
with tab_house:
    st.subheader("House price prediction (King County)")
    house_ok = True
    miss = missing_files(["house_price.keras", "house_scaler.pkl", "house_meta.json"])
    if miss:
        st.error(f"Yeh files is folder mein nahi mili: {', '.join(miss)}")
        st.caption(f"Folder: {BASE}")
        house_ok = False
    else:
        try:
            h_model, h_scaler, h_meta = load_house()
        except Exception as e:
            st.error(f"House files load nahi hui: {e}")
            house_ok = False

    if house_ok:
        cols = h_meta["columns"]
        ranges = h_meta["ranges"]  # {col: [min, max, median]}
        inputs = {}

        binary = {"waterfront", "was_renovated"}
        decimals = {"lat", "long"}
        half_steps = {"bathrooms", "floors"}

        grid = st.columns(3)
        for i, col in enumerate(cols):
            lo, hi, med = ranges[col]
            with grid[i % 3]:
                label = pretty(col)
                if col in binary:
                    inputs[col] = 1 if st.checkbox(label, value=bool(med), key=f"house_{col}") else 0
                elif col in decimals:
                    inputs[col] = st.number_input(label, min_value=lo, max_value=hi, value=med,
                                                  step=0.001, format="%.4f", key=f"house_{col}")
                elif col in half_steps:
                    inputs[col] = st.slider(label, float(lo), float(hi), float(med), 0.25 if col == "bathrooms" else 0.5,
                                            key=f"house_{col}")
                else:
                    inputs[col] = st.slider(label, int(lo), int(hi), int(med), key=f"house_{col}")

        if st.button("Predict price", type="primary"):
            row = pd.DataFrame([inputs])[cols].astype(float)
            scaled = h_scaler.transform(row)
            log_price = float(h_model.predict(scaled, verbose=0)[0][0])
            price = float(np.expm1(log_price))
            st.metric("Estimated price", f"${price:,.0f}")
            st.caption("Model ka typical error takreeban 10% hai (test set ka median).")