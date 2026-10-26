# Customer Churn and House Price Dashboard

Two Keras Functional API models with a Streamlit dashboard.

| Model | Task | Test result |
|---|---|---|
| Customer churn | Classification | Accuracy 96.9%, AUC 0.997, churn recall 0.98 |
| KC House price | Regression | R2 0.84, median error about 10% |

Both models use a skip connection: the input is concatenated with the output of the last hidden layer.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py`: Streamlit dashboard with two tabs
- `*.keras`: trained models
- `*_scaler.pkl`, `*_meta.json`: preprocessing (scaler and feature order)
- `notebooks/`: training notebooks