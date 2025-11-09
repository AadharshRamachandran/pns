# churn_insights

End-to-end Python pipeline for telecom churn analysis, explainability, and Gemini-powered business recommendations.

## Key Capabilities

- Load the Telco churn dataset from CSV or XLSX component tables.
- Run advanced exploratory data analysis (EDA) with saved figures and tables.
- Prepare stratified train/validation/test splits with leakage-aware preprocessing.
- Train classical interpretable models (Logistic Regression, Random Forest, XGBoost, CatBoost, Naive Bayes) via grid search.
- Generate SHAP global explanations and LIME local interpretations.
- Extract structured, leakage-free insights and call Gemini to turn them into actionable recommendations.
- Provide rule-based fallback and Streamlit workbench for interactive churn retention planning.
- Produce markdown reports summarising insights and recommendations.

## Project Layout

```
churn_insights/
├── data/
│   └── README.md            # Instructions for placing raw Telco churn data
├── scripts/
│   └── run_pipeline.py      # Single-command CLI orchestrator
├── src/
│   ├── data_loader.py       # CSV/XLSX ingestion and schema normalisation
│   ├── preprocess.py        # Leakage-aware preprocessing & splits
│   ├── model.py             # Training + evaluation of classical models
│   ├── eda.py               # EDA analytics and figures
│   ├── explain.py           # SHAP & LIME explainability utilities
│   ├── insight_extractor.py # Structured insight synthesis
│   ├── recommender.py       # Gemini prompt + recommendation generation
│   └── utils.py             # Shared helpers
├── notebooks/
│   └── demo.ipynb           # (Optional) interactive walkthrough
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10 or newer
- Google Gemini API key (`GEMINI_API_KEY` or `GOOGLE_API_KEY`)
- Telco churn dataset: either consolidated `telco_churn.csv` or component XLSX tables (`Telco_customer_churn_*.xlsx`)

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows PowerShell
   # source .venv/bin/activate      # macOS/Linux
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place data files**
   - Put `telco_churn.csv` (or the original XLSX tables) into `data/raw/`.
   - Review `data/README.md` for details.
4. **Configure environment variables**
   ```bash
   setx GEMINI_API_KEY "your_api_key"        # Windows
   # export GEMINI_API_KEY="your_api_key"   # macOS/Linux
   ```

## Running the Pipeline

```bash
python scripts/run_pipeline.py --data data/raw --out outputs --use-llm
```

- `--data`: folder containing the Telco churn CSV/XLSX files.
- `--out`: destination for all generated artifacts (figures, models, insights, reports).
- `--use-llm`: trigger Gemini recommendation generation. Omit to emit rule-based recommendations only.
- `--gemini-key`: optional inline API key; otherwise set `GEMINI_API_KEY` env var.
- Outputs always include rule-based actions. Gemini recommendations appear only when the flag is supplied and a valid key is available. See `outputs/artifacts_manifest.json` for paths.

### Streamlit Workbench

After running the pipeline, launch the interactive retention workbench:

```bash
streamlit run app/streamlit_app.py
```

- Enter the path used for `--out` to load trained artifacts.
- Score hypothetical customers, view SHAP/LIME drivers, and review rule-based and Gemini (if API key supplied) recommendations.

### Execution Order

1. **Dataset loading** via `src.data_loader`.
2. **EDA generation** storing analytics under `outputs/eda/`.
3. **Preprocessing + splits** with leakage removal.
4. **Model training & evaluation**; best model saved to `outputs/models/`.
5. **Explainability artifacts** (SHAP/LIME) in `outputs/explainability/`.
6. **Insight extraction** to `outputs/knowledge_base/`.
7. **Gemini recommendations** (if enabled) in `outputs/recommendations/`.

## Outputs

- `eda/figures`, `eda/tables`: churn distribution and cohort visuals.
- `models/`: tuned models, evaluation JSON.
- `explainability/`: SHAP value arrays, summary plots, LIME explanations.
- `knowledge_base/`: timestamped insight JSON and manifest.
- `recommendations/`: rule-based JSON + markdown summary, and optional Gemini outputs when enabled.
- `artifacts_manifest.json`: canonical lookup for all generated assets.

## Notebook Demo

(Optional) open `notebooks/demo.ipynb` after running the pipeline to explore artifacts interactively.

## Troubleshooting

- **Missing data files**: ensure all XLSX tables or the CSV exist in `data/raw/`.
- **Gemini errors**: confirm `GEMINI_API_KEY` is set and the model is available in your region.
- **Large dataset / long runtime**: adjust `--out` to a dedicated drive and consider reducing grid-search ranges in `src/model.py`.

Happy churn analyzing!
