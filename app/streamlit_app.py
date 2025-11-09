from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0,str(PROJECT_ROOT))
from src.rules import customer_rules
from src.recommender import customer_llm_recommendations
def read_json(path):
	if not path:
		return {}
	path=Path(path)
	if not path.exists():
		return {}
	return json.loads(path.read_text(encoding="utf-8"))
def load_artifacts(base_dir:str):
	base=Path(base_dir)
	manifest=read_json(base/"artifacts_manifest.json")
	if not manifest:
		raise FileNotFoundError("Run pipeline first and provide its output directory")
	preprocessor=joblib.load(manifest["preprocessor_path"])
	model=joblib.load(manifest["model_path"])
	metrics=read_json(manifest["metrics_path"])
	summary=read_json(manifest["summary_path"])
	shap_summary=read_json(manifest.get("shap_summary_path"))
	rule_based=read_json(manifest.get("rule_based_path"))
	background_path=manifest.get("shap_sample_path")
	background=pd.read_csv(background_path) if background_path and Path(background_path).exists() else pd.DataFrame()
	return {"preprocessor":preprocessor,"model":model,"metrics":metrics,"summary":summary,"shap_summary":shap_summary,"rule_recs":rule_based.get("recommendations",[]) if isinstance(rule_based,dict) else [],"background":background,"manifest":manifest}
def build_input_widgets(preprocessor, background):
    numeric = {}
    categorical = {}
    artifacts = preprocessor.artifacts
    feature_columns = list(artifacts.feature_columns)
    numeric_columns = list(artifacts.numeric_columns)
    categorical_columns = list(artifacts.categorical_columns)
    defaults = {}
    if not background.empty:
        sample_row = background.iloc[0]
        for col in categorical_columns:
            if col not in artifacts.encoders:
                continue
            encoder = artifacts.encoders[col]
            if col in sample_row:
                try:
                    defaults[col] = encoder.inverse_transform([int(sample_row[col])])[0]
                except Exception:
                    defaults[col] = encoder.classes_[0]
        for col in numeric_columns:
            if col in sample_row:
                idx = numeric_columns.index(col)
                scale = artifacts.scalers.scale_[idx] if hasattr(artifacts.scalers, "scale_") else 1.0
                defaults[col] = float(sample_row[col]) * scale + artifacts.numeric_means.get(col, 0.0)
    for col in numeric_columns:
        value = defaults.get(col, artifacts.numeric_means.get(col, 0.0))
        numeric[col] = st.number_input(col, value=float(value))
    for col in categorical_columns:
        options = artifacts.category_levels.get(col, ["Missing"])
        default = defaults.get(col, options[0] if options else "Missing")
        idx = options.index(default) if default in options else 0
        categorical[col] = st.selectbox(col, options, index=idx)
    data = {**numeric, **categorical}
    ordered = {col: data.get(col) for col in feature_columns if col in data}
    return pd.DataFrame([ordered])
def compute_prediction(model,preprocessor,customer_df):
	transformed=preprocessor.transform(customer_df)
	proba=float(model.predict_proba(transformed)[0,1]) if hasattr(model,"predict_proba") else float(model.predict(transformed)[0])
	return transformed,proba
def explain_instance(model,background,transformed_row,top_k:int=5):
	shap_details=[]
	lime_details=[]
	if background.empty:
		return shap_details,lime_details
	try:
		explainer=shap.Explainer(model,background)
		values=explainer(transformed_row)
		array=values.values[0] if hasattr(values,"values") else np.array(values)[0]
		names=transformed_row.columns.tolist()
		order=np.argsort(np.abs(array))[::-1][:top_k]
		shap_details=[{"feature":names[i],"value":float(array[i]),"abs":float(abs(array[i]))} for i in order]
	except Exception:
		shap_details=[]
	try:
		lime_explainer=LimeTabularExplainer(training_data=background.values,feature_names=background.columns.tolist(),class_names=["No","Yes"],discretize_continuous=True,mode="classification")
		explanation=lime_explainer.explain_instance(data_row=transformed_row.iloc[0].values,predict_fn=lambda data:model.predict_proba(pd.DataFrame(data,columns=background.columns.tolist())),num_features=top_k)
		lime_details=[{"feature":feature,"contribution":float(weight)} for feature,weight in explanation.as_list()[:top_k]]
	except Exception:
		lime_details=[]
	return shap_details,lime_details
def main():
	st.title("Churn Insights Workbench")
	out_dir=st.sidebar.text_input("Pipeline output directory")
	api_key=st.sidebar.text_input("Gemini API key",type="password")
	if not out_dir:
		st.info("Provide the pipeline output directory to load artifacts")
		return
	try:
		artifacts=load_artifacts(out_dir)
	except Exception as exc:
		st.error(str(exc))
		return
	preprocessor=artifacts["preprocessor"]
	model=artifacts["model"]
	metrics=artifacts["metrics"].get("models",{})
	best_name=artifacts["metrics"].get("best_model")
	st.subheader("Model performance")
	if best_name and best_name in metrics:
		st.metric("Best model",best_name)
		st.dataframe(pd.DataFrame(metrics[best_name],index=[0]))
	shap_top=artifacts["summary"].get("shap_top5",[])
	lime_top=artifacts["summary"].get("lime_top5",[])
	reasons=artifacts["summary"].get("churn_reasons",[])
	categories=artifacts["summary"].get("churn_categories",[])
	satisfaction=artifacts["summary"].get("satisfaction",{})
	st.subheader("Global drivers")
	if shap_top:
		st.dataframe(pd.DataFrame(shap_top))
	if lime_top:
		st.dataframe(pd.DataFrame(lime_top))
	cols=st.columns(2)
	with cols[0]:
		if reasons:
			st.dataframe(pd.DataFrame(reasons))
	with cols[1]:
		if categories:
			st.dataframe(pd.DataFrame(categories))
	if satisfaction:
		st.metric("Avg satisfaction",round(float(satisfaction.get("overall_mean",0.0)),2))
	st.subheader("Customer simulation")
	with st.form("customer_form"):
		customer_df=build_input_widgets(preprocessor,artifacts["background"])
		submitted=st.form_submit_button("Score customer")
	if not submitted:
		return
	transformed,probability=compute_prediction(model,preprocessor,customer_df)
	label="Churn" if probability>=0.5 else "Retain"
	st.metric("Churn probability",f"{probability:.3f}",delta=label)
	shap_local,lime_local=explain_instance(model,artifacts["background"],transformed)
	if shap_local:
		st.subheader("SHAP drivers")
		st.dataframe(pd.DataFrame(shap_local))
	if lime_local:
		st.subheader("LIME drivers")
		st.dataframe(pd.DataFrame(lime_local))
	customer_payload=customer_df.iloc[0].to_dict()
	customer_payload["churn_probability"]=probability
	rules=customer_rules(customer_payload)
	if rules:
		st.subheader("Rule-based actions")
		st.dataframe(pd.DataFrame(rules))
	if api_key:
		try:
			llm_recs=customer_llm_recommendations(customer_payload,shap_local if shap_local else shap_top,lime_local if lime_local else lime_top,reasons,satisfaction,{"api_key":api_key})
			if llm_recs:
				st.subheader("LLM recommendations")
				for rec in llm_recs:
					with st.expander(rec.get("id","recommendation")):
						for key,value in rec.items():
							if key=="implementation_steps":
								st.write("Steps:")
								for step in value:
									st.write(f"- {step}")
							elif key!="id":
								st.write(f"{key}: {value}")
		except Exception as exc:
			st.warning(f"LLM recommendations unavailable: {exc}")
if __name__=="__main__":
	main()
