from __future__ import annotations
import logging
from pathlib import Path
from typing import Any,Dict,List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import ClassifierMixin
from .utils import ensure_dir,save_json
LOGGER=logging.getLogger(__name__)
def shap_analysis(model:ClassifierMixin,X:pd.DataFrame,out_dir:str,top_n:int=15,sample_size:int=500)->Dict[str,Any]:
	ensure_dir(Path(out_dir))
	figures_dir=Path(out_dir)/"figures"
	data_dir=Path(out_dir)/"data"
	for directory in(figures_dir,data_dir):
		ensure_dir(directory)
	sample=X.sample(min(sample_size,len(X)),random_state=42)if len(X)>sample_size else X
	feature_names=sample.columns.tolist()
	try:
		explainer=shap.Explainer(model,sample)
		shap_values=explainer(sample)
		shap_array=shap_values.values if hasattr(shap_values,"values")else np.array(shap_values)
	except Exception as exc:
		LOGGER.warning("Default SHAP explainer failed (%s); falling back to KernelExplainer",exc)
		def predict_fn(data:np.ndarray)->np.ndarray:
			return model.predict_proba(pd.DataFrame(data,columns=feature_names))[:,1]
		background=sample.iloc[:min(100,len(sample))]
		kernel_explainer=shap.KernelExplainer(predict_fn,background)
		shap_array=np.array(kernel_explainer.shap_values(sample,nsamples=100))
		if shap_array.ndim>2:
			shap_array=shap_array[1]
	if shap_array.ndim==1:
		shap_array=shap_array.reshape(1,-1)
	mean_abs=np.abs(shap_array).mean(axis=0)
	ranked_idx=np.argsort(mean_abs)[::-1]
	top_idx=ranked_idx[:top_n]
	top_features=[{"feature":feature_names[i],"mean_abs_shap":float(mean_abs[i])}for i in top_idx]
	shap_values_path=data_dir/"shap_values.npy"
	np.save(shap_values_path,shap_array)
	summary_json_path=data_dir/"shap_summary.json"
	save_json({"top_features":top_features},summary_json_path)
	sample_path=data_dir/"shap_sample.csv"
	sample.to_csv(sample_path,index=False)
	plt.figure(figsize=(8,6))
	shap.summary_plot(shap_array,features=sample,feature_names=feature_names,plot_type="bar",show=False)
	bar_path=figures_dir/"shap_summary_bar.png"
	plt.tight_layout()
	plt.savefig(bar_path,dpi=200)
	plt.close()
	plt.figure(figsize=(10,6))
	shap.summary_plot(shap_array,features=sample,feature_names=feature_names,show=False)
	beeswarm_path=figures_dir/"shap_beeswarm.png"
	plt.tight_layout()
	plt.savefig(beeswarm_path,dpi=200)
	plt.close()
	dependence_paths=[]
	for feature in[item["feature"]for item in top_features[:5]]:
		plt.figure(figsize=(6,4))
		shap.dependence_plot(feature,shap_array,sample,feature_names=feature_names,show=False)
		dep_path=figures_dir/f"shap_dependence_{feature}.png"
		plt.tight_layout()
		plt.savefig(dep_path,dpi=200)
		plt.close()
		dependence_paths.append(str(dep_path))
	return{"top_features":top_features,"values_path":str(shap_values_path),"summary_json":str(summary_json_path),"sample_path":str(sample_path),"plots":{"bar":str(bar_path),"beeswarm":str(beeswarm_path),"dependence":dependence_paths}}
def lime_explain(model:ClassifierMixin,instance:pd.Series,X_train:pd.DataFrame,out_dir:str,num_features:int=10)->Dict[str,Any]:
	ensure_dir(Path(out_dir))
	figures_dir=Path(out_dir)/"figures"
	data_dir=Path(out_dir)/"data"
	ensure_dir(figures_dir)
	ensure_dir(data_dir)
	explainer=LimeTabularExplainer(training_data=X_train.values,feature_names=X_train.columns.tolist(),class_names=["No","Yes"],discretize_continuous=True,mode="classification")
	def predict_fn(data:np.ndarray)->np.ndarray:
		return model.predict_proba(pd.DataFrame(data,columns=X_train.columns))
	explanation=explainer.explain_instance(data_row=instance.values,predict_fn=predict_fn,num_features=num_features)
	explanation_list=explanation.as_list()
	explanation_json=[{"feature":feature,"contribution":float(weight)}for feature,weight in explanation_list]
	explanation_path=data_dir/"lime_explanation.json"
	save_json({"explanations":explanation_json},explanation_path)
	fig=explanation.as_pyplot_figure()
	fig_path=figures_dir/"lime_explanation.png"
	fig.tight_layout()
	fig.savefig(fig_path,dpi=200)
	plt.close(fig)
	return{"explanations":explanation_json,"json_path":str(explanation_path),"plot_path":str(fig_path)}
