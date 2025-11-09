"""Model training pipeline."""
from __future__ import annotations
import json
import logging
import warnings
from pathlib import Path
from typing import Any,Dict,List,Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from .preprocess import Preprocessor
from .utils import ensure_dir,save_json
LOGGER=logging.getLogger(__name__)
_GRID_LOGREG={"model__C":[0.25,0.5,1.0,2.0],"model__penalty":["l2"],"model__solver":["lbfgs","newton-cg"],"model__max_iter":[4000,8000,12000]}
_GRID_XGB={"model__max_depth":[4,6,8],"model__learning_rate":[0.05,0.1],"model__n_estimators":[300,500]}
_GRID_RF={"model__n_estimators":[200,400],"model__max_depth":[None,10,20],"model__min_samples_split":[2,5]}
_GRID_CATBOOST={"model__depth":[6,8],"model__learning_rate":[0.05,0.1],"model__iterations":[300,500]}
_GRID_NB={"model__var_smoothing":[1e-9,1e-8,1e-7]}
def train_and_eval(preprocessor:Preprocessor,X_train:pd.DataFrame,y_train:pd.Series,X_val:pd.DataFrame,y_val:pd.Series,X_test:pd.DataFrame,y_test:pd.Series,model_dir:str)->Dict[str,Any]:
	ensure_dir(Path(model_dir))
	results:Dict[str,Dict[str,float]]={}
	best_model_name=None
	best_auc=-np.inf
	best_pipeline:Pipeline|None=None
	model_artifacts:Dict[str,Any]={}
	candidates:List[Tuple[str,Pipeline,Dict[str,List[Any]]]]=[
		("logistic_regression",Pipeline([("model",LogisticRegression(class_weight="balanced",max_iter=4000,solver="lbfgs"))]),_GRID_LOGREG),
		("xgboost",Pipeline([("model",XGBClassifier(eval_metric="logloss"))]),_GRID_XGB),
		("random_forest",Pipeline([("model",RandomForestClassifier(class_weight="balanced",random_state=42))]),_GRID_RF),
		("catboost",Pipeline([("model",CatBoostClassifier(verbose=0,loss_function="Logloss",random_seed=42))]),_GRID_CATBOOST),
		("naive_bayes",Pipeline([("model",GaussianNB())]),_GRID_NB)
	]
	for name,pipeline,param_grid in candidates:
		LOGGER.info("Training %s",name)
		grid=GridSearchCV(pipeline,param_grid=param_grid,cv=3,scoring="roc_auc",n_jobs=-1)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			grid.fit(X_train,y_train)
		best_estimator=grid.best_estimator_
		val_pred_prob=_predict_proba_or_decision(best_estimator,X_val)
		val_pred=(val_pred_prob>=0.5).astype(int)
		metrics=_compute_metrics(y_val,val_pred,val_pred_prob)
		results[name]={"roc_auc":metrics["roc_auc"],"accuracy":metrics["accuracy"],"precision":metrics["precision"],"recall":metrics["recall"],"f1":metrics["f1"]}
		model_path=Path(model_dir)/f"{name}.joblib"
		joblib.dump(best_estimator,model_path)
		model_artifacts[name]={"model_path":str(model_path),"best_params":grid.best_params_}
		if metrics["roc_auc"]>best_auc:
			best_auc=metrics["roc_auc"]
			best_model_name=name
			best_pipeline=best_estimator
	if best_pipeline is None or best_model_name is None:
		raise RuntimeError("No model trained successfully")
	LOGGER.info("Refitting best model (%s) on train+val",best_model_name)
	X_combined=pd.concat([X_train,X_val],axis=0)
	y_combined=pd.concat([y_train,y_val],axis=0)
	best_pipeline.fit(X_combined,y_combined)
	test_pred_prob=_predict_proba_or_decision(best_pipeline,X_test)
	test_pred=(test_pred_prob>=0.5).astype(int)
	test_metrics=_compute_metrics(y_test,test_pred,test_pred_prob)
	results[best_model_name]["test_roc_auc"]=test_metrics["roc_auc"]
	results[best_model_name]["test_accuracy"]=test_metrics["accuracy"]
	results[best_model_name]["test_precision"]=test_metrics["precision"]
	results[best_model_name]["test_recall"]=test_metrics["recall"]
	results[best_model_name]["test_f1"]=test_metrics["f1"]
	final_model_path=Path(model_dir)/f"best_{best_model_name}.joblib"
	joblib.dump(best_pipeline,final_model_path)
	metrics_path=Path(model_dir)/"model_metrics.json"
	save_json({"models":results,"best_model":best_model_name},metrics_path)
	return{"best_model_name":best_model_name,"best_model_path":str(final_model_path),"metrics":results,"artifacts":model_artifacts,"metrics_path":str(metrics_path)}
def _compute_metrics(y_true:pd.Series,y_pred:np.ndarray,y_prob:np.ndarray)->Dict[str,float]:
	return{"roc_auc":float(roc_auc_score(y_true,y_prob)),"accuracy":float(accuracy_score(y_true,y_pred)),"precision":float(precision_score(y_true,y_pred,zero_division=0)),"recall":float(recall_score(y_true,y_pred,zero_division=0)),"f1":float(f1_score(y_true,y_pred,zero_division=0))}
def _predict_proba_or_decision(model:Pipeline,X:pd.DataFrame)->np.ndarray:
	if hasattr(model,"predict_proba"):
		proba=model.predict_proba(X)
		if proba.ndim==2:
			return proba[:,1]
	if hasattr(model,"decision_function"):
		decision=model.decision_function(X)
		if decision.ndim==1:
			decision=(decision-decision.min())/(decision.max()-decision.min()+1e-9)
		return decision
	predictions=model.predict(X)
	return predictions.astype(float)
