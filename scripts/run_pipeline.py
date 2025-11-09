from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Any,Dict
import joblib
import pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0,str(PROJECT_ROOT))
from src.data_loader import DataLoadError,load_churn_dataset
from src.eda import run_eda
from src.preprocess import fit_preprocessor
from src.model import train_and_eval
from src.explain import shap_analysis,lime_explain
from src.insight_extractor import extract_insights
from src.recommender import generate_recommendations
from src.rules import global_rules
from src.utils import ensure_dir,save_json
LOG_FORMAT="[%(levelname)s] %(name)s: %(message)s"
_MODULE_LOGGERS=("src",)
_NOISY_THIRD_PARTY_LOGGERS=("urllib3","google","googleapiclient","google.auth","matplotlib","shap","lime","joblib","cvxpy","xgboost",)
LOGGER=logging.getLogger("pipeline")
def configure_logging(verbose:bool)->None:
	base_level=logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(level=base_level,format=LOG_FORMAT)
	LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
	module_level=logging.INFO if verbose else logging.WARNING
	for name in _MODULE_LOGGERS:
		logging.getLogger(name).setLevel(module_level)
	for name in _NOISY_THIRD_PARTY_LOGGERS:
		logging.getLogger(name).setLevel(logging.WARNING)
def parse_args()->argparse.Namespace:
	parser=argparse.ArgumentParser(description="Run churn insights pipeline")
	parser.add_argument("--data",required=True,help="Path to data directory containing telco churn files")
	parser.add_argument("--out",required=True,help="Output directory for pipeline artifacts")
	parser.add_argument("--use-llm",action="store_true",help="Enable Gemini recommendations")
	parser.add_argument("--verbose",action="store_true",help="Show detailed logging from pipeline modules (defaults to quiet mode)")
	parser.add_argument("--gemini-key",default=None,help="Optional Gemini API key; otherwise GEMINI_API_KEY env var is used")
	return parser.parse_args()
def main()->None:
	args=parse_args()
	configure_logging(verbose=args.verbose)
	data_dir=Path(args.data)
	out_dir=Path(args.out)
	ensure_dir(out_dir)
	LOGGER.info("Loading dataset from %s",data_dir)
	try:
		df, _=load_churn_dataset(data_dir)
	except DataLoadError as exc:
		raise SystemExit(f"Failed to load dataset: {exc}") from exc
	LOGGER.info("Running exploratory analysis")
	eda_dir=out_dir/"eda"
	eda_results=run_eda(df,str(eda_dir))
	LOGGER.info("Fitting preprocessors and splits")
	preprocessor,X_train,X_val,y_train,y_val,X_test,y_test=fit_preprocessor(df,target="Churn",random_state=42)
	LOGGER.info("Training models")
	model_dir=out_dir/"models"
	ensure_dir(model_dir)
	preprocessor_path=model_dir/"preprocessor.joblib"
	joblib.dump(preprocessor,preprocessor_path)
	train_results=train_and_eval(preprocessor,X_train,y_train,X_val,y_val,X_test,y_test,model_dir=str(model_dir))
	best_model_path=Path(train_results["best_model_path"])
	LOGGER.info("Loading best model from %s",best_model_path)
	best_model=joblib.load(best_model_path)
	LOGGER.info("Generating explainability artifacts")
	explain_dir=out_dir/"explainability"
	shap_results=shap_analysis(best_model,X_test,str(explain_dir))
	lime_results:Dict[str,Any]
	if not X_test.empty:
		instance=X_test.iloc[0]
		lime_results=lime_explain(best_model,instance,X_train,str(explain_dir))
	else:
		lime_results={"explanations":[]}
	LOGGER.info("Extracting structured insights")
	insights_dir=out_dir/"knowledge_base"
	insights=extract_insights(eda_results=eda_results,shap_results=shap_results,lime_results=lime_results,out_dir=str(insights_dir))
	insights_path=str(insights_dir/"insights.json")
	ensure_dir(Path(insights_path).parent)
	save_json(insights,Path(insights_path))
	summary_dir=out_dir/"summaries"
	ensure_dir(summary_dir)
	shap_top=insights.get("shap_top",[])
	lime_top=insights.get("lime_top",[])
	summary_payload={
		"shap_top":shap_top,
		"shap_top5":shap_top,
		"lime_top":lime_top,
		"lime_top5":lime_top,
		"churn_reasons":insights.get("churn_reasons",[]),
		"churn_categories":insights.get("churn_categories",[]),
	}
	save_json(summary_payload,summary_dir/"retention_summary.json")
	recommendation_dir=out_dir/"recommendations"
	ensure_dir(recommendation_dir)
	rule_recs=global_rules(shap_top,insights.get("churn_reasons",[]),insights.get("churn_categories",[]),eda_results.get("satisfaction",{}))
	rule_path=recommendation_dir/"rule_based.json"
	rule_md_path=recommendation_dir/"rule_based.md"
	md_lines=["# Rule-based Recommendations"]
	for rec in rule_recs:
		md_lines.append("")
		md_lines.append(f"## {rec['focus']}")
		md_lines.append(f"- **Title:** {rec['title']}")
		md_lines.append(f"- **Action:** {rec['action']}")
		if rec.get("source")=="churn_reason" and rec.get("reason"):
			md_lines.append(f"- **Reason:** {rec['reason']}")
		if rec.get("source")=="churn_category" and rec.get("category"):
			md_lines.append(f"- **Category:** {rec['category']}")
		if rec.get("frequency") is not None:
			md_lines.append(f"- **Frequency:** {rec['frequency']}")
	rule_md_path.write_text("\n".join(md_lines),encoding="utf-8")
	save_json({"recommendations":rule_recs},rule_path)
	recommendation_results:Dict[str,Any]|None=None
	if args.use_llm:
		LOGGER.info("Generating LLM-aligned recommendations (mirrors rule-based drivers)")
		model_config:Dict[str,Any]={"summary":summary_payload}
		if args.gemini_key:
			model_config["api_key"]=args.gemini_key
		recommendation_results=generate_recommendations(insights_json_path=insights_path,model_config=model_config,out_dir=str(recommendation_dir))
		rec_json=recommendation_results["json_path"]
		LOGGER.info("Recommendations saved to %s using %s",rec_json,recommendation_results.get("source","unknown"))
	else:
		LOGGER.info("Skipping mirrored LLM recommendations; rule-based saved to %s",rule_path)

	manifest={"preprocessor_path":str(preprocessor_path),"model_path":train_results["best_model_path"],"best_model_name":train_results["best_model_name"],"metrics_path":train_results["metrics_path"],"summary_path":str(summary_dir/"retention_summary.json"),"rule_based_path":str(rule_path),"shap_summary_path":shap_results.get("summary_json"),"shap_values_path":shap_results.get("values_path"),"shap_sample_path":shap_results.get("sample_path"),"shap_plots":shap_results.get("plots"),"lime_summary_path":lime_results.get("json_path"),"lime_plot_path":lime_results.get("plot_path"),"eda_analytics_path":eda_results.get("analytics_path")}
	manifest["rule_based_markdown_path"]=str(rule_md_path)
	if recommendation_results:
		manifest["llm_recommendations_path"]=recommendation_results["json_path"]
		manifest["llm_recommendations_markdown_path"]=recommendation_results.get("markdown_path")
		manifest["recommendation_source"]=recommendation_results.get("source")
	save_json(manifest,out_dir/"artifacts_manifest.json")
	LOGGER.info("Pipeline complete")
if __name__=="__main__":
	main()
