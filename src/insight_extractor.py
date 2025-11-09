from __future__ import annotations
import logging
from pathlib import Path
from typing import Any,Dict,List
from .utils import ensure_dir,save_json
LOGGER=logging.getLogger(__name__)
_LEAKAGE_TERMS={"churn reason","churn category","churn value","churn label","churn score","customer status","satisfaction score","status id"}
def extract_insights(eda_results:Dict[str,Any],shap_results:Dict[str,Any],lime_results:Dict[str,Any],out_dir:str)->Dict[str,Any]:
	ensure_dir(Path(out_dir))
	shap_top=[item for item in shap_results.get("top_features",[])if not _contains_leakage(item.get("feature",""))][:5]
	ordered_features=[item["feature"]for item in shap_top if item.get("feature")]
	lime_map:Dict[str,float]={}
	for entry in lime_results.get("explanations",[]):
		feature_text=str(entry.get("feature",""))
		if not feature_text or _contains_leakage(feature_text):
			continue
		contribution=float(entry.get("contribution",0.0))
		lower_text=feature_text.lower()
		matched=False
		for name in ordered_features:
			if name.lower() in lower_text:
				lime_map[name]=lime_map.get(name,0.0)+contribution
				matched=True
		if matched:
			continue
	lime_top=[{"feature":name,"contribution":lime_map[name]}for name in ordered_features if name in lime_map]
	raw_reasons=eda_results.get("reasons",{}).get("top_reasons",[])[:5]
	reasons=[{"reason":item.get("reason"),"frequency":int(item.get("count",0))}for item in raw_reasons if item.get("reason")]
	raw_categories=eda_results.get("categories",{}).get("top_categories",[])[:5]
	categories=[{"category":item.get("category"),"frequency":int(item.get("count",0))}for item in raw_categories if item.get("category")]
	payload={"shap_top":shap_top,"lime_top":lime_top,"churn_reasons":reasons,"churn_categories":categories,"analytics_path":eda_results.get("analytics_path")}
	insights_path=Path(out_dir)/"insights.json"
	save_json(payload,insights_path)
	LOGGER.info("Insights written to %s",insights_path)
	return payload
def _contains_leakage(name:str)->bool:
	lower=name.lower()
	return any(term in lower for term in _LEAKAGE_TERMS)
