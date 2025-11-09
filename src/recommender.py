"""Recommendation generation with Gemini LLM aligned to rule-based format."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any,Dict,List
import requests
from .rules import customer_rules,global_rules
from .schema import validate_recommendations
from .utils import ensure_dir,save_json
LOGGER=logging.getLogger(__name__)
PROMPT_TEMPLATE="""
You are a telecom retention strategist. Using the JSON context, emit churn mitigation recommendations that cover every insight in `coverage`.
Output Requirements:
- Return a JSON array. Each item must contain exactly the keys: focus, title, action.
- Provide at least one recommendation for every driver in `coverage.drivers` and `coverage.lime_drivers` (a single recommendation may satisfy both lists when they share a name).
- Provide at least one recommendation for every churn reason in `coverage.churn_reasons` and every churn category in `coverage.churn_categories`.
- title must be <= 12 words, imperative tone.
- action must be <= 25 words, specific and outcome-oriented, matching the style in `rule_examples`.
- Keep tone decisive and practical. Do not mention that you are an AI.
- Do not add commentary—only emit valid JSON.
"""
def _sample_rule_examples()->List[Dict[str,str]]:
	return global_rules([{"feature":"MonthlyCharges"},{"feature":"tenure"}],[{"reason":"Competitor offers better price"}],[{"category":"Service"}],{"overall_mean":2.8})[:5]
def _context_from_summary(summary:Dict[str,Any])->Dict[str,Any]:
    shap_top=summary.get("shap_top")or summary.get("shap_top5",[])
    lime_top=summary.get("lime_top")or summary.get("lime_top5",[])
    churn_reasons=summary.get("churn_reasons",[])
    churn_categories=summary.get("churn_categories",[])
    drivers=[item.get("feature") for item in shap_top if item.get("feature")]
    lime_drivers=[item.get("feature") for item in lime_top if item.get("feature")]
    coverage={"drivers":drivers,"lime_drivers":lime_drivers,"churn_reasons":[item.get("reason") for item in churn_reasons if item.get("reason")],"churn_categories":[item.get("category") for item in churn_categories if item.get("category")]}
    return {"shap_top":shap_top,"lime_top":lime_top,"churn_reasons":churn_reasons,"churn_categories":churn_categories,"satisfaction":summary.get("satisfaction",{}),"coverage":coverage}
def _fallback_from_summary(summary:Dict[str,Any])->List[Dict[str,Any]]:
    return global_rules(summary.get("shap_top",summary.get("shap_top5",[])),summary.get("churn_reasons",[]),summary.get("churn_categories",[]),summary.get("satisfaction",{}))
def _call_gemini(context:Dict[str,Any],model_config:Dict[str,Any])->List[Dict[str,Any]]:
	api_key=model_config.get("api_key")or os.environ.get("GEMINI_API_KEY")or os.environ.get("GOOGLE_API_KEY")
	if not api_key:
		raise RuntimeError("Gemini API key missing")
	payload={"contents":[{"role":"user","parts":[{"text":PROMPT_TEMPLATE+"\nrule_examples="+json.dumps(_sample_rule_examples(),ensure_ascii=False)+"\ncontext="+json.dumps(context,ensure_ascii=False)}]}],"generationConfig":{"temperature":float(model_config.get("temperature",0.15)),"maxOutputTokens":int(model_config.get("max_tokens",1024))}}
	endpoint=model_config.get("endpoint","https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent")
	response=requests.post(endpoint,params={"key":api_key},json=payload,timeout=60)
	if response.status_code!=200:
		raise RuntimeError(f"Gemini error {response.status_code}: {response.text[:200]}")
	data=response.json()
	candidates=data.get("candidates")or[]
	text=""
	for candidate in candidates:
		parts=candidate.get("content",{}).get("parts",[])or[]
		for part in parts:
			candidate_text=part.get("text","").strip()
			if candidate_text:
				text=candidate_text
				break
		if text:
			break
	if not text:
		raise RuntimeError("Gemini returned empty text")
	parsed=_parse_json_array(text)
	if not parsed:
		raise RuntimeError("Gemini output not valid focus/title/action JSON")
	return parsed
def _parse_json_array(raw_text:str)->List[Dict[str,Any]]:
	cleaned=raw_text.strip()
	if cleaned.startswith("```"):
		lines=cleaned.splitlines()
		if lines and lines[0].strip().startswith("```"):
			lines=lines[1:]
		if lines and lines[-1].strip().startswith("```"):
			lines=lines[:-1]
		cleaned="\n".join(lines).strip()
	def _loads(text:str)->Any:
		try:return json.loads(text)
		except json.JSONDecodeError:return None
	payload=_loads(cleaned)
	if payload is None:
		start=cleaned.find("[")
		end=cleaned.rfind("]")
		if start==-1 or end==-1 or end<=start:
			return []
		payload=_loads(cleaned[start:end+1])
		if payload is None:
			return []
	if isinstance(payload,dict):
		for key in("recommendations","items","actions"):
			candidate=payload.get(key)
			if isinstance(candidate,list):
				payload=candidate
				break
		else:
			return []
	if not isinstance(payload,list):
		return []
	recommendations=[]
	for item in payload:
		if not isinstance(item,dict):continue
		focus=item.get("focus")
		title=item.get("title")
		action=item.get("action")
		if isinstance(focus,str)and isinstance(title,str)and isinstance(action,str):
			recommendations.append({"focus":focus.strip(),"title":title.strip(),"action":action.strip()})
	return recommendations
def _load_summary(insights_json_path:str,model_config:Dict[str,Any])->Dict[str,Any]:
	summary=model_config.get("summary")if isinstance(model_config,dict)else None
	if summary:return summary
	path=Path(insights_json_path)
	if path.exists():
		try:return json.loads(path.read_text(encoding="utf-8"))
		except json.JSONDecodeError:LOGGER.warning("Failed to parse insights JSON at %s",path)
	return {}
def generate_recommendations(insights_json_path:str,model_config:Dict[str,Any],out_dir:str)->Dict[str,Any]:
	ensure_dir(Path(out_dir))
	summary=_load_summary(insights_json_path,model_config)
	context=_context_from_summary(summary)
	source="gemini"
	try:recommendations=_call_gemini(context,model_config)
	except Exception as exc:
		LOGGER.warning("Falling back to rule-based recommendations: %s",exc)
		recommendations=_fallback_from_summary(context)
		source="rule_fallback"
	if not recommendations:
		recommendations=_fallback_from_summary(context)
		source="rule_fallback"
	validate_recommendations(recommendations)
	json_path=Path(out_dir)/"recommendations.json"
	save_json(recommendations,json_path)
	md_lines=["# Recommendations"]
	for rec in recommendations:
		md_lines.append(f"\n## {rec['focus']}")
		md_lines.append(f"- **Title:** {rec['title']}")
		md_lines.append(f"- **Action:** {rec['action']}")
	md_path=Path(out_dir)/"recommendations.md"
	md_path.write_text("\n".join(md_lines),encoding="utf-8")
	return{"recommendations":recommendations,"json_path":str(json_path),"markdown_path":str(md_path),"source":source}
def customer_llm_recommendations(customer:Dict[str,Any],shap_top:List[Dict[str,Any]],lime_top:List[Dict[str,Any]],reasons:List[Dict[str,Any]],satisfaction:Dict[str,Any],model_config:Dict[str,Any])->List[Dict[str,Any]]:
	summary={"shap_top":shap_top,"lime_top":lime_top,"churn_reasons":reasons,"churn_categories":[],"satisfaction":satisfaction,"customer":customer}
	source="gemini"
	try:recs=_call_gemini(summary,model_config)
	except Exception as exc:
		LOGGER.warning("Customer LLM fallback to rule-based: %s",exc)
		recs=customer_rules(customer)
		source="rule_fallback"
	if not recs:
		recs=customer_rules(customer)
		source="rule_fallback"
	validate_recommendations(recs)
	return recs
