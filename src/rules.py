from __future__ import annotations
from typing import Any,Dict,List
def _feature_action(name:str)->Dict[str,Any]:
	mapping={"MonthlyCharges":("Stabilize monthly spend","Offer targeted bill credit or bundle"),"TotalCharges":("Rebalance total charges","Grant loyalty credit tied to value"),"tenure":("Lift early tenure loyalty","Deliver concierge onboarding sprint"),"Contract":("Move to secure contract","Incentivize annual plan upgrade"),"PaymentMethod":("Strengthen payment experience","Promote auto-pay with fee waiver"),"TechSupport":("Reinforce tech support","Schedule proactive diagnostic outreach"),"InternetService":("Optimize connectivity","Run network quality review"),"OnlineSecurity":("Bundle protective add-ons","Offer complimentary security trial"),"StreamingTV":("Refresh entertainment bundle","Curate content set to usage"),"DeviceProtection":("Protect devices","Provide device health check upgrade"),"SeniorCitizen":("Assist seniors","Assign dedicated retention hotline"),"Dependents":("Family-centric plan","Introduce shared-benefit family bundle")}
	title,action=mapping.get(name,(f"Act on {name}","Review touchpoints and launch focused retention play"))
	return{"source":"feature","focus":name,"title":title,"action":action}
def _reason_action(reason:str,frequency:int|float|None=None)->Dict[str,Any]:
	lowered=reason.lower()
	if "competitor" in lowered:
		rec={"source":"churn_reason","focus":"Competition","title":"Counter competitor pull","action":"Deploy save offers combining price match and differentiated features"}
	elif "price" in lowered or "cost" in lowered:
		rec={"source":"churn_reason","focus":"Pricing","title":"Address price objections","action":"Extend retention credit and communicate value stack"}
	elif "service" in lowered:
		rec={"source":"churn_reason","focus":"Service","title":"Fix service friction","action":"Open rapid response case with root-cause fix and follow-up"}
	elif "support" in lowered:
		rec={"source":"churn_reason","focus":"Support","title":"Upgrade support journey","action":"Assign success manager and guarantee resolution SLAs"}
	else:
		rec={"source":"churn_reason","focus":"Experience","title":"Elevate experience","action":"Run empathy outreach and co-create fix roadmap"}
	rec["reason"]=reason
	if frequency is not None:
		rec["frequency"]=frequency
	return rec
def _category_action(category:str,frequency:int|float|None=None)->Dict[str,Any]:
	rec={"source":"churn_category","focus":category,"title":"Stabilize churn category","action":"Build dedicated playbook for this churn journey with cross-functional squad","category":category}
	if frequency is not None:
		rec["frequency"]=frequency
	return rec
def global_rules(top_features:List[Dict[str,Any]],reasons:List[Dict[str,Any]],categories:List[Dict[str,Any]],satisfaction:Dict[str,Any])->List[Dict[str,Any]]:
	recs:List[Dict[str,Any]]=[]
	seen=set()
	for item in top_features:
		name=item.get("feature")
		if not name or name in seen:
			continue
		recs.append(_feature_action(name))
		seen.add(name)
	for item in reasons[:3]:
		reason=item.get("reason")
		if reason:
			frequency=item.get("frequency") if "frequency" in item else item.get("count")
			recs.append(_reason_action(reason,frequency))
	for item in categories[:3]:
		category=item.get("category")
		if category:
			frequency=item.get("frequency") if "frequency" in item else item.get("count")
			recs.append(_category_action(category,frequency))
	mean=satisfaction.get("overall_mean") if satisfaction else None
	if isinstance(mean,(int,float)):
		focus="Satisfaction"
		if mean<3:
			recs.append({"source":"satisfaction","focus":focus,"title":"Rebuild satisfaction","action":"Launch rapid NPS recovery sprint with weekly listening posts"})
		elif mean<4:
			recs.append({"source":"satisfaction","focus":focus,"title":"Elevate sentiment","action":"Introduce surprise delight moments and proactive check-ins"})
	return recs
def customer_rules(customer:Dict[str,Any])->List[Dict[str,Any]]:
	recs=[]
	monthly=float(customer.get("MonthlyCharges",0)or 0)
	tenure=float(customer.get("tenure",0)or 0)
	payment=str(customer.get("PaymentMethod",""))
	contract=str(customer.get("Contract",""))
	satisfaction=float(customer.get("Satisfaction Score",customer.get("Satisfaction",0))or 0)
	if monthly>=80:
		recs.append({"focus":"MonthlyCharges","title":"Reduce bill shock","action":"Offer tailored discount or bundle to keep spend below threshold"})
	if tenure<12:
		recs.append({"focus":"tenure","title":"Strengthen onboarding","action":"Assign welcome specialist and trigger milestone rewards"})
	if payment.lower().strip()=="electronic check":
		recs.append({"focus":"PaymentMethod","title":"Shift to stickier payments","action":"Incentivize auto-pay enrollment with one-month credit"})
	if contract.lower().strip()=="month-to-month":
		recs.append({"focus":"Contract","title":"Promote term upgrade","action":"Provide device upgrade credit for switching to annual plan"})
	if satisfaction and satisfaction<=3:
		recs.append({"focus":"Satisfaction","title":"Close feedback loop","action":"Call back within 24 hours and resolve top pain point"})
	if not recs:
		recs.append({"focus":"Retention","title":"Personalize engagement","action":"Conduct needs assessment and tailor retention bundle"})
	return recs
