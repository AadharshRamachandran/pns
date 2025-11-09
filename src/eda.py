from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any,Dict,Iterable,List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .utils import ensure_dir,save_json
LOGGER=logging.getLogger(__name__)
_PLOT_STYLE={"axes.facecolor":"white","axes.grid":True,"grid.linestyle":"--","grid.color":"#dddddd"}
_SEGMENT_COLUMNS=("gender","SeniorCitizen","Contract","PaymentMethod","InternetService","TenureBucket","Partner","Dependents")
_NUMERIC_COLUMNS=(("MonthlyCharges","monthly charges"),("TotalCharges","total spend"),("CLTV","customer lifetime value"),("tenure","tenure (months)"))
_CORR_COLUMNS=("MonthlyCharges","TotalCharges","tenure","num_services_active","num_services_flagged","service_engagement_ratio","refund_intensity","auto_pay_flag","is_monthly_contract","payment_complexity","revenue_velocity")
_CHURN_REASON_COLUMNS=("Churn Reason","Churn reason","churn_reason")
_CHURN_CATEGORY_COLUMNS=("Churn Category","Churn category","churn_category")
_SATISFACTION_COLUMNS=("Satisfaction Score","Satisfaction score","satisfaction_score")
def run_eda(df:pd.DataFrame,out_dir:str,top_k:int=10)->Dict[str,Any]:
	if "Churn" not in df.columns:
		raise ValueError("run_eda requires a dataframe with a 'Churn' column")
	output_root=Path(out_dir).resolve()
	figures_dir=output_root/"figures"
	tables_dir=output_root/"tables"
	kb_dir=output_root/"knowledge_base"
	for directory in(output_root,figures_dir,tables_dir,kb_dir):
		ensure_dir(directory)
	sns.set_theme(style="whitegrid")
	for key,value in _PLOT_STYLE.items():
		plt.rcParams[key]=value
	eda_results={"population":{},"segments":{},"numeric_deltas":[],"cohorts":{},"correlations":{},"reasons":{},"categories":{},"satisfaction":{},"figures":[],"tables":[]}
	churn_series=df["Churn"].astype(str)
	churned=churn_series.eq("Yes")
	total_customers=int(len(df))
	churn_count=int(churned.sum())
	churn_rate=float(churn_count/total_customers) if total_customers else 0.0
	eda_results["population"]={"total":total_customers,"churned":churn_count,"overall_churn_rate":churn_rate}
	summary_path=tables_dir/"summary_stats.csv"
	df.describe(include="all").to_csv(summary_path)
	eda_results["tables"].append(str(summary_path))
	if "tenure" in df.columns and "TenureBucket" not in df.columns:
		tenure_bins=[0,6,12,24,48,72,np.inf]
		tenure_labels=["<6m","6-12m","12-24m","24-48m","48-72m","72m+"]
		df=df.assign(TenureBucket=pd.cut(df["tenure"],bins=tenure_bins,labels=tenure_labels,right=False))
	segment_stats:Dict[str,List[Dict[str,Any]]]={}
	for column in _SEGMENT_COLUMNS:
		if column not in df.columns:
			continue
		grouped=df.groupby(column,observed=True)["Churn"].value_counts(normalize=True).unstack().fillna(0)
		if "Yes" not in grouped.columns:
			continue
		ranked=grouped.sort_values("Yes",ascending=False).head(top_k)
		stats=[]
		for value,row in ranked.iterrows():
			subset=df[df[column]==value]
			stats.append({"value":str(value),"churn_rate":float(row["Yes"]),"sample_size":int(len(subset))})
		segment_stats[column]=stats
		fig,ax=plt.subplots(figsize=(8,4))
		sns.barplot(data=ranked.reset_index().rename(columns={"Yes":"churn_rate"}),x="churn_rate",y=column,hue=column,palette="viridis",dodge=False,legend=False,ax=ax)
		ax.set_xlim(0,1)
		ax.set_xlabel("Churn rate")
		ax.set_ylabel(column)
		ax.set_title(f"Churn rate by {column}")
		fig_path=figures_dir/f"churn_by_{column}.png"
		fig.tight_layout()
		fig.savefig(fig_path,dpi=200)
		plt.close(fig)
		eda_results["figures"].append(str(fig_path))
	eda_results["segments"]=segment_stats
	numeric_deltas=[]
	for column,label in _NUMERIC_COLUMNS:
		if column not in df.columns:
			continue
		grouped=df.groupby("Churn")[column].mean().dropna()
		if not {"Yes","No"}.issubset(grouped.index):
			continue
		delta=float(grouped["Yes"]-grouped["No"])
		numeric_deltas.append({"feature":column,"label":label,"churn":float(grouped["Yes"]),"retained":float(grouped["No"]),"delta":delta})
	eda_results["numeric_deltas"]=numeric_deltas
	if {"Contract","TenureBucket"}.issubset(df.columns):
		cohort_table=df.groupby(["Contract","TenureBucket"],observed=True)["Churn"].value_counts(normalize=True).unstack().fillna(0)
		if "Yes" in cohort_table.columns:
			heatmap_data=cohort_table["Yes"].unstack("Contract")
			cohort_path=tables_dir/"cohort_contract_tenure.csv"
			heatmap_data.to_csv(cohort_path)
			eda_results["tables"].append(str(cohort_path))
			eda_results["cohorts"]["contract_tenure"]=heatmap_data.to_dict()
			fig,ax=plt.subplots(figsize=(8,6))
			sns.heatmap(heatmap_data,annot=True,fmt=".2f",cmap="coolwarm",ax=ax)
			ax.set_title("Churn rate by Contract and Tenure bucket")
			fig_path=figures_dir/"heatmap_contract_tenure.png"
			fig.tight_layout()
			fig.savefig(fig_path,dpi=200)
			plt.close(fig)
			eda_results["figures"].append(str(fig_path))
	corr_cols=[col for col in _CORR_COLUMNS if col in df.columns]
	if corr_cols:
		corr_matrix=df[corr_cols].corr(numeric_only=True)
		corr_path=tables_dir/"correlations.csv"
		corr_matrix.to_csv(corr_path)
		eda_results["tables"].append(str(corr_path))
		eda_results["correlations"]["matrix"]=corr_matrix.round(4).to_dict()
		fig,ax=plt.subplots(figsize=(8,6))
		sns.heatmap(corr_matrix,annot=True,fmt=".2f",cmap="vlag",ax=ax)
		ax.set_title("Correlation heatmap (selected features)")
		fig_path=figures_dir/"correlations.png"
		fig.tight_layout()
		fig.savefig(fig_path,dpi=220)
		plt.close(fig)
		eda_results["figures"].append(str(fig_path))
	reason_col=_first_existing(df.columns,_CHURN_REASON_COLUMNS)
	if reason_col:
		reason_counts=df[churned][reason_col].dropna().astype(str).value_counts()
		total=int(reason_counts.sum())
		top_reasons=[{"reason":reason,"count":int(count),"share":float(count/total) if total else 0.0} for reason,count in reason_counts.head(top_k).items()]
		eda_results["reasons"]={"column":reason_col,"top_reasons":top_reasons,"total":total}
	satisfaction_col=_first_existing(df.columns,_SATISFACTION_COLUMNS)
	if satisfaction_col:
		satisfaction=pd.to_numeric(df[satisfaction_col],errors="coerce")
		stats=satisfaction.describe()
		by_churn=pd.concat({"score":satisfaction,"Churn":churn_series},axis=1).dropna().groupby("Churn")["score"].agg(["mean","median","count"])
		eda_results["satisfaction"]={"column":satisfaction_col,"overall_mean":float(stats["mean"]),"overall_std":float(stats["std"]),"by_churn":by_churn.round(3).to_dict()}
	category_col=_first_existing(df.columns,_CHURN_CATEGORY_COLUMNS)
	if category_col:
		category_counts=df[churned][category_col].dropna().astype(str).value_counts()
		total_categories=int(category_counts.sum())
		top_categories=[{"category":category,"count":int(count),"share":float(count/total_categories) if total_categories else 0.0} for category,count in category_counts.head(top_k).items()]
		eda_results["categories"]={"column":category_col,"top_categories":top_categories,"total":total_categories}
	analytics_path=kb_dir/"analytics.json"
	save_json(eda_results,analytics_path)
	eda_results["analytics_path"]=str(analytics_path)
	LOGGER.info("EDA analytics saved to %s",analytics_path)
	return eda_results
def _first_existing(columns:Iterable[str],candidates:Iterable[str])->str|None:
	lower={col.lower():col for col in columns}
	for candidate in candidates:
		match=lower.get(candidate.lower())
		if match:
			return match
	return None
