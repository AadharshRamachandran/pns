from __future__ import annotations
from pathlib import Path
from typing import Dict,Optional,Tuple
import pandas as pd
DEFAULT_CSV_NAME="telco_churn.csv"
TABLE_FILES={"demographics":"Telco_customer_churn_demographics.xlsx","location":"Telco_customer_churn_location.xlsx","population":"Telco_customer_churn_population.xlsx","services":"Telco_customer_churn_services.xlsx","status":"Telco_customer_churn_status.xlsx"}
_LEAKAGE_COLUMNS={"Churn Reason","Churn Category","Churn Score","Churn Value","Churn Label","Customer Status","Status ID","Satisfaction Score","Satisfaction Score Label"}
class DataLoadError(RuntimeError):
	pass
def load_churn_dataset(base_dir:str|Path,*,return_sources:bool=False)->Tuple[pd.DataFrame,Optional[Dict[str,pd.DataFrame]]]:
	base_path=Path(base_dir).resolve()
	csv_path=base_path/DEFAULT_CSV_NAME
	if csv_path.exists():
		df=pd.read_csv(csv_path)
		return _normalize_schema(df),{"csv":df} if return_sources else None
	tables:Dict[str,pd.DataFrame]={}
	missing=[]
	for key,filename in TABLE_FILES.items():
		file_path=base_path/filename
		if not file_path.exists():
			missing.append(filename)
			continue
		tables[key]=pd.read_excel(file_path)
	if missing:
		raise DataLoadError("Missing Telco churn tables: "+", ".join(missing)+". Provide the consolidated CSV or all component XLSX files.")
	merged=_merge_tables(tables)
	return merged,tables if return_sources else None
def _merge_tables(tables:Dict[str,pd.DataFrame])->pd.DataFrame:
	demographics=tables["demographics"].copy()
	location=tables["location"].copy()
	population=tables["population"].copy()
	services=tables["services"].copy()
	status=tables["status"].copy()
	for frame in(demographics,location,services,status):
		if "Customer ID" in frame.columns and "CustomerID" not in frame.columns:
			frame.rename(columns={"Customer ID":"CustomerID"},inplace=True)
	if {"Zip Code"}.issubset(location.columns) and {"Zip Code"}.issubset(population.columns):
		location=location.merge(population,on="Zip Code",how="left",suffixes=("","_population"))
	join_keys=[key for key in("CustomerID","Quarter") if key in services.columns and key in status.columns]
	merged=services.merge(status,on=join_keys or["CustomerID"],how="left")
	merged=merged.merge(demographics,on="CustomerID",how="left")
	location_columns=[col for col in location.columns if col!="CustomerID" and col not in merged.columns]
	merged=merged.merge(location[["CustomerID",*location_columns]],on="CustomerID",how="left")
	merged.rename(columns={"Gender":"gender","Senior Citizen":"SeniorCitizen","Phone Service":"PhoneService","Multiple Lines":"MultipleLines","Internet Service":"InternetService","Online Security":"OnlineSecurity","Online Backup":"OnlineBackup","Device Protection Plan":"DeviceProtection","Premium Tech Support":"TechSupport","Streaming TV":"StreamingTV","Streaming Movies":"StreamingMovies","Streaming Music":"StreamingMusic","Unlimited Data":"UnlimitedData","Paperless Billing":"PaperlessBilling","Payment Method":"PaymentMethod","Monthly Charge":"MonthlyCharges","Total Charges":"TotalCharges","Total Refunds":"TotalRefunds","Total Long Distance Charges":"TotalLongDistanceCharges","Total Extra Data Charges":"TotalExtraDataCharges","Tenure in Months":"tenure"},inplace=True)
	if "Churn" not in merged.columns:
		if "Churn Label" in merged.columns:
			merged["Churn"]=merged["Churn Label"]
		elif "Churn Value" in merged.columns:
			merged["Churn"]=merged["Churn Value"].map({1:"Yes",0:"No"})
	if "CustomerID" in merged.columns and "customerID" not in merged.columns:
		merged["customerID"]=merged["CustomerID"]
	return _normalize_schema(merged)
def _normalize_schema(df:pd.DataFrame)->pd.DataFrame:
	normalised=df.copy()
	if "TotalCharges" in normalised.columns:
		normalised["TotalCharges"]=pd.to_numeric(normalised["TotalCharges"],errors="coerce")
		total_charges_median=normalised["TotalCharges"].median()
		normalised["TotalCharges"]=normalised["TotalCharges"].fillna(total_charges_median)
	if "SeniorCitizen" in normalised.columns:
		normalised["SeniorCitizen"]=normalised["SeniorCitizen"].map({0:"No",1:"Yes","No":"No","Yes":"Yes"}).fillna(normalised["SeniorCitizen"])
	return normalised
