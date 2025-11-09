from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder,StandardScaler
_LEAKAGE_COLUMNS={"Churn Reason","Churn Category","Churn Score","Churn Value","Churn Label","Customer Status","Status ID","Satisfaction Score","Satisfaction Score Label"}
_ID_COLUMNS={"customerID","CustomerID","Customer ID"}
@dataclass
class PreprocessArtifacts:
	drop_columns:List[str]
	numeric_columns:List[str]
	categorical_columns:List[str]
	scalers:StandardScaler
	encoders:Dict[str,LabelEncoder]
	category_levels:Dict[str,List[str]]
	numeric_means:Dict[str,float]
	feature_columns:List[str]
class Preprocessor:
	def __init__(self,artifacts:PreprocessArtifacts)->None:
		self.artifacts=artifacts
	def transform(self,df:pd.DataFrame)->pd.DataFrame:
		X=df.drop(columns=self.artifacts.drop_columns,errors="ignore").copy()
		for column in self.artifacts.feature_columns:
			if column not in X.columns:
				X[column]=np.nan
		for column in self.artifacts.numeric_columns:
			series=pd.to_numeric(X[column],errors="coerce") if column in X.columns else pd.Series(np.nan,index=X.index)
			fill_value=self.artifacts.numeric_means.get(column,0.0)
			X[column]=series.fillna(fill_value)
		for column in self.artifacts.categorical_columns:
			values=X[column].astype(str) if column in X.columns else pd.Series("Missing",index=X.index)
			encoder=self.artifacts.encoders[column]
			X[column]=_encode_with_fallback(values,encoder)
		X=X[self.artifacts.feature_columns]
		if self.artifacts.numeric_columns:
			X.loc[:,self.artifacts.numeric_columns]=self.artifacts.scalers.transform(X[self.artifacts.numeric_columns])
		return X
def fit_preprocessor(df:pd.DataFrame,target:str,*,test_size:float=0.2,val_size:float=0.2,random_state:int=42)->Tuple[Preprocessor,pd.DataFrame,pd.DataFrame,pd.Series,pd.Series,pd.DataFrame,pd.Series]:
	if target not in df.columns:
		raise ValueError(f"Target column '{target}' missing for preprocessing")
	drop_columns=[target,*sorted(_ID_COLUMNS),*sorted(_LEAKAGE_COLUMNS)]
	X=df.drop(columns=drop_columns,errors="ignore").copy()
	y=df[target].map({"No":0,"Yes":1}).astype(int)
	numeric_columns=X.select_dtypes(include=["number"]).columns.tolist()
	categorical_columns=[col for col in X.columns if col not in numeric_columns]
	encoders:Dict[str,LabelEncoder]={}
	category_levels:Dict[str,List[str]]={}
	for column in categorical_columns:
		series=pd.Series(X[column]).astype(str).fillna("Missing")
		counts=series.value_counts()
		levels=counts.head(20).index.tolist()
		if "Missing" not in levels:
			levels.append("Missing")
		category_levels[column]=levels
		encoder=LabelEncoder()
		encoder.fit(list(counts.index)+["Missing"])
		X[column]=_encode_with_fallback(series,encoder)
		encoders[column]=encoder
	scaler=StandardScaler()
	numeric_means:Dict[str,float]={}
	if numeric_columns:
		X[numeric_columns]=scaler.fit_transform(X[numeric_columns].astype(float))
		numeric_means={col:float(mean) for col,mean in zip(numeric_columns,scaler.mean_)}
	feature_columns=X.columns.tolist()
	splitter=StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
	train_idx,test_idx=next(splitter.split(X,y))
	X_train=X.iloc[train_idx]
	y_train=y.iloc[train_idx]
	X_test=X.iloc[test_idx]
	y_test=y.iloc[test_idx]
	val_fraction=val_size/(1-test_size)
	splitter_val=StratifiedShuffleSplit(n_splits=1,test_size=val_fraction,random_state=random_state)
	train_idx2,val_idx=next(splitter_val.split(X_train,y_train))
	X_train_final=X_train.iloc[train_idx2]
	y_train_final=y_train.iloc[train_idx2]
	X_val=X_train.iloc[val_idx]
	y_val=y_train.iloc[val_idx]
	artifacts=PreprocessArtifacts(drop_columns=drop_columns,numeric_columns=numeric_columns,categorical_columns=categorical_columns,scalers=scaler,encoders=encoders,category_levels=category_levels,numeric_means=numeric_means,feature_columns=feature_columns)
	return(Preprocessor(artifacts),X_train_final,X_val,y_train_final,y_val,X_test,y_test)
def _encode_with_fallback(values:pd.Series,encoder:LabelEncoder)->pd.Series:
	filled=values.fillna("Missing").astype(str)
	if "Missing" not in encoder.classes_:
		encoder.classes_=np.append(encoder.classes_,"Missing")
	mask=~np.isin(filled,encoder.classes_)
	filled[mask]="Missing"
	return pd.Series(encoder.transform(filled),index=values.index)
