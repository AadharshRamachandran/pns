from __future__ import annotations
import json
from pathlib import Path
from typing import Any,Dict
from jsonschema import Draft7Validator
RECOMMENDATION_SCHEMA:Dict[str,Any]={"type":"array","items":{"type":"object","required":["focus","title","action"],"properties":{"focus":{"type":"string"},"title":{"type":"string"},"action":{"type":"string"}},"additionalProperties":False}}
_VALIDATOR=Draft7Validator(RECOMMENDATION_SCHEMA)
def validate_recommendations(payload:Any)->None:
	errors=list(_VALIDATOR.iter_errors(payload))
	if errors:
		messages=[error.message for error in errors]
		raise ValueError("Recommendation schema validation failed: "+"; ".join(messages))
def save_schema(path:Path)->None:
	path.write_text(json.dumps(RECOMMENDATION_SCHEMA,indent=2))
