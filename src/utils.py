from __future__ import annotations
import json
import logging
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Dict
LOGGER=logging.getLogger(__name__)
def ensure_dir(path:Path)->None:
	path.mkdir(parents=True,exist_ok=True)
def timestamped_name(prefix:str,suffix:str)->str:
	stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	return f"{prefix}_{stamp}.{suffix}"
def save_json(data:Dict[str,Any],path:Path)->None:
	ensure_dir(path.parent)
	path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
	LOGGER.debug("Wrote JSON artifact to %s",path)
def load_json(path:Path)->Dict[str,Any]:
	if not path.exists():
		LOGGER.warning("JSON file %s not found; returning empty dict",path)
		return {}
	return json.loads(path.read_text(encoding="utf-8"))
