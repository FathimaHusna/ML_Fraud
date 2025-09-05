from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class Box(BaseModel):
    page: int
    polygon: List[Tuple[float, float]]


class IdDocNormalized(BaseModel):
    submissionId: str
    docType: Optional[str]
    modelVersion: Optional[str]
    captureTs: Optional[str]
    firstName: Optional[str]
    lastName: Optional[str]
    fullName: Optional[str]
    dob: Optional[str]
    docNumber: Optional[str]
    nationality: Optional[str]
    countryRegion: Optional[str]
    issueDate: Optional[str]
    expiryDate: Optional[str]
    mrzLine1: Optional[str]
    mrzLine2: Optional[str]
    confidences: Dict[str, float]
    boxes: Dict[str, Box | dict]


def _val(fields: Dict[str, Any], name: str) -> Optional[str]:
    el = fields.get(name)
    if not el:
        return None
    return el.get("valueString") or el.get("content") or el.get("valueDate") or el.get("value")


def _conf(fields: Dict[str, Any], name: str) -> Optional[float]:
    el = fields.get(name)
    return el.get("confidence") if el else None


def _box(fields: Dict[str, Any], name: str) -> dict:
    el = fields.get(name) or {}
    br = (el.get("boundingRegions") or [{}])[0]
    poly = br.get("polygon") or []
    page = br.get("pageNumber", 1)
    return {"page": page, "polygon": poly}


def normalize_id(result: Dict[str, Any], submission_id: str) -> IdDocNormalized:
    documents = result.get("documents") or result.get("analyzeResult", {}).get("documents") or []
    doc = documents[0] if documents else {}
    fields: Dict[str, Any] = doc.get("fields", {})
    model_version = result.get("modelVersion") or result.get("apiVersion")
    capture_ts = result.get("createdDateTime") or result.get("timestamp")

    confidences = {k: _conf(fields, k) for k in fields.keys() if _conf(fields, k) is not None}
    boxes = {k: _box(fields, k) for k in fields.keys()}

    return IdDocNormalized(
        submissionId=submission_id,
        docType=doc.get("docType"),
        modelVersion=model_version,
        captureTs=capture_ts,
        firstName=_val(fields, "FirstName"),
        lastName=_val(fields, "LastName"),
        fullName=_val(fields, "FullName"),
        dob=_val(fields, "DateOfBirth"),
        docNumber=_val(fields, "DocumentNumber"),
        nationality=_val(fields, "Nationality"),
        countryRegion=_val(fields, "CountryRegion"),
        issueDate=_val(fields, "DateOfIssue"),
        expiryDate=_val(fields, "DateOfExpiration"),
        mrzLine1=_val(fields, "MachineReadableZoneLine1") or _val(fields, "MachineReadableZone1"),
        mrzLine2=_val(fields, "MachineReadableZoneLine2") or _val(fields, "MachineReadableZone2"),
        confidences=confidences,
        boxes=boxes,
    )

