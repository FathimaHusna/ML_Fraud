from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class TxnRaw(BaseModel):
    """Raw transaction schema (bronze). Fields align with product doc.

    All timestamps are ISO8601 UTC strings.
    Amount is in LKR.
    """

    transaction_id: str = Field(..., alias="transactionId")
    user_id: str = Field(..., alias="userId")
    payee_id: Optional[str] = Field(None, alias="payeeId")
    merchant_id: Optional[str] = Field(None, alias="merchantId")
    merchant_category: Optional[str] = Field(None, alias="merchantCategory")
    amount: float
    timestamp: str
    channel: Optional[str] = None  # e.g., web, mobile, ussd
    device_id: Optional[str] = None
    device_type: Optional[str] = None  # e.g., iPhone, Android, PC
    device_os: Optional[str] = None
    ip: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    # Optional training label, if known
    is_fraud: Optional[int] = None


class TxnNormalized(BaseModel):
    """Normalized transaction (silver) with minimal harmonization.

    Keep fields flat and consistently named for feature builders.
    """

    transactionId: str
    userId: str
    payeeId: Optional[str] = None
    merchantId: Optional[str] = None
    merchantCategory: Optional[str] = None
    amount: float
    timestamp: str  # ISO8601
    channel: Optional[str] = None
    deviceId: Optional[str] = None
    deviceType: Optional[str] = None
    deviceOs: Optional[str] = None
    ip: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    is_fraud: Optional[int] = None


def normalize_txn(bronze: TxnRaw) -> TxnNormalized:
    return TxnNormalized(
        transactionId=bronze.transaction_id,
        userId=bronze.user_id,
        payeeId=bronze.payee_id,
        merchantId=bronze.merchant_id,
        merchantCategory=bronze.merchant_category,
        amount=float(bronze.amount),
        timestamp=bronze.timestamp,
        channel=(bronze.channel or "").lower() or None,
        deviceId=bronze.device_id,
        deviceType=bronze.device_type,
        deviceOs=bronze.device_os,
        ip=bronze.ip,
        city=(bronze.city or None),
        region=(bronze.region or None),
        country=(bronze.country or None),
        lat=bronze.lat,
        lon=bronze.lon,
        is_fraud=bronze.is_fraud,
    )

