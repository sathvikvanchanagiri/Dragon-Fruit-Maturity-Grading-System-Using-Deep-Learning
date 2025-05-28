from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class WaterQualityDataBase(BaseModel):
    ph: float = Field(..., ge=0, le=14, description="pH level of water")
    temperature: float = Field(..., ge=0, le=100, description="Temperature in Celsius")
    turbidity: float = Field(..., ge=0, description="Turbidity in NTU")
    dissolved_oxygen: float = Field(..., ge=0, description="Dissolved oxygen in mg/L")
    tds: float = Field(..., ge=0, description="Total Dissolved Solids in ppm")
    conductivity: float = Field(..., ge=0, description="Conductivity in µS/cm")
    sensor_id: str
    location: str

class WaterQualityDataCreate(WaterQualityDataBase):
    pass

class WaterQualityData(WaterQualityDataBase):
    id: int
    timestamp: datetime
    is_alert: bool
    alert_message: Optional[str] = None

    class Config:
        from_attributes = True

class AlertBase(BaseModel):
    parameter: str
    value: float
    threshold: float
    severity: str
    message: str

class AlertCreate(AlertBase):
    pass

class Alert(AlertBase):
    id: int
    timestamp: datetime
    is_active: bool
    resolved_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class TreatmentRecommendation(BaseModel):
    recommendations: List[Dict[str, str]]
    priority: str
    estimated_impact: Dict[str, float]
    confidence_score: float

class TreatmentLogBase(BaseModel):
    treatment_type: str
    parameters_affected: List[str]
    initial_values: Dict[str, float]
    final_values: Dict[str, float]
    success: bool
    notes: Optional[str] = None

class TreatmentLogCreate(TreatmentLogBase):
    pass

class TreatmentLog(TreatmentLogBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True 