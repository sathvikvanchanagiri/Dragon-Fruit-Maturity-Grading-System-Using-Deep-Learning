from sqlalchemy import Column, Integer, Float, DateTime, String, Boolean
from sqlalchemy.sql import func
from database.database import Base

class WaterQualityData(Base):
    __tablename__ = "water_quality_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    ph = Column(Float)
    temperature = Column(Float)
    turbidity = Column(Float)
    dissolved_oxygen = Column(Float)
    tds = Column(Float)  # Total Dissolved Solids
    conductivity = Column(Float)
    sensor_id = Column(String)
    location = Column(String)
    is_alert = Column(Boolean, default=False)
    alert_message = Column(String, nullable=True)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    parameter = Column(String)
    value = Column(Float)
    threshold = Column(Float)
    severity = Column(String)  # 'low', 'medium', 'high'
    message = Column(String)
    is_active = Column(Boolean, default=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

class TreatmentLog(Base):
    __tablename__ = "treatment_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    treatment_type = Column(String)
    parameters_affected = Column(String)  # JSON string of affected parameters
    initial_values = Column(String)  # JSON string of initial values
    final_values = Column(String)  # JSON string of final values
    success = Column(Boolean)
    notes = Column(String, nullable=True) 