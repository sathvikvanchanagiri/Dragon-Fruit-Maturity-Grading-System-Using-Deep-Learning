from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import uvicorn

from database.database import get_db
from models.schemas import WaterQualityData, TreatmentRecommendation
from services.water_quality_service import WaterQualityService
from services.treatment_service import TreatmentService

app = FastAPI(title="Water Quality Monitoring System",
             description="API for monitoring and analyzing water quality parameters",
             version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
water_quality_service = WaterQualityService()
treatment_service = TreatmentService()

@app.get("/")
async def root():
    return {"message": "Welcome to Water Quality Monitoring System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/sensor-data", response_model=WaterQualityData)
async def add_sensor_data(data: WaterQualityData, db: Session = Depends(get_db)):
    """Add new water quality sensor data"""
    try:
        return water_quality_service.add_sensor_data(db, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sensor-data", response_model=List[WaterQualityData])
async def get_sensor_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get historical water quality data"""
    try:
        return water_quality_service.get_sensor_data(db, skip, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current-status")
async def get_current_status(db: Session = Depends(get_db)):
    """Get current water quality status"""
    try:
        return water_quality_service.get_current_status(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/treatment-recommendation", response_model=TreatmentRecommendation)
async def get_treatment_recommendation(db: Session = Depends(get_db)):
    """Get treatment recommendations based on current water quality"""
    try:
        current_status = water_quality_service.get_current_status(db)
        return treatment_service.get_recommendation(current_status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
async def get_alerts(db: Session = Depends(get_db)):
    """Get active alerts for critical water quality conditions"""
    try:
        return water_quality_service.get_active_alerts(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 