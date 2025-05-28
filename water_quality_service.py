from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
import json

from models.models import WaterQualityData, Alert
from models.schemas import WaterQualityDataCreate

class WaterQualityService:
    # Thresholds for different parameters
    THRESHOLDS = {
        'ph': {'min': 6.5, 'max': 8.5},
        'temperature': {'min': 10, 'max': 30},
        'turbidity': {'min': 0, 'max': 5},
        'dissolved_oxygen': {'min': 4, 'max': 10},
        'tds': {'min': 0, 'max': 500},
        'conductivity': {'min': 0, 'max': 1500}
    }

    def add_sensor_data(self, db: Session, data: WaterQualityDataCreate) -> WaterQualityData:
        # Create new water quality data entry
        db_data = WaterQualityData(**data.dict())
        
        # Check for alerts
        alerts = self._check_parameters(db_data)
        if alerts:
            db_data.is_alert = True
            db_data.alert_message = json.dumps(alerts)
            
            # Create alert entries
            for alert in alerts:
                db_alert = Alert(
                    parameter=alert['parameter'],
                    value=alert['value'],
                    threshold=alert['threshold'],
                    severity=alert['severity'],
                    message=alert['message']
                )
                db.add(db_alert)
        
        db.add(db_data)
        db.commit()
        db.refresh(db_data)
        return db_data

    def get_sensor_data(self, db: Session, skip: int = 0, limit: int = 100) -> List[WaterQualityData]:
        return db.query(WaterQualityData).offset(skip).limit(limit).all()

    def get_current_status(self, db: Session) -> Dict[str, Any]:
        # Get the latest reading for each sensor
        latest_data = db.query(WaterQualityData).order_by(
            WaterQualityData.timestamp.desc()
        ).first()
        
        if not latest_data:
            return {"error": "No data available"}
        
        return {
            "timestamp": latest_data.timestamp,
            "parameters": {
                "ph": latest_data.ph,
                "temperature": latest_data.temperature,
                "turbidity": latest_data.turbidity,
                "dissolved_oxygen": latest_data.dissolved_oxygen,
                "tds": latest_data.tds,
                "conductivity": latest_data.conductivity
            },
            "status": "alert" if latest_data.is_alert else "normal",
            "location": latest_data.location,
            "sensor_id": latest_data.sensor_id
        }

    def get_active_alerts(self, db: Session) -> List[Dict[str, Any]]:
        alerts = db.query(Alert).filter(
            Alert.is_active == True
        ).order_by(Alert.timestamp.desc()).all()
        
        return [
            {
                "id": alert.id,
                "parameter": alert.parameter,
                "value": alert.value,
                "threshold": alert.threshold,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp
            }
            for alert in alerts
        ]

    def _check_parameters(self, data: WaterQualityData) -> List[Dict[str, Any]]:
        alerts = []
        
        # Check pH
        if data.ph < self.THRESHOLDS['ph']['min']:
            alerts.append({
                'parameter': 'ph',
                'value': data.ph,
                'threshold': self.THRESHOLDS['ph']['min'],
                'severity': 'high',
                'message': f'pH level too low: {data.ph}'
            })
        elif data.ph > self.THRESHOLDS['ph']['max']:
            alerts.append({
                'parameter': 'ph',
                'value': data.ph,
                'threshold': self.THRESHOLDS['ph']['max'],
                'severity': 'high',
                'message': f'pH level too high: {data.ph}'
            })

        # Check temperature
        if data.temperature < self.THRESHOLDS['temperature']['min']:
            alerts.append({
                'parameter': 'temperature',
                'value': data.temperature,
                'threshold': self.THRESHOLDS['temperature']['min'],
                'severity': 'medium',
                'message': f'Temperature too low: {data.temperature}°C'
            })
        elif data.temperature > self.THRESHOLDS['temperature']['max']:
            alerts.append({
                'parameter': 'temperature',
                'value': data.temperature,
                'threshold': self.THRESHOLDS['temperature']['max'],
                'severity': 'high',
                'message': f'Temperature too high: {data.temperature}°C'
            })

        # Check other parameters similarly
        # ... (similar checks for turbidity, dissolved_oxygen, tds, conductivity)

        return alerts 