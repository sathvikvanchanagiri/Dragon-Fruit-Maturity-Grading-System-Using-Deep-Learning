# Intelligent Water Quality Monitoring and Treatment System

This project implements an IoT-based intelligent water quality monitoring and treatment system that can:
- Monitor water quality parameters in real-time
- Analyze water quality data using machine learning
- Provide intelligent treatment recommendations
- Generate alerts for critical conditions
- Store historical data for analysis

## Features

- Real-time monitoring of key water quality parameters:
  - pH levels
  - Temperature
  - Turbidity
  - Dissolved Oxygen
  - Total Dissolved Solids (TDS)
  - Conductivity
- Machine learning-based water quality analysis
- Automated treatment recommendations
- Real-time alerts and notifications
- Historical data visualization
- RESTful API for data access
- Web-based dashboard

## Project Structure

```
water-quality-system/
├── backend/
│   ├── api/              # REST API endpoints
│   ├── models/           # ML models for water quality analysis
│   ├── sensors/          # Sensor interface modules
│   └── utils/            # Utility functions
├── frontend/
│   ├── src/             # React frontend source
│   └── public/          # Static assets
├── database/            # Database schemas and migrations
├── docs/               # Documentation
└── tests/              # Test cases
```

## Prerequisites

- Python 3.8+
- Node.js 14+
- PostgreSQL
- Arduino IDE (for sensor programming)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/water-quality-system.git
cd water-quality-system
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Configure the database:
```bash
# Create a PostgreSQL database
createdb water_quality_db
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Access the web dashboard at `http://localhost:3000`

## Hardware Requirements

- Arduino/ESP32 microcontroller
- Water quality sensors:
  - pH sensor
  - Temperature sensor
  - Turbidity sensor
  - Dissolved Oxygen sensor
  - TDS sensor
  - Conductivity sensor
- Power supply
- Waterproof enclosures
- Connecting wires and cables

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 