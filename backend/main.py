#main.py
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from geopy.geocoders import Nominatim
import asyncio
import aiohttp
from cachetools import TTLCache
import logging
from dotenv import load_dotenv 
load_dotenv() 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forest_fire_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Forest Fire Risk Monitoring API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for API responses (TTL: 5 minutes)
cache = TTLCache(maxsize=100, ttl=300)

# Now, this line will work correctly because load_dotenv() has loaded the variable
NASA_FIRMS_API_KEY = os.getenv("NASA_FIRMS_API_KEY")
if not NASA_FIRMS_API_KEY:
    logger.error("❌ NASA FIRMS API Key is not set in environment variables.")
    logger.error("💡 Please create a .env file with: NASA_FIRMS_API_KEY=your_api_key_here")
    raise ValueError("NASA FIRMS API Key is required. Check your .env file.")

if len(NASA_FIRMS_API_KEY) < 20:  # Basic validation
    logger.warning("⚠️ API key seems too short. Please verify it's correct.")

logger.info(f"✅ NASA FIRMS API Key loaded successfully (length: {len(NASA_FIRMS_API_KEY)})")

# Enhanced configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 8000)),
    'debug': os.getenv('API_DEBUG', 'True').lower() == 'true',
    'cache_ttl': int(os.getenv('CACHE_TTL', 300)),
    'cache_maxsize': int(os.getenv('CACHE_MAXSIZE', 100))
}

logger.info(f"🔧 API Configuration: {API_CONFIG}")

# Updated to use AREA API instead of COUNTRY API
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# India bounding box coordinates - Updated for Area API format (west,south,east,north)
INDIA_COORDINATES = "68.1,6.5,97.4,35.7"  # Complete India territory

# Regional coordinate options
INDIA_REGIONS = {
    'complete': '68.1,6.5,97.4,35.7',  # Complete India
    'north': '68.1,24.0,88.2,37.0',     # North India
    'south': '68.1,8.0,80.3,20.0',      # South India  
    'west': '68.1,15.0,78.0,24.0',      # West India
    'east': '80.0,20.0,97.4,28.0',      # East India
}

class FireData(BaseModel):
    latitude: float
    longitude: float
    bright_ti4: float
    scan: float
    track: float
    acq_date: str
    acq_time: str
    satellite: str
    instrument: str
    confidence: str
    version: str
    bright_ti5: float
    frp: float
    daynight: str
    risk_level: int
    risk_label: str
    state: Optional[str] = None
    district: Optional[str] = None

class PredictionRequest(BaseModel):
    date: str
    day_range: int
    satellite: Optional[str] = None
    daynight: Optional[str] = None
    risk_level: Optional[int] = None
    region: Optional[str] = 'complete' 

class FireDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.confidence_mapping = {'l': 0, 'n': 1, 'h': 2}
        self.daynight_mapping = {'D': 1, 'N': 0}
        self.risk_labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
        self.load_model()
        
    def load_model(self):
        """Load the pre-trained model"""
        try:
            # In a real application, load from file
            # self.model = joblib.load('forest_fire_model.joblib')
            # For demonstration, we'll create a mock model
            self.model = self._create_mock_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a more sophisticated mock model for demonstration"""
        class MockModel:
            def predict(self, X):
                """Enhanced prediction logic based on multiple factors"""
                predictions = []
                for row in X:
                    try:
                        # Extract features safely
                        lat = row[0] if len(row) > 0 else 0
                        lon = row[1] if len(row) > 1 else 0
                        brightness = row[2] if len(row) > 2 else 0
                        bright_t31 = row[3] if len(row) > 3 else 0
                        scan = row[4] if len(row) > 4 else 0
                        track = row[5] if len(row) > 5 else 0
                        hour = row[6] if len(row) > 6 else 12
                        month = row[7] if len(row) > 7 else 6
                        weekday = row[8] if len(row) > 8 else 1
                        confidence = row[9] if len(row) > 9 else 0
                        daynight = row[10] if len(row) > 10 else 1

                        # More sophisticated risk assessment
                        risk_score = 0

                        # Brightness factor (most important)
                        if brightness > 0.8:
                            risk_score += 3
                        elif brightness > 0.6:
                            risk_score += 2
                        elif brightness > 0.4:
                            risk_score += 1

                        # Confidence factor
                        if confidence >= 2:  # High confidence
                            risk_score += 2
                        elif confidence >= 1:  # Normal confidence
                            risk_score += 1

                        # Temporal factors
                        if hour >= 10 and hour <= 16:  # Peak fire hours
                            risk_score += 1

                        if month in [3, 4, 5, 10, 11, 12]:  # Fire season months
                            risk_score += 1

                        # Geographic factors (basic)
                        if 15 <= lat <= 30 and 70 <= lon <= 90:  # High-risk regions
                            risk_score += 1

                        # Determine final risk level
                        if risk_score >= 5:
                            predictions.append(2)  # High risk
                        elif risk_score >= 3:
                            predictions.append(1)  # Moderate risk
                        else:
                            predictions.append(0)  # Low risk

                    except Exception as e:
                        logger.warning(f"Error in prediction for row: {e}")
                        predictions.append(0)  # Default to low risk

                return np.array(predictions)

        return MockModel()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for model prediction"""
        try:
            # Convert acq_time to 4-digit padded format
            df['acq_time'] = df['acq_time'].astype(str).str.zfill(4)
            
            # Create acq_datetime
            df['acq_datetime'] = pd.to_datetime(
                df['acq_date'] + ' ' + df['acq_time'].str[:2] + ':' + df['acq_time'].str[2:],
                format='%Y-%m-%d %H:%M'
            )
            
            # Normalize brightness, bright_t31, frp
            df['brightness'] = df['bright_ti4']
            df['bright_t31'] = df['bright_ti5']
            
            # Fit scaler on current data (in production, use pre-fitted scaler)
            numeric_cols = ['brightness', 'bright_t31', 'frp']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = self.scaler.fit_transform(df[[col]])
            
            # Extract temporal features
            df['year'] = df['acq_datetime'].dt.year
            df['month'] = df['acq_datetime'].dt.month
            df['day'] = df['acq_datetime'].dt.day
            df['hour'] = df['acq_datetime'].dt.hour
            df['weekday'] = df['acq_datetime'].dt.weekday
            
            # Encode categorical variables
            df['confidence_encoded'] = df['confidence'].map(self.confidence_mapping)
            df['daynight_encoded'] = df['daynight'].map(self.daynight_mapping)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict fire risk using the model"""
        try:
            # Select features for prediction
            feature_columns = [
                'latitude', 'longitude', 'brightness', 'bright_t31', 'scan', 'track',
                'hour', 'month', 'weekday', 'confidence_encoded', 'daynight_encoded'
            ]
            
            # Ensure all required columns exist
            for col in feature_columns:
                if col not in df.columns:
                    if col == 'bright_t31':
                        df[col] = 0  # Default value if missing
                    else:
                        logger.warning(f"Missing column: {col}")
                        return df
            
            X = df[feature_columns].values
            predictions = self.model.predict(X)
            
            df['risk_level'] = predictions
            df['risk_label'] = df['risk_level'].map(self.risk_labels)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Return default low risk if prediction fails
            df['risk_level'] = 0
            df['risk_label'] = 'Low'
            return df

class NASAFirmsAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = BASE_URL
        
    async def fetch_fire_data(self, date: str, day_range: int, region: str = 'complete') -> pd.DataFrame:
        """Fetch fire data from NASA FIRMS AREA API with improved error handling"""
        try:
            # Get coordinates for the specified region
            coordinates = INDIA_REGIONS.get(region, INDIA_REGIONS['complete'])
            
            # Construct the correct API URL for AREA endpoint
            url = f"{self.base_url}/{self.api_key}/VIIRS_SNPP_NRT/{coordinates}/{day_range}/{date}"
            
            # Check cache first
            cache_key = f"{region}/{day_range}/{date}"
            if cache_key in cache:
                logger.info(f"Returning cached data for {cache_key}")
                return cache[cache_key]
            
            logger.info(f"Fetching fire data from NASA FIRMS AREA API: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    logger.info(f"NASA API Response Status: {response.status}")
                    
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Received {len(content)} characters from NASA API")
                        
                        # Check if content contains "Invalid API call"
                        if "Invalid API call" in content:
                            logger.error("NASA API returned 'Invalid API call' error")
                            logger.error(f"URL attempted: {url}")
                            return pd.DataFrame()
                        
                        # Check if content is empty or just headers
                        lines = content.strip().split('\n')
                        if len(lines) <= 1:
                            logger.warning("NASA API returned no data (only headers or empty)")
                            return pd.DataFrame()
                        
                        # Parse CSV data
                        from io import StringIO
                        df = pd.read_csv(StringIO(content))
                        
                        if df.empty:
                            logger.warning("NASA API returned empty DataFrame")
                            return df
                        
                        logger.info(f"Successfully parsed {len(df)} fire data points")
                        
                        # Validate required columns
                        required_columns = ['latitude', 'longitude', 'bright_ti4', 'acq_date']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        
                        if missing_columns:
                            logger.error(f"Missing required columns: {missing_columns}")
                            logger.info(f"Available columns: {df.columns.tolist()}")
                            return pd.DataFrame()
                        
                        # Cache the result
                        cache[cache_key] = df
                        return df
                        
                    elif response.status == 404:
                        logger.warning(f"No fire data available for date: {date}, day_range: {day_range}, region: {region}")
                        return pd.DataFrame()
                    
                    elif response.status == 401 or response.status == 403:
                        logger.error("NASA API access forbidden - check your API key")
                        error_text = await response.text()
                        logger.error(f"Error details: {error_text}")
                        return pd.DataFrame()
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"NASA API request failed with status {response.status}: {error_text}")
                        return pd.DataFrame()
                        
        except asyncio.TimeoutError:
            logger.error("NASA API request timed out")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching fire data from NASA: {e}")
            return pd.DataFrame()

class LocationService:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="forest_fire_monitor")
        self.location_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
    
    def get_location_info(self, lat: float, lon: float) -> Dict[str, str]:
        """Get state and district information from coordinates"""
        try:
            cache_key = f"{lat:.2f},{lon:.2f}"
            if cache_key in self.location_cache:
                return self.location_cache[cache_key]
            
            location = self.geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
            
            if location and location.address:
                address_parts = location.address.split(', ')
                state = None
                district = None
                
                # Extract state and district from address
                for part in address_parts:
                    if any(keyword in part.lower() for keyword in ['state', 'pradesh', 'bihar', 'bengal', 'kerala', 'punjab', 'karnataka', 'maharashtra', 'gujarat', 'rajasthan', 'odisha', 'assam', 'haryana', 'himachal', 'jharkhand', 'meghalaya', 'tripura', 'uttarakhand', 'mizoram', 'nagaland', 'manipur', 'arunachal', 'sikkim', 'goa', 'delhi', 'chandigarh', 'puducherry']):
                        state = part
                        break
                
                result = {'state': state, 'district': district}
                self.location_cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.error(f"Error getting location info: {e}")
        
        return {'state': None, 'district': None}

# Initialize services
data_processor = FireDataProcessor()
nasa_api = NASAFirmsAPI(NASA_FIRMS_API_KEY)
location_service = LocationService()

@app.get("/")
async def root():
    return {
        "message": "Forest Fire Risk Monitoring API", 
        "version": "1.0.0",
        "api_type": "NASA FIRMS Area API",
        "supported_regions": list(INDIA_REGIONS.keys())
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/regions")
async def get_regions():
    """Get available Indian regions for fire data"""
    return {
        "regions": {
            "complete": "Complete India Territory",
            "north": "North India (Punjab, Himachal, Uttarakhand, UP)",
            "south": "South India (Karnataka, Tamil Nadu, Kerala, Andhra)",
            "west": "West India (Maharashtra, Gujarat, Rajasthan)",
            "east": "East India (West Bengal, Bihar, Jharkhand, Odisha)"
        },
        "coordinates": INDIA_REGIONS,
        "default": "complete"
    }

@app.post("/predict-fires")
async def predict_fires(request: PredictionRequest):
    """Fetch fire data and predict risk levels with better error handling"""
    try:
        logger.info(f"Received prediction request: {request}")
        
        # Validate region
        if request.region and request.region not in INDIA_REGIONS:
            raise HTTPException(status_code=400, detail=f"Invalid region. Available regions: {list(INDIA_REGIONS.keys())}")
        
        # Fetch data from NASA FIRMS AREA API
        df = await nasa_api.fetch_fire_data(request.date, request.day_range, request.region or 'complete')
        
        if df.empty:
            logger.warning(f"No fire data found for date: {request.date}, day_range: {request.day_range}, region: {request.region}")
            coordinates = INDIA_REGIONS.get(request.region or 'complete', INDIA_REGIONS['complete'])
            return {
                "message": f"No fire data found for {request.date} with {request.day_range} day range in {request.region or 'complete'} region",
                "data": [],
                "statistics": {
                    "total_points": 0,
                    "low_risk": 0,
                    "moderate_risk": 0,
                    "high_risk": 0
                },
                "debug_info": {
                    "date_requested": request.date,
                    "day_range": request.day_range,
                    "region": request.region or 'complete',
                    "coordinates": coordinates,
                    "nasa_api_url": f"{BASE_URL}/{NASA_FIRMS_API_KEY}/VIIRS_SNPP_NRT/{coordinates}/{request.day_range}/{request.date}"
                }
            }
        
        logger.info(f"Processing {len(df)} fire data points")
        
        # Preprocess data
        df = data_processor.preprocess_data(df)
        
        # Predict risk levels
        df = data_processor.predict_risk(df)
        
        logger.info(f"Risk level distribution: {df['risk_level'].value_counts().to_dict()}")
        
        # Apply filters
        original_count = len(df)
        
        if request.satellite:
            df = df[df['satellite'] == request.satellite]
            logger.info(f"After satellite filter ({request.satellite}): {len(df)} records")
        
        if request.daynight:
            df = df[df['daynight'] == request.daynight]
            logger.info(f"After day/night filter ({request.daynight}): {len(df)} records")
        
        if request.risk_level is not None:
            df = df[df['risk_level'] == request.risk_level]
            logger.info(f"After risk level filter ({request.risk_level}): {len(df)} records")
        
        # Convert to list of dictionaries
        results = []
        for _, row in df.iterrows():
            fire_data = {
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'bright_ti4': float(row['bright_ti4']),
                'scan': float(row['scan']),
                'track': float(row['track']),
                'acq_date': row['acq_date'],
                'acq_time': str(row['acq_time']).zfill(4),  
                'satellite': row['satellite'],
                'instrument': row['instrument'],
                'confidence': row['confidence'],
                'version': row['version'],
                'bright_ti5': float(row['bright_ti5']),
                'frp': float(row['frp']),
                'daynight': row['daynight'],
                'risk_level': int(row['risk_level']),
                'risk_label': row['risk_label']
            }
            results.append(fire_data)
        
        # Calculate statistics
        risk_counts = df['risk_level'].value_counts().to_dict()
        
        return {
            "message": f"Successfully processed {len(results)} fire data points",
            "data": results,
            "statistics": {
                "total_points": len(results),
                "low_risk": risk_counts.get(0, 0),
                "moderate_risk": risk_counts.get(1, 0),
                "high_risk": risk_counts.get(2, 0)
            },
            "debug_info": {
                "original_data_points": original_count,
                "filtered_data_points": len(results),
                "date_processed": request.date,
                "day_range": request.day_range,
                "region": request.region or 'complete'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in predict_fires: {e}")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "debug_info": {
                "date": request.date,
                "day_range": request.day_range,
                "region": request.region
            }
        })

@app.get("/map-data")
async def get_map_data(
    date: str = Query(..., description="date (YYYY-MM-DD)"),
    day_range: int = Query(..., description="Day Range 1 - 10"),
    satellite: Optional[str] = Query(None, description="Satellite filter"),
    daynight: Optional[str] = Query(None, description="Day/Night filter"),
    risk_level: Optional[int] = Query(None, description="Risk level filter (0=Low, 1=Moderate, 2=High)"),
    region: Optional[str] = Query('complete', description="Region filter (complete, north, south, east, west)")
):
    """Get fire data in GeoJSON format for map visualization"""
    try:
        request = PredictionRequest(
            date=date,
            day_range=day_range,
            satellite=satellite,
            daynight=daynight,
            risk_level=risk_level,
            region=region
        )
        
        result = await predict_fires(request)
        
        # Convert to GeoJSON
        features = []
        for point in result['data']:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [point['longitude'], point['latitude']]
                },
                "properties": {
                    "risk_level": point['risk_level'],
                    "risk_label": point['risk_label'],
                    "brightness": point['bright_ti4'],
                    "frp": point['frp'],
                    "confidence": point['confidence'],
                    "satellite": point['satellite'],
                    "acq_date": point['acq_date'],
                    "acq_time": point['acq_time'],
                    "daynight": point['daynight']
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return {
            "geojson": geojson,
            "statistics": result['statistics'],
            "region": region
        }
        
    except Exception as e:
        logger.error(f"Error in get_map_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/satellites")
async def get_satellites():
    """Get available satellites"""
    return {
        "satellites": ["N", "N20", "N21", "AQUA", "TERRA"],
        "description": "Available satellite options for filtering"
    }

@app.get("/statistics")
async def get_statistics(
    date: str = Query(..., description="date (YYYY-MM-DD)"),
    day_range: int = Query(..., description="day_range 1 - 10"),
    region: Optional[str] = Query('complete', description="Region filter")
):
    """Get fire risk statistics for the specified date range"""
    try:
        request = PredictionRequest(date=date, day_range=day_range, region=region)
        result = await predict_fires(request)
        
        # Calculate additional statistics
        data = result['data']
        
        satellite_stats = {}
        confidence_stats = {}
        daynight_stats = {}
        
        for point in data:
            # Satellite statistics
            sat = point['satellite']
            satellite_stats[sat] = satellite_stats.get(sat, 0) + 1
            
            # Confidence statistics
            conf = point['confidence']
            confidence_stats[conf] = confidence_stats.get(conf, 0) + 1
            
            # Day/Night statistics
            dn = point['daynight']
            daynight_stats[dn] = daynight_stats.get(dn, 0) + 1
        
        return {
            "basic_statistics": result['statistics'],
            "satellite_distribution": satellite_stats,
            "confidence_distribution": confidence_stats,
            "daynight_distribution": daynight_stats,
            "region": region,
            "date": f"{date}"
        }
        
    except Exception as e:
        logger.error(f"Error in get_statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)