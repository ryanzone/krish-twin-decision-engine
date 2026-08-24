import os
import sys
import json
import requests
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Setup resilient HTTP session with retry mechanism and browser User-Agent
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    raise_on_status=False
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json"
})

def fetch_sentinel2_ndvi(lat: float, lon: float) -> float:
    """Queries live Google Earth Engine Sentinel-2 SR collection to calculate mean NDVI for (lat, lon)."""
    try:
        import ee
        
        # Initialize GEE with service account or default credentials
        key_path = "gcp-key.json"
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                key_data = json.load(f)
            client_email = key_data.get("client_email")
            project_id = key_data.get("project_id")
            if client_email and project_id:
                credentials = ee.ServiceAccountCredentials(client_email, key_path)
                ee.Initialize(credentials, project=project_id)
            else:
                ee.Initialize()
        else:
            ee.Initialize()

        point = ee.Geometry.Point([lon, lat])
        start_date = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .sort("system:time_start", False)
        )

        image = s2_collection.first()
        ndvi_image = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        stats = ndvi_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(200),
            scale=10
        )
        res = stats.getInfo()
        mean_ndvi = res.get("NDVI") if res else None
        if mean_ndvi is not None:
            print(f"Sentinel-2 Live NDVI Fetched via GEE: {round(mean_ndvi, 4)}", flush=True)
            return round(float(mean_ndvi), 4)

    except Exception as e:
        print(f"Earth Engine NDVI Query Warning: {type(e).__name__} ({e}). Falling back to baseline model index 0.37.")

    return 0.37

def run_weather_engine(
    lat: float = 9.8821,
    lon: float = 78.0815,
    crop: str = "Tomato",
    district: str = "Madurai",
    acres: float = 2.0,
    symptom: str = "Leaf Blight",
    spray_cost_override: float = None,
    crop_loss_override: float = None,
    backend_url: str = "http://127.0.0.1:8080/api/v1/simulate"
) -> dict:
    """Executes telemetry pipeline with dynamic location, GEE satellite NDVI, live market prices, and backend simulation."""
    print(f"\n=== Running Krishi-Twin Weather & Telemetry Engine ===", flush=True)
    print(f"Target Location: ({lat}, {lon}) | District: {district} | Crop: {crop} ({acres} acres)", flush=True)

    # 1. Fetch live satellite NDVI via Google Earth Engine
    mean_ndvi = fetch_sentinel2_ndvi(lat, lon)

    # 2. Fetch live market commodity prices from data.gov.in
    gov_api_key = os.getenv("GOV_DATA_API_KEY")
    if not gov_api_key:
        print("Market Warning: GOV_DATA_API_KEY is not set in environment/.env file.", flush=True)
    gov_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    gov_params = {
        "api-key": gov_api_key,
        "format": "json",
        "filters[district]": district,
        "filters[commodity]": crop,
        "limit": 10
    }

    print(f"Fetching live market rates from data.gov.in for {crop} in {district}...", flush=True)
    market_info = {}
    modal_price_per_kg = 21.0  # default fallback if no record found

    try:
        market_response = session.get(gov_url, params=gov_params, timeout=10)
        market_response.raise_for_status()
        records = market_response.json().get("records", [])

        if records:
            latest = records[0]
            modal_price_quintal = float(latest.get("modal_price", 2100))
            modal_price_per_kg = round(modal_price_quintal / 100.0, 2)
            market_info = {
                "market": latest.get("market"),
                "commodity": latest.get("commodity"),
                "variety": latest.get("variety"),
                "modal_price_inr_quintal": modal_price_quintal,
                "modal_price_per_kg_inr": modal_price_per_kg,
                "arrival_date": latest.get("arrival_date")
            }
            print(f"Market Data Fetched: {crop} @ Rs.{modal_price_quintal}/quintal (Rs.{modal_price_per_kg}/kg) at {market_info['market']}", flush=True)
        else:
            print(f"Market Warning: No records found for {crop} in {district}. Using baseline Rs.{modal_price_per_kg}/kg.", flush=True)
    except Exception as e:
        print(f"Market Fetch Warning: {e}. Using baseline Rs.{modal_price_per_kg}/kg.", flush=True)

    # 3. Dynamic Financial Calculations
    spray_cost_inr = spray_cost_override if spray_cost_override is not None else round(acres * 425.0, 2)
    potential_crop_loss_inr = crop_loss_override if crop_loss_override is not None else round(acres * 100.0 * modal_price_per_kg, 2)

    print(f"Financial Calculations: Spray Cost = Rs.{spray_cost_inr} | Potential Crop Loss = Rs.{potential_crop_loss_inr}", flush=True)

    # 4. Fetch live meteorological telemetry from Open-Meteo
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation,precipitation_probability,wind_speed_10m&forecast_days=2&timezone=auto"
    weather_response = session.get(weather_url, timeout=10)
    weather_response.raise_for_status()
    weather_data = weather_response.json().get("hourly", {})

    rain_24h_mm = sum(weather_data.get("precipitation", [])[:24])
    max_rain_prob = max(weather_data.get("precipitation_probability", [])[:24], default=0)
    max_wind_kmh = max(weather_data.get("wind_speed_10m", [])[:24], default=0)
    wash_off_risk = "HIGH" if rain_24h_mm > 10.0 or max_rain_prob > 70 else "LOW"

    # 5. Build dynamic payload
    payload = {
        "engine": "Krishi-Twin-Decision-Core",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "farm_profile": {
            "coordinates": {"latitude": lat, "longitude": lon},
            "crop": crop,
            "farm_size_acres": acres,
            "detected_symptom": symptom
        },
        "geospatial_telemetry": {
            "mean_ndvi_index": mean_ndvi,
            "canopy_vigor": "Stressed" if mean_ndvi < 0.4 else "Healthy"
        },
        "meteorological_risk": {
            "rain_next_24h_mm": round(rain_24h_mm, 2),
            "rain_probability_pct": max_rain_prob,
            "max_wind_kmh": round(max_wind_kmh, 2),
            "computed_washoff_risk": wash_off_risk
        },
        "market_telemetry": market_info,
        "financial_inputs": {
            "spray_cost_inr": spray_cost_inr,
            "potential_crop_loss_inr": potential_crop_loss_inr
        }
    }

    # 6. POST live payload to backend simulation core
    print(f"Posting live payload to backend ({backend_url})...", flush=True)
    sim_response = requests.post(backend_url, json=payload, timeout=60)
    sim_response.raise_for_status()
    result = sim_response.json()

    print("\n--- Live Simulation Result from Backend ---", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Krishi-Twin Dynamic Telemetry & Simulation Engine")
    parser.add_argument("--lat", type=float, default=9.8821, help="Latitude coordinate (default: 9.8821)")
    parser.add_argument("--lon", type=float, default=78.0815, help="Longitude coordinate (default: 78.0815)")
    parser.add_argument("--crop", type=str, default="Tomato", help="Crop commodity name (default: Tomato)")
    parser.add_argument("--district", type=str, default="Madurai", help="District name (default: Madurai)")
    parser.add_argument("--acres", type=float, default=2.0, help="Farm size in acres (default: 2.0)")
    parser.add_argument("--symptom", type=str, default="Leaf Blight", help="Detected crop symptom (default: Leaf Blight)")
    parser.add_argument("--spray-cost", type=float, default=None, help="Optional spray cost override in INR")
    parser.add_argument("--crop-loss", type=float, default=None, help="Optional potential crop loss override in INR")
    parser.add_argument("--backend-url", type=str, default="http://127.0.0.1:8080/api/v1/simulate", help="FastAPI simulation endpoint URL")

    args = parser.parse_args()
    run_weather_engine(
        lat=args.lat,
        lon=args.lon,
        crop=args.crop,
        district=args.district,
        acres=args.acres,
        symptom=args.symptom,
        spray_cost_override=args.spray_cost,
        crop_loss_override=args.crop_loss,
        backend_url=args.backend_url
    )