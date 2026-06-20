import pandas as pd
import osmnx as ox
import json
import time

def geocode_stations():
    print("Loading events.csv...")
    df = pd.read_csv('data/events.csv')
    stations = [str(s) for s in df['police_station'].dropna().unique() if str(s).strip() and s != 'No Police Station']
    
    coords = {}
    print(f"Found {len(stations)} police stations to geocode.")
    
    # Predefined known coordinates for difficult ones or fallbacks
    # City center fallback
    default_lat, default_lon = 12.9716, 77.5946
    
    for s in stations:
        try:
            # Try specific query
            query = f"{s} Police Station, Bengaluru, India"
            lat, lon = ox.geocode(query)
            coords[s] = {'lat': lat, 'lon': lon}
            print(f"[OK] {s}: {lat}, {lon}")
            time.sleep(1) # respect Nominatim limits
        except Exception as e:
            try:
                # Try broader query
                query = f"{s}, Bengaluru, India"
                lat, lon = ox.geocode(query)
                coords[s] = {'lat': lat, 'lon': lon}
                print(f"[WARN] {s} (used neighborhood): {lat}, {lon}")
                time.sleep(1)
            except Exception as e2:
                print(f"[FAIL] {s}: Using default city center.")
                coords[s] = {'lat': default_lat, 'lon': default_lon}
                
    with open('data/station_coords.json', 'w') as f:
        json.dump(coords, f, indent=4)
        
    print("Finished writing data/station_coords.json")

if __name__ == "__main__":
    geocode_stations()
