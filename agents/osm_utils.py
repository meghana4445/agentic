import time
import requests
from typing import Optional, Tuple, List

USER_AGENT = "CampusGuide/1.0 (contact: you@example.com)"


def geocode_nominatim(query: str, country_codes: Optional[str] = None) -> Optional[Tuple[float, float]]:
	params = {
		"q": query,
		"format": "json",
		"limit": 1,
	}
	if country_codes:
		params["countrycodes"] = country_codes
	headers = {"User-Agent": USER_AGENT}
	r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=15)
	if r.status_code != 200:
		return None
	data = r.json()
	if not data:
		return None
	lat = float(data[0]["lat"])
	lon = float(data[0]["lon"])
	# be nice to Nominatim
	time.sleep(1)
	return (lat, lon)


def route_osrm(start: Tuple[float, float], end: Tuple[float, float], profile: str = "foot") -> Optional[List[Tuple[float, float]]]:
	# OSRM demo server expects lon,lat
	start_lonlat = f"{start[1]},{start[0]}"
	end_lonlat = f"{end[1]},{end[0]}"
	url = f"https://router.project-osrm.org/route/v1/{profile}/{start_lonlat};{end_lonlat}"
	params = {"overview": "full", "geometries": "geojson"}
	r = requests.get(url, params=params, timeout=20)
	if r.status_code != 200:
		return None
	data = r.json()
	if not data.get("routes"):
		return None
	coords = data["routes"][0]["geometry"]["coordinates"]  # list of [lon, lat]
	poly = [(lat, lon) for lon, lat in coords]
	return poly






