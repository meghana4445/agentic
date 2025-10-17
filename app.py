import os
import json
import pandas as pd
import streamlit as st
from typing import Optional

from agents.agent import CampusAIAgent, load_datasets
from agents.osm_utils import geocode_nominatim, route_osrm

st.set_page_config(page_title="Campus Guide – Agentic AI", page_icon="🎓", layout="wide")

@st.cache_data(show_spinner=False)
def get_data():
	return load_datasets()

buildings_df, departments_df, events_df, services_df = get_data()

st.sidebar.title("🎓 Campus Guide")
page = st.sidebar.radio(
	"Navigate",
	["Home", "Campus Map", "Departments", "Events", "Services", "Ask AI"],
)

with st.sidebar:
	st.markdown("---")
	st.markdown("**Setup**")
	st.caption("Set `OPENAI_API_KEY` env var or use a `.env` file.")
	user_key = st.text_input("OpenAI API Key (optional)", type="password")

# --- helper to parse 'lat, lon' strings ---
def parse_latlon(text: str):
	try:
		parts = [p.strip() for p in text.split(",")]
		if len(parts) == 2:
			lat = float(parts[0])
			lon = float(parts[1])
			return (lat, lon)
		return None
	except Exception:
		return None

class LocalFallbackAgent:
	def __init__(self):
		pass
	def search_all(self, q: str):
		return {
			"departments": self.filter_departments(q),
			"buildings": self.filter_buildings(q),
			"events": self.filter_events(q),
			"services": self.filter_services(q),
		}
	def filter_buildings(self, q: str):
		qq = q.lower()
		df = buildings_df
		mask = (
			df["name"].str.lower().str.contains(qq)
			| df["code"].str.lower().str.contains(qq)
			| df.get("zone", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(qq)
			| df.get("description", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(qq)
		)
		return df[mask]
	def filter_departments(self, q: str):
		qq = q.lower()
		df = departments_df
		mask = (
			df["name"].str.lower().str.contains(qq)
			| df["building"].str.lower().str.contains(qq)
			| df.get("contact", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(qq)
		)
		return df[mask]
	def filter_events(self, q: str, df: Optional[pd.DataFrame] = None):
		qq = q.lower()
		target = df if df is not None else events_df
		mask = (
			target["name"].str.lower().str.contains(qq)
			| target["venue"].str.lower().str.contains(qq)
			| target.get("description", pd.Series([""] * len(target))).astype(str).str.lower().str.contains(qq)
		)
		return target[mask]
	def filter_services(self, q: str):
		qq = q.lower()
		df = services_df
		mask = (
			df["name"].str.lower().str.contains(qq)
			| df.get("category", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(qq)
			| df.get("description", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(qq)
		)
		return df[mask]
	def get_directions(self, start_name: str, end_name: str) -> str:
		if start_name == end_name:
			return "You are already at the destination."
		return f"Route from {start_name} to {end_name} shown on the map."
	def chat(self, message: str) -> str:
		return "AI is not configured. Set OPENAI_API_KEY to enable chat."

@st.cache_resource(show_spinner=False)
def get_agent(user_key_input: Optional[str]):
	api_key = user_key_input or os.environ.get("OPENAI_API_KEY", "")
	try:
		return CampusAIAgent(
			api_key=api_key,
			buildings=buildings_df,
			departments=departments_df,
			events=events_df,
			services=services_df,
		)
	except Exception as e:
		st.sidebar.error(f"AI init failed: {e}. Using local features only.")
		return LocalFallbackAgent()

agent = get_agent(user_key)


def render_df(df: pd.DataFrame, use_container_width: bool = True):
	st.dataframe(df, use_container_width=use_container_width, hide_index=True)


if page == "Home":
	st.title("Campus Navigation & Information Overload – Agentic AI-powered Campus Guide")
	st.write("Find departments, events, and services, and get directions across campus. The AI summarizes to reduce information overload.")
	cols = st.columns(3)
	with cols[0]:
		st.metric("Buildings", len(buildings_df))
	with cols[1]:
		st.metric("Departments", len(departments_df))
	with cols[2]:
		st.metric("Events", len(events_df))
	st.markdown("---")
	st.subheader("Quick Search")
	q = st.text_input("Search departments, buildings, services or events")
	if q:
		res = agent.search_all(q)
		with st.expander("Departments"):
			render_df(res.get("departments", departments_df.iloc[0:0]))
		with st.expander("Buildings"):
			render_df(res.get("buildings", buildings_df.iloc[0:0]))
		with st.expander("Events"):
			render_df(res.get("events", events_df.iloc[0:0]))
		with st.expander("Services"):
			render_df(res.get("services", services_df.iloc[0:0]))

elif page == "Campus Map":
	st.title("Campus Map")

	map_source = st.radio("Map source", ["Campus image", "OpenStreetMap"], index=1, horizontal=True)

	if map_source == "Campus image":
		points_path = os.path.join("data", "map_points.json")
		points = {}
		if os.path.exists(points_path):
			with open(points_path, "r", encoding="utf-8") as f:
				cfg = json.load(f)
				points = cfg.get("points", {})
		from PIL import Image, ImageDraw
		img_file = None
		for p in [os.path.join("data", fn) for fn in ["campus_map.png", "campus_map.jpg", "campus_map.jpeg"]]:
			if os.path.exists(p):
				img_file = p
				break
		all_locations = sorted(points.keys()) if points else []
		dest = st.selectbox("Search destination", all_locations, index=all_locations.index("U-Block") if "U-Block" in all_locations else 0) if all_locations else ""
		if img_file and points and dest:
			img = Image.open(img_file).convert("RGBA")
			w, h = img.size
			def to_xy(pt):
				return (int(pt["x"] * w), int(pt["y"] * h))
			start_name = "N-Block"
			draw = ImageDraw.Draw(img)
			start_pt = points.get(start_name)
			dest_pt = points.get(dest)
			if start_pt:
				sx, sy = to_xy(start_pt)
				draw.ellipse((sx-10, sy-10, sx+10, sy+10), fill=(89,161,79,255))
				draw.text((sx+12, sy-10), start_name, fill=(0,0,0,255))
			if dest_pt:
				dx, dy = to_xy(dest_pt)
				draw.ellipse((dx-10, dy-10, dx+10, dy+10), fill=(225,87,89,255))
				draw.text((dx+12, dy-10), dest, fill=(0,0,0,255))
			if start_pt and dest_pt:
				draw.line([to_xy(start_pt), to_xy(dest_pt)], fill=(66,133,244,255), width=6)
			st.image(img, use_column_width=True)
		elif not img_file:
			st.warning("Add data/campus_map.png or .jpg to enable the campus image map.")
		elif not points:
			st.warning("map_points.json missing or empty. Share the labels you want included and I will add them.")

	else:
		# OpenStreetMap routing with tight campus view and lat,lon support
		default_suffix = ", Vignan University, Vadlamudi, Guntur, Andhra Pradesh"
		location_suffix = st.text_input("Location context (used to refine searches)", default_suffix)
		# Precise N-Block coordinates from your photo: 16°13'55.4268" N, 80°33'0.774" E -> 16.232063, 80.550215
		default_anchor = "16.232063, 80.550215"
		campus_name = st.text_input("Current location (accepts 'lat, lon' too)", default_anchor)
		if st.button("Set Anchor"):
			st.session_state["anchor"] = campus_name
		anchor_query = st.session_state.get("anchor", campus_name)
		anchor = parse_latlon(anchor_query) or geocode_nominatim(anchor_query)

		# Quick picks with known lat/lon (provided by you)
		quick_picks = {
			"A-Block": (16.232988, 80.547555),
			"Library": (16.232909, 80.548626),
		}
		qp = st.selectbox("Quick picks (OSM)", ["(none)"] + list(quick_picks.keys()))

		dest_query_raw = st.text_input("Search destination (accepts 'lat, lon')")
		dest_query = f"{dest_query_raw}{location_suffix}" if dest_query_raw else ""

		route_poly = None
		dest_coords = None
		if anchor:
			if qp != "(none)":
				dest_coords = quick_picks[qp]
			elif dest_query_raw:
				dest_coords = parse_latlon(dest_query_raw) or (geocode_nominatim(dest_query) if dest_query else None)
			if dest_coords:
				route_poly = route_osrm(anchor, dest_coords, profile="foot")

		import folium
		from streamlit.components.v1 import html
		center = anchor or (16.3067, 80.4365)
		m = folium.Map(location=center, zoom_start=18, control_scale=True)
		if anchor:
			folium.Marker(location=anchor, tooltip="Current location (N-Block)", icon=folium.Icon(color="green", icon="home")).add_to(m)
		if dest_coords:
			label = qp if qp != "(none)" else (dest_query_raw or "Destination")
			folium.Marker(location=dest_coords, tooltip=label, icon=folium.Icon(color="red", icon="flag")).add_to(m)
		if route_poly:
			folium.PolyLine(locations=route_poly, color="#4285F4", weight=6).add_to(m)
			m.fit_bounds(route_poly)
		elif anchor:
			lat, lon = anchor
			d = 0.003
			m.fit_bounds([[lat - d, lon - d], [lat + d, lon + d]])
		html(m._repr_html_(), height=600)
		st.markdown("---")
		if anchor and dest_coords and route_poly:
			st.success("Walking route shown (OSM+OSRM).")
		elif anchor and (dest_query_raw or qp != "(none)") and not dest_coords:
			st.warning("Destination not found on OSM. Try coordinates 'lat, lon' or refine the text.")
		elif not anchor:
			st.info("Set the current location first.")

elif page == "Departments":
	st.title("Departments")
	kw = st.text_input("Filter by name/building/contact")
	df = departments_df
	if kw:
		df = agent.filter_departments(kw)
	render_df(df)

elif page == "Events":
	st.title("Events & Schedules")
	c1, c2 = st.columns(2)
	with c1:
		date_filter = st.date_input("Filter by date", value=None)
	with c2:
		text_filter = st.text_input("Filter by name/venue/description")
	df = events_df
	if date_filter:
		df = df[df["date"] == pd.to_datetime(date_filter).date().isoformat()]
	if text_filter:
		df = agent.filter_events(text_filter, df)
	render_df(df)

elif page == "Services":
	st.title("Student Services")
	kw = st.text_input("Search services (e.g., counseling, IT help)")
	df = services_df
	if kw:
		df = agent.filter_services(kw)
	render_df(df)

elif page == "Ask AI":
	st.title("Ask the Campus AI")
	st.caption("The assistant summarizes to reduce information overload.")
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []
	for role, content in st.session_state.chat_history:
		with st.chat_message(role):
			st.markdown(content)
	user_msg = st.chat_input("Ask about directions, departments, events, or campus life...")
	if user_msg:
		st.session_state.chat_history.append(("user", user_msg))
		with st.chat_message("assistant"):
			with st.spinner("Thinking..."):
				response = agent.chat(user_msg)
				st.markdown(response)
		st.session_state.chat_history.append(("assistant", response))

else:
	st.write("Unknown page")
