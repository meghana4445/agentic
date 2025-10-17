import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_csv(path: str) -> pd.DataFrame:
	if os.path.exists(path):
		return pd.read_csv(path)
	return pd.DataFrame()


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	buildings = load_csv(os.path.join(DATA_DIR, "buildings.csv"))
	departments = load_csv(os.path.join(DATA_DIR, "departments.csv"))
	events = load_csv(os.path.join(DATA_DIR, "events.csv"))
	services = load_csv(os.path.join(DATA_DIR, "services.csv"))
	for df in [buildings, departments, events, services]:
		if "description" in df.columns:
			df["description"] = df["description"].fillna("")
	return buildings, departments, events, services


@dataclass
class CampusAIAgent:
	api_key: str
	buildings: pd.DataFrame
	departments: pd.DataFrame
	events: pd.DataFrame
	services: pd.DataFrame

	def __post_init__(self):
		load_dotenv(override=False)
		if self.api_key:
			os.environ["OPENAI_API_KEY"] = self.api_key

		# Try to build LangChain-based agent, but make it optional
		self.llm = None
		self.tools = []
		self.agent_executor = None
		try:
			from langchain_openai import ChatOpenAI
			from langchain_core.tools import tool
			from langchain.agents import AgentExecutor, create_tool_calling_agent
			from langchain_core.prompts import ChatPromptTemplate

			self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

			@tool("search_buildings", return_direct=False)
			def search_buildings_tool(query: str) -> str:
				res = self.filter_buildings(query)
				return res.to_json(orient="records")

			@tool("search_departments", return_direct=False)
			def search_departments_tool(query: str) -> str:
				res = self.filter_departments(query)
				return res.to_json(orient="records")

			@tool("search_events", return_direct=False)
			def search_events_tool(query: str) -> str:
				res = self.filter_events(query)
				return res.to_json(orient="records")

			@tool("search_services", return_direct=False)
			def search_services_tool(query: str) -> str:
				res = self.filter_services(query)
				return res.to_json(orient="records")

			@tool("directions", return_direct=True)
			def directions_tool(payload: str) -> str:
				try:
					obj = json.loads(payload)
					return self.get_directions(obj["start"], obj["end"])
				except Exception as e:
					return f"Invalid input. Use JSON with 'start' and 'end'. Error: {e}"

			self.tools = [
				search_buildings_tool,
				search_departments_tool,
				search_events_tool,
				search_services_tool,
				directions_tool,
			]

			system = (
				"You are CampusGuide, an assistant that helps students navigate campus and reduce information overload. "
				"Be concise and structured: summarize key points, include top 3 matches when listing, and provide short next steps. "
				"Use tools for searching data and directions. If you return JSON lists from tools, summarize them into readable bullets."
			)
			prompt = ChatPromptTemplate.from_messages([
				("system", system),
				("human", "{input}"),
				("placeholder", "{agent_scratchpad}"),
			])
			agent = create_tool_calling_agent(self.llm, self.tools, prompt)
			self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
		except Exception:
			# LangChain or OpenAI not installed; chat will use fallback
			self.llm = None
			self.agent_executor = None

	def search_all(self, query: str) -> Dict[str, pd.DataFrame]:
		q = query.lower()
		return {
			"departments": self.filter_departments(q),
			"buildings": self.filter_buildings(q),
			"events": self.filter_events(q),
			"services": self.filter_services(q),
		}

	def filter_buildings(self, query: str) -> pd.DataFrame:
		q = query.lower()
		df = self.buildings
		mask = (
			df["name"].str.lower().str.contains(q)
			| df["code"].str.lower().str.contains(q)
			| df.get("zone", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(q)
			| df.get("description", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(q)
		)
		return df[mask]

	def filter_departments(self, query: str) -> pd.DataFrame:
		q = query.lower()
		df = self.departments
		mask = (
			df["name"].str.lower().str.contains(q)
			| df["building"].str.lower().str.contains(q)
			| df.get("contact", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(q)
		)
		return df[mask]

	def filter_events(self, query: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
		q = query.lower()
		target = df if df is not None else self.events
		mask = (
			target["name"].str.lower().str.contains(q)
			| target["venue"].str.lower().str.contains(q)
			| target.get("description", pd.Series([""] * len(target))).astype(str).str.lower().str.contains(q)
		)
		return target[mask]

	def filter_services(self, query: str) -> pd.DataFrame:
		q = query.lower()
		df = self.services
		mask = (
			df["name"].str.lower().str.contains(q)
			| df.get("category", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(q)
			| df.get("description", pd.Series([""] * len(df))).astype(str).str.lower().str.contains(q)
		)
		return df[mask]

	def get_directions(self, start_name: str, end_name: str) -> str:
		if start_name == end_name:
			return "You are already at the destination."
		start = self.buildings[self.buildings["name"] == start_name].iloc[0]
		end = self.buildings[self.buildings["name"] == end_name].iloc[0]
		steps = [
			f"Start at {start['name']} ({start['code']})",
			f"Head towards zone {end['zone']} following campus wayfinding signs",
			f"Continue straight for ~{abs(int(end['x']) - int(start['x']))} blocks on X-axis",
			f"Turn and proceed ~{abs(int(end['y']) - int(start['y']))} blocks on Y-axis",
			f"Arrive at {end['name']} ({end['code']})",
		]
		return "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

	def chat(self, message: str) -> str:
		if self.agent_executor is None:
			return (
				"AI is not configured. Set OPENAI_API_KEY and install extras with:\n"
				"pip install langchain-openai openai langchain tiktoken\n"
				"You can still use search, filters, and directions."
			)
		try:
			result = self.agent_executor.invoke({"input": message})
			return result.get("output", "(No response)")
		except Exception as e:
			return f"There was an issue answering that: {e}"
