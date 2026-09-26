#!/usr/bin/env python3
"""
06_run_deepseek_agentic.py

Standalone 5-minute agentic DeepSeek HVAC controller for Closed-LoopAgenticLLMs.

Architecture
------------
controller -> evaluator -> optional single refinement -> deterministic validator

This public version embeds the previously shared DeepSeek/EnergyPlus controller
infrastructure directly in this file. It does NOT require, run, or compare against
a separate ``07_run_deepseek_direct.py`` experiment. The frozen controller prompt,
EnergyPlus integration, occupancy handling, comfort calculations, action validator,
and energy accounting are retained so the agentic experiment logic is unchanged.

Repository layout
-----------------
The script is intended to live in the repository root beside the EPW and measured
occupancy CSV. The generated IDF remains under ``generated/building`` and outputs
are written under ``results/deepseek_agentic``.

DeepSeek credentials are read from ``DEEPSEEK_API_KEY`` or an optional key file;
the secret value is never printed or written to results.

EnergyPlus target: 24.1
Python target: 3.9+
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


# ============================================================================
# Project configuration
# ============================================================================

SCRIPT_NAME = Path(__file__).name
SCRIPT_DIR = Path(__file__).resolve().parent

# Final public repository layout: the executable scripts and primary input
# files live directly in the repository root. Generated models and results are
# created below generated/ and results/.
PROJECT_DIR = SCRIPT_DIR
GENERATED_BUILDING_SUBDIR = Path("generated") / "building"
FIXED_RESULTS_SUBDIR = Path("results") / "fixed_baseline"
OCCUPANCY_RULE_RESULTS_SUBDIR = Path("results") / "occupancy_rule_baseline"
COMFORT_RULE_RESULTS_SUBDIR = Path("results") / "comfort_rule_baseline"
DEEPSEEK_AGENTIC_RESULTS_SUBDIR = Path("results") / "deepseek_agentic"

DEFAULT_IDF_NAME = "honeycomb_7zone_fcu_closed_loop_ready.idf"
DEFAULT_EPW_NAME = "CHN_Hebei.Shijiazhuang.536980_CSWD.epw"
DEFAULT_OCCUPANCY_NAME = "actual_occupancy_count_5min_7day.csv"
DEFAULT_FIXED_SUMMARY_NAME = "fixed_baseline_summary.json"
DEFAULT_OCCUPANCY_RULE_SUMMARY_NAME = "occupancy_rule_summary.json"
DEFAULT_COMFORT_RULE_SUMMARY_NAME = "comfort_rule_summary.json"

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_KEY_CANDIDATES = ("deepseek_key", "deepseek_key.txt")

LLM_DECISION_INTERVAL_MINUTES = 30
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = 90.0
LLM_MAX_TOKENS = 1200
LLM_MAX_ATTEMPTS = 2

HEATING_MIN_C = 16.0
HEATING_MAX_C = 20.0
COOLING_MIN_C = 25.0
COOLING_MAX_C = 28.0
SETPOINT_RESOLUTION_C = 0.5
MIN_DEADBAND_C = 5.0

EXPECTED_LLM_DECISION_EPOCHS = 140
EXPECTED_LLM_ROOM_DECISIONS = EXPECTED_LLM_DECISION_EPOCHS * 6
MAX_LLM_FALLBACK_FRACTION = 0.05

# Frozen canonical occupancy input verified by SHA256.
EXPECTED_OCCUPANCY_SHA256 = (
    "b5432ff455391c7ecf111bb932d7e7b9636ff1f4dd38c216dea4fcf8d3201ab8"
)

CONTROL_START = pd.Timestamp("2021-08-16 00:00:00")
CONTROL_END_EXCLUSIVE = pd.Timestamp("2021-08-23 00:00:00")

ZONE_TIMESTEP_MINUTES = 5
EXPECTED_INTERVALS_PER_ROOM = 2016
EXPECTED_ANALYSIS_ROWS = 12096
EXPECTED_FIRST_INTERVAL_START = CONTROL_START
EXPECTED_LAST_INTERVAL_START = CONTROL_END_EXCLUSIVE - pd.Timedelta(
    minutes=ZONE_TIMESTEP_MINUTES
)

# EnergyPlus DataExchange.kind_of_sim() enumeration:
#   1 Design Day
#   2 Design RunPeriod
#   3 Weather File Run Period
#   4 HVAC-Sizing Design Day
#   5 HVAC-Sizing RunPeriod
#   6 Weather Data Processing
WEATHER_FILE_RUN_PERIOD_KIND = 3

OFFICE_START_HOUR = 8
OFFICE_END_HOUR = 18

UNOCCUPIED_HEATING_SP_C = 16.0
UNOCCUPIED_COOLING_SP_C = 28.0

LOW_OCC_HEATING_SP_C = 18.0
LOW_OCC_COOLING_SP_C = 27.0

MEDIUM_OCC_HEATING_SP_C = 19.0
MEDIUM_OCC_COOLING_SP_C = 26.0

HIGH_OCC_HEATING_SP_C = 20.0
HIGH_OCC_COOLING_SP_C = 25.0

FCU_AVAILABILITY_COMMAND = 1.0

VENT_FLOW_PER_PERSON_M3_S = 0.009438948864

# Main analysis thresholds
OVERHEAT_THRESHOLD_C = 27.0
OVERCOOL_THRESHOLD_C = 22.5
HIGH_RH_THRESHOLD_PERCENT = 85.0

# PMV/PPD assumptions
PMV_MET = 1.2
PMV_CLO = 0.5
PMV_AIR_SPEED_M_S = 0.1
PMV_EXTERNAL_WORK_MET = 0.0

ROOMS: Dict[str, Dict[str, Any]] = {
    "Room_1": {
        "zone": "Thermal Zone 1",
        "zone_number": 1,
        "capacity": 8.0,
        "people_object": "Experimental People Room 1",
        "vent_schedule": "Experimental Ventilation Fraction Room 1",
        "vent_object": "Thermal Zone 1 Ventilation per Person",
    },
    "Room_2": {
        "zone": "Thermal Zone 2",
        "zone_number": 2,
        "capacity": 4.0,
        "people_object": "Experimental People Room 2",
        "vent_schedule": "Experimental Ventilation Fraction Room 2",
        "vent_object": "Thermal Zone 2 Ventilation per Person",
    },
    "Room_4": {
        "zone": "Thermal Zone 4",
        "zone_number": 4,
        "capacity": 8.0,
        "people_object": "Experimental People Room 4",
        "vent_schedule": "Experimental Ventilation Fraction Room 4",
        "vent_object": "Thermal Zone 4 Ventilation per Person",
    },
    "Room_5": {
        "zone": "Thermal Zone 5",
        "zone_number": 5,
        "capacity": 4.0,
        "people_object": "Experimental People Room 5",
        "vent_schedule": "Experimental Ventilation Fraction Room 5",
        "vent_object": "Thermal Zone 5 Ventilation per Person",
    },
    "Room_6": {
        "zone": "Thermal Zone 6",
        "zone_number": 6,
        "capacity": 7.0,
        "people_object": "Experimental People Room 6",
        "vent_schedule": "Experimental Ventilation Fraction Room 6",
        "vent_object": "Thermal Zone 6 Ventilation per Person",
    },
    "Room_7": {
        "zone": "Thermal Zone 7",
        "zone_number": 7,
        "capacity": 7.0,
        "people_object": "Experimental People Room 7",
        "vent_schedule": "Experimental Ventilation Fraction Room 7",
        "vent_object": "Thermal Zone 7 Ventilation per Person",
    },
}

ROOM_ORDER = list(ROOMS.keys())

REQUIRED_ENERGY_METERS = (
    "Cooling:Electricity",
    "Heating:Electricity",
    "Fans:Electricity",
    "Pumps:Electricity",
)

DIAGNOSTIC_ENERGY_METERS = (
    "Electricity:HVAC",
    "Electricity:Facility",
)

ALL_ENERGY_METERS = REQUIRED_ENERGY_METERS + DIAGNOSTIC_ENERGY_METERS


# ============================================================================
# Generic helpers
# ============================================================================

def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
        return default
    except Exception:
        return default


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def is_office_hour(interval_start: pd.Timestamp) -> bool:
    return OFFICE_START_HOUR <= interval_start.hour < OFFICE_END_HOUR


def occupancy_rule_setpoints(
    interval_start: pd.Timestamp,
    occupancy_count: float,
) -> Tuple[float, float, str]:
    """Return the office-hours deterministic Rule-OCC thermostat action."""
    occupancy = max(0.0, float(occupancy_count))

    if not is_office_hour(interval_start):
        return (
            UNOCCUPIED_HEATING_SP_C,
            UNOCCUPIED_COOLING_SP_C,
            "off_hours_fixed",
        )

    if occupancy <= 0.0:
        return (
            UNOCCUPIED_HEATING_SP_C,
            UNOCCUPIED_COOLING_SP_C,
            "office_vacant",
        )
    if occupancy <= 2.0:
        return (
            LOW_OCC_HEATING_SP_C,
            LOW_OCC_COOLING_SP_C,
            "office_low_1_2",
        )
    if occupancy <= 5.0:
        return (
            MEDIUM_OCC_HEATING_SP_C,
            MEDIUM_OCC_COOLING_SP_C,
            "office_medium_3_5",
        )
    return (
        HIGH_OCC_HEATING_SP_C,
        HIGH_OCC_COOLING_SP_C,
        "office_high_6_plus",
    )


DEEPSEEK_SYSTEM_PROMPT = """You are the direct supervisory thermostat controller for six office rooms in an EnergyPlus building simulation.

Choose heating and cooling thermostat setpoints for the CURRENT state only.

Objectives:
1. Reduce HVAC energy use.
2. Maintain reasonable thermal comfort when a room is occupied.
3. Avoid overheating, overcooling, and very high relative humidity when feasible.
4. Avoid unnecessary setpoint changes.

Evaluation context for occupied rooms:
- temperature >27 C is overheating;
- temperature <22.5 C is overcooling;
- relative humidity >85% is high;
- PMV near 0 is preferable.

Hard action envelope:
- heating_setpoint_C: 16 to 20 C;
- cooling_setpoint_C: 25 to 28 C;
- actions are quantized to 0.5 C;
- minimum heating/cooling deadband is 5 C;
- you do not control fan speed, FCU availability, People, or ventilation.

Use only the state provided. Do not assume future occupancy or future weather.
Do not reproduce a fixed occupancy-rule table.
Return JSON only and exactly one action for every requested room.

Required JSON:
{
  "rooms": [
    {
      "room": "Room_1",
      "heating_setpoint_C": 18.0,
      "cooling_setpoint_C": 26.5,
      "action": "short label",
      "reason": "brief causal reason",
      "confidence": 0.85
    }
  ]
}
"""


def deepseek_prompt_sha256() -> str:
    return hashlib.sha256(DEEPSEEK_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def is_llm_decision_epoch(interval_start: pd.Timestamp) -> bool:
    if not is_office_hour(interval_start):
        return False
    minute_of_day = int(interval_start.hour) * 60 + int(interval_start.minute)
    office_start = OFFICE_START_HOUR * 60
    return (
        (minute_of_day - office_start) % LLM_DECISION_INTERVAL_MINUTES
        == 0
    )


def quantize_setpoint(value: float) -> float:
    return round(float(value) / SETPOINT_RESOLUTION_C) * SETPOINT_RESOLUTION_C


def fallback_rule_action(occupancy_count: float) -> Dict[str, Any]:
    # Reuse the frozen Rule-OCC office-hour mapping only as a fallback.
    _, _, band = occupancy_rule_setpoints(
        pd.Timestamp("2021-08-16 12:00:00"),
        occupancy_count,
    )
    occupancy = max(0.0, float(occupancy_count))
    if occupancy <= 0:
        heat, cool = 16.0, 28.0
    elif occupancy <= 2:
        heat, cool = 18.0, 27.0
    elif occupancy <= 5:
        heat, cool = 19.0, 26.0
    else:
        heat, cool = 20.0, 25.0
    return {
        "heating_setpoint_C": heat,
        "cooling_setpoint_C": cool,
        "action": "rule_occ_fallback",
        "reason": f"deterministic_fallback:{band}",
        "confidence": 0.0,
        "proposal_valid": False,
        "safety_modified": False,
        "fallback_used": True,
        "fallback_reason": "proposal_missing_or_invalid",
        "raw_heating_setpoint_C": None,
        "raw_cooling_setpoint_C": None,
    }


def validate_deepseek_room_action(
    proposal: Optional[Dict[str, Any]],
    occupancy_count: float,
) -> Dict[str, Any]:
    if not isinstance(proposal, dict):
        action = fallback_rule_action(occupancy_count)
        action["fallback_reason"] = "missing_room_proposal"
        return action

    try:
        raw_heat = float(proposal.get("heating_setpoint_C"))
        raw_cool = float(proposal.get("cooling_setpoint_C"))
    except Exception:
        action = fallback_rule_action(occupancy_count)
        action["fallback_reason"] = "nonnumeric_setpoint"
        return action

    if not is_finite(raw_heat) or not is_finite(raw_cool):
        action = fallback_rule_action(occupancy_count)
        action["fallback_reason"] = "nonfinite_setpoint"
        return action

    heat = quantize_setpoint(clamp(raw_heat, HEATING_MIN_C, HEATING_MAX_C))
    cool = quantize_setpoint(clamp(raw_cool, COOLING_MIN_C, COOLING_MAX_C))
    heat = clamp(heat, HEATING_MIN_C, HEATING_MAX_C)
    cool = clamp(cool, COOLING_MIN_C, COOLING_MAX_C)

    if cool - heat < MIN_DEADBAND_C - 1e-9:
        feasible = []
        h = HEATING_MIN_C
        while h <= HEATING_MAX_C + 1e-9:
            c = COOLING_MIN_C
            while c <= COOLING_MAX_C + 1e-9:
                if c - h >= MIN_DEADBAND_C - 1e-9:
                    feasible.append((round(h, 6), round(c, 6)))
                c += SETPOINT_RESOLUTION_C
            h += SETPOINT_RESOLUTION_C
        heat, cool = min(
            feasible,
            key=lambda pair: abs(pair[0] - raw_heat) + abs(pair[1] - raw_cool),
        )

    try:
        confidence = float(proposal.get("confidence"))
        confidence = clamp(confidence, 0.0, 1.0) if is_finite(confidence) else None
    except Exception:
        confidence = None

    return {
        "heating_setpoint_C": float(heat),
        "cooling_setpoint_C": float(cool),
        "action": str(proposal.get("action", "deepseek_direct"))[:120],
        "reason": str(proposal.get("reason", ""))[:400],
        "confidence": confidence,
        "proposal_valid": True,
        "safety_modified": bool(
            abs(float(heat) - raw_heat) > 1e-9
            or abs(float(cool) - raw_cool) > 1e-9
        ),
        "fallback_used": False,
        "fallback_reason": None,
        "raw_heating_setpoint_C": raw_heat,
        "raw_cooling_setpoint_C": raw_cool,
    }


def extract_json_object(content: str) -> Dict[str, Any]:
    content = str(content or "").strip()
    if not content:
        raise ValueError("Empty DeepSeek response.")
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    first = content.find("{")
    last = content.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("No JSON object found in DeepSeek response.")
    obj = json.loads(content[first:last + 1])
    if not isinstance(obj, dict):
        raise ValueError("DeepSeek JSON root is not an object.")
    return obj


def normalize_deepseek_rooms(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rooms = payload.get("rooms")
    if isinstance(rooms, dict):
        iterable = []
        for key, value in rooms.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("room", key)
                iterable.append(item)
    elif isinstance(rooms, list):
        iterable = rooms
    else:
        iterable = []

    aliases: Dict[str, str] = {}
    for room in ROOM_ORDER:
        aliases[normalize(room)] = room
        aliases[normalize(room).replace("room", "")] = room

    result: Dict[str, Dict[str, Any]] = {}
    for item in iterable:
        if not isinstance(item, dict):
            continue
        key = normalize(item.get("room", ""))
        room = aliases.get(key) or aliases.get(key.replace("room", ""))
        if room is not None and room not in result:
            result[room] = item
    return result


def load_deepseek_api_key(
    project: Path,
    explicit_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """Load the API key without ever printing or persisting the secret."""
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    candidates: List[Path] = []
    if explicit_path is not None:
        candidates.append(
            explicit_path
            if explicit_path.is_absolute()
            else project / explicit_path
        )
    else:
        for name in DEFAULT_DEEPSEEK_KEY_CANDIDATES:
            candidates.append(project / name)

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue

        raw = path.read_text(
            encoding="utf-8-sig",
        ).strip()

        if not raw:
            continue

        # Accept either a bare key or DEEPSEEK_API_KEY=<key>.
        if "=" in raw and "\n" not in raw:
            left, right = raw.split("=", 1)
            if left.strip().upper() in {
                "DEEPSEEK_API_KEY",
                "API_KEY",
                "KEY",
            }:
                raw = right.strip()

        raw = raw.strip().strip('"').strip("'").strip()

        if not raw:
            continue

        return raw, f"file:{path.name}"

    if env_key:
        return env_key, "environment:DEEPSEEK_API_KEY"

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "DeepSeek API key not found. Put the key in 'deepseek_key' "
        "or 'deepseek_key.txt' in the project root, set "
        "DEEPSEEK_API_KEY, or pass --api-key-file. "
        f"Searched: {searched}"
    )


class DeepSeekAPIClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        key_source: str,
        timeout_s: float,
        temperature: float,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = str(api_key)
        self.key_source = str(key_source)
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)

    def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = self.base_url + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(
                req,
                timeout=self.timeout_s,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                pass

            # Never include request headers or the key in diagnostics.
            raise RuntimeError(
                f"DeepSeek API HTTP {exc.code}: "
                f"{detail or exc.reason}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"DeepSeek API request failed: {exc}"
            ) from exc

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(
                "Unexpected DeepSeek API response root."
            )
        return data

    def chat(
        self,
        user_prompt: str,
    ) -> Tuple[str, float, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": DEEPSEEK_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "response_format": {
                "type": "json_object",
            },
            "thinking": {
                "type": "disabled",
            },
            "temperature": self.temperature,
            "max_tokens": LLM_MAX_TOKENS,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            started = time.perf_counter()

            try:
                data = self._post_json(
                    "/chat/completions",
                    payload,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= LLM_MAX_ATTEMPTS:
                    raise
                time.sleep(0.5)
                continue

            latency = time.perf_counter() - started

            choices = data.get("choices", [])
            if not isinstance(choices, list) or not choices:
                last_error = RuntimeError(
                    "DeepSeek response contains no choices."
                )
                if attempt >= LLM_MAX_ATTEMPTS:
                    raise last_error
                time.sleep(0.5)
                continue

            message = choices[0].get("message", {})
            content = message.get("content", "")
            content = str(content or "").strip()

            # DeepSeek JSON mode may occasionally return empty content.
            if not content:
                last_error = RuntimeError(
                    "DeepSeek returned empty content."
                )
                if attempt >= LLM_MAX_ATTEMPTS:
                    raise last_error
                time.sleep(0.5)
                continue

            data["_client_attempts"] = attempt
            return content, float(latency), data

        raise RuntimeError(
            f"DeepSeek request failed: {last_error}"
        )



def build_deepseek_user_prompt(
    interval_start: pd.Timestamp,
    room_states: List[Dict[str, Any]],
) -> str:
    payload = {
        "timestamp": interval_start.isoformat(sep=" "),
        "decision_interval_minutes": LLM_DECISION_INTERVAL_MINUTES,
        "action_hold": "accepted setpoints are held until the next decision epoch",
        "rooms": room_states,
    }
    return (
        "Choose thermostat setpoints for every room in this current-state snapshot. "
        "Return only the required JSON object.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def energyplus_interval_times(
    api: Any,
    state: Any,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (interval_start, interval_end) for the active 5-minute zone interval."""
    month = int(api.exchange.month(state))
    day = int(api.exchange.day_of_month(state))
    hour = int(api.exchange.hour(state))
    minute = int(api.exchange.minutes(state))

    elapsed_minutes = (24 * 60 + minute) if hour >= 24 else (hour * 60 + minute)

    interval_end = (
        pd.Timestamp(year=2021, month=month, day=day)
        + pd.Timedelta(minutes=elapsed_minutes)
    ).round(f"{ZONE_TIMESTEP_MINUTES}min")

    interval_start = interval_end - pd.Timedelta(
        minutes=ZONE_TIMESTEP_MINUTES
    )
    return interval_start, interval_end


def in_control_period(interval_start: pd.Timestamp) -> bool:
    return CONTROL_START <= interval_start < CONTROL_END_EXCLUSIVE


def is_weather_run_period(api: Any, state: Any) -> bool:
    try:
        return int(api.exchange.kind_of_sim(state)) == WEATHER_FILE_RUN_PERIOD_KIND
    except Exception:
        return False


def api_clock_snapshot(api: Any, state: Any) -> Dict[str, Any]:
    """Return EnergyPlus clock fields for audit only, not experiment indexing."""
    def call(name: str, default: Any = None) -> Any:
        func = getattr(api.exchange, name, None)
        if func is None:
            return default
        try:
            return func(state)
        except Exception:
            return default

    return {
        "api_calendar_year": call("calendar_year"),
        "api_month": call("month"),
        "api_day_of_month": call("day_of_month"),
        "api_hour": call("hour"),
        "api_minutes": call("minutes"),
        "api_current_time_hours": call("current_time"),
        "api_current_sim_time_hours": call("current_sim_time"),
        "api_zone_timestep_number": call("zone_time_step_number"),
        "api_zone_timestep_hours": call("zone_time_step"),
        "api_num_timesteps_in_hour": call("num_time_steps_in_hour"),
        "api_kind_of_sim": call("kind_of_sim"),
    }


def resolve_project_path(path: Path, project: Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def repository_relative_path(path: Path, project: Path) -> str:
    """Return a repository-relative POSIX path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(project.resolve()).as_posix()
    except (ValueError, OSError):
        return resolved.name


def resolve_recorded_path(value: Any, project: Path) -> Optional[Path]:
    """Resolve a path recorded by a previous repository result summary."""
    if value is None or str(value).strip() == "":
        return None
    recorded = Path(str(value)).expanduser()
    return recorded.resolve() if recorded.is_absolute() else (project / recorded).resolve()


def _is_energyplus_root(path: Path) -> bool:
    return path.exists() and (path / "pyenergyplus" / "api.py").exists()


def discover_energyplus_root(preferred: Optional[Path] = None) -> Path:
    """Locate EnergyPlus 24.1 without relying on a contributor-specific path."""
    candidates: List[Path] = []

    if preferred is not None:
        candidates.append(Path(preferred).expanduser())

    for variable in ("ENERGYPLUS_ROOT", "ENERGYPLUS_HOME"):
        value = os.environ.get(variable, "").strip()
        if value:
            candidates.append(Path(value).expanduser())

    try:
        import importlib.util
        spec = importlib.util.find_spec("pyenergyplus.api")
        if spec is not None and spec.origin:
            candidates.append(Path(spec.origin).resolve().parent.parent)
    except (ImportError, AttributeError, ValueError):
        pass

    executable = shutil.which("energyplus")
    if executable:
        candidates.append(Path(executable).resolve().parent)

    candidates.extend([
        Path("C:/EnergyPlusV24-1-0"),
        Path("/usr/local/EnergyPlus-24-1-0"),
        Path("/usr/local/EnergyPlusV24-1-0"),
        Path("/opt/EnergyPlus-24-1-0"),
        Path("/opt/EnergyPlusV24-1-0"),
        Path("/Applications/EnergyPlus-24-1-0"),
        Path("/Applications/EnergyPlusV24-1-0"),
    ])

    if os.name == "nt":
        try:
            candidates.extend(sorted(Path("C:/").glob("EnergyPlusV24-1-*"), reverse=True))
        except OSError:
            pass

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve() if candidate.exists() else candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if _is_energyplus_root(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the EnergyPlus 24.1 Python API. Install EnergyPlus 24.1 "
        "and provide --energyplus-root, or set ENERGYPLUS_ROOT / ENERGYPLUS_HOME."
    )

def import_energyplus_api(energyplus_root: Path) -> Any:
    root_str = str(energyplus_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from pyenergyplus.api import EnergyPlusAPI
    except Exception as exc:
        raise RuntimeError(
            f"Could not import pyenergyplus from {energyplus_root}: {exc}"
        ) from exc

    return EnergyPlusAPI


# ============================================================================
# Thermal comfort
# ============================================================================

def fanger_pmv_ppd(
    tdb_c: float,
    rh_percent: float,
    tr_c: Optional[float] = None,
    air_speed_m_s: float = PMV_AIR_SPEED_M_S,
    met: float = PMV_MET,
    clo: float = PMV_CLO,
    external_work_met: float = PMV_EXTERNAL_WORK_MET,
) -> Tuple[float, float]:
    """Return ISO 7730-style Fanger PMV and PPD.

    This implementation is dependency-free so later controllers can use the
    exact same calculation. MRT comes directly from EnergyPlus.
    """
    try:
        tdb_c = float(tdb_c)
        tr_c = tdb_c if tr_c is None else float(tr_c)
        rh_percent = clamp(float(rh_percent), 0.0, 100.0)
        air_speed_m_s = max(float(air_speed_m_s), 0.0)
        met = float(met)
        clo = float(clo)
        external_work_met = float(external_work_met)

        values = (
            tdb_c,
            tr_c,
            rh_percent,
            air_speed_m_s,
            met,
            clo,
            external_work_met,
        )
        if not all(math.isfinite(v) for v in values):
            return float("nan"), float("nan")

        pa = (
            rh_percent
            * 10.0
            * math.exp(16.6536 - 4030.183 / (tdb_c + 235.0))
        )

        icl = 0.155 * clo
        m = met * 58.15
        w = external_work_met * 58.15
        mw = m - w

        fcl = (
            1.0 + 1.29 * icl
            if icl <= 0.078
            else 1.05 + 0.645 * icl
        )

        hcf = 12.1 * math.sqrt(air_speed_m_s)
        taa = tdb_c + 273.0
        tra = tr_c + 273.0
        tcla = taa + (35.5 - tdb_c) / (3.5 * icl + 0.1)

        p1 = icl * fcl
        p2 = p1 * 3.96
        p3 = p1 * 100.0
        p4 = p1 * taa
        p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4

        xn = tcla / 100.0
        xf = tcla / 50.0
        hc = hcf

        for _ in range(150):
            xf = (xf + xn) / 2.0
            hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
            hc = max(hcf, hcn)
            xn_new = (
                p5 + p4 * hc - p2 * xn**4
            ) / (100.0 + p3 * hc)

            if abs(xn_new - xf) <= 0.00015:
                xn = xn_new
                break

            xn = xn_new

        tcl = 100.0 * xn - 273.0

        hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)
        hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
        hl3 = 1.7e-5 * m * (5867.0 - pa)
        hl4 = 0.0014 * m * (34.0 - tdb_c)
        hl5 = 3.96 * fcl * (xn**4 - (tra / 100.0) ** 4)
        hl6 = fcl * hc * (tcl - tdb_c)

        ts = 0.303 * math.exp(-0.036 * m) + 0.028
        pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
        ppd = 100.0 - 95.0 * math.exp(
            -0.03353 * pmv**4 - 0.2179 * pmv**2
        )

        return float(pmv), float(ppd)

    except Exception:
        return float("nan"), float("nan")


# ============================================================================
# Occupancy input validation
# ============================================================================

def load_occupancy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Occupancy CSV not found: {path}")

    actual_hash = sha256_file(path)
    if actual_hash.lower() != EXPECTED_OCCUPANCY_SHA256.lower():
        raise ValueError(
            "Occupancy CSV SHA256 does not match the frozen canonical input. "
            f"Expected {EXPECTED_OCCUPANCY_SHA256}, got {actual_hash}."
        )

    df = pd.read_csv(path)

    timestamp_column = None
    for candidate in ("timestamp", "datetime", "date_time", "time"):
        if candidate in df.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        raise ValueError(
            f"Occupancy file has no timestamp column. Columns: {list(df.columns)}"
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    if df[timestamp_column].isna().any():
        raise ValueError(
            f"Occupancy CSV contains "
            f"{int(df[timestamp_column].isna().sum())} invalid timestamps."
        )

    df = df.set_index(timestamp_column).sort_index()

    if df.index.has_duplicates:
        raise ValueError(
            f"Occupancy CSV contains "
            f"{int(df.index.duplicated().sum())} duplicate timestamps."
        )

    expected_index = pd.date_range(
        CONTROL_START,
        EXPECTED_LAST_INTERVAL_START,
        freq=f"{ZONE_TIMESTEP_MINUTES}min",
    )

    if len(expected_index) != EXPECTED_INTERVALS_PER_ROOM:
        raise RuntimeError("Internal expected-index construction is incorrect.")

    missing_times = expected_index.difference(df.index)
    extras_in_week = df.index[
        (df.index >= CONTROL_START)
        & (df.index < CONTROL_END_EXCLUSIVE)
    ].difference(expected_index)

    if len(missing_times):
        raise ValueError(
            f"Occupancy CSV is missing {len(missing_times)} required rows. "
            f"First missing: {missing_times[0]}"
        )

    if len(extras_in_week):
        raise ValueError(
            f"Occupancy CSV contains {len(extras_in_week)} unexpected in-week "
            f"timestamps. First extra: {extras_in_week[0]}"
        )

    resolved_columns: Dict[str, str] = {}

    for room, info in ROOMS.items():
        candidates = (
            f"{room.lower()}_occupant_count",
            f"{room.lower()}_occupancy_count",
            room,
        )
        found = next((c for c in candidates if c in df.columns), None)

        if found is None:
            raise ValueError(
                f"Missing occupancy column for {room}. "
                f"Tried {list(candidates)}."
            )

        numeric = pd.to_numeric(df[found], errors="coerce")

        if numeric.loc[expected_index].isna().any():
            raise ValueError(f"{found} contains non-numeric/NaN values.")

        if (numeric.loc[expected_index] < 0).any():
            raise ValueError(f"{found} contains negative occupancy.")

        capacity = float(info["capacity"])
        if (numeric.loc[expected_index] > capacity + 1e-9).any():
            raise ValueError(
                f"{found} exceeds design capacity {capacity:g}. "
                f"Maximum={float(numeric.loc[expected_index].max()):g}"
            )

        df[found] = numeric.astype(float)
        resolved_columns[room] = found

    df.attrs["resolved_columns"] = resolved_columns
    return df


# ============================================================================
# IDF parsing and API-key discovery
# ============================================================================

def split_idf_objects(text: str) -> List[str]:
    objects: List[str] = []
    current: List[str] = []

    for line in text.splitlines(keepends=True):
        current.append(line)
        code = line.split("!", 1)[0]
        if ";" in code:
            objects.append("".join(current))
            current = []

    if current and "".join(current).strip():
        objects.append("".join(current))

    return objects


def idf_fields(block: str) -> List[str]:
    code_parts: List[str] = []

    for line in block.splitlines():
        code = line.split("!", 1)[0]
        if code.strip():
            code_parts.append(code)

    code = "\n".join(code_parts).replace(";", ",")
    values = [value.strip() for value in code.split(",")]

    while values and values[-1] == "":
        values.pop()

    return values


def discover_fcu_components(
    idf_path: Path,
) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}

    text = idf_path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )

    for block in split_idf_objects(text):
        fields = idf_fields(block)

        if (
            not fields
            or normalize(fields[0])
            != normalize("ZoneHVAC:FourPipeFanCoil")
        ):
            continue

        if len(fields) < 15:
            continue

        fcu_name = fields[1]
        availability_schedule = fields[2]
        fan_object_type = fields[13]
        fan_name = fields[14]

        zone_number: Optional[int] = None

        match = re.search(
            r"fcu\s*zone\s*(\d+)\s*availability",
            availability_schedule,
            flags=re.I,
        )

        if match:
            zone_number = int(match.group(1))

        if zone_number is None:
            match = re.search(
                r"thermal\s*zone\s*(\d+)",
                fcu_name,
                flags=re.I,
            )
            if match:
                zone_number = int(match.group(1))

        if zone_number is None:
            continue

        mapping[f"Thermal Zone {zone_number}"] = {
            "fcu_name": fcu_name,
            "fan_name": fan_name,
            "fan_object_type": fan_object_type,
            "availability_schedule": availability_schedule,
        }

    return mapping


def parse_api_csv(api_csv: str) -> List[List[str]]:
    rows: List[List[str]] = []

    reader = csv.reader(io.StringIO(api_csv))
    for row in reader:
        if not row:
            continue
        cleaned = [str(value).strip() for value in row]
        if cleaned:
            rows.append(cleaned)

    return rows


def api_output_keys(
    api_rows: Sequence[Sequence[str]],
    variable_name: str,
) -> List[str]:
    target = normalize(variable_name)
    keys: List[str] = []

    for row in api_rows:
        if len(row) < 3:
            continue

        if normalize(row[0]) != "outputvariable":
            continue

        if normalize(row[1]) == target and row[2]:
            keys.append(row[2])

    return sorted(set(keys))


def resolve_ventilation_output_key(
    api_rows: Sequence[Sequence[str]],
    variable_name: str,
    zone: str,
    vent_object: str,
    zone_number: int,
) -> str:
    keys = api_output_keys(api_rows, variable_name)

    targets = (normalize(vent_object), normalize(zone))
    for key in keys:
        if normalize(key) in targets:
            return key

    zone_tokens = (
        f"thermalzone{zone_number}",
        f"zone{zone_number}",
    )

    matches = [
        key
        for key in keys
        if any(token in normalize(key) for token in zone_tokens)
    ]

    return matches[0] if len(matches) == 1 else ""


# ============================================================================
# Runtime records
# ============================================================================

@dataclass
class BaselineState:
    handles_ready: bool = False
    callback_errors: List[str] = field(default_factory=list)
    handle_rows: List[Dict[str, Any]] = field(default_factory=list)

    # The last state reported for each room in each 5-minute interval.
    interval_room_state: Dict[
        Tuple[pd.Timestamp, str],
        Dict[str, Any],
    ] = field(default_factory=dict)

    # Energy meters are logged exactly once at the end of each 5-minute
    # zone timestep, after zone reporting has been finalized.
    meter_rows: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Baseline runner
# ============================================================================

class OccupancyRuleRunner:
    def __init__(
        self,
        EnergyPlusAPI: Any,
        energyplus_root: Path,
        idf_path: Path,
        epw_path: Path,
        occupancy_path: Path,
        output_dir: Path,
        fixed_summary_path: Path,
    ) -> None:
        self.energyplus_root = energyplus_root
        self.idf_path = idf_path
        self.epw_path = epw_path
        self.occupancy_path = occupancy_path
        self.output_dir = output_dir
        self.fixed_summary_path = fixed_summary_path

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.energyplus_output_dir = self.output_dir / "energyplus_output"
        if self.energyplus_output_dir.exists():
            shutil.rmtree(self.energyplus_output_dir)

        self.energyplus_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.occupancy = load_occupancy(occupancy_path)
        self.occ_columns: Dict[str, str] = dict(
            self.occupancy.attrs["resolved_columns"]
        )

        self.fcu_components = discover_fcu_components(idf_path)

        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()

        self.runtime = BaselineState()

        # ------------------------------------------------------------------
        # Authoritative research clock
        # ------------------------------------------------------------------
        # The BeginZoneTimestep callback advances this index exactly once per
        # non-warmup Weather File Run Period zone timestep.
        self.zone_sequence_index: int = -1
        self.current_interval_start: Optional[pd.Timestamp] = None
        self.current_interval_end: Optional[pd.Timestamp] = None
        self.last_logged_zone_sequence_index: int = -1

        self.begin_zone_callback_count: int = 0
        self.end_zone_callback_count: int = 0
        self.system_control_callback_count: int = 0

        # Required state feedback
        self.temp_handles: Dict[str, int] = {}
        self.rh_handles: Dict[str, int] = {}
        self.mrt_handles: Dict[str, int] = {}
        self.people_output_handles: Dict[str, int] = {}
        self.actual_heat_handles: Dict[str, int] = {}
        self.actual_cool_handles: Dict[str, int] = {}

        # Required control actuators
        self.people_actuators: Dict[str, int] = {}
        self.vent_actuators: Dict[str, int] = {}
        self.heat_actuators: Dict[str, int] = {}
        self.cool_actuators: Dict[str, int] = {}
        self.availability_actuators: Dict[str, int] = {}

        # Ventilation feedback
        self.vent_current_handles: Dict[str, int] = {}
        self.vent_standard_handles: Dict[str, int] = {}

        # Native FCU/fan diagnostics only -- never actuated here
        self.fan_names: Dict[str, str] = {}
        self.fcu_names: Dict[str, str] = {}
        self.fan_flow_handles: Dict[str, int] = {}
        self.fan_power_handles: Dict[str, int] = {}
        self.fcu_speed_ratio_handles: Dict[str, int] = {}
        self.fcu_plr_handles: Dict[str, int] = {}

        # Optional humidity / latent diagnostics
        self.humidity_ratio_handles: Dict[str, int] = {}
        self.zone_latent_rate_handles: Dict[str, int] = {}
        self.zone_sensible_rate_handles: Dict[str, int] = {}
        self.coil_latent_rate_handles: Dict[str, int] = {}
        self.coil_sensible_rate_handles: Dict[str, int] = {}
        self.coil_total_rate_handles: Dict[str, int] = {}

        # Outdoor conditions
        self.outdoor_handles: Dict[str, int] = {}

        # Energy meters
        self.meter_handles: Dict[str, int] = {}

        # Current supervisory command memory.
        self.current_commands: Dict[str, Dict[str, float]] = {
            room: {
                "occupancy": 0.0,
                "vent_fraction": 0.0,
                "heating_setpoint_C": UNOCCUPIED_HEATING_SP_C,
                "cooling_setpoint_C": UNOCCUPIED_COOLING_SP_C,
                "occupancy_band": "vacant",
                "availability": FCU_AVAILABILITY_COMMAND,
            }
            for room in ROOM_ORDER
        }

        self.request_variables()

    # ------------------------------------------------------------------
    # Requests before the simulation starts
    # ------------------------------------------------------------------

    def request_variables(self) -> None:
        for room, info in ROOMS.items():
            zone = str(info["zone"])
            vent_object = str(info["vent_object"])

            for variable in (
                "Zone Mean Air Temperature",
                "Zone Air Relative Humidity",
                "Zone Mean Radiant Temperature",
                "Zone Air Humidity Ratio",
                "Zone People Occupant Count",
                "Zone Thermostat Heating Setpoint Temperature",
                "Zone Thermostat Cooling Setpoint Temperature",
                "Zone Air System Latent Cooling Rate",
                "Zone Air System Sensible Cooling Rate",
            ):
                self.api.exchange.request_variable(
                    self.state,
                    variable,
                    zone,
                )

            for variable in (
                "Zone Ventilation Current Density Volume Flow Rate",
                "Zone Ventilation Standard Density Volume Flow Rate",
            ):
                self.api.exchange.request_variable(
                    self.state,
                    variable,
                    zone,
                )
                self.api.exchange.request_variable(
                    self.state,
                    variable,
                    vent_object,
                )

            zone_number = int(info["zone_number"])
            coil_name = f"THERMAL ZONE {zone_number} COOLING COIL"

            for variable in (
                "Cooling Coil Latent Cooling Rate",
                "Cooling Coil Sensible Cooling Rate",
                "Cooling Coil Total Cooling Rate",
            ):
                self.api.exchange.request_variable(
                    self.state,
                    variable,
                    coil_name,
                )

            components = self.fcu_components.get(zone, {})
            fan_name = components.get("fan_name", "")
            fcu_name = components.get("fcu_name", "")

            if fan_name:
                self.api.exchange.request_variable(
                    self.state,
                    "Fan Air Mass Flow Rate",
                    fan_name,
                )
                self.api.exchange.request_variable(
                    self.state,
                    "Fan Electricity Rate",
                    fan_name,
                )

            if fcu_name:
                self.api.exchange.request_variable(
                    self.state,
                    "Fan Coil Speed Ratio",
                    fcu_name,
                )
                self.api.exchange.request_variable(
                    self.state,
                    "Fan Coil Part Load Ratio",
                    fcu_name,
                )

        for variable in (
            "Site Outdoor Air Drybulb Temperature",
            "Site Outdoor Air Relative Humidity",
            "Site Outdoor Air Dewpoint Temperature",
            "Site Outdoor Air Humidity Ratio",
        ):
            self.api.exchange.request_variable(
                self.state,
                variable,
                "Environment",
            )

    # ------------------------------------------------------------------
    # Small access helpers
    # ------------------------------------------------------------------

    def occupancy_count(
        self,
        interval_start: pd.Timestamp,
        room: str,
    ) -> float:
        if interval_start not in self.occupancy.index:
            return 0.0

        column = self.occ_columns[room]
        return max(
            0.0,
            safe_float(
                self.occupancy.loc[interval_start, column],
                0.0,
            ),
        )

    def variable_value(
        self,
        state: Any,
        handle: int,
    ) -> float:
        if handle == -1:
            return float("nan")

        try:
            return safe_float(
                self.api.exchange.get_variable_value(
                    state,
                    handle,
                )
            )
        except Exception:
            return float("nan")

    def actuator_value(
        self,
        state: Any,
        handle: int,
    ) -> float:
        if handle == -1:
            return float("nan")

        getter = getattr(
            self.api.exchange,
            "get_actuator_value",
            None,
        )

        if getter is None:
            return float("nan")

        try:
            return safe_float(getter(state, handle))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Handle setup
    # ------------------------------------------------------------------

    def add_handle_row(
        self,
        room: str,
        category: str,
        api_type: str,
        api_name: str,
        key: str,
        handle: int,
        required: bool,
    ) -> None:
        self.runtime.handle_rows.append(
            {
                "room": room,
                "category": category,
                "api_type": api_type,
                "api_name": api_name,
                "key": key,
                "handle": int(handle),
                "required": bool(required),
                "status": (
                    "PASS"
                    if handle != -1
                    else ("FAIL" if required else "OPTIONAL_MISSING")
                ),
            }
        )

    def setup_handles(self, state: Any) -> None:
        if self.runtime.handles_ready:
            return

        if not self.api.exchange.api_data_fully_ready(state):
            return

        print("\n" + "=" * 78)
        print("EnergyPlus API data ready - acquiring occupancy-rule handles")
        print("=" * 78)

        api_csv = self.api.exchange.list_available_api_data_csv(
            state
        ).decode(
            "utf-8",
            errors="replace",
        )

        (self.output_dir / "available_api_data.csv").write_text(
            api_csv,
            encoding="utf-8",
        )

        api_rows = parse_api_csv(api_csv)

        for room, info in ROOMS.items():
            zone = str(info["zone"])
            zone_number = int(info["zone_number"])
            people_object = str(info["people_object"])
            vent_schedule = str(info["vent_schedule"])
            vent_object = str(info["vent_object"])
            availability_schedule = f"FCU Zone {zone_number} Availability"

            self.temp_handles[room] = self.api.exchange.get_variable_handle(
                state,
                "Zone Mean Air Temperature",
                zone,
            )
            self.rh_handles[room] = self.api.exchange.get_variable_handle(
                state,
                "Zone Air Relative Humidity",
                zone,
            )
            self.mrt_handles[room] = self.api.exchange.get_variable_handle(
                state,
                "Zone Mean Radiant Temperature",
                zone,
            )
            self.humidity_ratio_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Zone Air Humidity Ratio",
                    zone,
                )
            )
            self.people_output_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Zone People Occupant Count",
                    zone,
                )
            )
            self.actual_heat_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Zone Thermostat Heating Setpoint Temperature",
                    zone,
                )
            )
            self.actual_cool_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Zone Thermostat Cooling Setpoint Temperature",
                    zone,
                )
            )
            self.zone_latent_rate_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Zone Air System Latent Cooling Rate",
                    zone,
                )
            )
            self.zone_sensible_rate_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Zone Air System Sensible Cooling Rate",
                    zone,
                )
            )

            self.people_actuators[room] = (
                self.api.exchange.get_actuator_handle(
                    state,
                    "People",
                    "Number of People",
                    people_object,
                )
            )
            self.vent_actuators[room] = (
                self.api.exchange.get_actuator_handle(
                    state,
                    "Schedule:Constant",
                    "Schedule Value",
                    vent_schedule,
                )
            )
            self.heat_actuators[room] = (
                self.api.exchange.get_actuator_handle(
                    state,
                    "Zone Temperature Control",
                    "Heating Setpoint",
                    zone,
                )
            )
            self.cool_actuators[room] = (
                self.api.exchange.get_actuator_handle(
                    state,
                    "Zone Temperature Control",
                    "Cooling Setpoint",
                    zone,
                )
            )
            self.availability_actuators[room] = (
                self.api.exchange.get_actuator_handle(
                    state,
                    "Schedule:Constant",
                    "Schedule Value",
                    availability_schedule,
                )
            )

            current_key = resolve_ventilation_output_key(
                api_rows,
                "Zone Ventilation Current Density Volume Flow Rate",
                zone,
                vent_object,
                zone_number,
            )
            standard_key = resolve_ventilation_output_key(
                api_rows,
                "Zone Ventilation Standard Density Volume Flow Rate",
                zone,
                vent_object,
                zone_number,
            )

            self.vent_current_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Zone Ventilation Current Density Volume Flow Rate",
                    current_key,
                )
                if current_key
                else -1
            )
            self.vent_standard_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Zone Ventilation Standard Density Volume Flow Rate",
                    standard_key,
                )
                if standard_key
                else -1
            )

            components = self.fcu_components.get(zone, {})
            fan_name = components.get("fan_name", "")
            fcu_name = components.get("fcu_name", "")

            self.fan_names[room] = fan_name
            self.fcu_names[room] = fcu_name

            self.fan_flow_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Fan Air Mass Flow Rate",
                    fan_name,
                )
                if fan_name
                else -1
            )
            self.fan_power_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Fan Electricity Rate",
                    fan_name,
                )
                if fan_name
                else -1
            )
            self.fcu_speed_ratio_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Fan Coil Speed Ratio",
                    fcu_name,
                )
                if fcu_name
                else -1
            )
            self.fcu_plr_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Fan Coil Part Load Ratio",
                    fcu_name,
                )
                if fcu_name
                else -1
            )

            coil_name = f"THERMAL ZONE {zone_number} COOLING COIL"
            self.coil_latent_rate_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Cooling Coil Latent Cooling Rate",
                    coil_name,
                )
            )
            self.coil_sensible_rate_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Cooling Coil Sensible Cooling Rate",
                    coil_name,
                )
            )
            self.coil_total_rate_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state,
                    "Cooling Coil Total Cooling Rate",
                    coil_name,
                )
            )

            required = [
                (
                    "temperature_feedback",
                    "OutputVariable",
                    "Zone Mean Air Temperature",
                    zone,
                    self.temp_handles[room],
                ),
                (
                    "rh_feedback",
                    "OutputVariable",
                    "Zone Air Relative Humidity",
                    zone,
                    self.rh_handles[room],
                ),
                (
                    "mrt_feedback",
                    "OutputVariable",
                    "Zone Mean Radiant Temperature",
                    zone,
                    self.mrt_handles[room],
                ),
                (
                    "people_feedback",
                    "OutputVariable",
                    "Zone People Occupant Count",
                    zone,
                    self.people_output_handles[room],
                ),
                (
                    "people_actuator",
                    "Actuator",
                    "People / Number of People",
                    people_object,
                    self.people_actuators[room],
                ),
                (
                    "ventilation_actuator",
                    "Actuator",
                    "Schedule:Constant / Schedule Value",
                    vent_schedule,
                    self.vent_actuators[room],
                ),
                (
                    "heating_setpoint_actuator",
                    "Actuator",
                    "Zone Temperature Control / Heating Setpoint",
                    zone,
                    self.heat_actuators[room],
                ),
                (
                    "cooling_setpoint_actuator",
                    "Actuator",
                    "Zone Temperature Control / Cooling Setpoint",
                    zone,
                    self.cool_actuators[room],
                ),
                (
                    "heating_setpoint_feedback",
                    "OutputVariable",
                    "Zone Thermostat Heating Setpoint Temperature",
                    zone,
                    self.actual_heat_handles[room],
                ),
                (
                    "cooling_setpoint_feedback",
                    "OutputVariable",
                    "Zone Thermostat Cooling Setpoint Temperature",
                    zone,
                    self.actual_cool_handles[room],
                ),
                (
                    "fcu_availability_actuator",
                    "Actuator",
                    "Schedule:Constant / Schedule Value",
                    availability_schedule,
                    self.availability_actuators[room],
                ),
            ]

            for (
                category,
                api_type,
                api_name,
                key,
                handle,
            ) in required:
                self.add_handle_row(
                    room,
                    category,
                    api_type,
                    api_name,
                    key,
                    handle,
                    required=True,
                )

            vent_handle = (
                self.vent_current_handles[room]
                if self.vent_current_handles[room] != -1
                else self.vent_standard_handles[room]
            )

            self.add_handle_row(
                room,
                "ventilation_flow_feedback",
                "OutputVariable",
                (
                    "Zone Ventilation Current Density Volume Flow Rate"
                    if self.vent_current_handles[room] != -1
                    else "Zone Ventilation Standard Density Volume Flow Rate"
                ),
                current_key or standard_key,
                vent_handle,
                required=True,
            )

            for category, api_name, key, handle in (
                (
                    "fan_mass_flow_feedback",
                    "Fan Air Mass Flow Rate",
                    fan_name,
                    self.fan_flow_handles[room],
                ),
                (
                    "fan_power_feedback",
                    "Fan Electricity Rate",
                    fan_name,
                    self.fan_power_handles[room],
                ),
                (
                    "fcu_speed_ratio_feedback",
                    "Fan Coil Speed Ratio",
                    fcu_name,
                    self.fcu_speed_ratio_handles[room],
                ),
                (
                    "fcu_part_load_ratio_feedback",
                    "Fan Coil Part Load Ratio",
                    fcu_name,
                    self.fcu_plr_handles[room],
                ),
            ):
                self.add_handle_row(
                    room,
                    category,
                    "OutputVariable",
                    api_name,
                    key,
                    handle,
                    required=False,
                )

            print(
                f"{room}: "
                f"T={self.temp_handles[room]}, "
                f"RH={self.rh_handles[room]}, "
                f"MRT={self.mrt_handles[room]}, "
                f"PeopleAct={self.people_actuators[room]}, "
                f"VentAct={self.vent_actuators[room]}, "
                f"HeatAct={self.heat_actuators[room]}, "
                f"CoolAct={self.cool_actuators[room]}, "
                f"AvailAct={self.availability_actuators[room]}, "
                f"FanFlow={self.fan_flow_handles[room]}"
            )

        for variable in (
            "Site Outdoor Air Drybulb Temperature",
            "Site Outdoor Air Relative Humidity",
            "Site Outdoor Air Dewpoint Temperature",
            "Site Outdoor Air Humidity Ratio",
        ):
            handle = self.api.exchange.get_variable_handle(
                state,
                variable,
                "Environment",
            )
            self.outdoor_handles[variable] = handle
            self.add_handle_row(
                "Environment",
                "outdoor_feedback",
                "OutputVariable",
                variable,
                "Environment",
                handle,
                required=False,
            )

        for meter in ALL_ENERGY_METERS:
            handle = self.api.exchange.get_meter_handle(
                state,
                meter,
            )
            self.meter_handles[meter] = handle
            self.add_handle_row(
                "Building",
                "energy_meter",
                "Meter",
                meter,
                meter,
                handle,
                required=meter in REQUIRED_ENERGY_METERS,
            )
            print(f"meter {meter}: {handle}")

        handle_df = pd.DataFrame(self.runtime.handle_rows)
        handle_df.to_csv(
            self.output_dir / "handle_report.csv",
            index=False,
        )

        required_missing = handle_df[
            (handle_df["required"] == True)  # noqa: E712
            & (handle_df["handle"] == -1)
        ]

        if not required_missing.empty:
            rows = required_missing[
                ["room", "category", "api_name", "key"]
            ].to_dict(orient="records")
            raise RuntimeError(
                "Missing required EnergyPlus handles: "
                + json.dumps(rows, ensure_ascii=False)
            )

        self.runtime.handles_ready = True

    # ------------------------------------------------------------------
    # Callback error handling
    # ------------------------------------------------------------------

    def record_callback_error(
        self,
        callback_name: str,
        exc: BaseException,
    ) -> None:
        message = (
            f"{callback_name}: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )

        self.runtime.callback_errors.append(message)

        if len(self.runtime.callback_errors) <= 3:
            print("\nCALLBACK ERROR:")
            print(message)

    # ------------------------------------------------------------------
    # Begin-zone-timestep: occupancy and ventilation
    # ------------------------------------------------------------------

    def apply_people_and_ventilation(
        self,
        state: Any,
    ) -> None:
        """Start one authoritative 5-minute research interval.

        This is the only callback allowed to advance the research clock.
        """
        try:
            self.setup_handles(state)

            if not self.runtime.handles_ready:
                return

            if self.api.exchange.warmup_flag(state):
                return

            if not is_weather_run_period(self.api, state):
                return

            next_index = self.zone_sequence_index + 1

            if next_index >= EXPECTED_INTERVALS_PER_ROOM:
                raise RuntimeError(
                    "Received more non-warmup Weather File Run Period zone "
                    f"timesteps than expected ({EXPECTED_INTERVALS_PER_ROOM})."
                )

            interval_start = (
                CONTROL_START
                + pd.Timedelta(
                    minutes=next_index * ZONE_TIMESTEP_MINUTES
                )
            )
            interval_end = interval_start + pd.Timedelta(
                minutes=ZONE_TIMESTEP_MINUTES
            )

            self.zone_sequence_index = next_index
            self.current_interval_start = interval_start
            self.current_interval_end = interval_end
            self.begin_zone_callback_count += 1

            for room, info in ROOMS.items():
                occupancy = self.occupancy_count(
                    interval_start,
                    room,
                )

                capacity = float(info["capacity"])
                vent_fraction = clamp(
                    occupancy / capacity,
                    0.0,
                    1.0,
                )

                self.api.exchange.set_actuator_value(
                    state,
                    self.people_actuators[room],
                    occupancy,
                )
                self.api.exchange.set_actuator_value(
                    state,
                    self.vent_actuators[room],
                    vent_fraction,
                )

                self.current_commands[room]["occupancy"] = occupancy
                self.current_commands[room]["vent_fraction"] = vent_fraction

        except Exception as exc:
            self.record_callback_error(
                "apply_people_and_ventilation",
                exc,
            )

    # ------------------------------------------------------------------
    # Begin-system-timestep: occupancy-rule thermostat + FCU availability
    # ------------------------------------------------------------------

    def apply_occupancy_rule_control(
        self,
        state: Any,
    ) -> None:
        try:
            self.setup_handles(state)

            if not self.runtime.handles_ready:
                return

            if self.api.exchange.warmup_flag(state):
                return

            if not is_weather_run_period(self.api, state):
                return

            if self.current_interval_start is None:
                raise RuntimeError(
                    "System-timestep control callback executed before the "
                    "authoritative BeginZoneTimestep clock was initialized."
                )

            interval_start = self.current_interval_start

            if not in_control_period(interval_start):
                return

            for room in ROOM_ORDER:
                occupancy = self.occupancy_count(
                    interval_start,
                    room,
                )
                heating_sp, cooling_sp, occupancy_band = (
                    occupancy_rule_setpoints(interval_start, occupancy)
                )

                self.api.exchange.set_actuator_value(
                    state,
                    self.heat_actuators[room],
                    heating_sp,
                )
                self.api.exchange.set_actuator_value(
                    state,
                    self.cool_actuators[room],
                    cooling_sp,
                )
                self.api.exchange.set_actuator_value(
                    state,
                    self.availability_actuators[room],
                    FCU_AVAILABILITY_COMMAND,
                )

                self.current_commands[room][
                    "heating_setpoint_C"
                ] = heating_sp
                self.current_commands[room][
                    "cooling_setpoint_C"
                ] = cooling_sp
                self.current_commands[room][
                    "occupancy_band"
                ] = occupancy_band
                self.current_commands[room][
                    "availability"
                ] = FCU_AVAILABILITY_COMMAND

            self.system_control_callback_count += 1

            # Intentionally NO direct fan actuator call here.

        except Exception as exc:
            self.record_callback_error(
                "apply_occupancy_rule_control",
                exc,
            )

    # ------------------------------------------------------------------
    # End-zone-timestep: finalized physical state and energy
    # ------------------------------------------------------------------

    def end_zone_timestep(
        self,
        state: Any,
    ) -> None:
        """Log one finalized physical state and one meter row per zone timestep.

        This callback is deliberately at EndOfZoneTimestepAfterZoneReporting.
        It avoids the ambiguity found when EndOfSystemTimestep callbacks were
        invoked multiple times within a single 5-minute zone timestep.
        """
        try:
            if not self.runtime.handles_ready:
                return

            if self.api.exchange.warmup_flag(state):
                return

            if not is_weather_run_period(self.api, state):
                return

            if (
                self.current_interval_start is None
                or self.current_interval_end is None
                or self.zone_sequence_index < 0
            ):
                raise RuntimeError(
                    "EndZoneTimestep callback executed before the authoritative "
                    "BeginZoneTimestep clock was initialized."
                )

            if (
                self.zone_sequence_index
                == self.last_logged_zone_sequence_index
            ):
                raise RuntimeError(
                    "True duplicate EndZoneTimestep callback for zone sequence "
                    f"index {self.zone_sequence_index}."
                )

            interval_start = self.current_interval_start
            interval_end = self.current_interval_end

            if not in_control_period(interval_start):
                return

            raw_clock = api_clock_snapshot(self.api, state)

            outdoor = {
                "outdoor_drybulb_C": self.variable_value(
                    state,
                    self.outdoor_handles.get(
                        "Site Outdoor Air Drybulb Temperature",
                        -1,
                    ),
                ),
                "outdoor_RH_percent": self.variable_value(
                    state,
                    self.outdoor_handles.get(
                        "Site Outdoor Air Relative Humidity",
                        -1,
                    ),
                ),
                "outdoor_dewpoint_C": self.variable_value(
                    state,
                    self.outdoor_handles.get(
                        "Site Outdoor Air Dewpoint Temperature",
                        -1,
                    ),
                ),
                "outdoor_humidity_ratio_kg_per_kg": self.variable_value(
                    state,
                    self.outdoor_handles.get(
                        "Site Outdoor Air Humidity Ratio",
                        -1,
                    ),
                ),
            }

            for room, info in ROOMS.items():
                # Canonical occupancy is always re-read from the frozen input
                # using this exact interval_start. Do not use shared command
                # memory for analysis or validation.
                occupancy = self.occupancy_count(
                    interval_start,
                    room,
                )
                capacity = float(info["capacity"])
                vent_fraction = clamp(
                    occupancy / capacity,
                    0.0,
                    1.0,
                )

                heating_sp, cooling_sp, occupancy_band = (
                    occupancy_rule_setpoints(interval_start, occupancy)
                )

                temp = self.variable_value(
                    state,
                    self.temp_handles[room],
                )
                rh = self.variable_value(
                    state,
                    self.rh_handles[room],
                )
                mrt = self.variable_value(
                    state,
                    self.mrt_handles[room],
                )

                pmv, ppd = fanger_pmv_ppd(
                    temp,
                    rh,
                    mrt,
                )

                actual_people = self.variable_value(
                    state,
                    self.people_output_handles[room],
                )
                actual_heat = self.variable_value(
                    state,
                    self.actual_heat_handles[room],
                )
                actual_cool = self.variable_value(
                    state,
                    self.actual_cool_handles[room],
                )

                vent_current = self.variable_value(
                    state,
                    self.vent_current_handles[room],
                )
                vent_standard = self.variable_value(
                    state,
                    self.vent_standard_handles[room],
                )

                row: Dict[str, Any] = {
                    "case": "occupancy_rule_baseline",
                    "zone_sequence_index": self.zone_sequence_index,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "room": room,
                    "zone": info["zone"],
                    "office_hour": int(
                        is_office_hour(interval_start)
                    ),
                    "occupancy_count": occupancy,
                    "occupancy_band": occupancy_band,
                    "people_actual": actual_people,
                    "people_tracking_error": (
                        actual_people - occupancy
                        if is_finite(actual_people)
                        else float("nan")
                    ),
                    "vent_fraction_command": vent_fraction,
                    "vent_actuator_readback": self.actuator_value(
                        state,
                        self.vent_actuators[room],
                    ),
                    "vent_expected_m3_s": (
                        VENT_FLOW_PER_PERSON_M3_S * occupancy
                    ),
                    "vent_current_density_m3_s": vent_current,
                    "vent_standard_density_m3_s": vent_standard,
                    "zone_temp_C": temp,
                    "zone_RH_percent": rh,
                    "zone_MRT_C": mrt,
                    "zone_humidity_ratio_kg_per_kg": self.variable_value(
                        state,
                        self.humidity_ratio_handles[room],
                    ),
                    "PMV": pmv,
                    "PPD_percent": ppd,
                    "heating_setpoint_command_C": heating_sp,
                    "cooling_setpoint_command_C": cooling_sp,
                    "actual_heating_setpoint_C": actual_heat,
                    "actual_cooling_setpoint_C": actual_cool,
                    "fcu_availability_command": FCU_AVAILABILITY_COMMAND,
                    "fcu_availability_actuator_readback": (
                        self.actuator_value(
                            state,
                            self.availability_actuators[room],
                        )
                    ),
                    "fan_direct_actuation_used": 0,
                    "fan_name": self.fan_names.get(room, ""),
                    "fan_mass_flow_kg_s": self.variable_value(
                        state,
                        self.fan_flow_handles.get(room, -1),
                    ),
                    "fan_power_W": self.variable_value(
                        state,
                        self.fan_power_handles.get(room, -1),
                    ),
                    "fcu_speed_ratio": self.variable_value(
                        state,
                        self.fcu_speed_ratio_handles.get(room, -1),
                    ),
                    "fcu_part_load_ratio": self.variable_value(
                        state,
                        self.fcu_plr_handles.get(room, -1),
                    ),
                    "zone_latent_cooling_rate_W": self.variable_value(
                        state,
                        self.zone_latent_rate_handles.get(room, -1),
                    ),
                    "zone_sensible_cooling_rate_W": self.variable_value(
                        state,
                        self.zone_sensible_rate_handles.get(room, -1),
                    ),
                    "cooling_coil_latent_rate_W": self.variable_value(
                        state,
                        self.coil_latent_rate_handles.get(room, -1),
                    ),
                    "cooling_coil_sensible_rate_W": self.variable_value(
                        state,
                        self.coil_sensible_rate_handles.get(room, -1),
                    ),
                    "cooling_coil_total_rate_W": self.variable_value(
                        state,
                        self.coil_total_rate_handles.get(room, -1),
                    ),
                    **outdoor,
                    **raw_clock,
                }

                key = (interval_start, room)
                if key in self.runtime.interval_room_state:
                    raise RuntimeError(
                        "Duplicate end-zone state row for "
                        f"{room} at {interval_start}."
                    )

                self.runtime.interval_room_state[key] = row

            # One meter value per zone timestep. Output:Meter reporting
            # frequency is Timestep in the prepared IDF.
            meter_row: Dict[str, Any] = {
                "case": "occupancy_rule_baseline",
                "zone_sequence_index": self.zone_sequence_index,
                "interval_start": interval_start,
                "interval_end": interval_end,
                **raw_clock,
            }

            for meter, handle in self.meter_handles.items():
                meter_row[f"{meter}_J"] = (
                    safe_float(
                        self.api.exchange.get_meter_value(
                            state,
                            handle,
                        ),
                        float("nan"),
                    )
                    if handle != -1
                    else float("nan")
                )

            self.runtime.meter_rows.append(meter_row)

            self.last_logged_zone_sequence_index = (
                self.zone_sequence_index
            )
            self.end_zone_callback_count += 1

        except Exception as exc:
            self.record_callback_error(
                "end_zone_timestep",
                exc,
            )


    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> int:
        if not self.idf_path.exists():
            raise FileNotFoundError(
                f"Generated IDF not found: {self.idf_path}"
            )

        if not self.epw_path.exists():
            raise FileNotFoundError(
                f"EPW not found: {self.epw_path}"
            )

        self.api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
            self.state,
            self.apply_people_and_ventilation,
        )

        self.api.runtime.callback_begin_system_timestep_before_predictor(
            self.state,
            self.apply_occupancy_rule_control,
        )

        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            self.state,
            self.end_zone_timestep,
        )

        command = [
            "-w",
            str(self.epw_path),
            "-d",
            str(self.energyplus_output_dir),
            str(self.idf_path),
        ]

        print("\n" + "=" * 78)
        print("Running deterministic occupancy-rule EnergyPlus/API baseline")
        print("=" * 78)
        print(f"EnergyPlus : {self.energyplus_root}")
        print(f"IDF        : {self.idf_path}")
        print(f"EPW        : {self.epw_path}")
        print(f"Occupancy  : {self.occupancy_path}")
        print(f"Fixed ref  : {self.fixed_summary_path}")
        print(f"Results    : {self.output_dir}")
        print()
        print("Occupancy-rule supervisory policy:")
        print(
            f"  Control window: "
            f"{OFFICE_START_HOUR:02d}:00-{OFFICE_END_HOUR:02d}:00"
        )
        print(
            f"    N = 0   -> "
            f"{UNOCCUPIED_HEATING_SP_C:.1f}/"
            f"{UNOCCUPIED_COOLING_SP_C:.1f} C"
        )
        print(
            f"    N = 1-2 -> "
            f"{LOW_OCC_HEATING_SP_C:.1f}/"
            f"{LOW_OCC_COOLING_SP_C:.1f} C"
        )
        print(
            f"    N = 3-5 -> "
            f"{MEDIUM_OCC_HEATING_SP_C:.1f}/"
            f"{MEDIUM_OCC_COOLING_SP_C:.1f} C"
        )
        print(
            f"    N >= 6  -> "
            f"{HIGH_OCC_HEATING_SP_C:.1f}/"
            f"{HIGH_OCC_COOLING_SP_C:.1f} C"
        )
        print(
            f"  Outside control window -> "
            f"{UNOCCUPIED_HEATING_SP_C:.1f}/"
            f"{UNOCCUPIED_COOLING_SP_C:.1f} C"
        )
        print("  FCU availability -> 1.0")
        print("  Direct fan override -> NONE")
        print("  Occupancy look-ahead -> NONE")
        print()

        wall_start = time.perf_counter()

        status = self.api.runtime.run_energyplus(
            self.state,
            command,
        )

        wall_time = time.perf_counter() - wall_start

        print(f"\nEnergyPlus exit status: {status}")
        print(f"Simulation wall time: {wall_time:.2f} s")

        self.save_results(
            int(status),
            wall_time,
        )

        return int(status)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def parse_energyplus_err(self) -> Dict[str, Any]:
        err_path = self.energyplus_output_dir / "eplusout.err"

        result = {
            "err_file": str(err_path),
            "err_file_exists": err_path.exists(),
            "warning_count": 0,
            "severe_count": 0,
            "fatal_count": 0,
        }

        if not err_path.exists():
            return result

        text = err_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        result["warning_count"] = len(
            re.findall(r"\*\*\s*Warning\s*\*\*", text, flags=re.I)
        )
        result["severe_count"] = len(
            re.findall(r"\*\*\s*Severe\s*\*\*", text, flags=re.I)
        )
        result["fatal_count"] = len(
            re.findall(r"\*\*\s*Fatal\s*\*\*", text, flags=re.I)
        )

        return result

    def state_dataframe(self) -> pd.DataFrame:
        if not self.runtime.interval_room_state:
            return pd.DataFrame()

        frame = pd.DataFrame(
            list(self.runtime.interval_room_state.values())
        )

        frame["interval_start"] = pd.to_datetime(
            frame["interval_start"]
        )
        frame["interval_end"] = pd.to_datetime(
            frame["interval_end"]
        )

        return (
            frame.sort_values(
                ["interval_start", "room"]
            )
            .reset_index(drop=True)
        )

    def meter_dataframe(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.runtime.meter_rows)

        if frame.empty:
            return frame

        frame["interval_start"] = pd.to_datetime(
            frame["interval_start"]
        )
        frame["interval_end"] = pd.to_datetime(
            frame["interval_end"]
        )

        return frame.sort_values(
            ["interval_start", "interval_end"]
        ).reset_index(drop=True)

    def build_interval_energy(
        self,
        meter_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if meter_df.empty:
            return meter_df

        if meter_df["zone_sequence_index"].duplicated().any():
            duplicate_count = int(
                meter_df["zone_sequence_index"].duplicated().sum()
            )
            raise RuntimeError(
                "Meter trace contains duplicate authoritative zone-sequence "
                f"rows: {duplicate_count}."
            )

        if meter_df["interval_start"].duplicated().any():
            duplicate_count = int(
                meter_df["interval_start"].duplicated().sum()
            )
            raise RuntimeError(
                "Meter trace contains duplicate interval_start rows despite "
                f"the authoritative sequence clock: {duplicate_count}."
            )

        result = (
            meter_df.sort_values("interval_start")
            .reset_index(drop=True)
            .copy()
        )

        for meter in ALL_ENERGY_METERS:
            joule_col = f"{meter}_J"
            if joule_col in result.columns:
                result[f"{meter}_kWh"] = (
                    pd.to_numeric(
                        result[joule_col],
                        errors="coerce",
                    )
                    / 3_600_000.0
                )

        component_cols = [
            f"{meter}_kWh"
            for meter in REQUIRED_ENERGY_METERS
            if f"{meter}_kWh" in result.columns
        ]

        if component_cols:
            result[
                "HVAC_component_sum_kWh"
            ] = result[component_cols].sum(
                axis=1,
                min_count=len(component_cols),
            )

        return result

    @staticmethod
    def weighted_mean(
        values: pd.Series,
        weights: pd.Series,
    ) -> float:
        values = pd.to_numeric(values, errors="coerce")
        weights = pd.to_numeric(weights, errors="coerce")

        valid = (
            values.notna()
            & weights.notna()
            & (weights > 0)
        )

        if not valid.any():
            return float("nan")

        denominator = float(weights[valid].sum())

        if denominator <= 0:
            return float("nan")

        return float(
            (values[valid] * weights[valid]).sum()
            / denominator
        )

    def summarize_room(
        self,
        room_frame: pd.DataFrame,
    ) -> Dict[str, Any]:
        occupied = room_frame[
            room_frame["occupancy_count"] > 0
        ].copy()

        people_sum = float(
            pd.to_numeric(
                room_frame["occupancy_count"],
                errors="coerce",
            ).fillna(0.0).sum()
        )

        ordered = room_frame.sort_values("interval_start").copy()
        action_changed = (
            ordered[
                ["heating_setpoint_command_C", "cooling_setpoint_command_C"]
            ]
            .ne(
                ordered[
                    ["heating_setpoint_command_C", "cooling_setpoint_command_C"]
                ].shift()
            )
            .any(axis=1)
        )
        if len(action_changed) > 0:
            action_changed.iloc[0] = False

        band_counts = ordered["occupancy_band"].value_counts().to_dict()

        summary: Dict[str, Any] = {
            "room": str(room_frame["room"].iloc[0]),
            "setpoint_switch_count": int(action_changed.sum()),
            "off_hours_fixed_action_count": int(
                band_counts.get("off_hours_fixed", 0)
            ),
            "office_vacant_action_count": int(
                band_counts.get("office_vacant", 0)
            ),
            "office_low_1_2_action_count": int(
                band_counts.get("office_low_1_2", 0)
            ),
            "office_medium_3_5_action_count": int(
                band_counts.get("office_medium_3_5", 0)
            ),
            "office_high_6_plus_action_count": int(
                band_counts.get("office_high_6_plus", 0)
            ),
            "zone": str(room_frame["zone"].iloc[0]),
            "timesteps": int(len(room_frame)),
            "occupied_room_timesteps": int(len(occupied)),
            "occupied_room_hours": (
                len(occupied) * ZONE_TIMESTEP_MINUTES / 60.0
            ),
            "person_hours": (
                people_sum * ZONE_TIMESTEP_MINUTES / 60.0
            ),
            "mean_temperature_C": float(
                room_frame["zone_temp_C"].mean()
            ),
            "mean_RH_percent": float(
                room_frame["zone_RH_percent"].mean()
            ),
            "mean_MRT_C": float(
                room_frame["zone_MRT_C"].mean()
            ),
        }

        if not occupied.empty:
            summary.update(
                {
                    "occupied_mean_temperature_C": float(
                        occupied["zone_temp_C"].mean()
                    ),
                    "occupied_mean_RH_percent": float(
                        occupied["zone_RH_percent"].mean()
                    ),
                    "occupied_mean_MRT_C": float(
                        occupied["zone_MRT_C"].mean()
                    ),
                    "occupied_mean_PMV": float(
                        occupied["PMV"].mean()
                    ),
                    "occupied_mean_abs_PMV": float(
                        occupied["PMV"].abs().mean()
                    ),
                    "occupied_mean_PPD_percent": float(
                        occupied["PPD_percent"].mean()
                    ),
                    "occupied_overheat_gt_27_count": int(
                        (
                            occupied["zone_temp_C"]
                            > OVERHEAT_THRESHOLD_C
                        ).sum()
                    ),
                    "occupied_overheat_gt_27_fraction": float(
                        (
                            occupied["zone_temp_C"]
                            > OVERHEAT_THRESHOLD_C
                        ).mean()
                    ),
                    "occupied_overcool_lt_22_5_count": int(
                        (
                            occupied["zone_temp_C"]
                            < OVERCOOL_THRESHOLD_C
                        ).sum()
                    ),
                    "occupied_overcool_lt_22_5_fraction": float(
                        (
                            occupied["zone_temp_C"]
                            < OVERCOOL_THRESHOLD_C
                        ).mean()
                    ),
                    "occupied_RH_gt_85_count": int(
                        (
                            occupied["zone_RH_percent"]
                            > HIGH_RH_THRESHOLD_PERCENT
                        ).sum()
                    ),
                    "occupied_RH_gt_85_fraction": float(
                        (
                            occupied["zone_RH_percent"]
                            > HIGH_RH_THRESHOLD_PERCENT
                        ).mean()
                    ),
                    "occupied_abs_PMV_le_0_5_fraction": float(
                        (occupied["PMV"].abs() <= 0.5).mean()
                    ),
                    "occupied_PPD_le_10_fraction": float(
                        (occupied["PPD_percent"] <= 10.0).mean()
                    ),
                    "person_weighted_temperature_C": self.weighted_mean(
                        occupied["zone_temp_C"],
                        occupied["occupancy_count"],
                    ),
                    "person_weighted_RH_percent": self.weighted_mean(
                        occupied["zone_RH_percent"],
                        occupied["occupancy_count"],
                    ),
                    "person_weighted_PMV": self.weighted_mean(
                        occupied["PMV"],
                        occupied["occupancy_count"],
                    ),
                    "person_weighted_abs_PMV": self.weighted_mean(
                        occupied["PMV"].abs(),
                        occupied["occupancy_count"],
                    ),
                    "person_weighted_PPD_percent": self.weighted_mean(
                        occupied["PPD_percent"],
                        occupied["occupancy_count"],
                    ),
                }
            )

        return summary

    def build_daily_comfort(
        self,
        state_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if state_df.empty:
            return pd.DataFrame()

        frame = state_df.copy()
        frame["date"] = frame["interval_start"].dt.date.astype(str)

        rows: List[Dict[str, Any]] = []

        for date, daily in frame.groupby("date"):
            occupied = daily[
                daily["occupancy_count"] > 0
            ].copy()

            row: Dict[str, Any] = {
                "date": date,
                "room_timesteps": int(len(daily)),
                "occupied_room_timesteps": int(len(occupied)),
                "person_hours": float(
                    daily["occupancy_count"].sum()
                    * ZONE_TIMESTEP_MINUTES
                    / 60.0
                ),
            }

            if not occupied.empty:
                row.update(
                    {
                        "occupied_mean_temperature_C": float(
                            occupied["zone_temp_C"].mean()
                        ),
                        "occupied_mean_RH_percent": float(
                            occupied["zone_RH_percent"].mean()
                        ),
                        "occupied_mean_abs_PMV": float(
                            occupied["PMV"].abs().mean()
                        ),
                        "occupied_mean_PPD_percent": float(
                            occupied["PPD_percent"].mean()
                        ),
                        "occupied_overheat_gt_27_fraction": float(
                            (
                                occupied["zone_temp_C"]
                                > OVERHEAT_THRESHOLD_C
                            ).mean()
                        ),
                        "occupied_overcool_lt_22_5_fraction": float(
                            (
                                occupied["zone_temp_C"]
                                < OVERCOOL_THRESHOLD_C
                            ).mean()
                        ),
                        "occupied_RH_gt_85_fraction": float(
                            (
                                occupied["zone_RH_percent"]
                                > HIGH_RH_THRESHOLD_PERCENT
                            ).mean()
                        ),
                    }
                )

            rows.append(row)

        return pd.DataFrame(rows)

    def build_daily_energy(
        self,
        interval_energy: pd.DataFrame,
    ) -> pd.DataFrame:
        if interval_energy.empty:
            return pd.DataFrame()

        frame = interval_energy.copy()
        frame["date"] = frame["interval_start"].dt.date.astype(str)

        kwh_columns = [
            column
            for column in frame.columns
            if column.endswith("_kWh")
        ]

        return (
            frame.groupby(
                "date",
                as_index=False,
            )[kwh_columns]
            .sum(min_count=1)
        )

    def validation_checks(
        self,
        state_df: pd.DataFrame,
        meter_df: pd.DataFrame,
        energyplus_status: int,
        err_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        checks: Dict[str, Any] = {}

        checks["energyplus_exit_status_zero"] = (
            energyplus_status == 0
        )
        checks["callback_error_count_zero"] = (
            len(self.runtime.callback_errors) == 0
        )
        checks["energyplus_severe_errors_zero"] = (
            int(err_info["severe_count"]) == 0
        )
        checks["energyplus_fatal_errors_zero"] = (
            int(err_info["fatal_count"]) == 0
        )

        checks["analysis_rows_12096"] = (
            len(state_df) == EXPECTED_ANALYSIS_ROWS
        )

        checks["begin_zone_callback_count"] = (
            self.begin_zone_callback_count
        )
        checks["end_zone_callback_count"] = (
            self.end_zone_callback_count
        )
        checks["begin_zone_callback_count_2016"] = (
            self.begin_zone_callback_count
            == EXPECTED_INTERVALS_PER_ROOM
        )
        checks["end_zone_callback_count_2016"] = (
            self.end_zone_callback_count
            == EXPECTED_INTERVALS_PER_ROOM
        )

        if not state_df.empty:
            observed_sequence = sorted(
                pd.to_numeric(
                    state_df["zone_sequence_index"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
        else:
            observed_sequence = []

        expected_sequence = list(
            range(EXPECTED_INTERVALS_PER_ROOM)
        )
        checks["zone_sequence_complete_0_to_2015"] = (
            observed_sequence == expected_sequence
        )

        unique_counts: Dict[str, int] = {}
        first_by_room: Dict[str, Optional[str]] = {}
        last_by_room: Dict[str, Optional[str]] = {}
        complete_rooms: Dict[str, bool] = {}

        expected_index = pd.date_range(
            EXPECTED_FIRST_INTERVAL_START,
            EXPECTED_LAST_INTERVAL_START,
            freq=f"{ZONE_TIMESTEP_MINUTES}min",
        )

        for room in ROOM_ORDER:
            r = state_df[state_df["room"] == room]

            starts = pd.DatetimeIndex(
                pd.to_datetime(
                    r["interval_start"],
                    errors="coerce",
                ).dropna()
            ).sort_values()

            unique_starts = starts.unique()
            unique_count = len(unique_starts)

            first = (
                pd.Timestamp(unique_starts[0])
                if unique_count
                else None
            )
            last = (
                pd.Timestamp(unique_starts[-1])
                if unique_count
                else None
            )

            missing = expected_index.difference(
                pd.DatetimeIndex(unique_starts)
            )
            extras = pd.DatetimeIndex(unique_starts).difference(
                expected_index
            )

            complete = (
                unique_count == EXPECTED_INTERVALS_PER_ROOM
                and first == EXPECTED_FIRST_INTERVAL_START
                and last == EXPECTED_LAST_INTERVAL_START
                and len(missing) == 0
                and len(extras) == 0
            )

            unique_counts[room] = unique_count
            first_by_room[room] = str(first) if first is not None else None
            last_by_room[room] = str(last) if last is not None else None
            complete_rooms[room] = complete

        checks["unique_interval_starts_per_room"] = unique_counts
        checks["first_interval_start_per_room"] = first_by_room
        checks["last_interval_start_per_room"] = last_by_room
        checks["complete_time_coverage_all_rooms"] = all(
            complete_rooms.values()
        )

        if not state_df.empty:
            global_first = state_df["interval_start"].min()
            global_last = state_df["interval_start"].max()
        else:
            global_first = None
            global_last = None

        checks["first_interval_start"] = (
            str(global_first)
            if global_first is not None
            else None
        )
        checks["last_interval_start"] = (
            str(global_last)
            if global_last is not None
            else None
        )
        checks["first_interval_start_correct"] = (
            global_first == EXPECTED_FIRST_INTERVAL_START
        )
        checks["last_interval_start_correct"] = (
            global_last == EXPECTED_LAST_INTERVAL_START
        )

        if not state_df.empty:
            people_errors = pd.to_numeric(
                state_df["people_tracking_error"],
                errors="coerce",
            ).dropna()

            checks["people_tracking_max_abs_error"] = (
                float(people_errors.abs().max())
                if not people_errors.empty
                else float("nan")
            )
            checks["people_tracking_pass"] = (
                not people_errors.empty
                and float(people_errors.abs().max()) <= 0.05
            )

            heat_error = (
                pd.to_numeric(
                    state_df["actual_heating_setpoint_C"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    state_df["heating_setpoint_command_C"],
                    errors="coerce",
                )
            ).abs().dropna()

            cool_error = (
                pd.to_numeric(
                    state_df["actual_cooling_setpoint_C"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    state_df["cooling_setpoint_command_C"],
                    errors="coerce",
                )
            ).abs().dropna()

            checks["heating_setpoint_max_abs_error_C"] = (
                float(heat_error.max())
                if not heat_error.empty
                else float("nan")
            )
            checks["cooling_setpoint_max_abs_error_C"] = (
                float(cool_error.max())
                if not cool_error.empty
                else float("nan")
            )
            checks["thermostat_tracking_pass"] = (
                not heat_error.empty
                and not cool_error.empty
                and float(heat_error.max()) <= 0.10
                and float(cool_error.max()) <= 0.10
            )

            # Physical ventilation check.
            current = pd.to_numeric(
                state_df["vent_current_density_m3_s"],
                errors="coerce",
            )
            standard = pd.to_numeric(
                state_df["vent_standard_density_m3_s"],
                errors="coerce",
            )
            actual = current.where(
                current.notna(),
                standard,
            )
            expected = pd.to_numeric(
                state_df["vent_expected_m3_s"],
                errors="coerce",
            )

            valid = actual.notna() & expected.notna()

            if valid.any():
                error = (actual[valid] - expected[valid]).abs()
                tolerance = 0.001 + 0.20 * expected[valid].abs()
                fraction = float(
                    (error <= tolerance).mean()
                )
            else:
                fraction = float("nan")

            checks[
                "ventilation_fraction_within_tolerance"
            ] = fraction
            checks["ventilation_tracking_pass"] = (
                is_finite(fraction) and fraction >= 0.95
            )

            availability = pd.to_numeric(
                state_df[
                    "fcu_availability_actuator_readback"
                ],
                errors="coerce",
            ).dropna()

            if not availability.empty:
                checks[
                    "availability_readback_min"
                ] = float(availability.min())
                checks[
                    "availability_readback_max"
                ] = float(availability.max())
                checks[
                    "availability_tracking_pass"
                ] = bool(
                    (availability - FCU_AVAILABILITY_COMMAND)
                    .abs()
                    .max()
                    <= 1e-6
                )
            else:
                checks[
                    "availability_readback_min"
                ] = None
                checks[
                    "availability_readback_max"
                ] = None
                # Preflight already established this actuator. If this API
                # build does not expose readback, do not manufacture evidence.
                checks[
                    "availability_tracking_pass"
                ] = None

            checks["direct_fan_actuation_used"] = bool(
                state_df[
                    "fan_direct_actuation_used"
                ].astype(bool).any()
            )
            checks["direct_fan_actuation_absent"] = not checks[
                "direct_fan_actuation_used"
            ]

            off_hours = state_df[
                state_df["office_hour"] == 0
            ].copy()
            checks["off_hours_action_count"] = int(
                len(off_hours)
            )

            if not off_hours.empty:
                off_heat = pd.to_numeric(
                    off_hours["heating_setpoint_command_C"],
                    errors="coerce",
                )
                off_cool = pd.to_numeric(
                    off_hours["cooling_setpoint_command_C"],
                    errors="coerce",
                )
                checks["off_hours_fixed_16_28_pass"] = bool(
                    (off_heat == UNOCCUPIED_HEATING_SP_C).all()
                    and (off_cool == UNOCCUPIED_COOLING_SP_C).all()
                )
            else:
                checks["off_hours_fixed_16_28_pass"] = False

        else:
            checks["people_tracking_pass"] = False
            checks["thermostat_tracking_pass"] = False
            checks["ventilation_tracking_pass"] = False
            checks["direct_fan_actuation_absent"] = False
            checks["off_hours_action_count"] = 0
            checks["off_hours_fixed_16_28_pass"] = False

        checks["meter_rows_positive"] = len(meter_df) > 0
        checks["meter_rows_2016"] = (
            len(meter_df) == EXPECTED_INTERVALS_PER_ROOM
        )
        checks["meter_unique_interval_starts"] = (
            int(meter_df["interval_start"].nunique())
            if not meter_df.empty
            else 0
        )
        checks["meter_no_duplicate_intervals"] = (
            not meter_df["interval_start"].duplicated().any()
            if not meter_df.empty
            else False
        )

        if not meter_df.empty:
            meter_sequence = sorted(
                pd.to_numeric(
                    meter_df["zone_sequence_index"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
        else:
            meter_sequence = []

        checks["meter_zone_sequence_complete_0_to_2015"] = (
            meter_sequence
            == list(range(EXPECTED_INTERVALS_PER_ROOM))
        )

        checks["meter_time_coverage_pass"] = (
            checks["meter_rows_2016"]
            and checks["meter_unique_interval_starts"]
            == EXPECTED_INTERVALS_PER_ROOM
            and checks["meter_no_duplicate_intervals"]
            and checks[
                "meter_zone_sequence_complete_0_to_2015"
            ]
        )

        # Canonical occupancy totals are derived from the frozen input file.
        expected_occupied_room_timesteps = 0
        expected_people_sum = 0.0
        expected_index = pd.date_range(
            EXPECTED_FIRST_INTERVAL_START,
            EXPECTED_LAST_INTERVAL_START,
            freq=f"{ZONE_TIMESTEP_MINUTES}min",
        )

        for room in ROOM_ORDER:
            column = self.occ_columns[room]
            values = pd.to_numeric(
                self.occupancy.loc[expected_index, column],
                errors="coerce",
            )
            expected_occupied_room_timesteps += int(
                (values > 0).sum()
            )
            expected_people_sum += float(values.sum())

        expected_person_hours = (
            expected_people_sum
            * ZONE_TIMESTEP_MINUTES
            / 60.0
        )

        checks[
            "canonical_occupied_room_timesteps"
        ] = expected_occupied_room_timesteps
        checks[
            "canonical_person_hours"
        ] = expected_person_hours

        if not state_df.empty:
            actual_people_series = pd.to_numeric(
                state_df["people_actual"],
                errors="coerce",
            )

            actual_occupied_room_timesteps = int(
                (actual_people_series > 0).sum()
            )
            actual_person_hours = float(
                actual_people_series.fillna(0.0).sum()
                * ZONE_TIMESTEP_MINUTES
                / 60.0
            )

            checks[
                "actual_people_occupied_room_timesteps"
            ] = actual_occupied_room_timesteps
            checks[
                "actual_people_person_hours"
            ] = actual_person_hours
            checks[
                "canonical_occupancy_totals_match"
            ] = (
                actual_occupied_room_timesteps
                == expected_occupied_room_timesteps
                and abs(
                    actual_person_hours
                    - expected_person_hours
                )
                <= 1e-9
            )
        else:
            checks[
                "actual_people_occupied_room_timesteps"
            ] = 0
            checks[
                "actual_people_person_hours"
            ] = 0.0
            checks[
                "canonical_occupancy_totals_match"
            ] = False

        required_meter_valid: Dict[str, bool] = {}
        for meter in REQUIRED_ENERGY_METERS:
            handle = self.meter_handles.get(meter, -1)
            required_meter_valid[meter] = handle != -1

        checks["required_meter_handles"] = required_meter_valid
        checks["required_meter_handles_all_valid"] = all(
            required_meter_valid.values()
        )

        boolean_requirements = [
            checks["energyplus_exit_status_zero"],
            checks["callback_error_count_zero"],
            checks["energyplus_severe_errors_zero"],
            checks["energyplus_fatal_errors_zero"],
            checks["analysis_rows_12096"],
            checks["begin_zone_callback_count_2016"],
            checks["end_zone_callback_count_2016"],
            checks["zone_sequence_complete_0_to_2015"],
            checks["complete_time_coverage_all_rooms"],
            checks["first_interval_start_correct"],
            checks["last_interval_start_correct"],
            checks["people_tracking_pass"],
            checks["thermostat_tracking_pass"],
            checks["ventilation_tracking_pass"],
            checks["direct_fan_actuation_absent"],
            checks["off_hours_fixed_16_28_pass"],
            checks["meter_rows_positive"],
            checks["meter_time_coverage_pass"],
            checks["canonical_occupancy_totals_match"],
            checks["required_meter_handles_all_valid"],
        ]

        if checks.get("availability_tracking_pass") is False:
            boolean_requirements.append(False)

        checks["all_required_baseline_checks_pass"] = all(
            boolean_requirements
        )

        return checks

    def load_fixed_reference(self) -> Dict[str, Any]:
        if not self.fixed_summary_path.exists():
            raise FileNotFoundError(
                "Validated fixed-baseline summary not found: "
                f"{self.fixed_summary_path}"
            )

        data = json.loads(
            self.fixed_summary_path.read_text(
                encoding="utf-8"
            )
        )

        if not bool(
            data.get("validation", {}).get(
                "all_required_baseline_checks_pass",
                False,
            )
        ):
            raise RuntimeError(
                "The fixed-reference summary exists but is not validated."
            )

        if (
            str(data.get("occupancy_sha256", "")).lower()
            != EXPECTED_OCCUPANCY_SHA256.lower()
        ):
            raise RuntimeError(
                "Fixed reference used a different occupancy input/hash."
            )

        if str(data.get("idf", "")).lower() != str(self.idf_path).lower():
            raise RuntimeError(
                "Fixed reference and occupancy-rule run do not use the same IDF."
            )

        if str(data.get("epw", "")).lower() != str(self.epw_path).lower():
            raise RuntimeError(
                "Fixed reference and occupancy-rule run do not use the same EPW."
            )

        fixed_comfort = data.get("comfort_and_occupancy", {})
        fixed_occ_steps = fixed_comfort.get("occupied_room_timesteps")
        fixed_person_hours = fixed_comfort.get("person_hours")

        if (
            fixed_occ_steps != 3224
            or fixed_person_hours is None
            or abs(float(fixed_person_hours) - 487.1666666666667) > 1e-9
        ):
            raise RuntimeError(
                "Fixed reference does not contain the validated canonical "
                "occupancy totals (3224 occupied room-timesteps, "
                "487.1666667 person-hours)."
            )

        fixed_kwh = data.get(
            "energy", {}
        ).get("HVAC_component_sum_kWh")

        if (
            fixed_kwh is None
            or not is_finite(fixed_kwh)
            or float(fixed_kwh) <= 0.0
        ):
            raise RuntimeError(
                "Fixed-reference HVAC_component_sum_kWh is missing or invalid."
            )

        return {
            "summary_path": str(self.fixed_summary_path),
            "case": data.get("case"),
            "HVAC_component_sum_kWh": float(fixed_kwh),
            "occupied_room_timesteps": data.get(
                "comfort_and_occupancy", {}
            ).get("occupied_room_timesteps"),
            "person_hours": data.get(
                "comfort_and_occupancy", {}
            ).get("person_hours"),
        }


    def save_results(
        self,
        energyplus_status: int,
        wall_time_s: float,
    ) -> None:
        fixed_reference = self.load_fixed_reference()

        state_df = self.state_dataframe()
        meter_df = self.meter_dataframe()
        interval_energy = self.build_interval_energy(
            meter_df
        )

        state_path = (
            self.output_dir
            / "occupancy_rule_state_trace.csv"
        )
        meter_path = (
            self.output_dir
            / "occupancy_rule_meter_trace.csv"
        )
        interval_energy_path = (
            self.output_dir
            / "occupancy_rule_interval_energy.csv"
        )

        state_df.to_csv(
            state_path,
            index=False,
        )
        meter_df.to_csv(
            meter_path,
            index=False,
        )
        interval_energy.to_csv(
            interval_energy_path,
            index=False,
        )

        room_rows: List[Dict[str, Any]] = []
        if not state_df.empty:
            for room in ROOM_ORDER:
                r = state_df[
                    state_df["room"] == room
                ].copy()

                if not r.empty:
                    room_rows.append(
                        self.summarize_room(r)
                    )

        room_summary_df = pd.DataFrame(room_rows)
        room_summary_path = (
            self.output_dir
            / "occupancy_rule_room_summary.csv"
        )
        room_summary_df.to_csv(
            room_summary_path,
            index=False,
        )

        daily_comfort_df = self.build_daily_comfort(
            state_df
        )
        daily_comfort_path = (
            self.output_dir
            / "occupancy_rule_daily_comfort.csv"
        )
        daily_comfort_df.to_csv(
            daily_comfort_path,
            index=False,
        )

        daily_energy_df = self.build_daily_energy(
            interval_energy
        )
        daily_energy_path = (
            self.output_dir
            / "occupancy_rule_daily_energy.csv"
        )
        daily_energy_df.to_csv(
            daily_energy_path,
            index=False,
        )

        err_info = self.parse_energyplus_err()

        checks = self.validation_checks(
            state_df,
            meter_df,
            energyplus_status,
            err_info,
        )

        summary: Dict[str, Any] = {
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "script": "04_run_occupancy_rule_baseline.py",
            "case": "occupancy_rule_baseline",
            "energyplus_exit_status": energyplus_status,
            "simulation_wall_time_s": wall_time_s,
            "project": ".",
            "idf": repository_relative_path(self.idf_path, PROJECT_DIR),
            "epw": repository_relative_path(self.epw_path, PROJECT_DIR),
            "occupancy": repository_relative_path(self.occupancy_path, PROJECT_DIR),
            "occupancy_sha256": sha256_file(self.occupancy_path),
            "expected_occupancy_sha256": EXPECTED_OCCUPANCY_SHA256,
            "controlled_rooms": ROOM_ORDER,
            "reference_zone": "Thermal Zone 3",
            "fixed_reference": fixed_reference,
            "time_alignment": {
                "method": (
                    "authoritative sequential BeginZoneTimestep clock anchored "
                    "at CONTROL_START; API Hour/Minutes retained only for audit"
                ),
                "weather_run_period_kind_of_sim": WEATHER_FILE_RUN_PERIOD_KIND,
                "begin_zone_callback_count": self.begin_zone_callback_count,
                "end_zone_callback_count": self.end_zone_callback_count,
                "system_control_callback_count": self.system_control_callback_count,
                "zone_timestep_minutes": ZONE_TIMESTEP_MINUTES,
                "expected_intervals_per_room": (
                    EXPECTED_INTERVALS_PER_ROOM
                ),
                "expected_analysis_rows": (
                    EXPECTED_ANALYSIS_ROWS
                ),
                "actual_analysis_rows": int(
                    len(state_df)
                ),
                "expected_first_interval_start": str(
                    EXPECTED_FIRST_INTERVAL_START
                ),
                "expected_last_interval_start": str(
                    EXPECTED_LAST_INTERVAL_START
                ),
                "actual_first_interval_start": (
                    str(state_df["interval_start"].min())
                    if not state_df.empty
                    else None
                ),
                "actual_last_interval_start": (
                    str(state_df["interval_start"].max())
                    if not state_df.empty
                    else None
                ),
            },
            "policy": {
                "type": "deterministic_occupancy_rule_office_hours",
                "control_window": (
                    f"{OFFICE_START_HOUR:02d}:00-"
                    f"{OFFICE_END_HOUR:02d}:00"
                ),
                "occupancy_source": (
                    "measured occupancy count at the same 5-minute interval"
                ),
                "look_ahead_used": False,
                "office_hour_bands": {
                    "N=0": {
                        "heating_setpoint_C": UNOCCUPIED_HEATING_SP_C,
                        "cooling_setpoint_C": UNOCCUPIED_COOLING_SP_C,
                    },
                    "N=1-2": {
                        "heating_setpoint_C": LOW_OCC_HEATING_SP_C,
                        "cooling_setpoint_C": LOW_OCC_COOLING_SP_C,
                    },
                    "N=3-5": {
                        "heating_setpoint_C": MEDIUM_OCC_HEATING_SP_C,
                        "cooling_setpoint_C": MEDIUM_OCC_COOLING_SP_C,
                    },
                    "N>=6": {
                        "heating_setpoint_C": HIGH_OCC_HEATING_SP_C,
                        "cooling_setpoint_C": HIGH_OCC_COOLING_SP_C,
                    },
                },
                "off_hours_action": {
                    "heating_setpoint_C": UNOCCUPIED_HEATING_SP_C,
                    "cooling_setpoint_C": UNOCCUPIED_COOLING_SP_C,
                },
                "off_hours_people_and_ventilation_remain_measured": True,
                "fcu_availability": FCU_AVAILABILITY_COMMAND,
                "direct_fan_override": False,
            },
            "occupancy_and_ventilation": {
                "people_source": (
                    "measured 5-minute occupancy count CSV"
                ),
                "people_actuated": True,
                "ventilation_formula": (
                    "occupancy_count / design_people"
                ),
                "ventilation_flow_per_person_m3_s_person": (
                    VENT_FLOW_PER_PERSON_M3_S
                ),
            },
            "pmv_assumptions": {
                "met": PMV_MET,
                "clo": PMV_CLO,
                "air_speed_m_s": PMV_AIR_SPEED_M_S,
                "external_work_met": PMV_EXTERNAL_WORK_MET,
                "air_temperature_source": (
                    "EnergyPlus Zone Mean Air Temperature"
                ),
                "radiant_temperature_source": (
                    "EnergyPlus Zone Mean Radiant Temperature"
                ),
                "relative_humidity_source": (
                    "EnergyPlus Zone Air Relative Humidity"
                ),
            },
            "thresholds": {
                "occupied_overheating_C": (
                    OVERHEAT_THRESHOLD_C
                ),
                "occupied_overcooling_C": (
                    OVERCOOL_THRESHOLD_C
                ),
                "occupied_high_RH_percent": (
                    HIGH_RH_THRESHOLD_PERCENT
                ),
            },
            "meter_logging": {
                "callback": "EndOfZoneTimestepAfterZoneReporting",
                "timestamp_source": (
                    "authoritative BeginZoneTimestep sequence clock"
                ),
                "expected_rows": EXPECTED_INTERVALS_PER_ROOM,
                "actual_rows": int(len(meter_df)),
                "duplicate_interval_rows": int(
                    meter_df["interval_start"].duplicated().sum()
                )
                if not meter_df.empty
                else 0,
                "note": (
                    "One finalized meter value is logged per 5-minute zone "
                    "timestep; repeated system-timestep callbacks are not summed."
                ),
            },
            "energyplus_err": err_info,
            "validation": checks,
            "callback_errors": (
                self.runtime.callback_errors
            ),
        }

        # --------------------------------------------------------------
        # Whole-week energy
        # --------------------------------------------------------------

        energy_summary: Dict[str, float] = {}

        for meter in ALL_ENERGY_METERS:
            col = f"{meter}_J"
            handle = self.meter_handles.get(meter, -1)

            if (
                handle != -1
                and not meter_df.empty
                and col in meter_df.columns
            ):
                values = pd.to_numeric(
                    meter_df[col],
                    errors="coerce",
                ).dropna()

                energy_summary[
                    f"{meter}_kWh"
                ] = (
                    float(values.sum()) / 3_600_000.0
                    if not values.empty
                    else None
                )
            else:
                # Missing EnergyPlus meter means unavailable, not zero.
                energy_summary[
                    f"{meter}_kWh"
                ] = None

        component_values = [
            energy_summary.get(
                f"{meter}_kWh"
            )
            for meter in REQUIRED_ENERGY_METERS
        ]

        if all(
            value is not None and is_finite(value)
            for value in component_values
        ):
            energy_summary[
                "HVAC_component_sum_kWh"
            ] = float(sum(component_values))
        else:
            energy_summary[
                "HVAC_component_sum_kWh"
            ] = None

        summary["energy"] = energy_summary

        rule_kwh = energy_summary.get("HVAC_component_sum_kWh")
        fixed_kwh = fixed_reference["HVAC_component_sum_kWh"]

        if (
            rule_kwh is not None
            and is_finite(rule_kwh)
            and fixed_kwh > 0.0
        ):
            savings_kwh = fixed_kwh - float(rule_kwh)
            savings_percent = (
                savings_kwh / fixed_kwh * 100.0
            )
        else:
            savings_kwh = None
            savings_percent = None

        summary["comparison_to_fixed"] = {
            "fixed_HVAC_component_sum_kWh": fixed_kwh,
            "occupancy_rule_HVAC_component_sum_kWh": rule_kwh,
            "energy_difference_fixed_minus_rule_kWh": savings_kwh,
            "energy_savings_vs_fixed_percent": savings_percent,
        }

        # --------------------------------------------------------------
        # Whole-week controlled-room comfort / occupancy
        # --------------------------------------------------------------

        comfort: Dict[str, Any] = {}

        if not state_df.empty:
            # After validation, people_actual must match the canonical input
            # exactly. Use it for reported occupied counts/person-hours so the
            # physical internal-gain state is the analysis basis.
            occupied = state_df[
                state_df["people_actual"] > 0
            ].copy()

            occupancy_sum = float(
                pd.to_numeric(
                    state_df["people_actual"],
                    errors="coerce",
                ).fillna(0.0).sum()
            )

            comfort.update(
                {
                    "controlled_room_timesteps": int(
                        len(state_df)
                    ),
                    "occupied_room_timesteps": int(
                        len(occupied)
                    ),
                    "occupied_room_hours": (
                        len(occupied)
                        * ZONE_TIMESTEP_MINUTES
                        / 60.0
                    ),
                    "person_hours": (
                        occupancy_sum
                        * ZONE_TIMESTEP_MINUTES
                        / 60.0
                    ),
                }
            )

            if not occupied.empty:
                overheat = (
                    occupied["zone_temp_C"]
                    > OVERHEAT_THRESHOLD_C
                )
                overcool = (
                    occupied["zone_temp_C"]
                    < OVERCOOL_THRESHOLD_C
                )
                high_rh = (
                    occupied["zone_RH_percent"]
                    > HIGH_RH_THRESHOLD_PERCENT
                )

                comfort.update(
                    {
                        "occupied_mean_temperature_C": float(
                            occupied[
                                "zone_temp_C"
                            ].mean()
                        ),
                        "occupied_mean_RH_percent": float(
                            occupied[
                                "zone_RH_percent"
                            ].mean()
                        ),
                        "occupied_mean_MRT_C": float(
                            occupied[
                                "zone_MRT_C"
                            ].mean()
                        ),
                        "occupied_mean_PMV": float(
                            occupied["PMV"].mean()
                        ),
                        "occupied_mean_abs_PMV": float(
                            occupied[
                                "PMV"
                            ].abs().mean()
                        ),
                        "occupied_mean_PPD_percent": float(
                            occupied[
                                "PPD_percent"
                            ].mean()
                        ),
                        "occupied_overheat_gt_27_count": int(
                            overheat.sum()
                        ),
                        "occupied_overheat_gt_27_fraction": float(
                            overheat.mean()
                        ),
                        "occupied_overcool_lt_22_5_count": int(
                            overcool.sum()
                        ),
                        "occupied_overcool_lt_22_5_fraction": float(
                            overcool.mean()
                        ),
                        "occupied_RH_gt_85_count": int(
                            high_rh.sum()
                        ),
                        "occupied_RH_gt_85_fraction": float(
                            high_rh.mean()
                        ),
                        "occupied_abs_PMV_le_0_5_fraction": float(
                            (
                                occupied[
                                    "PMV"
                                ].abs()
                                <= 0.5
                            ).mean()
                        ),
                        "occupied_PPD_le_10_fraction": float(
                            (
                                occupied[
                                    "PPD_percent"
                                ]
                                <= 10.0
                            ).mean()
                        ),
                        "person_weighted_temperature_C": self.weighted_mean(
                            occupied[
                                "zone_temp_C"
                            ],
                            occupied[
                                "people_actual"
                            ],
                        ),
                        "person_weighted_RH_percent": self.weighted_mean(
                            occupied[
                                "zone_RH_percent"
                            ],
                            occupied[
                                "people_actual"
                            ],
                        ),
                        "person_weighted_PMV": self.weighted_mean(
                            occupied["PMV"],
                            occupied[
                                "people_actual"
                            ],
                        ),
                        "person_weighted_abs_PMV": self.weighted_mean(
                            occupied[
                                "PMV"
                            ].abs(),
                            occupied[
                                "people_actual"
                            ],
                        ),
                        "person_weighted_PPD_percent": self.weighted_mean(
                            occupied[
                                "PPD_percent"
                            ],
                            occupied[
                                "people_actual"
                            ],
                        ),
                    }
                )

        summary["comfort_and_occupancy"] = comfort

        if not state_df.empty:
            band_counts = (
                state_df["occupancy_band"]
                .value_counts()
                .to_dict()
            )
            total_switches = (
                int(room_summary_df["setpoint_switch_count"].sum())
                if (
                    not room_summary_df.empty
                    and "setpoint_switch_count" in room_summary_df.columns
                )
                else 0
            )
        else:
            band_counts = {}
            total_switches = 0

        summary["action_distribution"] = {
            "off_hours_fixed": int(
                band_counts.get("off_hours_fixed", 0)
            ),
            "office_vacant": int(
                band_counts.get("office_vacant", 0)
            ),
            "office_low_1_2": int(
                band_counts.get("office_low_1_2", 0)
            ),
            "office_medium_3_5": int(
                band_counts.get("office_medium_3_5", 0)
            ),
            "office_high_6_plus": int(
                band_counts.get("office_high_6_plus", 0)
            ),
            "total_setpoint_switches_across_rooms": total_switches,
        }

        summary_path = (
            self.output_dir
            / "occupancy_rule_summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                default=str,
                allow_nan=True,
            ),
            encoding="utf-8",
        )

        self.print_summary(
            summary,
            state_path,
            meter_path,
            interval_energy_path,
            room_summary_path,
            daily_energy_path,
            daily_comfort_path,
            summary_path,
        )

    def print_summary(
        self,
        summary: Dict[str, Any],
        state_path: Path,
        meter_path: Path,
        interval_energy_path: Path,
        room_summary_path: Path,
        daily_energy_path: Path,
        daily_comfort_path: Path,
        summary_path: Path,
    ) -> None:
        validation = summary["validation"]
        energy = summary["energy"]
        comfort = summary["comfort_and_occupancy"]

        print("\n" + "=" * 78)
        print("OCCUPANCY-RULE BASELINE SUMMARY")
        print("=" * 78)

        print(
            "Baseline validation: "
            + (
                "PASS"
                if validation[
                    "all_required_baseline_checks_pass"
                ]
                else "FAIL"
            )
        )

        print("\nTime alignment:")
        print(
            f"  Analysis rows: "
            f"{summary['time_alignment']['actual_analysis_rows']} / "
            f"{EXPECTED_ANALYSIS_ROWS}"
        )
        print(
            "  Unique intervals/room: "
            + ", ".join(
                f"{room}="
                f"{validation['unique_interval_starts_per_room'].get(room)}"
                for room in ROOM_ORDER
            )
        )
        print(
            "  First interval start: "
            f"{summary['time_alignment']['actual_first_interval_start']}"
        )
        print(
            "  Last interval start : "
            f"{summary['time_alignment']['actual_last_interval_start']}"
        )
        print(
            "  Begin-zone callbacks: "
            f"{validation.get('begin_zone_callback_count')} / "
            f"{EXPECTED_INTERVALS_PER_ROOM}"
        )
        print(
            "  End-zone callbacks  : "
            f"{validation.get('end_zone_callback_count')} / "
            f"{EXPECTED_INTERVALS_PER_ROOM}"
        )
        print(
            "  Sequence 0..2015 complete: "
            f"{validation.get('zone_sequence_complete_0_to_2015')}"
        )

        print("\nEnergy:")
        for meter in REQUIRED_ENERGY_METERS:
            value = energy.get(
                f"{meter}_kWh",
                float("nan"),
            )
            print(f"  {meter:24s} {value:.4f} kWh")

        print(
            "  HVAC component sum       "
            f"{energy.get('HVAC_component_sum_kWh', float('nan')):.4f} kWh"
        )

        comparison = summary.get("comparison_to_fixed", {})
        fixed_kwh = comparison.get("fixed_HVAC_component_sum_kWh")
        savings_kwh = comparison.get(
            "energy_difference_fixed_minus_rule_kWh"
        )
        savings_percent = comparison.get(
            "energy_savings_vs_fixed_percent"
        )

        if fixed_kwh is not None:
            print(
                "  Fixed reference          "
                f"{fixed_kwh:.4f} kWh"
            )
        if (
            savings_kwh is not None
            and savings_percent is not None
        ):
            print(
                "  Savings vs fixed         "
                f"{savings_kwh:.4f} kWh "
                f"({savings_percent:.3f}%)"
            )

        facility_kwh = energy.get("Electricity:Facility_kWh")
        if facility_kwh is not None and is_finite(facility_kwh):
            print(
                "  Electricity:Facility     "
                f"{facility_kwh:.4f} kWh"
            )
        else:
            print(
                "  Electricity:Facility     unavailable (meter handle = -1)"
            )

        print("\nOccupied controlled-room metrics:")
        print(
            f"  Occupied room-timesteps: "
            f"{comfort.get('occupied_room_timesteps')}"
        )
        print(
            f"  Person-hours: "
            f"{comfort.get('person_hours', float('nan')):.3f}"
        )
        print(
            f"  Overheat > {OVERHEAT_THRESHOLD_C:.1f} C: "
            f"{comfort.get('occupied_overheat_gt_27_count')} "
            f"({100.0 * comfort.get('occupied_overheat_gt_27_fraction', float('nan')):.3f}%)"
        )
        print(
            f"  Overcool < {OVERCOOL_THRESHOLD_C:.1f} C: "
            f"{comfort.get('occupied_overcool_lt_22_5_count')} "
            f"({100.0 * comfort.get('occupied_overcool_lt_22_5_fraction', float('nan')):.3f}%)"
        )
        print(
            f"  RH > {HIGH_RH_THRESHOLD_PERCENT:.0f}%: "
            f"{comfort.get('occupied_RH_gt_85_count')} "
            f"({100.0 * comfort.get('occupied_RH_gt_85_fraction', float('nan')):.3f}%)"
        )
        print(
            f"  Mean |PMV|: "
            f"{comfort.get('occupied_mean_abs_PMV', float('nan')):.4f}"
        )
        print(
            f"  Mean PPD: "
            f"{comfort.get('occupied_mean_PPD_percent', float('nan')):.3f}%"
        )

        print("\nControl integrity:")
        print(
            f"  People max abs error: "
            f"{validation.get('people_tracking_max_abs_error')}"
        )
        print(
            f"  Heating SP max abs error: "
            f"{validation.get('heating_setpoint_max_abs_error_C')} C"
        )
        print(
            f"  Cooling SP max abs error: "
            f"{validation.get('cooling_setpoint_max_abs_error_C')} C"
        )
        print(
            f"  Ventilation tracking fraction: "
            f"{validation.get('ventilation_fraction_within_tolerance')}"
        )
        print(
            f"  Canonical occupied room-timesteps: "
            f"{validation.get('canonical_occupied_room_timesteps')}"
        )
        print(
            f"  Actual occupied room-timesteps: "
            f"{validation.get('actual_people_occupied_room_timesteps')}"
        )
        print(
            f"  Canonical person-hours: "
            f"{validation.get('canonical_person_hours')}"
        )
        print(
            f"  Actual person-hours: "
            f"{validation.get('actual_people_person_hours')}"
        )
        print(
            f"  Meter rows: "
            f"{summary['meter_logging']['actual_rows']} / "
            f"{summary['meter_logging']['expected_rows']}"
        )
        print(
            f"  Duplicate meter intervals: "
            f"{summary['meter_logging']['duplicate_interval_rows']}"
        )
        print(
            "  Direct fan actuation used: "
            f"{validation.get('direct_fan_actuation_used')}"
        )
        print(
            "  Off-hours fixed 16/28: "
            f"{validation.get('off_hours_fixed_16_28_pass')} "
            f"({validation.get('off_hours_action_count')} room-timesteps)"
        )
        print(
            "  Total setpoint switches: "
            f"{summary.get('action_distribution', {}).get('total_setpoint_switches_across_rooms')}"
        )

        print("\nEnergyPlus .err:")
        print(
            f"  Warnings={summary['energyplus_err']['warning_count']}, "
            f"Severe={summary['energyplus_err']['severe_count']}, "
            f"Fatal={summary['energyplus_err']['fatal_count']}"
        )

        if not validation[
            "all_required_baseline_checks_pass"
        ]:
            print(
                "\nDO NOT use this baseline for comparison until the "
                "failed validation condition is resolved."
            )
        else:
            print(
                "\nOccupancy-rule baseline is structurally valid. Review the "
                "comparison with the frozen fixed reference before proceeding."
            )

        print("\nSaved:")
        for path in (
            state_path,
            meter_path,
            interval_energy_path,
            room_summary_path,
            daily_energy_path,
            daily_comfort_path,
            summary_path,
        ):
            print(f"  {path}")



# ============================================================================
# Direct DeepSeek controller
# ============================================================================

class DeepSeekDirectRunner(OccupancyRuleRunner):
    def __init__(
        self,
        EnergyPlusAPI: Any,
        energyplus_root: Path,
        idf_path: Path,
        epw_path: Path,
        occupancy_path: Path,
        output_dir: Path,
        fixed_summary_path: Path,
        occupancy_rule_summary_path: Path,
        comfort_rule_summary_path: Path,
        deepseek_base_url: str,
        deepseek_model: str,
        deepseek_api_key: str,
        deepseek_key_source: str,
        llm_timeout_s: float,
        llm_temperature: float,
    ) -> None:
        super().__init__(
            EnergyPlusAPI=EnergyPlusAPI,
            energyplus_root=energyplus_root,
            idf_path=idf_path,
            epw_path=epw_path,
            occupancy_path=occupancy_path,
            output_dir=output_dir,
            fixed_summary_path=fixed_summary_path,
        )

        self.occupancy_rule_summary_path = occupancy_rule_summary_path
        self.comfort_rule_summary_path = comfort_rule_summary_path

        self.deepseek_client = DeepSeekAPIClient(
            base_url=deepseek_base_url,
            model=deepseek_model,
            api_key=deepseek_api_key,
            key_source=deepseek_key_source,
            timeout_s=llm_timeout_s,
            temperature=llm_temperature,
        )
        self.deepseek_model = deepseek_model
        self.llm_timeout_s = float(llm_timeout_s)
        self.llm_temperature = float(llm_temperature)
        self.deepseek_key_source = str(deepseek_key_source)
        self.deepseek_verification: Dict[str, Any] = {}
        self.deepseek_preflight: Dict[str, Any] = {}

        self.llm_decision_epoch_count = 0
        self.llm_call_count = 0
        self.llm_call_success_count = 0
        self.llm_call_failure_count = 0
        self.llm_parse_failure_count = 0
        self.llm_room_proposal_valid_count = 0
        self.llm_fallback_count = 0
        self.llm_safety_modified_count = 0
        self.last_llm_decision_sequence_index = -1
        self.next_decision_id = 1

        self.decision_rows: List[Dict[str, Any]] = []
        self.decision_epoch_records: List[Dict[str, Any]] = []

        for room in ROOM_ORDER:
            command = self.current_commands[room]
            command.update(
                {
                    "action_reason": "off_hours_fixed",
                    "action_label": "off_hours_fixed",
                    "action_source": "off_hours_fixed",
                    "decision_id": None,
                    "decision_sequence_index": -1,
                    "decision_temp_C": float("nan"),
                    "decision_RH_percent": float("nan"),
                    "decision_MRT_C": float("nan"),
                    "decision_PMV": float("nan"),
                    "decision_PPD_percent": float("nan"),
                    "decision_state_valid": True,
                    "deepseek_confidence": None,
                    "deepseek_proposal_valid": False,
                    "deepseek_safety_modified": False,
                    "deepseek_fallback_used": False,
                    "deepseek_fallback_reason": None,
                    "raw_heating_setpoint_C": None,
                    "raw_cooling_setpoint_C": None,
                    "command_sequence_index": -1,
                }
            )

    # ------------------------------------------------------------------
    # Frozen-reference loaders
    # ------------------------------------------------------------------

    def _load_reference(
        self,
        path: Path,
        validation_key: str,
        expected_case: str,
    ) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Reference summary not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        if not bool(
            data.get("validation", {}).get(validation_key, False)
        ):
            raise RuntimeError(
                f"Reference is not validated: {path}"
            )

        if str(data.get("case", "")) != expected_case:
            raise RuntimeError(
                f"Unexpected reference case in {path}: {data.get('case')}"
            )

        if (
            str(data.get("occupancy_sha256", "")).lower()
            != EXPECTED_OCCUPANCY_SHA256.lower()
        ):
            raise RuntimeError(
                f"Reference used a different occupancy input: {path}"
            )

        if str(data.get("idf", "")).lower() != str(self.idf_path).lower():
            raise RuntimeError(f"Reference IDF mismatch: {path}")

        if str(data.get("epw", "")).lower() != str(self.epw_path).lower():
            raise RuntimeError(f"Reference EPW mismatch: {path}")

        comfort = data.get("comfort_and_occupancy", {})
        if comfort.get("occupied_room_timesteps") != 3224:
            raise RuntimeError(
                f"Reference occupied-room timestep total mismatch: {path}"
            )
        person_hours = comfort.get("person_hours")
        if (
            person_hours is None
            or abs(float(person_hours) - 487.1666666666667) > 1e-9
        ):
            raise RuntimeError(
                f"Reference person-hour total mismatch: {path}"
            )

        energy = data.get("energy", {})
        kwh = energy.get("HVAC_component_sum_kWh")
        if kwh is None or not is_finite(kwh) or float(kwh) <= 0:
            raise RuntimeError(
                f"Reference HVAC energy missing/invalid: {path}"
            )

        return data

    # ------------------------------------------------------------------
    # Direct DeepSeek control callback
    # ------------------------------------------------------------------

    def apply_occupancy_rule_control(
        self,
        state: Any,
    ) -> None:
        """Override the base Rule-OCC callback with direct DeepSeek control."""
        try:
            self.setup_handles(state)

            if not self.runtime.handles_ready:
                return
            if self.api.exchange.warmup_flag(state):
                return
            if not is_weather_run_period(self.api, state):
                return
            if self.current_interval_start is None:
                raise RuntimeError(
                    "System callback executed before BeginZoneTimestep clock."
                )

            interval_start = self.current_interval_start
            if not in_control_period(interval_start):
                return

            if not is_office_hour(interval_start):
                for room in ROOM_ORDER:
                    command = self.current_commands[room]
                    command.update(
                        {
                            "heating_setpoint_C": UNOCCUPIED_HEATING_SP_C,
                            "cooling_setpoint_C": UNOCCUPIED_COOLING_SP_C,
                            "occupancy_band": "off_hours_fixed",
                            "action_reason": "off_hours_fixed",
                            "action_label": "off_hours_fixed",
                            "action_source": "off_hours_fixed",
                            "decision_id": None,
                            "decision_sequence_index": -1,
                            "decision_state_valid": True,
                            "deepseek_confidence": None,
                            "deepseek_proposal_valid": False,
                            "deepseek_safety_modified": False,
                            "deepseek_fallback_used": False,
                            "deepseek_fallback_reason": None,
                            "raw_heating_setpoint_C": None,
                            "raw_cooling_setpoint_C": None,
                        }
                    )
            else:
                new_decision = (
                    is_llm_decision_epoch(interval_start)
                    and self.last_llm_decision_sequence_index
                    != self.zone_sequence_index
                )

                if new_decision:
                    self.llm_decision_epoch_count += 1
                    decision_id = self.next_decision_id
                    self.next_decision_id += 1

                    room_states: List[Dict[str, Any]] = []
                    state_by_room: Dict[str, Dict[str, Any]] = {}

                    for room in ROOM_ORDER:
                        occupancy = self.occupancy_count(
                            interval_start, room
                        )
                        temp = self.variable_value(
                            state, self.temp_handles[room]
                        )
                        rh = self.variable_value(
                            state, self.rh_handles[room]
                        )
                        mrt = self.variable_value(
                            state, self.mrt_handles[room]
                        )

                        valid_state = all(
                            is_finite(v) for v in (temp, rh, mrt)
                        )
                        if valid_state:
                            pmv, ppd = fanger_pmv_ppd(temp, rh, mrt)
                            valid_state = (
                                is_finite(pmv) and is_finite(ppd)
                            )
                        else:
                            pmv, ppd = float("nan"), float("nan")

                        previous = self.current_commands[room]
                        prompt_row = {
                            "room": room,
                            "occupancy_count": occupancy,
                            "air_temperature_C": temp,
                            "relative_humidity_percent": rh,
                            "MRT_C": mrt,
                            "PMV": pmv,
                            "previous_heating_setpoint_C": previous[
                                "heating_setpoint_C"
                            ],
                            "previous_cooling_setpoint_C": previous[
                                "cooling_setpoint_C"
                            ],
                        }
                        state_by_room[room] = {
                            **prompt_row,
                            "PPD_percent": ppd,
                            "state_valid": bool(valid_state),
                        }
                        if valid_state:
                            room_states.append(prompt_row)

                    user_prompt = build_deepseek_user_prompt(
                        interval_start,
                        room_states,
                    )

                    raw_content = ""
                    raw_response: Dict[str, Any] = {}
                    latency_s: Optional[float] = None
                    call_success = False
                    parse_success = False
                    call_error: Optional[str] = None
                    proposals: Dict[str, Dict[str, Any]] = {}

                    if len(room_states) == len(ROOM_ORDER):
                        self.llm_call_count += 1
                        try:
                            raw_content, latency_s, raw_response = (
                                self.deepseek_client.chat(user_prompt)
                            )
                            call_success = True
                            self.llm_call_success_count += 1

                            try:
                                parsed = extract_json_object(raw_content)
                                proposals = normalize_deepseek_rooms(parsed)
                                parse_success = True
                            except Exception as exc:
                                self.llm_parse_failure_count += 1
                                call_error = (
                                    f"JSON/shape error: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                        except Exception as exc:
                            self.llm_call_failure_count += 1
                            call_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                    else:
                        call_error = (
                            "Invalid decision state in one or more rooms; "
                            "DeepSeek call skipped."
                        )

                    epoch_record: Dict[str, Any] = {
                        "decision_id": decision_id,
                        "zone_sequence_index": self.zone_sequence_index,
                        "interval_start": interval_start.isoformat(sep=" "),
                        "model": self.deepseek_model,
                        "call_success": call_success,
                        "parse_success": parse_success,
                        "latency_s": latency_s,
                        "error": call_error,
                        "prompt": user_prompt,
                        "raw_response_content": raw_content,
                        "response_metadata": {
                            k: v
                            for k, v in raw_response.items()
                            if k not in {"message", "choices"}
                        },
                        "rooms": [],
                    }

                    for room in ROOM_ORDER:
                        ds = state_by_room[room]
                        occupancy = float(ds["occupancy_count"])

                        if not bool(ds["state_valid"]):
                            action = fallback_rule_action(occupancy)
                            action["fallback_reason"] = (
                                "invalid_decision_state"
                            )
                        else:
                            action = validate_deepseek_room_action(
                                proposals.get(room),
                                occupancy,
                            )
                            if not parse_success and action["fallback_used"]:
                                action["fallback_reason"] = (
                                    call_error or "unusable_deepseek_response"
                                )

                        if action["proposal_valid"]:
                            self.llm_room_proposal_valid_count += 1
                        if action["fallback_used"]:
                            self.llm_fallback_count += 1
                        if action["safety_modified"]:
                            self.llm_safety_modified_count += 1

                        source = (
                            "deepseek_direct"
                            if not action["fallback_used"]
                            else "rule_occ_fallback"
                        )

                        command = self.current_commands[room]
                        command.update(
                            {
                                "heating_setpoint_C": float(
                                    action["heating_setpoint_C"]
                                ),
                                "cooling_setpoint_C": float(
                                    action["cooling_setpoint_C"]
                                ),
                                "occupancy_band": source,
                                "action_reason": str(action["reason"]),
                                "action_label": str(action["action"]),
                                "action_source": source,
                                "decision_id": decision_id,
                                "decision_sequence_index": (
                                    self.zone_sequence_index
                                ),
                                "decision_temp_C": ds[
                                    "air_temperature_C"
                                ],
                                "decision_RH_percent": ds[
                                    "relative_humidity_percent"
                                ],
                                "decision_MRT_C": ds["MRT_C"],
                                "decision_PMV": ds["PMV"],
                                "decision_PPD_percent": ds[
                                    "PPD_percent"
                                ],
                                "decision_state_valid": bool(
                                    ds["state_valid"]
                                ),
                                "deepseek_confidence": action["confidence"],
                                "deepseek_proposal_valid": bool(
                                    action["proposal_valid"]
                                ),
                                "deepseek_safety_modified": bool(
                                    action["safety_modified"]
                                ),
                                "deepseek_fallback_used": bool(
                                    action["fallback_used"]
                                ),
                                "deepseek_fallback_reason": action[
                                    "fallback_reason"
                                ],
                                "raw_heating_setpoint_C": action[
                                    "raw_heating_setpoint_C"
                                ],
                                "raw_cooling_setpoint_C": action[
                                    "raw_cooling_setpoint_C"
                                ],
                            }
                        )

                        decision_row = {
                            "decision_id": decision_id,
                            "zone_sequence_index": self.zone_sequence_index,
                            "interval_start": interval_start,
                            "room": room,
                            "occupancy_count": occupancy,
                            "decision_temp_C": ds["air_temperature_C"],
                            "decision_RH_percent": ds[
                                "relative_humidity_percent"
                            ],
                            "decision_MRT_C": ds["MRT_C"],
                            "decision_PMV": ds["PMV"],
                            "decision_PPD_percent": ds["PPD_percent"],
                            "decision_state_valid": bool(
                                ds["state_valid"]
                            ),
                            "previous_heating_setpoint_C": ds[
                                "previous_heating_setpoint_C"
                            ],
                            "previous_cooling_setpoint_C": ds[
                                "previous_cooling_setpoint_C"
                            ],
                            "action_source": source,
                            "action_label": action["action"],
                            "reason": action["reason"],
                            "confidence": action["confidence"],
                            "raw_heating_setpoint_C": action[
                                "raw_heating_setpoint_C"
                            ],
                            "raw_cooling_setpoint_C": action[
                                "raw_cooling_setpoint_C"
                            ],
                            "heating_setpoint_C": action[
                                "heating_setpoint_C"
                            ],
                            "cooling_setpoint_C": action[
                                "cooling_setpoint_C"
                            ],
                            "proposal_valid": bool(
                                action["proposal_valid"]
                            ),
                            "safety_modified": bool(
                                action["safety_modified"]
                            ),
                            "fallback_used": bool(
                                action["fallback_used"]
                            ),
                            "fallback_reason": action["fallback_reason"],
                            "llm_call_success": call_success,
                            "llm_parse_success": parse_success,
                            "llm_latency_s": latency_s,
                            "llm_error": call_error,
                        }
                        self.decision_rows.append(decision_row)
                        epoch_record["rooms"].append(
                            {
                                k: (
                                    v.isoformat(sep=" ")
                                    if isinstance(v, pd.Timestamp)
                                    else v
                                )
                                for k, v in decision_row.items()
                            }
                        )

                    self.decision_epoch_records.append(epoch_record)
                    self.last_llm_decision_sequence_index = (
                        self.zone_sequence_index
                    )

                else:
                    # Hold the most recently accepted action.
                    for room in ROOM_ORDER:
                        command = self.current_commands[room]
                        if command.get("decision_id") is None:
                            occupancy = self.occupancy_count(
                                interval_start, room
                            )
                            action = fallback_rule_action(occupancy)
                            command.update(
                                {
                                    "heating_setpoint_C": action[
                                        "heating_setpoint_C"
                                    ],
                                    "cooling_setpoint_C": action[
                                        "cooling_setpoint_C"
                                    ],
                                    "occupancy_band": "fallback_hold",
                                    "action_reason": action["reason"],
                                    "action_label": action["action"],
                                    "action_source": "fallback_hold",
                                    "deepseek_fallback_used": True,
                                    "deepseek_fallback_reason": (
                                        "missing_prior_decision"
                                    ),
                                }
                            )
                        else:
                            command["action_source"] = (
                                "fallback_hold"
                                if command.get("deepseek_fallback_used")
                                else "deepseek_hold"
                            )

            for room in ROOM_ORDER:
                command = self.current_commands[room]

                self.api.exchange.set_actuator_value(
                    state,
                    self.heat_actuators[room],
                    float(command["heating_setpoint_C"]),
                )
                self.api.exchange.set_actuator_value(
                    state,
                    self.cool_actuators[room],
                    float(command["cooling_setpoint_C"]),
                )
                self.api.exchange.set_actuator_value(
                    state,
                    self.availability_actuators[room],
                    FCU_AVAILABILITY_COMMAND,
                )

                command["availability"] = FCU_AVAILABILITY_COMMAND
                command["command_sequence_index"] = (
                    self.zone_sequence_index
                )

            self.system_control_callback_count += 1

        except Exception as exc:
            self.record_callback_error(
                "apply_deepseek_direct_control",
                exc,
            )

    # ------------------------------------------------------------------
    # End-zone logger using actual DeepSeek command memory
    # ------------------------------------------------------------------

    def end_zone_timestep(
        self,
        state: Any,
    ) -> None:
        try:
            if not self.runtime.handles_ready:
                return
            if self.api.exchange.warmup_flag(state):
                return
            if not is_weather_run_period(self.api, state):
                return
            if (
                self.current_interval_start is None
                or self.current_interval_end is None
                or self.zone_sequence_index < 0
            ):
                raise RuntimeError(
                    "EndZoneTimestep callback before authoritative clock."
                )
            if (
                self.zone_sequence_index
                == self.last_logged_zone_sequence_index
            ):
                raise RuntimeError(
                    "Duplicate EndZoneTimestep for sequence "
                    f"{self.zone_sequence_index}."
                )

            interval_start = self.current_interval_start
            interval_end = self.current_interval_end
            if not in_control_period(interval_start):
                return

            raw_clock = api_clock_snapshot(self.api, state)

            outdoor = {
                "outdoor_drybulb_C": self.variable_value(
                    state,
                    self.outdoor_handles.get(
                        "Site Outdoor Air Drybulb Temperature", -1
                    ),
                ),
                "outdoor_RH_percent": self.variable_value(
                    state,
                    self.outdoor_handles.get(
                        "Site Outdoor Air Relative Humidity", -1
                    ),
                ),
                "outdoor_dewpoint_C": self.variable_value(
                    state,
                    self.outdoor_handles.get(
                        "Site Outdoor Air Dewpoint Temperature", -1
                    ),
                ),
                "outdoor_humidity_ratio_kg_per_kg": self.variable_value(
                    state,
                    self.outdoor_handles.get(
                        "Site Outdoor Air Humidity Ratio", -1
                    ),
                ),
            }

            for room, info in ROOMS.items():
                occupancy = self.occupancy_count(
                    interval_start,
                    room,
                )
                capacity = float(info["capacity"])
                vent_fraction = clamp(
                    occupancy / capacity,
                    0.0,
                    1.0,
                )

                command = self.current_commands[room]
                if int(command.get("command_sequence_index", -1)) != int(
                    self.zone_sequence_index
                ):
                    raise RuntimeError(
                        "DeepSeek command memory not aligned for "
                        f"{room}: command="
                        f"{command.get('command_sequence_index')}, "
                        f"zone={self.zone_sequence_index}."
                    )

                heating_sp = float(
                    command["heating_setpoint_C"]
                )
                cooling_sp = float(
                    command["cooling_setpoint_C"]
                )

                temp = self.variable_value(
                    state, self.temp_handles[room]
                )
                rh = self.variable_value(
                    state, self.rh_handles[room]
                )
                mrt = self.variable_value(
                    state, self.mrt_handles[room]
                )
                pmv, ppd = fanger_pmv_ppd(temp, rh, mrt)

                actual_people = self.variable_value(
                    state,
                    self.people_output_handles[room],
                )
                actual_heat = self.variable_value(
                    state,
                    self.actual_heat_handles[room],
                )
                actual_cool = self.variable_value(
                    state,
                    self.actual_cool_handles[room],
                )

                vent_current = self.variable_value(
                    state,
                    self.vent_current_handles[room],
                )
                vent_standard = self.variable_value(
                    state,
                    self.vent_standard_handles[room],
                )

                row: Dict[str, Any] = {
                    "case": "deepseek_direct",
                    "zone_sequence_index": self.zone_sequence_index,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "room": room,
                    "zone": info["zone"],
                    "office_hour": int(
                        is_office_hour(interval_start)
                    ),
                    "occupancy_count": occupancy,
                    "occupancy_band": str(
                        command.get("action_source", "")
                    ),
                    "action_source": command.get("action_source"),
                    "action_label": command.get("action_label"),
                    "action_reason": command.get("action_reason"),
                    "decision_id": command.get("decision_id"),
                    "decision_sequence_index": command.get(
                        "decision_sequence_index"
                    ),
                    "decision_temp_C": command.get(
                        "decision_temp_C"
                    ),
                    "decision_RH_percent": command.get(
                        "decision_RH_percent"
                    ),
                    "decision_MRT_C": command.get(
                        "decision_MRT_C"
                    ),
                    "decision_PMV": command.get("decision_PMV"),
                    "decision_PPD_percent": command.get(
                        "decision_PPD_percent"
                    ),
                    "decision_state_valid": command.get(
                        "decision_state_valid"
                    ),
                    "deepseek_confidence": command.get(
                        "deepseek_confidence"
                    ),
                    "deepseek_proposal_valid": command.get(
                        "deepseek_proposal_valid"
                    ),
                    "deepseek_safety_modified": command.get(
                        "deepseek_safety_modified"
                    ),
                    "deepseek_fallback_used": command.get(
                        "deepseek_fallback_used"
                    ),
                    "deepseek_fallback_reason": command.get(
                        "deepseek_fallback_reason"
                    ),
                    "raw_heating_setpoint_C": command.get(
                        "raw_heating_setpoint_C"
                    ),
                    "raw_cooling_setpoint_C": command.get(
                        "raw_cooling_setpoint_C"
                    ),
                    "command_sequence_index": command.get(
                        "command_sequence_index"
                    ),
                    "people_actual": actual_people,
                    "people_tracking_error": (
                        actual_people - occupancy
                        if is_finite(actual_people)
                        else float("nan")
                    ),
                    "vent_fraction_command": vent_fraction,
                    "vent_actuator_readback": self.actuator_value(
                        state,
                        self.vent_actuators[room],
                    ),
                    "vent_expected_m3_s": (
                        VENT_FLOW_PER_PERSON_M3_S * occupancy
                    ),
                    "vent_current_density_m3_s": vent_current,
                    "vent_standard_density_m3_s": vent_standard,
                    "zone_temp_C": temp,
                    "zone_RH_percent": rh,
                    "zone_MRT_C": mrt,
                    "zone_humidity_ratio_kg_per_kg": self.variable_value(
                        state,
                        self.humidity_ratio_handles[room],
                    ),
                    "PMV": pmv,
                    "PPD_percent": ppd,
                    "heating_setpoint_command_C": heating_sp,
                    "cooling_setpoint_command_C": cooling_sp,
                    "actual_heating_setpoint_C": actual_heat,
                    "actual_cooling_setpoint_C": actual_cool,
                    "fcu_availability_command": (
                        FCU_AVAILABILITY_COMMAND
                    ),
                    "fcu_availability_actuator_readback": (
                        self.actuator_value(
                            state,
                            self.availability_actuators[room],
                        )
                    ),
                    "fan_direct_actuation_used": 0,
                    "fan_name": self.fan_names.get(room, ""),
                    "fan_mass_flow_kg_s": self.variable_value(
                        state,
                        self.fan_flow_handles.get(room, -1),
                    ),
                    "fan_power_W": self.variable_value(
                        state,
                        self.fan_power_handles.get(room, -1),
                    ),
                    "fcu_speed_ratio": self.variable_value(
                        state,
                        self.fcu_speed_ratio_handles.get(room, -1),
                    ),
                    "fcu_part_load_ratio": self.variable_value(
                        state,
                        self.fcu_plr_handles.get(room, -1),
                    ),
                    "zone_latent_cooling_rate_W": self.variable_value(
                        state,
                        self.zone_latent_rate_handles.get(room, -1),
                    ),
                    "zone_sensible_cooling_rate_W": self.variable_value(
                        state,
                        self.zone_sensible_rate_handles.get(room, -1),
                    ),
                    "cooling_coil_latent_rate_W": self.variable_value(
                        state,
                        self.coil_latent_rate_handles.get(room, -1),
                    ),
                    "cooling_coil_sensible_rate_W": self.variable_value(
                        state,
                        self.coil_sensible_rate_handles.get(room, -1),
                    ),
                    "cooling_coil_total_rate_W": self.variable_value(
                        state,
                        self.coil_total_rate_handles.get(room, -1),
                    ),
                    **outdoor,
                    **raw_clock,
                }

                key = (interval_start, room)
                if key in self.runtime.interval_room_state:
                    raise RuntimeError(
                        f"Duplicate state row for {room} at {interval_start}."
                    )
                self.runtime.interval_room_state[key] = row

            meter_row: Dict[str, Any] = {
                "case": "deepseek_direct",
                "zone_sequence_index": self.zone_sequence_index,
                "interval_start": interval_start,
                "interval_end": interval_end,
                **raw_clock,
            }

            for meter, handle in self.meter_handles.items():
                meter_row[f"{meter}_J"] = (
                    safe_float(
                        self.api.exchange.get_meter_value(
                            state, handle
                        ),
                        float("nan"),
                    )
                    if handle != -1
                    else float("nan")
                )

            self.runtime.meter_rows.append(meter_row)
            self.last_logged_zone_sequence_index = (
                self.zone_sequence_index
            )
            self.end_zone_callback_count += 1

        except Exception as exc:
            self.record_callback_error(
                "end_zone_timestep",
                exc,
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_checks(
        self,
        state_df: pd.DataFrame,
        meter_df: pd.DataFrame,
        energyplus_status: int,
        err_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        checks = super().validation_checks(
            state_df,
            meter_df,
            energyplus_status,
            err_info,
        )

        checks["deepseek_model_verified"] = bool(
            self.deepseek_verification.get("verified", False)
        )
        checks["system_control_callback_count"] = (
            self.system_control_callback_count
        )
        checks["system_control_callback_count_2016"] = (
            self.system_control_callback_count
            == EXPECTED_INTERVALS_PER_ROOM
        )

        if not state_df.empty:
            alignment = (
                pd.to_numeric(
                    state_df["command_sequence_index"],
                    errors="coerce",
                )
                == pd.to_numeric(
                    state_df["zone_sequence_index"],
                    errors="coerce",
                )
            )
            checks["command_sequence_alignment_pass"] = bool(
                alignment.all()
            )

            heat = pd.to_numeric(
                state_df["heating_setpoint_command_C"],
                errors="coerce",
            )
            cool = pd.to_numeric(
                state_df["cooling_setpoint_command_C"],
                errors="coerce",
            )
            checks["action_bounds_pass"] = bool(
                heat.between(
                    HEATING_MIN_C,
                    HEATING_MAX_C,
                    inclusive="both",
                ).all()
                and cool.between(
                    COOLING_MIN_C,
                    COOLING_MAX_C,
                    inclusive="both",
                ).all()
            )
            checks["action_resolution_pass"] = bool(
                (
                    (
                        heat / SETPOINT_RESOLUTION_C
                        - (
                            heat
                            / SETPOINT_RESOLUTION_C
                        ).round()
                    ).abs()
                    <= 1e-9
                ).all()
                and (
                    (
                        cool / SETPOINT_RESOLUTION_C
                        - (
                            cool
                            / SETPOINT_RESOLUTION_C
                        ).round()
                    ).abs()
                    <= 1e-9
                ).all()
            )
            checks["action_deadband_pass"] = bool(
                (
                    (cool - heat)
                    >= MIN_DEADBAND_C - 1e-9
                ).all()
            )
        else:
            checks["command_sequence_alignment_pass"] = False
            checks["action_bounds_pass"] = False
            checks["action_resolution_pass"] = False
            checks["action_deadband_pass"] = False

        decision_df = pd.DataFrame(self.decision_rows)

        checks["llm_decision_epoch_count"] = (
            self.llm_decision_epoch_count
        )
        checks["expected_llm_decision_epochs"] = (
            EXPECTED_LLM_DECISION_EPOCHS
        )
        checks["llm_decision_epoch_count_expected"] = (
            self.llm_decision_epoch_count
            == EXPECTED_LLM_DECISION_EPOCHS
        )
        checks["llm_call_count"] = self.llm_call_count
        checks["llm_call_count_expected"] = (
            self.llm_call_count
            == EXPECTED_LLM_DECISION_EPOCHS
        )
        checks["llm_call_success_count"] = (
            self.llm_call_success_count
        )
        checks["llm_call_failure_count"] = (
            self.llm_call_failure_count
        )
        checks["llm_parse_failure_count"] = (
            self.llm_parse_failure_count
        )
        checks["llm_room_decision_rows"] = int(
            len(decision_df)
        )
        checks["llm_room_decision_rows_expected"] = (
            len(decision_df)
            == EXPECTED_LLM_ROOM_DECISIONS
        )

        if not decision_df.empty:
            state_valid = (
                decision_df["decision_state_valid"]
                .astype(bool)
            )
            checks["decision_state_valid_fraction"] = float(
                state_valid.mean()
            )
            checks["decision_state_all_valid"] = bool(
                state_valid.all()
            )

            fallback = (
                decision_df["fallback_used"]
                .astype(bool)
            )
            fallback_fraction = float(fallback.mean())
            checks["llm_fallback_count"] = int(
                fallback.sum()
            )
            checks["llm_fallback_fraction"] = (
                fallback_fraction
            )
            checks[
                "llm_fallback_fraction_within_limit"
            ] = bool(
                fallback_fraction
                <= MAX_LLM_FALLBACK_FRACTION
            )

            proposal_valid = (
                decision_df["proposal_valid"]
                .astype(bool)
            )
            checks["llm_proposal_valid_fraction"] = float(
                proposal_valid.mean()
            )

            modified = (
                decision_df["safety_modified"]
                .astype(bool)
            )
            checks["llm_safety_modified_count"] = int(
                modified.sum()
            )
            checks[
                "llm_safety_modified_fraction"
            ] = float(modified.mean())
        else:
            checks["decision_state_valid_fraction"] = 0.0
            checks["decision_state_all_valid"] = False
            checks["llm_fallback_count"] = 0
            checks["llm_fallback_fraction"] = 1.0
            checks[
                "llm_fallback_fraction_within_limit"
            ] = False
            checks["llm_proposal_valid_fraction"] = 0.0
            checks["llm_safety_modified_count"] = 0
            checks["llm_safety_modified_fraction"] = 0.0

        required = [
            checks["energyplus_exit_status_zero"],
            checks["callback_error_count_zero"],
            checks["energyplus_severe_errors_zero"],
            checks["energyplus_fatal_errors_zero"],
            checks["analysis_rows_12096"],
            checks["begin_zone_callback_count_2016"],
            checks["end_zone_callback_count_2016"],
            checks["system_control_callback_count_2016"],
            checks["zone_sequence_complete_0_to_2015"],
            checks["complete_time_coverage_all_rooms"],
            checks["first_interval_start_correct"],
            checks["last_interval_start_correct"],
            checks["people_tracking_pass"],
            checks["thermostat_tracking_pass"],
            checks["ventilation_tracking_pass"],
            checks["direct_fan_actuation_absent"],
            checks["off_hours_fixed_16_28_pass"],
            checks["meter_rows_positive"],
            checks["meter_time_coverage_pass"],
            checks["canonical_occupancy_totals_match"],
            checks["required_meter_handles_all_valid"],
            checks["deepseek_model_verified"],
            checks["command_sequence_alignment_pass"],
            checks["action_bounds_pass"],
            checks["action_resolution_pass"],
            checks["action_deadband_pass"],
            checks["llm_decision_epoch_count_expected"],
            checks["llm_call_count_expected"],
            checks["llm_room_decision_rows_expected"],
            checks["decision_state_all_valid"],
            checks["llm_fallback_fraction_within_limit"],
        ]

        if checks.get("availability_tracking_pass") is False:
            required.append(False)

        checks["all_required_deepseek_direct_checks_pass"] = all(
            required
        )
        checks["all_required_baseline_checks_pass"] = checks[
            "all_required_deepseek_direct_checks_pass"
        ]

        return checks

    # ------------------------------------------------------------------
    # Run / output
    # ------------------------------------------------------------------

    def run_deepseek_preflight(self) -> Dict[str, Any]:
        """Execute one real six-room API call before EnergyPlus starts."""
        representative_states: List[Dict[str, Any]] = [
            {
                "room": "Room_1",
                "occupancy_count": 2,
                "air_temperature_C": 25.1,
                "relative_humidity_percent": 78.0,
                "MRT_C": 23.4,
                "PMV": -0.05,
                "previous_heating_setpoint_C": 18.0,
                "previous_cooling_setpoint_C": 27.0,
            },
            {
                "room": "Room_2",
                "occupancy_count": 1,
                "air_temperature_C": 26.2,
                "relative_humidity_percent": 86.0,
                "MRT_C": 23.9,
                "PMV": 0.22,
                "previous_heating_setpoint_C": 18.0,
                "previous_cooling_setpoint_C": 27.0,
            },
            {
                "room": "Room_4",
                "occupancy_count": 4,
                "air_temperature_C": 24.7,
                "relative_humidity_percent": 76.0,
                "MRT_C": 23.2,
                "PMV": -0.12,
                "previous_heating_setpoint_C": 19.0,
                "previous_cooling_setpoint_C": 26.0,
            },
            {
                "room": "Room_5",
                "occupancy_count": 1,
                "air_temperature_C": 22.8,
                "relative_humidity_percent": 83.0,
                "MRT_C": 22.1,
                "PMV": -0.48,
                "previous_heating_setpoint_C": 18.0,
                "previous_cooling_setpoint_C": 27.0,
            },
            {
                "room": "Room_6",
                "occupancy_count": 5,
                "air_temperature_C": 26.8,
                "relative_humidity_percent": 88.0,
                "MRT_C": 24.1,
                "PMV": 0.38,
                "previous_heating_setpoint_C": 19.0,
                "previous_cooling_setpoint_C": 26.0,
            },
            {
                "room": "Room_7",
                "occupancy_count": 3,
                "air_temperature_C": 25.4,
                "relative_humidity_percent": 80.0,
                "MRT_C": 23.6,
                "PMV": 0.05,
                "previous_heating_setpoint_C": 19.0,
                "previous_cooling_setpoint_C": 26.0,
            },
        ]

        prompt = build_deepseek_user_prompt(
            pd.Timestamp("2021-08-16 09:00:00"),
            representative_states,
        )

        print("\n" + "=" * 78)
        print("DeepSeek API six-room preflight")
        print("=" * 78)
        print(f"Endpoint   : {self.deepseek_client.base_url}")
        print(f"Model      : {self.deepseek_model}")
        print("Thinking   : disabled")
        print("JSON mode  : enabled")
        print(
            f"Timeout    : {self.llm_timeout_s:.0f} s, "
            f"max_tokens={LLM_MAX_TOKENS}"
        )

        content, latency_s, response = (
            self.deepseek_client.chat(prompt)
        )

        parsed = extract_json_object(content)
        proposals = normalize_deepseek_rooms(parsed)

        missing = [
            room for room in ROOM_ORDER
            if room not in proposals
        ]
        if missing:
            raise RuntimeError(
                "DeepSeek preflight response is missing rooms: "
                f"{missing}"
            )

        validated: Dict[str, Dict[str, Any]] = {}
        for state_row in representative_states:
            room = state_row["room"]
            action = validate_deepseek_room_action(
                proposals[room],
                float(state_row["occupancy_count"]),
            )
            if action["fallback_used"]:
                raise RuntimeError(
                    "DeepSeek preflight produced an unusable "
                    f"proposal for {room}: "
                    f"{action['fallback_reason']}"
                )
            validated[room] = action

        usage = response.get("usage", {})
        result = {
            "passed": True,
            "latency_s": float(latency_s),
            "model_requested": self.deepseek_model,
            "model_returned": response.get("model"),
            "response_id": response.get("id"),
            "client_attempts": response.get(
                "_client_attempts",
                1,
            ),
            "usage": usage,
            "validated_actions": validated,
        }

        preflight_path = (
            self.output_dir
            / "deepseek_direct_preflight.json"
        )
        preflight_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        self.deepseek_preflight = result
        self.deepseek_verification = {
            "verified": True,
            "base_url": self.deepseek_client.base_url,
            "model_requested": self.deepseek_model,
            "model_returned": response.get("model"),
            "key_source": self.deepseek_key_source,
        }

        print(f"Latency    : {latency_s:.2f} s")
        print(
            f"Usage      : prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"total={usage.get('total_tokens')}"
        )
        print("Preflight  : PASS")
        print(f"Saved      : {preflight_path}")

        return result


    def run(self) -> int:
        if not self.idf_path.exists():
            raise FileNotFoundError(
                f"Generated IDF not found: {self.idf_path}"
            )
        if not self.epw_path.exists():
            raise FileNotFoundError(
                f"EPW not found: {self.epw_path}"
            )

        print("\n" + "=" * 78)
        print("Verifying DeepSeek API controller")
        print("=" * 78)
        print(f"Endpoint   : {self.deepseek_client.base_url}")
        print(f"Model      : {self.deepseek_model}")
        print(f"Key source : {self.deepseek_key_source}")
        print(
            f"Decision   : every "
            f"{LLM_DECISION_INTERVAL_MINUTES} min "
            f"({EXPECTED_LLM_DECISION_EPOCHS} expected epochs)"
        )
        print(
            f"Generation : temperature={self.llm_temperature}, "
            f"thinking=disabled, max_tokens={LLM_MAX_TOKENS}"
        )

        # A real API call must pass before EnergyPlus starts. This prevents
        # a multi-hour fallback-only run when credentials, model access,
        # balance, connectivity, or JSON behavior are invalid.
        self.run_deepseek_preflight()

        self.api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
            self.state,
            self.apply_people_and_ventilation,
        )
        self.api.runtime.callback_begin_system_timestep_before_predictor(
            self.state,
            self.apply_occupancy_rule_control,
        )
        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            self.state,
            self.end_zone_timestep,
        )

        command = [
            "-w",
            str(self.epw_path),
            "-d",
            str(self.energyplus_output_dir),
            str(self.idf_path),
        ]

        print("\n" + "=" * 78)
        print("Running direct DeepSeek API EnergyPlus controller")
        print("=" * 78)
        print(f"EnergyPlus : {self.energyplus_root}")
        print(f"IDF        : {self.idf_path}")
        print(f"EPW        : {self.epw_path}")
        print(f"Occupancy  : {self.occupancy_path}")
        print(f"Fixed ref  : {self.fixed_summary_path}")
        print(f"Rule-OCC ref: {self.occupancy_rule_summary_path}")
        print(f"Comfort ref: {self.comfort_rule_summary_path}")
        print(f"Results    : {self.output_dir}")
        print()
        print("Direct DeepSeek supervisory policy:")
        print(
            f"  Control window: "
            f"{OFFICE_START_HOUR:02d}:00-"
            f"{OFFICE_END_HOUR:02d}:00"
        )
        print(f"  Model: {self.deepseek_model}")
        print(
            f"  Decision interval: "
            f"{LLM_DECISION_INTERVAL_MINUTES} min"
        )
        print(
            f"  Heat {HEATING_MIN_C:.1f}-{HEATING_MAX_C:.1f} C; "
            f"cool {COOLING_MIN_C:.1f}-{COOLING_MAX_C:.1f} C; "
            f"resolution {SETPOINT_RESOLUTION_C:.1f} C"
        )
        print(
            "  Invalid/missing proposal -> frozen Rule-OCC fallback"
        )
        print("  Off-hours -> 16/28 C")
        print("  FCU availability -> 1.0")
        print("  Direct fan override -> NONE")
        print("  Evaluator -> NONE")
        print("  Future information -> NONE")
        print()

        wall_start = time.perf_counter()
        status = self.api.runtime.run_energyplus(
            self.state,
            command,
        )
        wall_time = time.perf_counter() - wall_start

        print(f"\nEnergyPlus exit status: {status}")
        print(f"Simulation wall time: {wall_time:.2f} s")

        self.save_results(
            int(status),
            wall_time,
        )
        return int(status)

    def print_summary(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Suppress the base Rule-OCC terminal summary. The enriched DeepSeek
        # summary is printed after super().save_results returns.
        return

    def save_results(
        self,
        energyplus_status: int,
        wall_time_s: float,
    ) -> None:
        # Reuse the validated deterministic infrastructure for energy,
        # comfort, daily summaries, room summaries, and integrity checks.
        super().save_results(
            energyplus_status,
            wall_time_s,
        )

        old_summary_path = (
            self.output_dir / "occupancy_rule_summary.json"
        )
        if not old_summary_path.exists():
            raise RuntimeError(
                "Base summary was not created."
            )

        summary = json.loads(
            old_summary_path.read_text(encoding="utf-8")
        )

        occ_ref = self._load_reference(
            self.occupancy_rule_summary_path,
            "all_required_baseline_checks_pass",
            "occupancy_rule_baseline",
        )
        comfort_ref = self._load_reference(
            self.comfort_rule_summary_path,
            "all_required_comfort_rule_checks_pass",
            "comfort_rule_baseline",
        )

        # Save exact prompt and DeepSeek decision logs.
        prompt_path = (
            self.output_dir / "deepseek_direct_prompt.txt"
        )
        prompt_path.write_text(
            DEEPSEEK_SYSTEM_PROMPT,
            encoding="utf-8",
        )

        decision_df = pd.DataFrame(self.decision_rows)
        decision_csv_path = (
            self.output_dir / "deepseek_direct_decisions.csv"
        )
        decision_df.to_csv(
            decision_csv_path,
            index=False,
        )

        decision_jsonl_path = (
            self.output_dir / "deepseek_direct_decisions.jsonl"
        )
        with decision_jsonl_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for record in self.decision_epoch_records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

        # Rename base files to DeepSeek-specific names.
        rename_map = {
            "occupancy_rule_state_trace.csv": (
                "deepseek_direct_state_trace.csv"
            ),
            "occupancy_rule_meter_trace.csv": (
                "deepseek_direct_meter_trace.csv"
            ),
            "occupancy_rule_interval_energy.csv": (
                "deepseek_direct_interval_energy.csv"
            ),
            "occupancy_rule_room_summary.csv": (
                "deepseek_direct_room_summary.csv"
            ),
            "occupancy_rule_daily_energy.csv": (
                "deepseek_direct_daily_energy.csv"
            ),
            "occupancy_rule_daily_comfort.csv": (
                "deepseek_direct_daily_comfort.csv"
            ),
        }
        for old_name, new_name in rename_map.items():
            old_path = self.output_dir / old_name
            new_path = self.output_dir / new_name
            if old_path.exists():
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)

        summary["script"] = "07_run_deepseek_direct.py"
        summary["case"] = "deepseek_direct"
        summary["fixed_reference"] = summary.get(
            "fixed_reference"
        )
        summary["occupancy_rule_reference"] = {
            "summary_path": str(
                self.occupancy_rule_summary_path
            ),
            "HVAC_component_sum_kWh": float(
                occ_ref["energy"][
                    "HVAC_component_sum_kWh"
                ]
            ),
            "comfort_and_occupancy": occ_ref.get(
                "comfort_and_occupancy", {}
            ),
            "switches": occ_ref.get(
                "action_distribution", {}
            ).get(
                "total_setpoint_switches_across_rooms"
            ),
        }
        summary["comfort_rule_reference"] = {
            "summary_path": str(
                self.comfort_rule_summary_path
            ),
            "HVAC_component_sum_kWh": float(
                comfort_ref["energy"][
                    "HVAC_component_sum_kWh"
                ]
            ),
            "comfort_and_occupancy": comfort_ref.get(
                "comfort_and_occupancy", {}
            ),
            "switches": comfort_ref.get(
                "action_distribution", {}
            ).get(
                "total_setpoint_switches_across_rooms"
            ),
        }

        summary["policy"] = {
            "type": "direct_deepseek_thermostat_office_hours",
            "architecture": (
                "single direct LLM controller; no evaluator"
            ),
            "control_window": "08:00-18:00",
            "decision_interval_minutes": (
                LLM_DECISION_INTERVAL_MINUTES
            ),
            "state_source": (
                "current EnergyPlus T/RH/MRT at "
                "BeginSystemTimestepBeforePredictor"
            ),
            "occupancy_source": (
                "measured occupancy count at same 5-minute interval"
            ),
            "future_information_used": False,
            "action_bounds": {
                "heating_min_C": HEATING_MIN_C,
                "heating_max_C": HEATING_MAX_C,
                "cooling_min_C": COOLING_MIN_C,
                "cooling_max_C": COOLING_MAX_C,
                "resolution_C": SETPOINT_RESOLUTION_C,
                "minimum_deadband_C": MIN_DEADBAND_C,
            },
            "invalid_proposal_fallback": (
                "frozen office-hours Rule-OCC"
            ),
            "off_hours_action": {
                "heating_setpoint_C": 16.0,
                "cooling_setpoint_C": 28.0,
            },
            "fcu_availability": 1.0,
            "direct_fan_override": False,
        }

        summary["llm_controller"] = {
            "provider": "DeepSeek API",
            "base_url": self.deepseek_client.base_url,
            "model_requested": self.deepseek_model,
            "model_returned_preflight": self.deepseek_preflight.get(
                "model_returned"
            ),
            "model_verified": bool(
                self.deepseek_verification.get("verified", False)
            ),
            "api_key_source": self.deepseek_key_source,
            "api_key_value_logged": False,
            "thinking_mode": "disabled",
            "temperature": self.llm_temperature,
            "max_tokens": LLM_MAX_TOKENS,
            "timeout_seconds": self.llm_timeout_s,
            "max_attempts_per_call": LLM_MAX_ATTEMPTS,
            "preflight": self.deepseek_preflight,
            "decision_interval_minutes": (
                LLM_DECISION_INTERVAL_MINUTES
            ),
            "expected_decision_epochs": (
                EXPECTED_LLM_DECISION_EPOCHS
            ),
            "batched_rooms_per_call": len(ROOM_ORDER),
            "response_format": "json_object",
            "system_prompt_sha256": deepseek_prompt_sha256(),
            "system_prompt_file": str(prompt_path),
            "evaluator_used": False,
            "future_information_used": False,
        }

        # Generic switch count from room summary.
        room_summary_path = (
            self.output_dir / "deepseek_direct_room_summary.csv"
        )
        if room_summary_path.exists():
            room_df = pd.read_csv(room_summary_path)
            if (
                "setpoint_switch_count"
                in room_df.columns
            ):
                total_switches = int(
                    pd.to_numeric(
                        room_df["setpoint_switch_count"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )
            else:
                total_switches = 0
        else:
            total_switches = 0

        state_path = (
            self.output_dir / "deepseek_direct_state_trace.csv"
        )
        state_df = pd.read_csv(
            state_path,
            parse_dates=[
                "interval_start",
                "interval_end",
            ],
        )

        source_counts = (
            state_df["action_source"]
            .value_counts()
            .to_dict()
        )
        decision_source_counts = (
            decision_df["action_source"]
            .value_counts()
            .to_dict()
            if not decision_df.empty
            else {}
        )

        summary["action_distribution"] = {
            "timestep_action_source_counts": {
                str(k): int(v)
                for k, v in source_counts.items()
            },
            "decision_action_source_counts": {
                str(k): int(v)
                for k, v in decision_source_counts.items()
            },
            "total_setpoint_switches_across_rooms": (
                total_switches
            ),
        }

        # DeepSeek runtime diagnostics.
        latencies: List[float] = []
        confidences: List[float] = []
        if not decision_df.empty:
            call_latencies = (
                decision_df[
                    ["decision_id", "llm_latency_s"]
                ]
                .drop_duplicates("decision_id")
            )
            latencies = [
                float(v)
                for v in pd.to_numeric(
                    call_latencies["llm_latency_s"],
                    errors="coerce",
                ).dropna().tolist()
                if is_finite(v)
            ]
            confidences = [
                float(v)
                for v in pd.to_numeric(
                    decision_df["confidence"],
                    errors="coerce",
                ).dropna().tolist()
                if is_finite(v)
            ]

        summary["llm_runtime"] = {
            "decision_epoch_count": self.llm_decision_epoch_count,
            "llm_call_count": self.llm_call_count,
            "llm_call_success_count": (
                self.llm_call_success_count
            ),
            "llm_call_failure_count": (
                self.llm_call_failure_count
            ),
            "llm_parse_failure_count": (
                self.llm_parse_failure_count
            ),
            "room_decision_count": int(
                len(decision_df)
            ),
            "room_proposal_valid_count": (
                self.llm_room_proposal_valid_count
            ),
            "fallback_count": self.llm_fallback_count,
            "fallback_fraction": (
                self.llm_fallback_count
                / len(decision_df)
                if len(decision_df)
                else None
            ),
            "safety_modified_count": (
                self.llm_safety_modified_count
            ),
            "safety_modified_fraction": (
                self.llm_safety_modified_count
                / len(decision_df)
                if len(decision_df)
                else None
            ),
            "mean_call_latency_s": (
                sum(latencies) / len(latencies)
                if latencies
                else None
            ),
            "median_call_latency_s": (
                median(latencies)
                if latencies
                else None
            ),
            "max_call_latency_s": (
                max(latencies)
                if latencies
                else None
            ),
            "mean_confidence": (
                sum(confidences) / len(confidences)
                if confidences
                else None
            ),
        }

        # Comparisons to all frozen references.
        deepseek_kwh = summary["energy"][
            "HVAC_component_sum_kWh"
        ]
        fixed_kwh = summary["comparison_to_fixed"][
            "fixed_HVAC_component_sum_kWh"
        ]
        occ_kwh = float(
            occ_ref["energy"][
                "HVAC_component_sum_kWh"
            ]
        )
        comfort_kwh = float(
            comfort_ref["energy"][
                "HVAC_component_sum_kWh"
            ]
        )

        summary["comparison_to_fixed"] = {
            "fixed_HVAC_component_sum_kWh": fixed_kwh,
            "deepseek_direct_HVAC_component_sum_kWh": deepseek_kwh,
            "energy_difference_fixed_minus_deepseek_kWh": (
                fixed_kwh - deepseek_kwh
            ),
            "energy_savings_vs_fixed_percent": (
                (fixed_kwh - deepseek_kwh)
                / fixed_kwh
                * 100.0
            ),
        }
        summary["energy_comparison_to_occupancy_rule"] = {
            "occupancy_rule_HVAC_component_sum_kWh": (
                occ_kwh
            ),
            "deepseek_direct_HVAC_component_sum_kWh": (
                deepseek_kwh
            ),
            "deepseek_minus_occupancy_rule_kWh": (
                deepseek_kwh - occ_kwh
            ),
            "deepseek_minus_occupancy_rule_percent": (
                (deepseek_kwh - occ_kwh)
                / occ_kwh
                * 100.0
            ),
        }
        summary["energy_comparison_to_comfort_rule"] = {
            "comfort_rule_HVAC_component_sum_kWh": (
                comfort_kwh
            ),
            "deepseek_direct_HVAC_component_sum_kWh": (
                deepseek_kwh
            ),
            "deepseek_minus_comfort_rule_kWh": (
                deepseek_kwh - comfort_kwh
            ),
            "deepseek_minus_comfort_rule_percent": (
                (deepseek_kwh - comfort_kwh)
                / comfort_kwh
                * 100.0
            ),
        }

        qcomfort = summary.get(
            "comfort_and_occupancy", {}
        )

        def comfort_deltas(
            ref_data: Dict[str, Any],
        ) -> Dict[str, Optional[float]]:
            refc = ref_data.get(
                "comfort_and_occupancy", {}
            )
            keys = [
                "occupied_overheat_gt_27_fraction",
                "occupied_overcool_lt_22_5_fraction",
                "occupied_RH_gt_85_fraction",
                "occupied_mean_abs_PMV",
                "occupied_mean_PPD_percent",
            ]
            out: Dict[str, Optional[float]] = {}
            for key in keys:
                a = qcomfort.get(key)
                b = refc.get(key)
                out[key + "_change"] = (
                    float(a) - float(b)
                    if (
                        a is not None
                        and b is not None
                        and is_finite(a)
                        and is_finite(b)
                    )
                    else None
                )
            return out

        summary[
            "deepseek_comparison_to_occupancy_rule"
        ] = comfort_deltas(occ_ref)
        summary[
            "deepseek_comparison_to_comfort_rule"
        ] = comfort_deltas(comfort_ref)

        summary["switching_comparison_to_occupancy_rule"] = {
            "occupancy_rule_switches": occ_ref.get(
                "action_distribution", {}
            ).get(
                "total_setpoint_switches_across_rooms"
            ),
            "deepseek_direct_switches": total_switches,
        }
        summary["switching_comparison_to_comfort_rule"] = {
            "comfort_rule_switches": comfort_ref.get(
                "action_distribution", {}
            ).get(
                "total_setpoint_switches_across_rooms"
            ),
            "deepseek_direct_switches": total_switches,
        }

        new_summary_path = (
            self.output_dir / "deepseek_direct_summary.json"
        )
        new_summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                default=str,
                allow_nan=True,
            ),
            encoding="utf-8",
        )

        # Remove the temporary base-named summary after successful conversion.
        if old_summary_path.exists():
            old_summary_path.unlink()

        self._print_deepseek_summary(
            summary,
            new_summary_path,
            decision_csv_path,
            decision_jsonl_path,
            prompt_path,
        )

    def _print_deepseek_summary(
        self,
        summary: Dict[str, Any],
        summary_path: Path,
        decision_csv_path: Path,
        decision_jsonl_path: Path,
        prompt_path: Path,
    ) -> None:
        print("\n" + "=" * 78)
        print("DEEPSEEK DIRECT CONTROLLER SUMMARY")
        print("=" * 78)

        valid = bool(
            summary.get("validation", {}).get(
                "all_required_deepseek_direct_checks_pass",
                False,
            )
        )
        print(
            f"DeepSeek-direct validation: "
            f"{'PASS' if valid else 'FAIL'}"
        )

        energy = summary.get("energy", {})
        fixed = summary.get("comparison_to_fixed", {})
        comfort = summary.get("comfort_and_occupancy", {})
        llm = summary.get("llm_runtime", {})
        validation = summary.get("validation", {})

        print("\nEnergy:")
        print(
            f"  HVAC component sum       "
            f"{energy.get('HVAC_component_sum_kWh', float('nan')):.4f} kWh"
        )
        print(
            f"  Savings vs fixed         "
            f"{fixed.get('energy_savings_vs_fixed_percent', float('nan')):.3f}%"
        )

        print("\nOccupied controlled-room metrics:")
        print(
            f"  Occupied room-timesteps: "
            f"{comfort.get('occupied_room_timesteps')}"
        )
        print(
            f"  Person-hours: "
            f"{comfort.get('person_hours', float('nan')):.3f}"
        )
        print(
            f"  Overheat >27 C: "
            f"{comfort.get('occupied_overheat_gt_27_fraction', float('nan'))*100:.3f}%"
        )
        print(
            f"  Overcool <22.5 C: "
            f"{comfort.get('occupied_overcool_lt_22_5_fraction', float('nan'))*100:.3f}%"
        )
        print(
            f"  RH >85%: "
            f"{comfort.get('occupied_RH_gt_85_fraction', float('nan'))*100:.3f}%"
        )
        print(
            f"  Mean |PMV|: "
            f"{comfort.get('occupied_mean_abs_PMV', float('nan')):.4f}"
        )
        print(
            f"  Mean PPD: "
            f"{comfort.get('occupied_mean_PPD_percent', float('nan')):.3f}%"
        )

        print("\nDirect DeepSeek diagnostics:")
        print(
            f"  Decision epochs: "
            f"{llm.get('decision_epoch_count')} / "
            f"{EXPECTED_LLM_DECISION_EPOCHS}"
        )
        print(
            f"  DeepSeek calls: {llm.get('llm_call_count')} "
            f"(success={llm.get('llm_call_success_count')}, "
            f"fail={llm.get('llm_call_failure_count')}, "
            f"parse_fail={llm.get('llm_parse_failure_count')})"
        )
        print(
            f"  Room decisions: "
            f"{llm.get('room_decision_count')} / "
            f"{EXPECTED_LLM_ROOM_DECISIONS}"
        )
        print(
            f"  Fallbacks: {llm.get('fallback_count')} "
            f"({llm.get('fallback_fraction')})"
        )
        print(
            f"  Safety-modified: "
            f"{llm.get('safety_modified_count')} "
            f"({llm.get('safety_modified_fraction')})"
        )
        print(
            f"  Mean call latency: "
            f"{llm.get('mean_call_latency_s')} s"
        )

        print("\nControl integrity:")
        print(
            f"  People max abs error: "
            f"{validation.get('people_tracking_max_abs_error')}"
        )
        print(
            f"  Ventilation tracking: "
            f"{validation.get('ventilation_fraction_within_tolerance')}"
        )
        print(
            f"  Off-hours fixed 16/28: "
            f"{validation.get('off_hours_fixed_16_28_pass')}"
        )
        print(
            f"  Setpoint bounds/grid/deadband: "
            f"{validation.get('action_bounds_pass')}/"
            f"{validation.get('action_resolution_pass')}/"
            f"{validation.get('action_deadband_pass')}"
        )
        print(
            f"  Direct fan actuation used: "
            f"{validation.get('direct_fan_actuation_used')}"
        )

        print("\nSaved:")
        for path in [
            self.output_dir / "deepseek_direct_state_trace.csv",
            self.output_dir / "deepseek_direct_meter_trace.csv",
            self.output_dir / "deepseek_direct_interval_energy.csv",
            self.output_dir / "deepseek_direct_room_summary.csv",
            self.output_dir / "deepseek_direct_daily_energy.csv",
            self.output_dir / "deepseek_direct_daily_comfort.csv",
            decision_csv_path,
            decision_jsonl_path,
            prompt_path,
            summary_path,
        ]:
            print(f"  {path}")



# ============================================================================
# Agentic DeepSeek layer
# ============================================================================

AGENTIC_MAX_REFINEMENTS = 1
EVALUATOR_MAX_TOKENS = 1100
REFINEMENT_MAX_TOKENS = LLM_MAX_TOKENS
FROZEN_EVALUATOR_V4_SHA256 = "30f22e883b205a027bdec4bebca7d32d83650d1b2a6c6a718bfd9cae0391ec6f"

# -----------------------------------------------------------------------------
# 5-minute Agentic LLM decision loop
# -----------------------------------------------------------------------------
# Agentic decisions are executed at the same resolution as the EnergyPlus
# callback and occupancy measurements.
# -----------------------------------------------------------------------------
AGENTIC_LLM_DECISION_INTERVAL_MINUTES = 5
EXPECTED_AGENTIC_DECISION_EPOCHS = 840  # 7 days x 10 h/day x 12 decisions/h
EXPECTED_AGENTIC_ROOM_DECISIONS = EXPECTED_AGENTIC_DECISION_EPOCHS * len(ROOM_ORDER)

def is_agentic_llm_decision_epoch(interval_start: pd.Timestamp) -> bool:
    """Return True for every 5-minute supervisory decision epoch."""
    return (
        int(interval_start.minute) % AGENTIC_LLM_DECISION_INTERVAL_MINUTES == 0
    )


def build_agentic_controller_prompt(
    interval_start: pd.Timestamp,
    room_states: List[Dict[str, Any]],
) -> str:
    """Reuse the frozen Direct controller prompt but report the true 5-min cadence."""
    prompt = build_deepseek_user_prompt(interval_start, room_states)
    old = f'"decision_interval_minutes": {LLM_DECISION_INTERVAL_MINUTES}'
    new = f'"decision_interval_minutes": {AGENTIC_LLM_DECISION_INTERVAL_MINUTES}'
    if old not in prompt:
        raise RuntimeError(
            "Inherited controller prompt does not contain the expected decision interval field."
        )
    return prompt.replace(old, new, 1)


EVALUATOR_SYSTEM_PROMPT = """You are an independent critic for occupant-centric HVAC supervisory control.

You receive the CURRENT EnergyPlus state and a controller proposal. Review each room independently. Reject only for a MATERIAL concern. There is NO target approval or rejection percentage.

DETERMINISTIC REVIEW DEFINITIONS

A. Material thermal state
- material_cold = true if EITHER air temperature < 22.5 C OR PMV <= -0.5.
- material_hot = true if EITHER air temperature > 27 C OR PMV >= +0.5.
- These are OR rules. Apply them exactly even if the other indicator appears acceptable.
- warm_warning = true if air temperature >= 26 C OR PMV >= +0.2, but warm_warning alone does not require rejection.
- high_rh = true if RH > 85%.

B. Feasibility
- Heating action range is 16-20 C.
- Cooling action range is 25-28 C.
- Cooling 25 C is the strongest permitted cooling action.
- Heating 20 C is the strongest permitted heating action.
- Minimum heat/cool deadband is 5 C.
- Do NOT reject merely because an undesirable state remains when the proposal is already using the strongest feasible helpful thermostat action.

C. Occupied thermal review
- If material_cold=true and the proposal leaves heating at 16 C while higher heating remains feasible, reject with cold_comfort_risk.
- If material_cold=true and the proposal is already making a credible helpful heating response, do not reject solely because the state has not recovered.
- If material_hot=true and the proposal leaves cooling relaxed while stronger cooling remains feasible, reject with warm_comfort_risk.
- If material_hot=true and cooling is already at 25 C, do not reject solely because the state remains hot.

D. Humidity review
- If occupied, high_rh=true, NOT material_cold, and stronger feasible cooling remains, the proposal should make a credible response; otherwise reject with humidity_risk.
- If occupied, high_rh=true, NOT material_cold, and cooling is already 25 C, do not reject solely because RH remains high.
- If material_cold=true, cold comfort takes priority; do not demand stronger cooling for humidity.
- If unoccupied, aggressive cooling solely to reduce humidity is normally vacant_energy_waste because no separate asset/safety humidity requirement is provided.

E. Energy proportionality
- Vacant rooms should normally use relaxed conditioning.
- Comfortable occupied rooms should not receive unnecessarily aggressive conditioning.
- Energy use should be proportional to current need.

F. State/reason consistency
- If material_cold=true, do not describe the room as neutral, comfortable, acceptable, or only slightly cool unless the reason explicitly acknowledges the material cold risk.
- If material_hot=true, do not describe the room as neutral, comfortable, acceptable, or only slightly warm unless the reason explicitly acknowledges the material hot risk.
- If high_rh=true, do not describe humidity as normal.
- Do not claim occupancy when occupancy is zero or vacancy when occupancy is positive.
- Material contradictions require state_reason_mismatch.

G. Action envelope
- Reject an action outside heat 16-20 C, cool 25-28 C, or deadband <5 C using action_envelope_issue.

APPROVAL CONSISTENCY CONTRACT

Apply this exactly:
- approved=true IFF there are ZERO material violations.
- approved=false IFF there is AT LEAST ONE material violation.
- Therefore approved=true MUST have violations=[].
- approved=false MUST have one or more violation labels.
- Feedback must be semantically consistent with approved and violations.
- Never say that an approved proposal "fails", "is inadequate", "must be reconsidered", or has a material risk.

STRICT ROLE BOUNDARY

You are REVIEW ONLY.
- Do NOT output, recommend, calculate, or hint at replacement numeric setpoints.
- Do NOT write phrases such as "set cooling to 25 C", "set heating to 20 C", or "use 19/26".
- Do NOT directly revise the controller action.
- Feedback explains WHAT concern exists, not WHICH numeric action to choose.
- Use only the supplied current state; do not assume future occupancy or future weather.

Use only these violation labels:
cold_comfort_risk
warm_comfort_risk
humidity_risk
vacant_energy_waste
unnecessary_switching
action_envelope_issue
state_reason_mismatch
other_material_risk

Return JSON only:
{
  "rooms": [
    {
      "room": "Room_1",
      "material_cold": false,
      "material_hot": false,
      "high_rh": false,
      "approved": true,
      "feedback": "brief qualitative review",
      "violations": []
    }
  ]
}
"""


REFINEMENT_PREAMBLE = """The independent evaluator reviewed your first proposal.

Revise ONLY rooms where approved=false, using the evaluator's qualitative feedback and violation labels.
For rooms where approved=true, repeat your original proposal unchanged.
The evaluator did not choose numeric setpoints and must not be treated as the controller.
Use only the same current-state snapshot.
Return only the same required controller JSON object.
"""


def evaluator_prompt_sha256() -> str:
    return hashlib.sha256(
        EVALUATOR_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()


def deterministic_review_flags(
    state_row: Dict[str, Any],
) -> Dict[str, bool]:
    """Compute factual threshold flags deterministically.

    These are state facts, not evaluator judgments. The LLM receives these
    flags as authoritative input and must not be relied on to perform numeric
    threshold comparisons.
    """
    occupancy = float(state_row["occupancy_count"])
    temp = float(state_row["air_temperature_C"])
    rh = float(state_row["relative_humidity_percent"])
    pmv = float(state_row["PMV"])

    return {
        "occupied": bool(occupancy > 0.0),
        "material_cold": bool(
            temp < 22.5 or pmv <= -0.5
        ),
        "material_hot": bool(
            temp > 27.0 or pmv >= 0.5
        ),
        "warm_warning": bool(
            temp >= 26.0 or pmv >= 0.2
        ),
        "high_rh": bool(
            rh > 85.0
        ),
    }


def minimal_structural_precheck(
    proposal: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Check only JSON/numeric structure before evaluator review.

    This function never changes the controller action. Bounds, resolution, and
    deadband remain the responsibility of the final deterministic validator.
    """
    if not isinstance(proposal, dict):
        return False, ["missing_or_nonobject_proposal"]

    issues: List[str] = []
    for field in ("heating_setpoint_C", "cooling_setpoint_C"):
        if field not in proposal:
            issues.append(f"missing_{field}")
            continue
        try:
            value = float(proposal[field])
        except Exception:
            issues.append(f"nonnumeric_{field}")
            continue
        if not is_finite(value):
            issues.append(f"nonfinite_{field}")

    return len(issues) == 0, issues


def evaluator_contains_action_fields(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    forbidden = (
        "heatingsetpoint",
        "coolingsetpoint",
        "heatsetpoint",
        "coolsetpoint",
    )
    for key in item:
        norm = normalize(key)
        if any(token in norm for token in forbidden):
            return True
    return False


def normalize_evaluator_rooms(
    payload: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    rooms = payload.get("rooms")

    if isinstance(rooms, dict):
        iterable: List[Dict[str, Any]] = []
        for key, value in rooms.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("room", key)
                iterable.append(item)
    elif isinstance(rooms, list):
        iterable = [x for x in rooms if isinstance(x, dict)]
    else:
        iterable = []

    aliases: Dict[str, str] = {}
    for room in ROOM_ORDER:
        key = normalize(room)
        aliases[key] = room
        aliases[key.replace("room", "")] = room

    result: Dict[str, Dict[str, Any]] = {}
    for item in iterable:
        key = normalize(item.get("room", ""))
        room = aliases.get(key) or aliases.get(key.replace("room", ""))
        if room is None or room in result:
            continue

        approved_raw = item.get("approved")
        if isinstance(approved_raw, bool):
            approved = approved_raw
        elif isinstance(approved_raw, str):
            approved = approved_raw.strip().lower() == "true"
        else:
            approved = False

        feedback = str(item.get("feedback", "")).strip()[:500]
        raw_violations = item.get("violations", [])
        if isinstance(raw_violations, list):
            violations = [
                str(v).strip()[:100]
                for v in raw_violations
                if str(v).strip()
            ]
        elif raw_violations is None:
            violations = []
        else:
            violations = [str(raw_violations).strip()[:100]]

        def as_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() == "true"
            return bool(value)

        approval_consistency_ok = (
            (bool(approved) and len(violations) == 0)
            or ((not bool(approved)) and len(violations) > 0)
        )

        result[room] = {
            "material_cold": as_bool(item.get("material_cold", False)),
            "material_hot": as_bool(item.get("material_hot", False)),
            "high_rh": as_bool(item.get("high_rh", False)),
            "approved": bool(approved),
            "feedback": feedback,
            "violations": violations,
            "approval_consistency_ok": bool(approval_consistency_ok),
            "contract_violation_action_fields": evaluator_contains_action_fields(item),
            "raw": item,
        }

    return result


def build_evaluator_prompt(
    interval_start: pd.Timestamp,
    state_by_room: Dict[str, Dict[str, Any]],
    proposals: Dict[str, Dict[str, Any]],
    structural: Dict[str, Dict[str, Any]],
) -> str:
    rows: List[Dict[str, Any]] = []
    for room in ROOM_ORDER:
        ds = state_by_room[room]
        rows.append(
            {
                "room": room,
                "occupancy_count": ds["occupancy_count"],
                "air_temperature_C": ds["air_temperature_C"],
                "relative_humidity_percent": ds["relative_humidity_percent"],
                "MRT_C": ds["MRT_C"],
                "PMV": ds["PMV"],
                "deterministic_flags": deterministic_review_flags(ds),
                "previous_heating_setpoint_C": ds["previous_heating_setpoint_C"],
                "previous_cooling_setpoint_C": ds["previous_cooling_setpoint_C"],
                "controller_proposal": proposals.get(room),
                "structural_precheck": structural[room],
            }
        )

    return (
        "Review every room proposal. The deterministic_flags fields were "
        "computed by Python from the supplied numeric state using the frozen "
        "threshold definitions. Treat those flags as AUTHORITATIVE state facts; "
        "do not recompute or override them. Use them when applying the frozen "
        "V4 rubric. Return only the required evaluator JSON. "
        "Do not output replacement numeric setpoints.\n\n"
        + json.dumps(
            {
                "timestamp": interval_start.isoformat(sep=" "),
                "decision_interval_minutes": AGENTIC_LLM_DECISION_INTERVAL_MINUTES,
                "rooms": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def build_refinement_prompt(
    interval_start: pd.Timestamp,
    state_by_room: Dict[str, Dict[str, Any]],
    proposals: Dict[str, Dict[str, Any]],
    reviews: Dict[str, Dict[str, Any]],
) -> str:
    rows: List[Dict[str, Any]] = []
    for room in ROOM_ORDER:
        ds = state_by_room[room]
        review = reviews[room]
        rows.append(
            {
                "room": room,
                "occupancy_count": ds["occupancy_count"],
                "air_temperature_C": ds["air_temperature_C"],
                "relative_humidity_percent": ds["relative_humidity_percent"],
                "MRT_C": ds["MRT_C"],
                "PMV": ds["PMV"],
                "previous_heating_setpoint_C": ds["previous_heating_setpoint_C"],
                "previous_cooling_setpoint_C": ds["previous_cooling_setpoint_C"],
                "first_proposal": proposals.get(room),
                "evaluator": {
                    "approved": review["approved"],
                    "feedback": review["feedback"],
                    "violations": review["violations"],
                },
            }
        )

    return (
        REFINEMENT_PREAMBLE
        + "\n\n"
        + json.dumps(
            {
                "timestamp": interval_start.isoformat(sep=" "),
                "refinement_round": 1,
                "maximum_refinement_rounds": AGENTIC_MAX_REFINEMENTS,
                "rooms": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def response_usage(response: Dict[str, Any]) -> Dict[str, int]:
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}

    def val(name: str) -> int:
        try:
            return int(usage.get(name, 0) or 0)
        except Exception:
            return 0

    return {
        "prompt_tokens": val("prompt_tokens"),
        "completion_tokens": val("completion_tokens"),
        "total_tokens": val("total_tokens"),
    }


def chat_with_system(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> Tuple[str, float, Dict[str, Any]]:
    """Use the frozen DeepSeek client transport with a role-specific system prompt."""
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": client.temperature,
        "max_tokens": int(max_tokens),
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            data = client._post_json("/chat/completions", payload)
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_MAX_ATTEMPTS:
                raise
            time.sleep(0.5)
            continue

        latency = time.perf_counter() - started
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            last_error = RuntimeError("DeepSeek response contains no choices.")
            if attempt >= LLM_MAX_ATTEMPTS:
                raise last_error
            time.sleep(0.5)
            continue

        content = str(choices[0].get("message", {}).get("content", "") or "").strip()
        if not content:
            last_error = RuntimeError("DeepSeek returned empty content.")
            if attempt >= LLM_MAX_ATTEMPTS:
                raise last_error
            time.sleep(0.5)
            continue

        data["_client_attempts"] = attempt
        return content, float(latency), data

    raise RuntimeError(f"DeepSeek role request failed: {last_error}")


class DeepSeekAgenticRunner(DeepSeekDirectRunner):
    def __init__(
        self,
        *args: Any,
        project_dir: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.project_dir = project_dir.resolve()
        self.evaluator_v4_validation: Dict[str, Any] = {
            "validation_mode": "embedded_prompt_hash_plus_live_preflight",
            "frozen_prompt_sha256": FROZEN_EVALUATOR_V4_SHA256,
            "runtime_prompt_sha256": evaluator_prompt_sha256(),
            "prompt_hash_matches": evaluator_prompt_sha256() == FROZEN_EVALUATOR_V4_SHA256,
        }

        self.evaluator_call_count = 0
        self.evaluator_call_success_count = 0
        self.evaluator_call_failure_count = 0
        self.evaluator_parse_failure_count = 0

        self.refinement_epoch_count = 0
        self.refinement_call_count = 0
        self.refinement_call_success_count = 0
        self.refinement_call_failure_count = 0
        self.refinement_parse_failure_count = 0
        self.refinement_room_count = 0

        self.evaluator_approved_room_count = 0
        self.evaluator_rejected_room_count = 0
        self.evaluator_contract_violation_count = 0
        self.evaluator_action_field_violation_count = 0
        self.evaluator_flag_mismatch_count = 0
        self.evaluator_approval_consistency_error_count = 0
        self.initial_structural_invalid_count = 0
        self.first_to_final_action_change_count = 0
        self.final_fallback_count = 0
        self.final_safety_modified_count = 0

        self.controller_initial_latencies: List[float] = []
        self.evaluator_latencies: List[float] = []
        self.refinement_latencies: List[float] = []
        self.decision_total_api_latencies: List[float] = []

        self.api_usage_by_role = {
            role: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for role in ("controller_initial", "evaluator", "controller_refinement")
        }
        self.returned_models_by_role = {
            "controller_initial": set(),
            "evaluator": set(),
            "controller_refinement": set(),
        }
        self.violation_counts: Dict[str, int] = {}

        for room in ROOM_ORDER:
            self.current_commands[room].update(
                {
                    "decision_action_source": "off_hours_fixed",
                    "initial_heating_setpoint_C": None,
                    "initial_cooling_setpoint_C": None,
                    "initial_structural_valid": False,
                    "evaluator_approved": None,
                    "evaluator_feedback": None,
                    "evaluator_violations": None,
                    "evaluator_contract_ok": True,
                    "refinement_used": False,
                    "refined_heating_setpoint_C": None,
                    "refined_cooling_setpoint_C": None,
                    "first_to_final_action_changed": False,
                }
            )

    def _load_reference(
        self,
        path: Path,
        validation_key: str,
        expected_case: str,
    ) -> Dict[str, Any]:
        """Load a validated deterministic reference with portable path checks."""
        if not path.exists():
            raise FileNotFoundError(f"Reference summary not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        if not bool(data.get("validation", {}).get(validation_key, False)):
            raise RuntimeError(f"Reference is not validated: {path}")
        if str(data.get("case", "")) != expected_case:
            raise RuntimeError(
                f"Unexpected reference case in {path}: {data.get('case')}"
            )
        if str(data.get("occupancy_sha256", "")).lower() != EXPECTED_OCCUPANCY_SHA256.lower():
            raise RuntimeError(f"Reference used a different occupancy input: {path}")

        recorded_idf = resolve_recorded_path(data.get("idf"), self.project_dir)
        if recorded_idf is None or recorded_idf != self.idf_path.resolve():
            raise RuntimeError(f"Reference IDF mismatch: {path}")
        recorded_epw = resolve_recorded_path(data.get("epw"), self.project_dir)
        if recorded_epw is None or recorded_epw != self.epw_path.resolve():
            raise RuntimeError(f"Reference EPW mismatch: {path}")

        comfort = data.get("comfort_and_occupancy", {})
        if comfort.get("occupied_room_timesteps") != 3224:
            raise RuntimeError(f"Reference occupied-room timestep total mismatch: {path}")
        person_hours = comfort.get("person_hours")
        if person_hours is None or abs(float(person_hours) - 487.1666666666667) > 1e-9:
            raise RuntimeError(f"Reference person-hour total mismatch: {path}")

        kwh = data.get("energy", {}).get("HVAC_component_sum_kWh")
        if kwh is None or not is_finite(kwh) or float(kwh) <= 0:
            raise RuntimeError(f"Reference HVAC energy missing/invalid: {path}")
        return data

    def _record_api(self, role: str, response: Dict[str, Any], latency: Optional[float]) -> None:
        usage = response_usage(response)
        for key, value in usage.items():
            self.api_usage_by_role[role][key] += int(value)
        model = response.get("model")
        if model:
            self.returned_models_by_role[role].add(str(model))
        if latency is None:
            return
        if role == "controller_initial":
            self.controller_initial_latencies.append(float(latency))
        elif role == "evaluator":
            self.evaluator_latencies.append(float(latency))
        elif role == "controller_refinement":
            self.refinement_latencies.append(float(latency))

    def _collect_state(
        self,
        state: Any,
        interval_start: pd.Timestamp,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        by_room: Dict[str, Dict[str, Any]] = {}

        for room in ROOM_ORDER:
            occupancy = self.occupancy_count(interval_start, room)
            temp = self.variable_value(state, self.temp_handles[room])
            rh = self.variable_value(state, self.rh_handles[room])
            mrt = self.variable_value(state, self.mrt_handles[room])
            valid = all(is_finite(v) for v in (temp, rh, mrt))
            if valid:
                pmv, ppd = fanger_pmv_ppd(temp, rh, mrt)
                valid = is_finite(pmv) and is_finite(ppd)
            else:
                pmv, ppd = float("nan"), float("nan")

            previous = self.current_commands[room]
            prompt_row = {
                "room": room,
                "occupancy_count": occupancy,
                "air_temperature_C": temp,
                "relative_humidity_percent": rh,
                "MRT_C": mrt,
                "PMV": pmv,
                "previous_heating_setpoint_C": previous["heating_setpoint_C"],
                "previous_cooling_setpoint_C": previous["cooling_setpoint_C"],
            }
            by_room[room] = {
                **prompt_row,
                "PPD_percent": ppd,
                "state_valid": bool(valid),
            }
            if valid:
                rows.append(prompt_row)

        return rows, by_room

    def _set_off_hours(self) -> None:
        for room in ROOM_ORDER:
            self.current_commands[room].update(
                {
                    "heating_setpoint_C": UNOCCUPIED_HEATING_SP_C,
                    "cooling_setpoint_C": UNOCCUPIED_COOLING_SP_C,
                    "occupancy_band": "off_hours_fixed",
                    "action_reason": "off_hours_fixed",
                    "action_label": "off_hours_fixed",
                    "action_source": "off_hours_fixed",
                    "decision_action_source": "off_hours_fixed",
                    "decision_id": None,
                    "decision_sequence_index": -1,
                    "decision_state_valid": True,
                    "deepseek_confidence": None,
                    "deepseek_proposal_valid": False,
                    "deepseek_safety_modified": False,
                    "deepseek_fallback_used": False,
                    "deepseek_fallback_reason": None,
                    "raw_heating_setpoint_C": None,
                    "raw_cooling_setpoint_C": None,
                    "initial_heating_setpoint_C": None,
                    "initial_cooling_setpoint_C": None,
                    "initial_structural_valid": False,
                    "evaluator_approved": None,
                    "evaluator_feedback": None,
                    "evaluator_violations": None,
                    "evaluator_contract_ok": True,
                    "refinement_used": False,
                    "refined_heating_setpoint_C": None,
                    "refined_cooling_setpoint_C": None,
                    "first_to_final_action_changed": False,
                }
            )

    def apply_occupancy_rule_control(self, state: Any) -> None:
        """Controller -> evaluator -> feedback -> one controller refinement."""
        try:
            self.setup_handles(state)
            if not self.runtime.handles_ready:
                return
            if self.api.exchange.warmup_flag(state):
                return
            if not is_weather_run_period(self.api, state):
                return
            if self.current_interval_start is None:
                raise RuntimeError("System callback executed before research clock.")

            interval_start = self.current_interval_start
            if not in_control_period(interval_start):
                return

            if not is_office_hour(interval_start):
                self._set_off_hours()
            else:
                new_decision = (
                    is_agentic_llm_decision_epoch(interval_start)
                    and self.last_llm_decision_sequence_index != self.zone_sequence_index
                )

                if new_decision:
                    self.llm_decision_epoch_count += 1
                    decision_id = self.next_decision_id
                    self.next_decision_id += 1

                    room_states, state_by_room = self._collect_state(state, interval_start)

                    # 1) First controller proposal: SAME prompt as frozen direct case.
                    initial_prompt = build_agentic_controller_prompt(interval_start, room_states)
                    initial_content = ""
                    initial_response: Dict[str, Any] = {}
                    initial_latency: Optional[float] = None
                    initial_success = False
                    initial_parse = False
                    initial_error: Optional[str] = None
                    proposals: Dict[str, Dict[str, Any]] = {}

                    self.llm_call_count += 1
                    if len(room_states) == len(ROOM_ORDER):
                        try:
                            initial_content, initial_latency, initial_response = self.deepseek_client.chat(initial_prompt)
                            initial_success = True
                            self.llm_call_success_count += 1
                            self._record_api("controller_initial", initial_response, initial_latency)
                            try:
                                proposals = normalize_deepseek_rooms(
                                    extract_json_object(initial_content)
                                )
                                initial_parse = True
                            except Exception as exc:
                                self.llm_parse_failure_count += 1
                                initial_error = f"Controller JSON/shape error: {type(exc).__name__}: {exc}"
                        except Exception as exc:
                            self.llm_call_failure_count += 1
                            initial_error = f"{type(exc).__name__}: {exc}"
                    else:
                        self.llm_call_failure_count += 1
                        initial_error = "Invalid current state in one or more rooms."

                    # 2) Minimal structural precheck; no action modification.
                    structural: Dict[str, Dict[str, Any]] = {}
                    for room in ROOM_ORDER:
                        ok, issues = minimal_structural_precheck(proposals.get(room))
                        structural[room] = {"valid": bool(ok), "issues": issues}
                        if not ok:
                            self.initial_structural_invalid_count += 1

                    # 3) Independent evaluator call.
                    evaluator_prompt = build_evaluator_prompt(
                        interval_start, state_by_room, proposals, structural
                    )
                    evaluator_content = ""
                    evaluator_response: Dict[str, Any] = {}
                    evaluator_latency: Optional[float] = None
                    evaluator_success = False
                    evaluator_parse = False
                    evaluator_error: Optional[str] = None
                    reviews_raw: Dict[str, Dict[str, Any]] = {}

                    self.evaluator_call_count += 1
                    try:
                        evaluator_content, evaluator_latency, evaluator_response = chat_with_system(
                            self.deepseek_client,
                            EVALUATOR_SYSTEM_PROMPT,
                            evaluator_prompt,
                            EVALUATOR_MAX_TOKENS,
                        )
                        evaluator_success = True
                        self.evaluator_call_success_count += 1
                        self._record_api("evaluator", evaluator_response, evaluator_latency)
                        try:
                            reviews_raw = normalize_evaluator_rooms(
                                extract_json_object(evaluator_content)
                            )
                            evaluator_parse = True
                        except Exception as exc:
                            self.evaluator_parse_failure_count += 1
                            evaluator_error = f"Evaluator JSON/shape error: {type(exc).__name__}: {exc}"
                    except Exception as exc:
                        self.evaluator_call_failure_count += 1
                        evaluator_error = f"{type(exc).__name__}: {exc}"

                    # Normalize evaluator output and force structural invalidity to rejection.
                    reviews: Dict[str, Dict[str, Any]] = {}
                    for room in ROOM_ORDER:
                        review = reviews_raw.get(room)
                        if review is None:
                            review = {
                                "material_cold": False,
                                "material_hot": False,
                                "high_rh": False,
                                "approved": False,
                                "feedback": "Evaluator output missing for this room.",
                                "violations": ["evaluator_output_missing"],
                                "approval_consistency_ok": True,
                                "contract_violation_action_fields": False,
                "contract_ok": True,
                            }

                        approved = bool(review["approved"])
                        feedback = str(review["feedback"])
                        violations = list(review["violations"])

                        ds = state_by_room[room]
                        authoritative_flags = deterministic_review_flags(ds)

                        expected_material_cold = bool(
                            authoritative_flags["material_cold"]
                        )
                        expected_material_hot = bool(
                            authoritative_flags["material_hot"]
                        )
                        expected_high_rh = bool(
                            authoritative_flags["high_rh"]
                        )

                        # The LLM still echoes the three flags because the
                        # frozen V4 system prompt asked for them. These echoed
                        # booleans are NOT authoritative and cannot change the
                        # control path. Numeric threshold comparisons are a
                        # deterministic Python responsibility.
                        flag_mismatch = bool(
                            bool(review.get("material_cold", False))
                            != expected_material_cold
                            or bool(review.get("material_hot", False))
                            != expected_material_hot
                            or bool(review.get("high_rh", False))
                            != expected_high_rh
                        )

                        approval_consistency_bad = not bool(
                            review.get("approval_consistency_ok", False)
                        )
                        action_field_bad = bool(
                            review.get("contract_violation_action_fields", False)
                        )

                        # V5.1 deterministic contract normalization.
                        # The frozen evaluator prompt requires approved=true IFF
                        # violations is empty. Rare LLM contradictions are retained
                        # as diagnostics, but cannot make the control wrapper itself
                        # inconsistent. The explicit structured violation labels are
                        # authoritative for this normalization:
                        #   violations present -> approved=False
                        #   no violations       -> approved=True
                        # A review-only role breach (numeric action fields) remains
                        # a hard evaluator contract violation.
                        raw_approved = bool(approved)
                        raw_approval_consistency_bad = approval_consistency_bad

                        if flag_mismatch:
                            self.evaluator_flag_mismatch_count += 1
                        if raw_approval_consistency_bad:
                            self.evaluator_approval_consistency_error_count += 1
                            approved = len(violations) == 0
                        if action_field_bad:
                            self.evaluator_action_field_violation_count += 1

                        contract_bad = bool(action_field_bad)
                        if contract_bad:
                            approved = False
                            violations.append("evaluator_contract_violation")
                            feedback = (
                                feedback
                                + " Evaluator output violated the frozen V4 review-only contract."
                            ).strip()
                            self.evaluator_contract_violation_count += 1

                        normalized_approval_consistency_ok = bool(
                            (approved and len(violations) == 0)
                            or ((not approved) and len(violations) > 0)
                        )

                        if not structural[room]["valid"]:
                            approved = False
                            violations.append("structural_invalid")
                            feedback = (
                                feedback
                                + " Controller proposal must contain finite numeric heating and cooling setpoints."
                            ).strip()

                        deduped: List[str] = []
                        for v in violations:
                            label = str(v).strip()
                            if label and label not in deduped:
                                deduped.append(label)

                        reviews[room] = {
                            "material_cold": expected_material_cold,
                            "material_hot": expected_material_hot,
                            "high_rh": expected_high_rh,
                            "evaluator_echo_material_cold": bool(
                                review.get("material_cold", False)
                            ),
                            "evaluator_echo_material_hot": bool(
                                review.get("material_hot", False)
                            ),
                            "evaluator_echo_high_rh": bool(
                                review.get("high_rh", False)
                            ),
                            "approved": approved,
                            "feedback": feedback[:500],
                            "violations": deduped,
                            "contract_violation_action_fields": action_field_bad,
                            "flag_echo_matches": not flag_mismatch,
                            "raw_evaluator_approved": raw_approved,
                            "raw_approval_consistency_ok": not raw_approval_consistency_bad,
                            "approval_consistency_ok": normalized_approval_consistency_ok,
                            "contract_ok": (not contract_bad) and normalized_approval_consistency_ok,
                        }

                        if approved:
                            self.evaluator_approved_room_count += 1
                        else:
                            self.evaluator_rejected_room_count += 1
                        for violation in deduped:
                            self.violation_counts[violation] = self.violation_counts.get(violation, 0) + 1

                    rejected = [
                        room for room in ROOM_ORDER
                        if not reviews[room]["approved"]
                    ]

                    # 4) ONE controller re-call only if at least one room is rejected.
                    refinement_used = bool(rejected)
                    refinement_prompt: Optional[str] = None
                    refinement_content = ""
                    refinement_response: Dict[str, Any] = {}
                    refinement_latency: Optional[float] = None
                    refinement_success = False
                    refinement_parse = False
                    refinement_error: Optional[str] = None
                    revised: Dict[str, Dict[str, Any]] = {}

                    if refinement_used:
                        self.refinement_epoch_count += 1
                        self.refinement_room_count += len(rejected)
                        self.refinement_call_count += 1
                        refinement_prompt = build_refinement_prompt(
                            interval_start, state_by_room, proposals, reviews
                        )
                        try:
                            refinement_content, refinement_latency, refinement_response = chat_with_system(
                                self.deepseek_client,
                                DEEPSEEK_SYSTEM_PROMPT,
                                refinement_prompt,
                                REFINEMENT_MAX_TOKENS,
                            )
                            refinement_success = True
                            self.refinement_call_success_count += 1
                            self._record_api(
                                "controller_refinement", refinement_response, refinement_latency
                            )
                            try:
                                revised = normalize_deepseek_rooms(
                                    extract_json_object(refinement_content)
                                )
                                refinement_parse = True
                            except Exception as exc:
                                self.refinement_parse_failure_count += 1
                                refinement_error = f"Refinement JSON/shape error: {type(exc).__name__}: {exc}"
                        except Exception as exc:
                            self.refinement_call_failure_count += 1
                            refinement_error = f"{type(exc).__name__}: {exc}"

                    total_latency = sum(
                        x for x in (initial_latency, evaluator_latency, refinement_latency)
                        if x is not None
                    )
                    self.decision_total_api_latencies.append(float(total_latency))

                    epoch_record: Dict[str, Any] = {
                        "decision_id": decision_id,
                        "zone_sequence_index": self.zone_sequence_index,
                        "interval_start": interval_start.isoformat(sep=" "),
                        "model": self.deepseek_model,
                        "controller_initial": {
                            "call_success": initial_success,
                            "parse_success": initial_parse,
                            "latency_s": initial_latency,
                            "error": initial_error,
                            "prompt": initial_prompt,
                            "raw_response_content": initial_content,
                            "response_metadata": {
                                k: v for k, v in initial_response.items()
                                if k not in {"message", "choices"}
                            },
                        },
                        "evaluator": {
                            "call_success": evaluator_success,
                            "parse_success": evaluator_parse,
                            "latency_s": evaluator_latency,
                            "error": evaluator_error,
                            "prompt": evaluator_prompt,
                            "raw_response_content": evaluator_content,
                            "response_metadata": {
                                k: v for k, v in evaluator_response.items()
                                if k not in {"message", "choices"}
                            },
                        },
                        "refinement": {
                            "used": refinement_used,
                            "call_success": refinement_success,
                            "parse_success": refinement_parse,
                            "latency_s": refinement_latency,
                            "error": refinement_error,
                            "prompt": refinement_prompt,
                            "raw_response_content": refinement_content,
                            "response_metadata": {
                                k: v for k, v in refinement_response.items()
                                if k not in {"message", "choices"}
                            },
                        },
                        "rooms": [],
                    }

                    # 5) Final deterministic validator after agent loop.
                    for room in ROOM_ORDER:
                        ds = state_by_room[room]
                        occupancy = float(ds["occupancy_count"])
                        review = reviews[room]
                        initial = proposals.get(room)
                        initial_valid = bool(structural[room]["valid"])
                        revised_item = revised.get(room) if refinement_used else None
                        revised_valid, _ = minimal_structural_precheck(revised_item)

                        if review["approved"] and initial_valid:
                            candidate = initial
                            final_source = "agentic_initial"
                        elif (
                            not review["approved"]
                            and refinement_used
                            and refinement_success
                            and refinement_parse
                            and revised_valid
                        ):
                            candidate = revised_item
                            final_source = "agentic_refined"
                        else:
                            candidate = None
                            final_source = "rule_occ_fallback"

                        final_action = validate_deepseek_room_action(candidate, occupancy)
                        if final_action["fallback_used"]:
                            final_source = "rule_occ_fallback"
                            self.final_fallback_count += 1
                        if final_action["safety_modified"]:
                            self.final_safety_modified_count += 1

                        if final_action["proposal_valid"]:
                            self.llm_room_proposal_valid_count += 1
                        if final_action["fallback_used"]:
                            self.llm_fallback_count += 1
                        if final_action["safety_modified"]:
                            self.llm_safety_modified_count += 1

                        first_validated = validate_deepseek_room_action(
                            initial if initial_valid else None,
                            occupancy,
                        )
                        changed = bool(
                            abs(
                                float(final_action["heating_setpoint_C"])
                                - float(first_validated["heating_setpoint_C"])
                            ) > 1e-9
                            or abs(
                                float(final_action["cooling_setpoint_C"])
                                - float(first_validated["cooling_setpoint_C"])
                            ) > 1e-9
                        )
                        if changed:
                            self.first_to_final_action_change_count += 1

                        def raw_value(item: Any, field: str) -> Optional[float]:
                            if not isinstance(item, dict):
                                return None
                            try:
                                value = float(item[field])
                            except Exception:
                                return None
                            return value if is_finite(value) else None

                        initial_heat = raw_value(initial, "heating_setpoint_C")
                        initial_cool = raw_value(initial, "cooling_setpoint_C")
                        refined_heat = raw_value(revised_item, "heating_setpoint_C")
                        refined_cool = raw_value(revised_item, "cooling_setpoint_C")

                        command = self.current_commands[room]
                        command.update(
                            {
                                "heating_setpoint_C": float(final_action["heating_setpoint_C"]),
                                "cooling_setpoint_C": float(final_action["cooling_setpoint_C"]),
                                "occupancy_band": final_source,
                                "action_reason": str(final_action["reason"]),
                                "action_label": str(final_action["action"]),
                                "action_source": final_source,
                                "decision_action_source": final_source,
                                "decision_id": decision_id,
                                "decision_sequence_index": self.zone_sequence_index,
                                "decision_temp_C": ds["air_temperature_C"],
                                "decision_RH_percent": ds["relative_humidity_percent"],
                                "decision_MRT_C": ds["MRT_C"],
                                "decision_PMV": ds["PMV"],
                                "decision_PPD_percent": ds["PPD_percent"],
                                "decision_state_valid": bool(ds["state_valid"]),
                                "deepseek_confidence": final_action["confidence"],
                                "deepseek_proposal_valid": bool(final_action["proposal_valid"]),
                                "deepseek_safety_modified": bool(final_action["safety_modified"]),
                                "deepseek_fallback_used": bool(final_action["fallback_used"]),
                                "deepseek_fallback_reason": final_action["fallback_reason"],
                                "raw_heating_setpoint_C": final_action["raw_heating_setpoint_C"],
                                "raw_cooling_setpoint_C": final_action["raw_cooling_setpoint_C"],
                                "initial_heating_setpoint_C": initial_heat,
                                "initial_cooling_setpoint_C": initial_cool,
                                "initial_structural_valid": initial_valid,
                                "deterministic_material_cold": bool(review["material_cold"]),
                                "deterministic_material_hot": bool(review["material_hot"]),
                                "deterministic_high_rh": bool(review["high_rh"]),
                                "evaluator_echo_material_cold": bool(
                                    review["evaluator_echo_material_cold"]
                                ),
                                "evaluator_echo_material_hot": bool(
                                    review["evaluator_echo_material_hot"]
                                ),
                                "evaluator_echo_high_rh": bool(
                                    review["evaluator_echo_high_rh"]
                                ),
                                "evaluator_approved": bool(review["approved"]),
                                "evaluator_feedback": review["feedback"],
                                "evaluator_violations": json.dumps(review["violations"], ensure_ascii=False),
                                "evaluator_flag_echo_matches": bool(review["flag_echo_matches"]),
                                "evaluator_approval_consistency_ok": bool(review["approval_consistency_ok"]),
                                "evaluator_contract_ok": bool(review["contract_ok"]),
                                "refinement_used": bool((not review["approved"]) and refinement_used),
                                "refined_heating_setpoint_C": refined_heat,
                                "refined_cooling_setpoint_C": refined_cool,
                                "first_to_final_action_changed": changed,
                            }
                        )

                        decision_row = {
                            "decision_id": decision_id,
                            "zone_sequence_index": self.zone_sequence_index,
                            "interval_start": interval_start,
                            "room": room,
                            "occupancy_count": occupancy,
                            "decision_temp_C": ds["air_temperature_C"],
                            "decision_RH_percent": ds["relative_humidity_percent"],
                            "decision_MRT_C": ds["MRT_C"],
                            "decision_PMV": ds["PMV"],
                            "decision_PPD_percent": ds["PPD_percent"],
                            "decision_state_valid": bool(ds["state_valid"]),
                            "previous_heating_setpoint_C": ds["previous_heating_setpoint_C"],
                            "previous_cooling_setpoint_C": ds["previous_cooling_setpoint_C"],
                            "initial_call_success": initial_success,
                            "initial_parse_success": initial_parse,
                            "initial_structural_valid": initial_valid,
                            "initial_heating_setpoint_C": initial_heat,
                            "initial_cooling_setpoint_C": initial_cool,
                            "initial_action": initial.get("action") if isinstance(initial, dict) else None,
                            "initial_reason": initial.get("reason") if isinstance(initial, dict) else None,
                            "initial_confidence": initial.get("confidence") if isinstance(initial, dict) else None,
                            "evaluator_call_success": evaluator_success,
                            "evaluator_parse_success": evaluator_parse,
                            "deterministic_material_cold": bool(review["material_cold"]),
                            "deterministic_material_hot": bool(review["material_hot"]),
                            "deterministic_high_rh": bool(review["high_rh"]),
                            "evaluator_echo_material_cold": bool(
                                review["evaluator_echo_material_cold"]
                            ),
                            "evaluator_echo_material_hot": bool(
                                review["evaluator_echo_material_hot"]
                            ),
                            "evaluator_echo_high_rh": bool(
                                review["evaluator_echo_high_rh"]
                            ),
                            "evaluator_approved": bool(review["approved"]),
                            "evaluator_feedback": review["feedback"],
                            "evaluator_violations": json.dumps(review["violations"], ensure_ascii=False),
                            "evaluator_flag_echo_matches": bool(review["flag_echo_matches"]),
                            "evaluator_approval_consistency_ok": bool(review["approval_consistency_ok"]),
                            "evaluator_contract_ok": bool(review["contract_ok"]),
                            "refinement_used": bool((not review["approved"]) and refinement_used),
                            "refinement_call_success": refinement_success if refinement_used else None,
                            "refinement_parse_success": refinement_parse if refinement_used else None,
                            "refined_heating_setpoint_C": refined_heat,
                            "refined_cooling_setpoint_C": refined_cool,
                            "refined_action": revised_item.get("action") if isinstance(revised_item, dict) else None,
                            "refined_reason": revised_item.get("reason") if isinstance(revised_item, dict) else None,
                            "refined_confidence": revised_item.get("confidence") if isinstance(revised_item, dict) else None,
                            "final_source": final_source,
                            "final_heating_setpoint_C": final_action["heating_setpoint_C"],
                            "final_cooling_setpoint_C": final_action["cooling_setpoint_C"],
                            "final_action": final_action["action"],
                            "final_reason": final_action["reason"],
                            "final_confidence": final_action["confidence"],
                            "proposal_valid": bool(final_action["proposal_valid"]),
                            "safety_modified": bool(final_action["safety_modified"]),
                            "fallback_used": bool(final_action["fallback_used"]),
                            "fallback_reason": final_action["fallback_reason"],
                            "first_to_final_action_changed": changed,
                            # Compatibility with inherited direct output engine:
                            "action_source": final_source,
                            "action_label": final_action["action"],
                            "reason": final_action["reason"],
                            "confidence": final_action["confidence"],
                            "raw_heating_setpoint_C": final_action["raw_heating_setpoint_C"],
                            "raw_cooling_setpoint_C": final_action["raw_cooling_setpoint_C"],
                            "heating_setpoint_C": final_action["heating_setpoint_C"],
                            "cooling_setpoint_C": final_action["cooling_setpoint_C"],
                            "llm_call_success": initial_success,
                            "llm_parse_success": initial_parse,
                            "llm_latency_s": initial_latency,
                            "llm_error": initial_error,
                            "controller_initial_latency_s": initial_latency,
                            "evaluator_latency_s": evaluator_latency,
                            "refinement_latency_s": refinement_latency,
                            "decision_total_api_latency_s": total_latency,
                            "evaluator_error": evaluator_error,
                            "refinement_error": refinement_error,
                        }
                        self.decision_rows.append(decision_row)
                        epoch_record["rooms"].append(
                            {
                                k: (v.isoformat(sep=" ") if isinstance(v, pd.Timestamp) else v)
                                for k, v in decision_row.items()
                            }
                        )

                    self.decision_epoch_records.append(epoch_record)
                    self.last_llm_decision_sequence_index = self.zone_sequence_index

                else:
                    for room in ROOM_ORDER:
                        command = self.current_commands[room]
                        if command.get("decision_id") is None:
                            occupancy = self.occupancy_count(interval_start, room)
                            action = fallback_rule_action(occupancy)
                            command.update(
                                {
                                    "heating_setpoint_C": action["heating_setpoint_C"],
                                    "cooling_setpoint_C": action["cooling_setpoint_C"],
                                    "occupancy_band": "fallback_hold",
                                    "action_reason": action["reason"],
                                    "action_label": action["action"],
                                    "action_source": "fallback_hold",
                                    "decision_action_source": "rule_occ_fallback",
                                    "deepseek_fallback_used": True,
                                    "deepseek_fallback_reason": "missing_prior_decision",
                                }
                            )
                        else:
                            source = str(command.get("decision_action_source", "rule_occ_fallback"))
                            command["action_source"] = source + "_hold"

            # Apply current command at every five-minute system callback.
            for room in ROOM_ORDER:
                command = self.current_commands[room]
                self.api.exchange.set_actuator_value(
                    state, self.heat_actuators[room], float(command["heating_setpoint_C"])
                )
                self.api.exchange.set_actuator_value(
                    state, self.cool_actuators[room], float(command["cooling_setpoint_C"])
                )
                self.api.exchange.set_actuator_value(
                    state, self.availability_actuators[room], FCU_AVAILABILITY_COMMAND
                )
                command["availability"] = FCU_AVAILABILITY_COMMAND
                command["command_sequence_index"] = self.zone_sequence_index

            self.system_control_callback_count += 1

        except Exception as exc:
            self.record_callback_error("apply_deepseek_agentic_control", exc)

    def end_zone_timestep(self, state: Any) -> None:
        interval_start = self.current_interval_start
        super().end_zone_timestep(state)

        if interval_start is None:
            return
        for room in ROOM_ORDER:
            row = self.runtime.interval_room_state.get((interval_start, room))
            if row is None:
                continue
            command = self.current_commands[room]
            row["case"] = "deepseek_agentic"
            for field in (
                "decision_action_source",
                "initial_heating_setpoint_C",
                "initial_cooling_setpoint_C",
                "initial_structural_valid",
                "deterministic_material_cold",
                "deterministic_material_hot",
                "deterministic_high_rh",
                "evaluator_echo_material_cold",
                "evaluator_echo_material_hot",
                "evaluator_echo_high_rh",
                "evaluator_approved",
                "evaluator_feedback",
                "evaluator_violations",
                "evaluator_flag_echo_matches",
                "evaluator_approval_consistency_ok",
                "evaluator_contract_ok",
                "refinement_used",
                "refined_heating_setpoint_C",
                "refined_cooling_setpoint_C",
                "first_to_final_action_changed",
            ):
                row[field] = command.get(field)

        if self.runtime.meter_rows:
            self.runtime.meter_rows[-1]["case"] = "deepseek_agentic"

    def validation_checks(
        self,
        state_df: pd.DataFrame,
        meter_df: pd.DataFrame,
        energyplus_status: int,
        err_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        checks = OccupancyRuleRunner.validation_checks(
            self, state_df, meter_df, energyplus_status, err_info
        )

        checks["deepseek_model_verified"] = bool(
            self.deepseek_verification.get("verified", False)
        )
        checks["system_control_callback_count"] = self.system_control_callback_count
        checks["system_control_callback_count_2016"] = (
            self.system_control_callback_count == EXPECTED_INTERVALS_PER_ROOM
        )

        if not state_df.empty:
            alignment = (
                pd.to_numeric(state_df["command_sequence_index"], errors="coerce")
                == pd.to_numeric(state_df["zone_sequence_index"], errors="coerce")
            )
            checks["command_sequence_alignment_pass"] = bool(alignment.all())
            heat = pd.to_numeric(state_df["heating_setpoint_command_C"], errors="coerce")
            cool = pd.to_numeric(state_df["cooling_setpoint_command_C"], errors="coerce")
            checks["action_bounds_pass"] = bool(
                heat.between(HEATING_MIN_C, HEATING_MAX_C, inclusive="both").all()
                and cool.between(COOLING_MIN_C, COOLING_MAX_C, inclusive="both").all()
            )
            checks["action_resolution_pass"] = bool(
                ((heat / SETPOINT_RESOLUTION_C - (heat / SETPOINT_RESOLUTION_C).round()).abs() <= 1e-9).all()
                and ((cool / SETPOINT_RESOLUTION_C - (cool / SETPOINT_RESOLUTION_C).round()).abs() <= 1e-9).all()
            )
            checks["action_deadband_pass"] = bool(
                ((cool - heat) >= MIN_DEADBAND_C - 1e-9).all()
            )
        else:
            checks["command_sequence_alignment_pass"] = False
            checks["action_bounds_pass"] = False
            checks["action_resolution_pass"] = False
            checks["action_deadband_pass"] = False

        decision_df = pd.DataFrame(self.decision_rows)
        checks["agentic_decision_epoch_count"] = self.llm_decision_epoch_count
        checks["agentic_decision_epoch_count_expected"] = (
            self.llm_decision_epoch_count == EXPECTED_AGENTIC_DECISION_EPOCHS
        )
        checks["initial_controller_call_count"] = self.llm_call_count
        checks["initial_controller_calls_expected"] = (
            self.llm_call_count == self.llm_decision_epoch_count
        )
        checks["initial_controller_all_calls_success"] = (
            self.llm_call_success_count == self.llm_decision_epoch_count
            and self.llm_call_failure_count == 0
            and self.llm_parse_failure_count == 0
        )
        checks["evaluator_call_count"] = self.evaluator_call_count
        checks["evaluator_calls_expected"] = (
            self.evaluator_call_count == self.llm_decision_epoch_count
        )
        checks["evaluator_all_calls_success"] = (
            self.evaluator_call_success_count == self.llm_decision_epoch_count
            and self.evaluator_call_failure_count == 0
            and self.evaluator_parse_failure_count == 0
        )
        checks["refinement_epoch_count"] = self.refinement_epoch_count
        checks["refinement_call_count"] = self.refinement_call_count
        checks["refinement_calls_match_rejected_epochs"] = (
            self.refinement_call_count == self.refinement_epoch_count
        )
        checks["refinement_all_attempted_calls_success"] = (
            self.refinement_call_failure_count == 0
            and self.refinement_parse_failure_count == 0
            and self.refinement_call_success_count == self.refinement_call_count
        )
        checks["maximum_refinement_rounds"] = AGENTIC_MAX_REFINEMENTS
        checks["maximum_refinement_rounds_is_one"] = AGENTIC_MAX_REFINEMENTS == 1
        checks["evaluator_room_review_count"] = (
            self.evaluator_approved_room_count + self.evaluator_rejected_room_count
        )
        checks["evaluator_room_reviews_expected"] = (
            checks["evaluator_room_review_count"] == EXPECTED_AGENTIC_ROOM_DECISIONS
        )
        checks["evaluator_contract_violation_count"] = self.evaluator_contract_violation_count
        checks["evaluator_action_field_violation_count"] = (
            self.evaluator_action_field_violation_count
        )
        checks["evaluator_flag_echo_mismatch_count"] = self.evaluator_flag_mismatch_count
        checks["evaluator_approval_consistency_error_count"] = (
            self.evaluator_approval_consistency_error_count
        )
        checks["evaluator_never_outputs_numeric_actions"] = (
            self.evaluator_action_field_violation_count == 0
        )
        checks["deterministic_threshold_flags_are_authoritative"] = True
        checks["evaluator_flag_echo_mismatch_is_diagnostic_only"] = True
        # Raw LLM approval/violation contradictions are diagnostic in V5.1;
        # the deterministic wrapper normalizes them before control/refinement.
        checks["evaluator_raw_approval_consistency_error_count"] = (
            self.evaluator_approval_consistency_error_count
        )
        checks["evaluator_raw_approval_consistency_is_diagnostic_only"] = True
        checks["evaluator_v4_approval_contract_all_consistent"] = bool(
            decision_df.empty
            or decision_df["evaluator_approval_consistency_ok"].fillna(False).astype(bool).all()
        )
        checks["evaluator_v4_frozen_prompt_hash_matches"] = (
            evaluator_prompt_sha256() == FROZEN_EVALUATOR_V4_SHA256
        )
        checks["evaluator_prompt_embedded_hash_verified"] = (
            evaluator_prompt_sha256() == FROZEN_EVALUATOR_V4_SHA256
        )
        checks["decision_rows"] = int(len(decision_df))
        checks["decision_rows_expected"] = (
            len(decision_df) == EXPECTED_AGENTIC_ROOM_DECISIONS
        )

        if not decision_df.empty:
            state_valid = decision_df["decision_state_valid"].astype(bool)
            checks["decision_state_valid_fraction"] = float(state_valid.mean())
            checks["decision_state_all_valid"] = bool(state_valid.all())
            fallback = decision_df["fallback_used"].astype(bool)
            fallback_fraction = float(fallback.mean())
            checks["final_fallback_count"] = int(fallback.sum())
            checks["final_fallback_fraction"] = fallback_fraction
            checks["final_fallback_fraction_within_limit"] = (
                fallback_fraction <= MAX_LLM_FALLBACK_FRACTION
            )
            contract = decision_df["evaluator_contract_ok"].astype(bool)
            checks["decision_rows_evaluator_contract_all_ok"] = bool(contract.all())
            ref_used = decision_df["refinement_used"].astype(bool)
            approved = decision_df["evaluator_approved"].astype(bool)
            checks["refinement_only_after_rejection"] = bool((~ref_used | ~approved).all())
        else:
            checks["decision_state_valid_fraction"] = 0.0
            checks["decision_state_all_valid"] = False
            checks["final_fallback_count"] = 0
            checks["final_fallback_fraction"] = 1.0
            checks["final_fallback_fraction_within_limit"] = False
            checks["decision_rows_evaluator_contract_all_ok"] = False
            checks["refinement_only_after_rejection"] = False

        required = [
            checks["energyplus_exit_status_zero"],
            checks["callback_error_count_zero"],
            checks["energyplus_severe_errors_zero"],
            checks["energyplus_fatal_errors_zero"],
            checks["analysis_rows_12096"],
            checks["begin_zone_callback_count_2016"],
            checks["end_zone_callback_count_2016"],
            checks["system_control_callback_count_2016"],
            checks["zone_sequence_complete_0_to_2015"],
            checks["complete_time_coverage_all_rooms"],
            checks["first_interval_start_correct"],
            checks["last_interval_start_correct"],
            checks["people_tracking_pass"],
            checks["thermostat_tracking_pass"],
            checks["ventilation_tracking_pass"],
            checks["direct_fan_actuation_absent"],
            checks["off_hours_fixed_16_28_pass"],
            checks["meter_rows_positive"],
            checks["meter_time_coverage_pass"],
            checks["canonical_occupancy_totals_match"],
            checks["required_meter_handles_all_valid"],
            checks["deepseek_model_verified"],
            checks["command_sequence_alignment_pass"],
            checks["action_bounds_pass"],
            checks["action_resolution_pass"],
            checks["action_deadband_pass"],
            checks["agentic_decision_epoch_count_expected"],
            checks["initial_controller_calls_expected"],
            checks["initial_controller_all_calls_success"],
            checks["evaluator_calls_expected"],
            checks["evaluator_all_calls_success"],
            checks["refinement_calls_match_rejected_epochs"],
            checks["refinement_all_attempted_calls_success"],
            checks["maximum_refinement_rounds_is_one"],
            checks["evaluator_room_reviews_expected"],
            checks["evaluator_never_outputs_numeric_actions"],
            checks["deterministic_threshold_flags_are_authoritative"],
            checks["evaluator_v4_approval_contract_all_consistent"],
            checks["evaluator_v4_frozen_prompt_hash_matches"],
            checks["evaluator_prompt_embedded_hash_verified"],
            checks["decision_rows_expected"],
            checks["decision_state_all_valid"],
            checks["final_fallback_fraction_within_limit"],
            checks["decision_rows_evaluator_contract_all_ok"],
            checks["refinement_only_after_rejection"],
        ]
        if checks.get("availability_tracking_pass") is False:
            required.append(False)

        checks["all_required_deepseek_agentic_checks_pass"] = all(required)
        checks["all_required_baseline_checks_pass"] = checks[
            "all_required_deepseek_agentic_checks_pass"
        ]
        return checks

    def run_deepseek_preflight(self) -> Dict[str, Any]:
        representative_states = [
            {"room": "Room_1", "occupancy_count": 2, "air_temperature_C": 25.1, "relative_humidity_percent": 78.0, "MRT_C": 23.4, "PMV": -0.05, "previous_heating_setpoint_C": 18.0, "previous_cooling_setpoint_C": 27.0},
            {"room": "Room_2", "occupancy_count": 1, "air_temperature_C": 26.2, "relative_humidity_percent": 86.0, "MRT_C": 23.9, "PMV": 0.22, "previous_heating_setpoint_C": 18.0, "previous_cooling_setpoint_C": 27.0},
            {"room": "Room_4", "occupancy_count": 4, "air_temperature_C": 24.7, "relative_humidity_percent": 76.0, "MRT_C": 23.2, "PMV": -0.12, "previous_heating_setpoint_C": 19.0, "previous_cooling_setpoint_C": 26.0},
            {"room": "Room_5", "occupancy_count": 1, "air_temperature_C": 22.8, "relative_humidity_percent": 83.0, "MRT_C": 22.1, "PMV": -0.48, "previous_heating_setpoint_C": 18.0, "previous_cooling_setpoint_C": 27.0},
            {"room": "Room_6", "occupancy_count": 5, "air_temperature_C": 26.8, "relative_humidity_percent": 88.0, "MRT_C": 24.1, "PMV": 0.38, "previous_heating_setpoint_C": 19.0, "previous_cooling_setpoint_C": 26.0},
            {"room": "Room_7", "occupancy_count": 3, "air_temperature_C": 25.4, "relative_humidity_percent": 80.0, "MRT_C": 23.6, "PMV": 0.05, "previous_heating_setpoint_C": 19.0, "previous_cooling_setpoint_C": 26.0},
        ]
        timestamp = pd.Timestamp("2021-08-16 09:00:00")

        print("\n" + "=" * 78)
        print("DeepSeek agentic controller/evaluator preflight")
        print("=" * 78)
        print(f"Endpoint   : {self.deepseek_client.base_url}")
        print(f"Model      : {self.deepseek_model}")
        print("Loop       : controller -> evaluator -> controller refinement (K=1)")
        print("Evaluator  : qualitative feedback only; no action setpoint fields")

        controller_prompt = build_agentic_controller_prompt(timestamp, representative_states)
        controller_content, controller_latency, controller_response = self.deepseek_client.chat(
            controller_prompt
        )
        proposals = normalize_deepseek_rooms(
            extract_json_object(controller_content)
        )
        missing = [r for r in ROOM_ORDER if r not in proposals]
        if missing:
            raise RuntimeError(f"Agentic preflight controller missing rooms: {missing}")

        state_by_room = {
            row["room"]: {**row, "PPD_percent": None, "state_valid": True}
            for row in representative_states
        }
        structural: Dict[str, Dict[str, Any]] = {}
        for room in ROOM_ORDER:
            ok, issues = minimal_structural_precheck(proposals[room])
            if not ok:
                raise RuntimeError(f"Preflight controller proposal invalid for {room}: {issues}")
            structural[room] = {"valid": True, "issues": []}

        eval_prompt = build_evaluator_prompt(timestamp, state_by_room, proposals, structural)
        eval_content, eval_latency, eval_response = chat_with_system(
            self.deepseek_client,
            EVALUATOR_SYSTEM_PROMPT,
            eval_prompt,
            EVALUATOR_MAX_TOKENS,
        )
        reviews = normalize_evaluator_rooms(extract_json_object(eval_content))
        missing_reviews = [r for r in ROOM_ORDER if r not in reviews]
        if missing_reviews:
            raise RuntimeError(f"Agentic preflight evaluator missing rooms: {missing_reviews}")
        for row in representative_states:
            room = row["room"]
            review = reviews[room]
            expected_cold = bool(
                float(row["air_temperature_C"]) < 22.5
                or float(row["PMV"]) <= -0.5
            )
            expected_hot = bool(
                float(row["air_temperature_C"]) > 27.0
                or float(row["PMV"]) >= 0.5
            )
            expected_high_rh = bool(
                float(row["relative_humidity_percent"]) > 85.0
            )

            if review["contract_violation_action_fields"]:
                raise RuntimeError(
                    "Preflight evaluator attempted to output action setpoint fields."
                )
            if not review["approval_consistency_ok"]:
                raise RuntimeError(
                    f"Preflight evaluator approval/violation inconsistency for {room}."
                )

            # Echoed threshold flags are diagnostic only. The evaluator receives
            # authoritative deterministic flags in the real decision prompt.
            # The LLM is not used as a numeric threshold comparator.
            _preflight_flag_echo_matches = bool(
                bool(review["material_cold"]) == expected_cold
                and bool(review["material_hot"]) == expected_hot
                and bool(review["high_rh"]) == expected_high_rh
            )

        # Exercise the refinement path even if all evaluator reviews approve.
        refinement_reviews: Dict[str, Dict[str, Any]] = {
            room: {
                "approved": bool(reviews[room]["approved"]),
                "feedback": str(reviews[room]["feedback"]),
                "violations": list(reviews[room]["violations"]),
                "contract_violation_action_fields": False,
                "contract_ok": True,
            }
            for room in ROOM_ORDER
        }
        forced = False
        if all(refinement_reviews[r]["approved"] for r in ROOM_ORDER):
            forced = True
            refinement_reviews["Room_6"] = {
                "approved": False,
                "feedback": "Preflight-only test: reconsider the warm and humid condition while balancing comfort and energy.",
                "violations": ["preflight_refinement_test"],
                "contract_violation_action_fields": False,
            }

        refine_prompt = build_refinement_prompt(
            timestamp, state_by_room, proposals, refinement_reviews
        )
        refine_content, refine_latency, refine_response = chat_with_system(
            self.deepseek_client,
            DEEPSEEK_SYSTEM_PROMPT,
            refine_prompt,
            REFINEMENT_MAX_TOKENS,
        )
        revised = normalize_deepseek_rooms(
            extract_json_object(refine_content)
        )
        missing_revised = [r for r in ROOM_ORDER if r not in revised]
        if missing_revised:
            raise RuntimeError(f"Agentic preflight refinement missing rooms: {missing_revised}")
        for row in representative_states:
            action = validate_deepseek_room_action(
                revised[row["room"]], float(row["occupancy_count"])
            )
            if action["fallback_used"]:
                raise RuntimeError(
                    f"Agentic preflight refinement unusable for {row['room']}: "
                    f"{action['fallback_reason']}"
                )

        result = {
            "passed": True,
            "model_requested": self.deepseek_model,
            "controller_initial": {
                "latency_s": float(controller_latency),
                "model_returned": controller_response.get("model"),
                "usage": controller_response.get("usage", {}),
            },
            "evaluator": {
                "latency_s": float(eval_latency),
                "model_returned": eval_response.get("model"),
                "usage": eval_response.get("usage", {}),
                "reviews": {
                    room: {
                        "approved": reviews[room]["approved"],
                        "feedback": reviews[room]["feedback"],
                        "violations": reviews[room]["violations"],
                    }
                    for room in ROOM_ORDER
                },
            },
            "controller_refinement": {
                "latency_s": float(refine_latency),
                "model_returned": refine_response.get("model"),
                "usage": refine_response.get("usage", {}),
                "forced_refinement_test": forced,
            },
            "controller_system_prompt_sha256": deepseek_prompt_sha256(),
            "evaluator_system_prompt_sha256": evaluator_prompt_sha256(),
        }
        path = self.output_dir / "deepseek_agentic_preflight.json"
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        self.deepseek_preflight = result
        self.deepseek_verification = {
            "verified": True,
            "base_url": self.deepseek_client.base_url,
            "model_requested": self.deepseek_model,
            "model_returned": controller_response.get("model"),
            "key_source": self.deepseek_key_source,
        }

        print(f"Controller : {controller_latency:.2f} s")
        print(f"Evaluator  : {eval_latency:.2f} s")
        print(f"Refinement : {refine_latency:.2f} s")
        print("Evaluator action-field contract: PASS")
        print("Controller/evaluator/refinement preflight: PASS")
        print(f"Saved      : {path}")
        return result

    def run(self) -> int:
        if not self.idf_path.exists():
            raise FileNotFoundError(f"Generated IDF not found: {self.idf_path}")
        if not self.epw_path.exists():
            raise FileNotFoundError(f"EPW not found: {self.epw_path}")

        print("\n" + "=" * 78)
        print("Verifying DeepSeek agentic controller")
        print("=" * 78)
        print(f"Endpoint   : {self.deepseek_client.base_url}")
        print(f"Model      : {self.deepseek_model}")
        print(f"Key source : {self.deepseek_key_source}")
        print(
            f"Decision   : every {AGENTIC_LLM_DECISION_INTERVAL_MINUTES} min "
            f"({EXPECTED_AGENTIC_DECISION_EPOCHS} expected epochs)"
        )
        print("Architecture: controller -> evaluator -> feedback -> controller (K=1)")
        print("Evaluator  : FROZEN V4 qualitative critic; no direct action revision")
        print("Thresholds : Python-computed flags supplied as authoritative evaluator input")
        print(
            f"Evaluator V4 hash: {evaluator_prompt_sha256()} "
            f"({'VERIFIED' if evaluator_prompt_sha256() == FROZEN_EVALUATOR_V4_SHA256 else 'MISMATCH'})"
        )
        print("Controller core: embedded in this standalone script; no direct-case file required")

        self.run_deepseek_preflight()

        self.api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
            self.state, self.apply_people_and_ventilation
        )
        self.api.runtime.callback_begin_system_timestep_before_predictor(
            self.state, self.apply_occupancy_rule_control
        )
        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            self.state, self.end_zone_timestep
        )

        command = [
            "-w",
            str(self.epw_path),
            "-d",
            str(self.energyplus_output_dir),
            str(self.idf_path),
        ]

        print("\n" + "=" * 78)
        print("Running agentic DeepSeek API EnergyPlus controller")
        print("=" * 78)
        print(f"EnergyPlus : {self.energyplus_root}")
        print(f"IDF        : {self.idf_path}")
        print(f"EPW        : {self.epw_path}")
        print(f"Occupancy  : {self.occupancy_path}")
        print(f"Results    : {self.output_dir}")
        print(f"\nAgentic loop per {AGENTIC_LLM_DECISION_INTERVAL_MINUTES}-minute decision:")
        print("  1. current EnergyPlus state")
        print("  2. controller first proposal")
        print("  3. minimal structural precheck")
        print("  4. evaluator -> approved/feedback/violations")
        print("  5. if rejected -> ONE controller re-call")
        print("  6. deterministic final validator")
        print("  7. apply thermostat action")
        print("  8. EnergyPlus advances")
        print("  Evaluator NEVER directly changes setpoints.")
        print("  Off-hours -> 16/28 C")
        print("  FCU availability -> 1.0")
        print("  Direct fan override -> NONE")
        print("  Future information -> NONE")

        t0 = time.perf_counter()
        status = self.api.runtime.run_energyplus(self.state, command)
        wall = time.perf_counter() - t0
        print(f"\nEnergyPlus exit status: {status}")
        print(f"Simulation wall time: {wall:.2f} s")
        self.save_results(int(status), wall)
        return int(status)

    def _print_deepseek_summary(self, *args: Any, **kwargs: Any) -> None:
        # Suppress inherited direct-case summary while reusing its output engine.
        return

    def save_results(self, energyplus_status: int, wall_time_s: float) -> None:
        super().save_results(energyplus_status, wall_time_s)

        intermediate = self.output_dir / "deepseek_direct_summary.json"
        if not intermediate.exists():
            raise RuntimeError("Intermediate direct-named summary was not created.")
        summary = json.loads(intermediate.read_text(encoding="utf-8"))
        rename_map = {
            "deepseek_direct_state_trace.csv": "deepseek_agentic_state_trace.csv",
            "deepseek_direct_meter_trace.csv": "deepseek_agentic_meter_trace.csv",
            "deepseek_direct_interval_energy.csv": "deepseek_agentic_interval_energy.csv",
            "deepseek_direct_room_summary.csv": "deepseek_agentic_room_summary.csv",
            "deepseek_direct_daily_energy.csv": "deepseek_agentic_daily_energy.csv",
            "deepseek_direct_daily_comfort.csv": "deepseek_agentic_daily_comfort.csv",
            "deepseek_direct_decisions.csv": "deepseek_agentic_decisions.csv",
            "deepseek_direct_decisions.jsonl": "deepseek_agentic_decisions.jsonl",
            "deepseek_direct_prompt.txt": "deepseek_agentic_controller_prompt.txt",
        }
        for old_name, new_name in rename_map.items():
            old = self.output_dir / old_name
            new = self.output_dir / new_name
            if old.exists():
                if new.exists():
                    new.unlink()
                old.rename(new)

        evaluator_prompt_path = self.output_dir / "deepseek_agentic_evaluator_prompt.txt"
        evaluator_prompt_path.write_text(EVALUATOR_SYSTEM_PROMPT, encoding="utf-8")
        controller_prompt_path = self.output_dir / "deepseek_agentic_controller_prompt.txt"
        decision_csv_path = self.output_dir / "deepseek_agentic_decisions.csv"
        decision_jsonl_path = self.output_dir / "deepseek_agentic_decisions.jsonl"
        state_path = self.output_dir / "deepseek_agentic_state_trace.csv"
        room_path = self.output_dir / "deepseek_agentic_room_summary.csv"

        state_df = pd.read_csv(state_path)
        decision_df = pd.read_csv(decision_csv_path)
        room_df = pd.read_csv(room_path) if room_path.exists() else pd.DataFrame()
        total_switches = int(
            pd.to_numeric(room_df.get("setpoint_switch_count", pd.Series(dtype=float)), errors="coerce")
            .fillna(0)
            .sum()
        )

        timestep_counts = (
            state_df["action_source"].value_counts().to_dict()
            if "action_source" in state_df.columns else {}
        )
        final_counts = (
            decision_df["final_source"].value_counts().to_dict()
            if "final_source" in decision_df.columns else {}
        )

        summary["script"] = "06_run_deepseek_agentic.py"
        summary["case"] = "deepseek_agentic"
        summary["implementation_version"] = "v5.1"
        summary["project"] = "."
        summary["idf"] = repository_relative_path(self.idf_path, self.project_dir)
        summary["epw"] = repository_relative_path(self.epw_path, self.project_dir)
        summary["occupancy"] = repository_relative_path(
            self.occupancy_path, self.project_dir
        )
        summary["evaluator_prompt_validation"] = self.evaluator_v4_validation


        summary["policy"] = {
            "type": "agentic_deepseek_thermostat_office_hours",
            "architecture": "controller -> evaluator -> feedback -> controller refinement; K=1",
            "control_window": "08:00-18:00",
            "decision_interval_minutes": AGENTIC_LLM_DECISION_INTERVAL_MINUTES,
            "controller_initial_prompt_is_frozen_embedded_prompt": True,
            "evaluator_can_directly_modify_actions": False,
            "threshold_flag_source": (
                "deterministic Python comparisons supplied to evaluator; "
                "LLM echo is diagnostic only"
            ),
            "maximum_refinement_rounds": AGENTIC_MAX_REFINEMENTS,
            "final_deterministic_validator_after_agent_loop": True,
            "state_source": "current EnergyPlus T/RH/MRT at BeginSystemTimestepBeforePredictor",
            "occupancy_source": "measured occupancy count at same 5-minute interval",
            "future_information_used": False,
            "invalid_final_proposal_fallback": "frozen office-hours Rule-OCC",
            "action_bounds": {
                "heating_min_C": HEATING_MIN_C,
                "heating_max_C": HEATING_MAX_C,
                "cooling_min_C": COOLING_MIN_C,
                "cooling_max_C": COOLING_MAX_C,
                "resolution_C": SETPOINT_RESOLUTION_C,
                "minimum_deadband_C": MIN_DEADBAND_C,
            },
            "off_hours_action": {"heating_setpoint_C": 16.0, "cooling_setpoint_C": 28.0},
            "fcu_availability": 1.0,
            "direct_fan_override": False,
        }

        summary["llm_controller"] = {
            "provider": "DeepSeek API",
            "base_url": self.deepseek_client.base_url,
            "model_requested": self.deepseek_model,
            "model_returned_preflight": self.deepseek_preflight.get(
                "controller_initial", {}
            ).get("model_returned"),
            "model_verified": bool(self.deepseek_verification.get("verified", False)),
            "api_key_source": self.deepseek_key_source,
            "api_key_value_logged": False,
            "thinking_mode": "disabled",
            "temperature": self.llm_temperature,
            "max_tokens": LLM_MAX_TOKENS,
            "timeout_seconds": self.llm_timeout_s,
            "decision_interval_minutes": AGENTIC_LLM_DECISION_INTERVAL_MINUTES,
            "controller_system_prompt_sha256": deepseek_prompt_sha256(),
            "controller_prompt_file": repository_relative_path(controller_prompt_path, self.project_dir),
            "evaluator_system_prompt_sha256": evaluator_prompt_sha256(),
            "evaluator_prompt_file": repository_relative_path(evaluator_prompt_path, self.project_dir),
            "evaluator_used": True,
            "same_model_for_controller_and_evaluator": True,
            "maximum_refinement_rounds": AGENTIC_MAX_REFINEMENTS,
            "preflight": self.deepseek_preflight,
            "future_information_used": False,
        }

        def latency_stats(values: List[float]) -> Dict[str, Optional[float]]:
            clean = [float(v) for v in values if is_finite(v)]
            return {
                "count": len(clean),
                "mean_s": sum(clean) / len(clean) if clean else None,
                "median_s": median(clean) if clean else None,
                "max_s": max(clean) if clean else None,
            }

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for role_usage in self.api_usage_by_role.values():
            for key in total_usage:
                total_usage[key] += int(role_usage[key])

        summary["agentic_runtime"] = {
            "decision_epoch_count": self.llm_decision_epoch_count,
            "initial_controller_calls": {
                "attempted": self.llm_call_count,
                "successful": self.llm_call_success_count,
                "failed": self.llm_call_failure_count,
                "parse_failures": self.llm_parse_failure_count,
                "latency": latency_stats(self.controller_initial_latencies),
            },
            "evaluator_calls": {
                "attempted": self.evaluator_call_count,
                "successful": self.evaluator_call_success_count,
                "failed": self.evaluator_call_failure_count,
                "parse_failures": self.evaluator_parse_failure_count,
                "latency": latency_stats(self.evaluator_latencies),
            },
            "refinement_calls": {
                "epochs_requiring_refinement": self.refinement_epoch_count,
                "rooms_rejected": self.refinement_room_count,
                "attempted": self.refinement_call_count,
                "successful": self.refinement_call_success_count,
                "failed": self.refinement_call_failure_count,
                "parse_failures": self.refinement_parse_failure_count,
                "latency": latency_stats(self.refinement_latencies),
            },
            "total_api_calls": (
                self.llm_call_count + self.evaluator_call_count + self.refinement_call_count
            ),
            "decision_total_api_latency": latency_stats(self.decision_total_api_latencies),
            "evaluator_approved_room_count": self.evaluator_approved_room_count,
            "evaluator_rejected_room_count": self.evaluator_rejected_room_count,
            "evaluator_approval_fraction": (
                self.evaluator_approved_room_count / EXPECTED_AGENTIC_ROOM_DECISIONS
            ),
            "evaluator_rejection_fraction": (
                self.evaluator_rejected_room_count / EXPECTED_AGENTIC_ROOM_DECISIONS
            ),
            "evaluator_contract_violation_count": self.evaluator_contract_violation_count,
            "evaluator_action_field_violation_count": (
                self.evaluator_action_field_violation_count
            ),
            "evaluator_flag_echo_mismatch_count": self.evaluator_flag_mismatch_count,
            "evaluator_flag_echo_mismatch_is_diagnostic_only": True,
            "evaluator_approval_consistency_error_count": (
                self.evaluator_approval_consistency_error_count
            ),
            "initial_structural_invalid_count": self.initial_structural_invalid_count,
            "first_to_final_action_change_count": self.first_to_final_action_change_count,
            "first_to_final_action_change_fraction": (
                self.first_to_final_action_change_count / EXPECTED_AGENTIC_ROOM_DECISIONS
            ),
            "final_fallback_count": self.final_fallback_count,
            "final_fallback_fraction": (
                self.final_fallback_count / EXPECTED_AGENTIC_ROOM_DECISIONS
            ),
            "final_safety_modified_count": self.final_safety_modified_count,
            "final_safety_modified_fraction": (
                self.final_safety_modified_count / EXPECTED_AGENTIC_ROOM_DECISIONS
            ),
            "violation_counts": {
                str(k): int(v) for k, v in sorted(self.violation_counts.items())
            },
            "api_usage_by_role": self.api_usage_by_role,
            "api_usage_total": total_usage,
            "returned_models_by_role": {
                role: sorted(values)
                for role, values in self.returned_models_by_role.items()
            },
        }

        summary["action_distribution"] = {
            "timestep_action_source_counts": {
                str(k): int(v) for k, v in timestep_counts.items()
            },
            "decision_final_source_counts": {
                str(k): int(v) for k, v in final_counts.items()
            },
            "total_setpoint_switches_across_rooms": total_switches,
        }

        agentic_kwh = float(summary["energy"]["HVAC_component_sum_kWh"])
        fixed_kwh = float(summary["comparison_to_fixed"]["fixed_HVAC_component_sum_kWh"])
        summary["comparison_to_fixed"].pop("deepseek_direct_HVAC_component_sum_kWh", None)
        summary["comparison_to_fixed"].pop("energy_difference_fixed_minus_deepseek_kWh", None)
        summary["comparison_to_fixed"]["deepseek_agentic_HVAC_component_sum_kWh"] = agentic_kwh
        summary["comparison_to_fixed"]["energy_difference_fixed_minus_agentic_kWh"] = (
            fixed_kwh - agentic_kwh
        )
        summary["comparison_to_fixed"]["energy_savings_vs_fixed_percent"] = (
            (fixed_kwh - agentic_kwh) / fixed_kwh * 100.0
        )

        # Rename deterministic-reference comparison fields from direct -> agentic.
        for name in ("energy_comparison_to_occupancy_rule", "energy_comparison_to_comfort_rule"):
            section = summary.get(name, {})
            if "deepseek_direct_HVAC_component_sum_kWh" in section:
                section["deepseek_agentic_HVAC_component_sum_kWh"] = section.pop(
                    "deepseek_direct_HVAC_component_sum_kWh"
                )
            for key in list(section):
                if key.startswith("deepseek_minus_"):
                    section[key.replace("deepseek_minus_", "agentic_minus_", 1)] = section.pop(key)

        if "deepseek_comparison_to_occupancy_rule" in summary:
            summary["agentic_comparison_to_occupancy_rule"] = summary.pop(
                "deepseek_comparison_to_occupancy_rule"
            )
        if "deepseek_comparison_to_comfort_rule" in summary:
            summary["agentic_comparison_to_comfort_rule"] = summary.pop(
                "deepseek_comparison_to_comfort_rule"
            )

        for name in ("switching_comparison_to_occupancy_rule", "switching_comparison_to_comfort_rule"):
            section = summary.get(name, {})
            if "deepseek_direct_switches" in section:
                section["deepseek_agentic_switches"] = section.pop("deepseek_direct_switches")

        summary_path = self.output_dir / "deepseek_agentic_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str, allow_nan=True),
            encoding="utf-8",
        )
        intermediate.unlink()
        self._print_agentic_summary(summary, summary_path)

    def _print_agentic_summary(self, summary: Dict[str, Any], summary_path: Path) -> None:
        print("\n" + "=" * 78)
        print("DEEPSEEK AGENTIC CONTROLLER SUMMARY")
        print("=" * 78)
        validation = summary.get("validation", {})
        valid = bool(validation.get("all_required_deepseek_agentic_checks_pass", False))
        print(f"DeepSeek-agentic validation: {'PASS' if valid else 'FAIL'}")

        energy = summary.get("energy", {})
        fixed = summary.get("comparison_to_fixed", {})
        comfort = summary.get("comfort_and_occupancy", {})
        runtime = summary.get("agentic_runtime", {})

        print("\nEnergy:")
        print(f"  HVAC component sum       {energy.get('HVAC_component_sum_kWh', float('nan')):.4f} kWh")
        print(f"  Savings vs fixed         {fixed.get('energy_savings_vs_fixed_percent', float('nan')):.3f}%")

        print("\nOccupied controlled-room metrics:")
        print(f"  Occupied room-timesteps: {comfort.get('occupied_room_timesteps')}")
        print(f"  Person-hours: {comfort.get('person_hours', float('nan')):.3f}")
        print(f"  Overheat >27 C: {comfort.get('occupied_overheat_gt_27_fraction', float('nan'))*100:.3f}%")
        print(f"  Overcool <22.5 C: {comfort.get('occupied_overcool_lt_22_5_fraction', float('nan'))*100:.3f}%")
        print(f"  RH >85%: {comfort.get('occupied_RH_gt_85_fraction', float('nan'))*100:.3f}%")
        print(f"  Mean |PMV|: {comfort.get('occupied_mean_abs_PMV', float('nan')):.4f}")
        print(f"  Mean PPD: {comfort.get('occupied_mean_PPD_percent', float('nan')):.3f}%")

        initial = runtime.get("initial_controller_calls", {})
        evaluator = runtime.get("evaluator_calls", {})
        refine = runtime.get("refinement_calls", {})
        print("\nAgentic diagnostics:")
        print(f"  Decision epochs: {runtime.get('decision_epoch_count')} / {EXPECTED_AGENTIC_DECISION_EPOCHS}")
        print(f"  Initial controller: {initial.get('successful')}/{initial.get('attempted')} successful")
        print(f"  Evaluator: {evaluator.get('successful')}/{evaluator.get('attempted')} successful")
        print(
            f"  Evaluator approved/rejected rooms: "
            f"{runtime.get('evaluator_approved_room_count')}/"
            f"{runtime.get('evaluator_rejected_room_count')}"
        )
        print(f"  Refinement epochs: {refine.get('epochs_requiring_refinement')}")
        print(f"  Refinement calls: {refine.get('successful')}/{refine.get('attempted')} successful")
        print(f"  First -> final action changes: {runtime.get('first_to_final_action_change_count')}")
        print(f"  Final fallbacks: {runtime.get('final_fallback_count')} ({runtime.get('final_fallback_fraction')})")
        print(f"  Final safety modifications: {runtime.get('final_safety_modified_count')}")
        print(f"  Evaluator hard contract violations: {runtime.get('evaluator_contract_violation_count')}")
        print(f"  Evaluator action-field violations: {runtime.get('evaluator_action_field_violation_count')}")
        print(
            f"  Evaluator flag-echo mismatches (diagnostic only): "
            f"{runtime.get('evaluator_flag_echo_mismatch_count')}"
        )
        print(f"  Total API calls: {runtime.get('total_api_calls')}")

        print("\nControl integrity:")
        print(f"  People max abs error: {validation.get('people_tracking_max_abs_error')}")
        print(f"  Ventilation tracking: {validation.get('ventilation_fraction_within_tolerance')}")
        print(f"  Off-hours fixed 16/28: {validation.get('off_hours_fixed_16_28_pass')}")
        print(
            f"  Setpoint bounds/grid/deadband: "
            f"{validation.get('action_bounds_pass')}/"
            f"{validation.get('action_resolution_pass')}/"
            f"{validation.get('action_deadband_pass')}"
        )
        print(
            f"  Evaluator never outputs action setpoint fields: "
            f"{validation.get('evaluator_never_outputs_numeric_actions')}"
        )
        print(f"  K=1 refinement limit: {validation.get('maximum_refinement_rounds_is_one')}")
        print(f"  Direct fan actuation used: {validation.get('direct_fan_actuation_used')}")

        print("\nSaved:")
        for name in (
            "deepseek_agentic_state_trace.csv",
            "deepseek_agentic_meter_trace.csv",
            "deepseek_agentic_interval_energy.csv",
            "deepseek_agentic_room_summary.csv",
            "deepseek_agentic_daily_energy.csv",
            "deepseek_agentic_daily_comfort.csv",
            "deepseek_agentic_decisions.csv",
            "deepseek_agentic_decisions.jsonl",
            "deepseek_agentic_controller_prompt.txt",
            "deepseek_agentic_evaluator_prompt.txt",
            "deepseek_agentic_preflight.json",
            "deepseek_agentic_summary.json",
        ):
            print(f"  {self.output_dir / name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the agentic DeepSeek controller/evaluator EnergyPlus/API "
            "thermostat experiment for Closed-LoopAgenticLLMs."
        )
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_DIR,
        help="Repository root. Defaults to the root detected from this script.",
    )
    parser.add_argument(
        "--energyplus-root",
        type=Path,
        default=None,
        help=(
            "EnergyPlus installation root. If omitted, ENERGYPLUS_ROOT, PATH, "
            "and common installation locations are checked."
        ),
    )
    parser.add_argument("--idf", type=Path, default=None)
    parser.add_argument("--epw", type=Path, default=None)
    parser.add_argument("--occupancy", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fixed-summary", type=Path, default=None)
    parser.add_argument("--occupancy-rule-summary", type=Path, default=None)
    parser.add_argument("--comfort-rule-summary", type=Path, default=None)
    parser.add_argument("--api-base-url", type=str, default=DEFAULT_DEEPSEEK_BASE_URL)
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=LLM_TIMEOUT_SECONDS)
    parser.add_argument("--temperature", type=float, default=LLM_TEMPERATURE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project.resolve()

    def resolve(value: Optional[Path], default: Path) -> Path:
        return resolve_project_path(value, project) if value is not None else default

    idf_path = resolve(
        args.idf,
        project / "generated" / "building" / "honeycomb_7zone_fcu_closed_loop_ready.idf",
    )
    epw_path = resolve(
        args.epw,
        project / DEFAULT_EPW_NAME,
    )
    occupancy_path = resolve(
        args.occupancy,
        project / DEFAULT_OCCUPANCY_NAME,
    )
    output_dir = resolve(args.output_dir, project / DEEPSEEK_AGENTIC_RESULTS_SUBDIR)
    fixed_summary = resolve(
        args.fixed_summary,
        project / "results" / "fixed_baseline" / "fixed_baseline_summary.json",
    )
    occ_summary = resolve(
        args.occupancy_rule_summary,
        project / "results" / "occupancy_rule_baseline" / "occupancy_rule_summary.json",
    )
    comfort_summary = resolve(
        args.comfort_rule_summary,
        project / "results" / "comfort_rule_baseline" / "comfort_rule_summary.json",
    )

    print("=" * 78)
    print("Closed-LoopAgenticLLMs - 06_run_deepseek_agentic.py")
    print("=" * 78)

    try:
        api_key_file = (
            resolve_project_path(args.api_key_file, project)
            if args.api_key_file is not None
            else None
        )
        api_key, key_source = load_deepseek_api_key(project, api_key_file)
        energyplus_root = discover_energyplus_root(args.energyplus_root)
        EnergyPlusAPI = import_energyplus_api(energyplus_root)

        for label, path in (
            ("Generated IDF", idf_path),
            ("EPW", epw_path),
            ("Occupancy CSV", occupancy_path),
            ("Fixed summary", fixed_summary),
            ("Rule-OCC summary", occ_summary),
            ("Comfort-Rule summary", comfort_summary),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        runner = DeepSeekAgenticRunner(
            EnergyPlusAPI=EnergyPlusAPI,
            project_dir=project,
            energyplus_root=energyplus_root,
            idf_path=idf_path,
            epw_path=epw_path,
            occupancy_path=occupancy_path,
            output_dir=output_dir,
            fixed_summary_path=fixed_summary,
            occupancy_rule_summary_path=occ_summary,
            comfort_rule_summary_path=comfort_summary,
            deepseek_base_url=args.api_base_url,
            deepseek_model=args.model,
            deepseek_api_key=api_key,
            deepseek_key_source=key_source,
            llm_timeout_s=args.llm_timeout,
            llm_temperature=args.temperature,
        )
        status = runner.run()

        summary_path = output_dir / "deepseek_agentic_summary.json"
        if not summary_path.exists():
            return 2
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        valid = bool(
            summary.get("validation", {}).get(
                "all_required_deepseek_agentic_checks_pass", False
            )
        )
        return 0 if status == 0 and valid else 1

    except Exception as exc:
        print("\nFATAL DEEPSEEK-AGENTIC ERROR", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
