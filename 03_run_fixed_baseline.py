#!/usr/bin/env python3
r"""
03_run_fixed_baseline.py

Fixed-policy EnergyPlus/API baseline for the Closed-LoopAgenticLLMs project.

Purpose
-------
Run the first publication-quality reference case after
``01_prepare_energyplus_model.py`` and ``02_preflight_energyplus_api.py`` have
passed.

This is NOT an LLM controller. It uses the same runtime infrastructure that
later deterministic and LLM cases should use so that comparisons differ only
in the supervisory control policy.

Controlled rooms
----------------
Rooms 1, 2, 4, 5, 6 and 7.

Thermal Zone 3 remains the conventional/reference zone and is not actuated by
this script.

Runtime inputs for the six controlled rooms
--------------------------------------------
1. Measured 5-minute occupancy count:
       inputs/occupancy/actual_occupancy_count_5min_7day.csv

2. People internal gains:
       EnergyPlus People -> Number of People actuator
       commanded exactly from the measured occupancy count.

3. Occupancy-proportional outdoor ventilation:
       ventilation_fraction = occupancy_count / design_people
   applied to each experimental Schedule:Constant prepared by
   ``01_prepare_energyplus_model.py``.

4. Fixed thermostat policy:
       08:00 <= interval_start < 18:00:
           heating = 20 C
           cooling = 25 C
       otherwise:
           heating = 16 C
           cooling = 28 C

5. FCU availability:
       1.0 for all six controlled rooms at all times.

6. Direct fan actuation:
       NONE.
   EnergyPlus' native four-pipe FCU / MultiSpeedFan logic determines fan flow.
   The preflight proves whether direct fan actuation is available, but it is
   intentionally NOT overridden in this baseline.

Time alignment
--------------
The six controlled rooms use one authoritative zone-timestep sequence clock.

The weather-file RunPeriod is identified with EnergyPlus ``kind_of_sim == 3``.
At the first non-warmup BeginZoneTimestep callback, sequence index 0 is anchored
to 2021-08-16 00:00. Each later BeginZoneTimestep advances the clock by exactly
5 minutes:

    interval_start = CONTROL_START + sequence_index * 5 minutes
    interval_end   = interval_start + 5 minutes

The same interval_start is then reused by:
    - measured occupancy lookup,
    - People actuation,
    - ventilation actuation,
    - fixed thermostat control,
    - end-zone physical-state logging,
    - meter logging,
    - final analysis.

This deliberately does NOT reconstruct the research clock from Hour/Minutes at
different EnergyPlus calling points. EnergyPlus current/date/weather values may
align differently at EndOfZoneTimestepAfterZoneReporting, while the sequential
zone-timestep clock provides one unambiguous research index.

The run must contain exactly:
    2016 unique 5-minute interval starts per controlled room
    12096 controlled-room analysis rows (= 2016 x 6)
    first interval start = 2021-08-16 00:00
    last interval start  = 2021-08-22 23:55

Primary energy metric
---------------------
HVAC component electricity is reported as:

    Cooling:Electricity
  + Heating:Electricity
  + Fans:Electricity
  + Pumps:Electricity

Electricity:HVAC and Electricity:Facility are retained as diagnostic meters,
but are NOT added to the component sum to avoid double counting.

Comfort / IEQ diagnostics
-------------------------
For occupied controlled-room timesteps (occupancy > 0), the script reports:

    temperature > 27.0 C
    temperature < 22.5 C
    RH > 85 %
    PMV / PPD

Fanger PMV/PPD uses:
    air temperature = EnergyPlus zone mean air temperature
    radiant temperature = EnergyPlus zone mean radiant temperature
    relative humidity = EnergyPlus zone air RH
    air speed = 0.1 m/s
    metabolic rate = 1.2 met
    clothing = 0.5 clo
    external work = 0 met

Both room-timestep-weighted and person-weighted metrics are written so the
weighting convention is explicit.

Default repository inputs
-------------------------
Generated IDF:
    generated/building/honeycomb_7zone_fcu_closed_loop_ready.idf

Weather:
    inputs/weather/CHN_Hebei.Shijiazhuang.536980_CSWD.epw

Occupancy:
    inputs/occupancy/actual_occupancy_count_5min_7day.csv

EnergyPlus installation
-----------------------
The EnergyPlus 24.1 installation is discovered from, in order:
1. ``--energyplus-root`` when supplied.
2. ``ENERGYPLUS_ROOT`` or ``ENERGYPLUS_HOME`` environment variables.
3. An already importable ``pyenergyplus`` package.
4. The ``energyplus`` executable on PATH.
5. Common EnergyPlus installation locations for Windows, Linux, and macOS.

Outputs
-------
    results/fixed_baseline/
        available_api_data.csv
        handle_report.csv
        fixed_baseline_state_trace.csv
        fixed_baseline_meter_trace.csv
        fixed_baseline_interval_energy.csv
        fixed_baseline_daily_energy.csv
        fixed_baseline_room_summary.csv
        fixed_baseline_daily_comfort.csv
        fixed_baseline_summary.json
        energyplus_output/...

EnergyPlus target: 24.1
Python target: 3.9+

The repository root is detected from this script location and can be overridden
with ``--project``. Default repository paths are portable across operating
systems and user accounts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


# ============================================================================
# Project configuration
# ============================================================================

SCRIPT_NAME = Path(__file__).name
SCRIPT_DIR = Path(__file__).resolve().parent

# Support the recommended repository layout (script stored in ``scripts/``)
# while remaining usable if the file is temporarily kept in the repository root.
DEFAULT_PROJECT_DIR = (
    SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() in {"scripts", "tools"} else SCRIPT_DIR
)

GENERATED_BUILDING_SUBDIR = Path("generated") / "building"
WEATHER_SUBDIR = Path("inputs") / "weather"
OCCUPANCY_SUBDIR = Path("inputs") / "occupancy"
FIXED_RESULTS_SUBDIR = Path("results") / "fixed_baseline"

DEFAULT_IDF_NAME = "honeycomb_7zone_fcu_closed_loop_ready.idf"
DEFAULT_EPW_NAME = "CHN_Hebei.Shijiazhuang.536980_CSWD.epw"
DEFAULT_OCCUPANCY_NAME = "actual_occupancy_count_5min_7day.csv"

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

OFFICE_HEATING_SP_C = 20.0
OFFICE_COOLING_SP_C = 25.0

OFF_HOURS_HEATING_SP_C = 16.0
OFF_HOURS_COOLING_SP_C = 28.0

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


def fixed_setpoints(interval_start: pd.Timestamp) -> Tuple[float, float]:
    if is_office_hour(interval_start):
        return OFFICE_HEATING_SP_C, OFFICE_COOLING_SP_C
    return OFF_HOURS_HEATING_SP_C, OFF_HOURS_COOLING_SP_C


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
    """Resolve a path relative to the repository root when it is not absolute."""
    return path.expanduser().resolve() if path.is_absolute() else (project / path).resolve()


def portable_path(path: Path, project: Path) -> str:
    """Return a repository-relative path when possible.

    This prevents generated result manifests from embedding a contributor's
    machine-specific filesystem layout.
    """
    resolved_path = path.resolve()
    resolved_project = project.resolve()
    try:
        return resolved_path.relative_to(resolved_project).as_posix()
    except ValueError:
        return str(resolved_path)


def _is_energyplus_root(path: Path) -> bool:
    """Return True when *path* appears to contain the EnergyPlus Python API."""
    return path.exists() and (path / "pyenergyplus" / "api.py").exists()


def discover_energyplus_root(preferred: Optional[Path] = None) -> Path:
    """Locate EnergyPlus 24.1 without relying on a user-specific path."""
    candidates: List[Path] = []

    if preferred is not None:
        candidates.append(preferred.expanduser())

    for variable in ("ENERGYPLUS_ROOT", "ENERGYPLUS_HOME"):
        value = os.environ.get(variable, "").strip()
        if value:
            candidates.append(Path(value).expanduser())

    # If pyenergyplus is already importable, infer its installation root.
    try:
        spec = importlib.util.find_spec("pyenergyplus.api")
        if spec is not None and spec.origin:
            candidates.append(Path(spec.origin).resolve().parent.parent)
    except (ImportError, AttributeError, ValueError):
        pass

    # If the EnergyPlus executable is on PATH, its parent is usually the install root.
    executable = shutil.which("energyplus")
    if executable:
        candidates.append(Path(executable).resolve().parent)

    # Generic fallback locations for major desktop/server platforms.
    candidates.extend(
        [
            Path("C:/EnergyPlusV24-1-0"),
            Path("/usr/local/EnergyPlus-24-1-0"),
            Path("/usr/local/EnergyPlusV24-1-0"),
            Path("/opt/EnergyPlus-24-1-0"),
            Path("/opt/EnergyPlusV24-1-0"),
            Path("/Applications/EnergyPlus-24-1-0"),
            Path("/Applications/EnergyPlusV24-1-0"),
        ]
    )

    # Discover other matching Windows 24.1 installations when present.
    if os.name == "nt":
        try:
            candidates.extend(sorted(Path("C:/").glob("EnergyPlusV24-1-*"), reverse=True))
        except OSError:
            pass

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate.resolve() if candidate.exists() else candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if _is_energyplus_root(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the EnergyPlus 24.1 Python API. Install EnergyPlus 24.1 "
        "and provide --energyplus-root, or set the ENERGYPLUS_ROOT environment "
        "variable to the EnergyPlus installation directory."
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

class FixedBaselineRunner:
    def __init__(
        self,
        EnergyPlusAPI: Any,
        energyplus_root: Path,
        project_dir: Path,
        idf_path: Path,
        epw_path: Path,
        occupancy_path: Path,
        output_dir: Path,
    ) -> None:
        self.energyplus_root = energyplus_root
        self.project_dir = project_dir.resolve()
        self.idf_path = idf_path
        self.epw_path = epw_path
        self.occupancy_path = occupancy_path
        self.output_dir = output_dir

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
                "heating_setpoint_C": OFF_HOURS_HEATING_SP_C,
                "cooling_setpoint_C": OFF_HOURS_COOLING_SP_C,
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
        print("EnergyPlus API data ready - acquiring fixed-baseline handles")
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
    # Begin-system-timestep: fixed thermostat + FCU availability
    # ------------------------------------------------------------------

    def apply_fixed_control(
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

            heating_sp, cooling_sp = fixed_setpoints(
                interval_start
            )

            for room in ROOM_ORDER:
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
                    "availability"
                ] = FCU_AVAILABILITY_COMMAND

            self.system_control_callback_count += 1

            # Intentionally NO direct fan actuator call here.

        except Exception as exc:
            self.record_callback_error(
                "apply_fixed_control",
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

            heating_sp, cooling_sp = fixed_setpoints(interval_start)

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
                    "case": "fixed_baseline",
                    "zone_sequence_index": self.zone_sequence_index,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "room": room,
                    "zone": info["zone"],
                    "office_hour": int(
                        is_office_hour(interval_start)
                    ),
                    "occupancy_count": occupancy,
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
                "case": "fixed_baseline",
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
            self.apply_fixed_control,
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
        print("Running fixed EnergyPlus/API baseline")
        print("=" * 78)
        print(f"EnergyPlus : {self.energyplus_root}")
        print(f"IDF        : {self.idf_path}")
        print(f"EPW        : {self.epw_path}")
        print(f"Occupancy  : {self.occupancy_path}")
        print(f"Results    : {self.output_dir}")
        print()
        print("Fixed supervisory policy:")
        print(
            f"  Office {OFFICE_START_HOUR:02d}:00-"
            f"{OFFICE_END_HOUR:02d}:00 -> "
            f"{OFFICE_HEATING_SP_C:.1f}/"
            f"{OFFICE_COOLING_SP_C:.1f} C"
        )
        print(
            f"  Off-hours -> "
            f"{OFF_HOURS_HEATING_SP_C:.1f}/"
            f"{OFF_HOURS_COOLING_SP_C:.1f} C"
        )
        print("  FCU availability -> 1.0")
        print("  Direct fan override -> NONE")
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

        summary: Dict[str, Any] = {
            "room": str(room_frame["room"].iloc[0]),
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

        else:
            checks["people_tracking_pass"] = False
            checks["thermostat_tracking_pass"] = False
            checks["ventilation_tracking_pass"] = False
            checks["direct_fan_actuation_absent"] = False

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

    def save_results(
        self,
        energyplus_status: int,
        wall_time_s: float,
    ) -> None:
        state_df = self.state_dataframe()
        meter_df = self.meter_dataframe()
        interval_energy = self.build_interval_energy(
            meter_df
        )

        state_path = (
            self.output_dir
            / "fixed_baseline_state_trace.csv"
        )
        meter_path = (
            self.output_dir
            / "fixed_baseline_meter_trace.csv"
        )
        interval_energy_path = (
            self.output_dir
            / "fixed_baseline_interval_energy.csv"
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
            / "fixed_baseline_room_summary.csv"
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
            / "fixed_baseline_daily_comfort.csv"
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
            / "fixed_baseline_daily_energy.csv"
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
            "script": SCRIPT_NAME,
            "case": "fixed_baseline",
            "energyplus_exit_status": energyplus_status,
            "simulation_wall_time_s": wall_time_s,
            "project": ".",
            "repository": "Closed-LoopAgenticLLMs",
            "idf": portable_path(self.idf_path, self.project_dir),
            "epw": portable_path(self.epw_path, self.project_dir),
            "occupancy": portable_path(self.occupancy_path, self.project_dir),
            "occupancy_sha256": sha256_file(self.occupancy_path),
            "expected_occupancy_sha256": EXPECTED_OCCUPANCY_SHA256,
            "controlled_rooms": ROOM_ORDER,
            "reference_zone": "Thermal Zone 3",
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
                "office_hours": (
                    f"{OFFICE_START_HOUR:02d}:00-"
                    f"{OFFICE_END_HOUR:02d}:00"
                ),
                "office_heating_setpoint_C": (
                    OFFICE_HEATING_SP_C
                ),
                "office_cooling_setpoint_C": (
                    OFFICE_COOLING_SP_C
                ),
                "off_hours_heating_setpoint_C": (
                    OFF_HOURS_HEATING_SP_C
                ),
                "off_hours_cooling_setpoint_C": (
                    OFF_HOURS_COOLING_SP_C
                ),
                "fcu_availability": (
                    FCU_AVAILABILITY_COMMAND
                ),
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

        summary_path = (
            self.output_dir
            / "fixed_baseline_summary.json"
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
        print("FIXED BASELINE SUMMARY")
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
                "\nFixed baseline is structurally valid. Review the "
                "energy/comfort values before freezing it as the reference."
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
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-policy EnergyPlus/API baseline for the "
            "Closed-LoopAgenticLLMs project."
        )
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Repository root. By default it is inferred from the script location.",
    )
    parser.add_argument(
        "--energyplus-root",
        type=Path,
        default=None,
        help=(
            "EnergyPlus 24.1 installation directory. If omitted, the script checks "
            "environment variables, PATH, and common installation locations."
        ),
    )
    parser.add_argument(
        "--idf",
        type=Path,
        default=None,
        help=(
            "Generated closed-loop-ready IDF path. Relative paths are resolved "
            "against --project."
        ),
    )
    parser.add_argument(
        "--epw",
        type=Path,
        default=None,
        help="Weather EPW path. Relative paths are resolved against --project.",
    )
    parser.add_argument(
        "--occupancy",
        type=Path,
        default=None,
        help=(
            "Measured 5-minute occupancy-count CSV path. Relative paths are "
            "resolved against --project."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Fixed-baseline result directory. Relative paths are resolved against "
            "--project."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project.expanduser().resolve()

    idf_path = (
        resolve_project_path(args.idf, project)
        if args.idf is not None
        else (project / GENERATED_BUILDING_SUBDIR / DEFAULT_IDF_NAME).resolve()
    )
    epw_path = (
        resolve_project_path(args.epw, project)
        if args.epw is not None
        else (project / WEATHER_SUBDIR / DEFAULT_EPW_NAME).resolve()
    )
    occupancy_path = (
        resolve_project_path(args.occupancy, project)
        if args.occupancy is not None
        else (project / OCCUPANCY_SUBDIR / DEFAULT_OCCUPANCY_NAME).resolve()
    )
    output_dir = (
        resolve_project_path(args.output_dir, project)
        if args.output_dir is not None
        else (project / FIXED_RESULTS_SUBDIR).resolve()
    )

    print("=" * 78)
    print(f"Closed-LoopAgenticLLMs - {SCRIPT_NAME}")
    print("=" * 78)
    print(f"Project    : {project}")

    try:
        energyplus_root = discover_energyplus_root(args.energyplus_root)
        EnergyPlusAPI = import_energyplus_api(energyplus_root)

        if not idf_path.exists():
            raise FileNotFoundError(
                f"Generated IDF not found: {idf_path}\n"
                "Run 01_prepare_energyplus_model.py first."
            )
        if not epw_path.exists():
            raise FileNotFoundError(f"EPW not found: {epw_path}")
        if not occupancy_path.exists():
            raise FileNotFoundError(f"Occupancy CSV not found: {occupancy_path}")

        runner = FixedBaselineRunner(
            EnergyPlusAPI=EnergyPlusAPI,
            energyplus_root=energyplus_root,
            project_dir=project,
            idf_path=idf_path,
            epw_path=epw_path,
            occupancy_path=occupancy_path,
            output_dir=output_dir,
        )

        status = runner.run()
        summary_path = output_dir / "fixed_baseline_summary.json"

        if not summary_path.exists():
            return 2

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        valid = bool(
            summary.get("validation", {}).get(
                "all_required_baseline_checks_pass",
                False,
            )
        )

        # Exit 0 only when EnergyPlus succeeds and baseline integrity checks pass.
        return 0 if status == 0 and valid else 1

    except Exception as exc:
        print("\nFATAL FIXED-BASELINE ERROR", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
