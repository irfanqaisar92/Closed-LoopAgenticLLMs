#!/usr/bin/env python3
r"""
02_preflight_energyplus_api.py

Strict EnergyPlus Python-API preflight for the Closed-LoopAgenticLLMs project.

Purpose
-------
This is NOT a benchmark and its results must not be used as research results.
It deliberately perturbs the model to verify that the runtime control pathway
is real before any fixed/rule/LLM controller is evaluated.

The script verifies, for Rooms 1, 2, 4, 5, 6 and 7:

REQUIRED
1. Zone temperature, RH and MRT feedback handles.
2. Independent heating and cooling thermostat actuators.
3. Thermostat readback follows deliberately different per-room commands.
4. Six experimental People actuators.
5. EnergyPlus Zone People Occupant Count follows the measured 5-minute counts.
6. Six experimental ventilation Schedule:Constant actuators.
7. Ventilation schedule commands follow measured occupancy / design_people.
8. Zone ventilation-flow feedback is available and responds to occupancy.
9. Six FCU availability schedule actuators.
10. FCU OFF -> ON behavior is physically checked under a forced cooling load.

OPTIONAL / CHARACTERIZED
11. Direct supply-fan air-mass-flow actuator availability.
12. If available, low/high fan-flow probes test whether the reported fan flow
    follows the requested fractions of Fan Maximum Mass Flow Rate.

Zone 3
------
Thermal Zone 3 is not controlled by this script. It remains the reference zone
using conventional People and ventilation schedules prepared by
``01_prepare_energyplus_model.py``.

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
    results/preflight/
        available_api_data.csv
        handle_report.csv
        preflight_trace.csv
        preflight_checks.csv
        preflight_summary.json
        energyplus_output/...

Important
---------
The ventilation Schedule:Constant objects have a fail-safe IDF default of 1.0.
During the weather-file RunPeriod this script overwrites each one every zone
timestep with:

    measured_occupancy / design_people

This script runs the full seven-day simulation, but special actuator probes are
limited to 16 August around midday. Outside the probe windows, it uses the
transparent fixed thermostat policy intended for the first benchmark:
    08:00-18:00 -> 20 C heating / 25 C cooling
    otherwise   -> 16 C heating / 28 C cooling

During the FCU availability and fan probes only, cooling is temporarily forced
to 22 C to ensure a clear cooling load. These probe periods are diagnostic only.

Time alignment
--------------
EnergyPlus reports the current simulation clock at the END of the active
5-minute zone timestep. The occupancy CSV is indexed by the START of each
5-minute interval. Therefore this script explicitly uses:

    interval_end   = EnergyPlus reported clock
    interval_start = interval_end - 5 minutes

All occupancy lookup, People actuation, ventilation actuation, controller
logic, probe windows, and analysis use interval_start.

The preflight requires exactly:
    2016 unique interval starts per controlled room
    12096 analysis rows total (= 2016 x 6 rooms)
    first interval start = 2021-08-16 00:00
    last interval start  = 2021-08-22 23:55

EnergyPlus target: 24.1
Python target: 3.9+

The repository root is detected from this script location and can be overridden
with ``--project``. Default repository paths are portable across operating
systems and user accounts.
"""

from __future__ import annotations

import argparse
import csv
import io
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
PREFLIGHT_RESULTS_SUBDIR = Path("results") / "preflight"

DEFAULT_IDF_NAME = "honeycomb_7zone_fcu_closed_loop_ready.idf"
DEFAULT_EPW_NAME = "CHN_Hebei.Shijiazhuang.536980_CSWD.epw"
DEFAULT_OCCUPANCY_NAME = "actual_occupancy_count_5min_7day.csv"

CONTROL_START = pd.Timestamp("2021-08-16 00:00:00")
CONTROL_END_EXCLUSIVE = pd.Timestamp("2021-08-23 00:00:00")

ZONE_TIMESTEP_MINUTES = 5
EXPECTED_INTERVALS_PER_ROOM = 7 * 24 * (60 // ZONE_TIMESTEP_MINUTES)  # 2016
EXPECTED_ANALYSIS_ROWS = 12096
EXPECTED_FIRST_INTERVAL_START = CONTROL_START
EXPECTED_LAST_INTERVAL_START = CONTROL_END_EXCLUSIVE - pd.Timedelta(
    minutes=ZONE_TIMESTEP_MINUTES
)

OFFICE_START_HOUR = 8
OFFICE_END_HOUR = 18

NORMAL_OFFICE_HEAT_C = 20.0
NORMAL_OFFICE_COOL_C = 25.0
NORMAL_OFF_HOURS_HEAT_C = 16.0
NORMAL_OFF_HOURS_COOL_C = 28.0

# Diagnostic-only forced cooling load for FCU/fan physical-response tests.
PROBE_HEAT_C = 16.0
PROBE_COOL_C = 22.0

VENT_FLOW_PER_PERSON_M3_S = 0.009438948864

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

# Unique thermostat commands used only in the independent-setpoint probe.
THERMOSTAT_PROBE: Dict[str, Tuple[float, float]] = {
    "Room_1": (16.0, 25.0),
    "Room_2": (16.5, 25.5),
    "Room_4": (17.0, 26.0),
    "Room_5": (17.5, 26.5),
    "Room_6": (18.0, 27.0),
    "Room_7": (18.5, 27.5),
}

# Probe windows on the first simulation day.
PROBE_WINDOWS = {
    "thermostat_probe": (
        pd.Timestamp("2021-08-16 12:00:00"),
        pd.Timestamp("2021-08-16 12:30:00"),
    ),
    "availability_on_probe": (
        pd.Timestamp("2021-08-16 12:30:00"),
        pd.Timestamp("2021-08-16 12:50:00"),
    ),
    "availability_off_probe": (
        pd.Timestamp("2021-08-16 12:50:00"),
        pd.Timestamp("2021-08-16 13:10:00"),
    ),
    "availability_recovery_probe": (
        pd.Timestamp("2021-08-16 13:10:00"),
        pd.Timestamp("2021-08-16 13:30:00"),
    ),
    "fan_low_probe": (
        pd.Timestamp("2021-08-16 13:30:00"),
        pd.Timestamp("2021-08-16 13:50:00"),
    ),
    "fan_high_probe": (
        pd.Timestamp("2021-08-16 13:50:00"),
        pd.Timestamp("2021-08-16 14:10:00"),
    ),
}

FAN_LOW_RATIO = 0.33
FAN_HIGH_RATIO = 1.00


# ============================================================================
# Generic helpers
# ============================================================================

def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def phase_for_timestamp(timestamp: pd.Timestamp) -> str:
    for name, (start, end) in PROBE_WINDOWS.items():
        if start <= timestamp < end:
            return name
    return "normal"


def fixed_setpoints(timestamp: pd.Timestamp) -> Tuple[float, float]:
    if OFFICE_START_HOUR <= timestamp.hour < OFFICE_END_HOUR:
        return NORMAL_OFFICE_HEAT_C, NORMAL_OFFICE_COOL_C
    return NORMAL_OFF_HOURS_HEAT_C, NORMAL_OFF_HOURS_COOL_C


def energyplus_interval_times(
    api: Any,
    state: Any,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (interval_start, interval_end) for the active 5-minute timestep.

    EnergyPlus' Hour/Minutes exchange values correspond to the END of the
    current timestep. The measured occupancy CSV is indexed by interval START.
    Example:
        EnergyPlus clock 00:05 -> occupancy/data interval 00:00-00:05.
        EnergyPlus clock 24:00 -> occupancy/data interval 23:55-24:00.
    """
    month = int(api.exchange.month(state))
    day = int(api.exchange.day_of_month(state))
    hour = int(api.exchange.hour(state))
    minute = int(api.exchange.minutes(state))

    elapsed_minutes = (24 * 60 + minute) if hour >= 24 else (hour * 60 + minute)
    interval_end = (
        pd.Timestamp(year=2021, month=month, day=day)
        + pd.Timedelta(minutes=elapsed_minutes)
    ).round(f"{ZONE_TIMESTEP_MINUTES}min")
    interval_start = interval_end - pd.Timedelta(minutes=ZONE_TIMESTEP_MINUTES)
    return interval_start, interval_end


def timestamp_from_energyplus(api: Any, state: Any) -> pd.Timestamp:
    """Return the START timestamp of the active 5-minute interval."""
    interval_start, _ = energyplus_interval_times(api, state)
    return interval_start


def in_control_period(interval_start: pd.Timestamp) -> bool:
    return CONTROL_START <= interval_start < CONTROL_END_EXCLUSIVE


def resolve_project_path(path: Path, project: Path) -> Path:
    """Resolve a path relative to the repository root when it is not absolute."""
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def portable_path(path: Path, project: Path) -> str:
    """Return a repository-relative path when possible.

    This keeps generated JSON summaries portable and avoids embedding a user's
    local filesystem layout in files that may be archived or shared.
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
    """Locate an EnergyPlus installation without relying on a user-specific path."""
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

    # Common installation locations. These are generic fallbacks only.
    common_candidates = [
        Path("C:/EnergyPlusV24-1-0"),
        Path("/usr/local/EnergyPlus-24-1-0"),
        Path("/usr/local/EnergyPlusV24-1-0"),
        Path("/opt/EnergyPlus-24-1-0"),
        Path("/opt/EnergyPlusV24-1-0"),
        Path("/Applications/EnergyPlus-24-1-0"),
        Path("/Applications/EnergyPlusV24-1-0"),
    ]
    candidates.extend(common_candidates)

    # Also discover versioned Windows installations without hard-coding a username.
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
# Occupancy input
# ============================================================================

def load_occupancy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Occupancy CSV not found: {path}")

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

    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
    if df[timestamp_column].isna().any():
        bad = int(df[timestamp_column].isna().sum())
        raise ValueError(f"Occupancy CSV has {bad} invalid timestamps.")

    df = df.set_index(timestamp_column).sort_index()

    if df.index.has_duplicates:
        duplicates = int(df.index.duplicated().sum())
        raise ValueError(f"Occupancy CSV has {duplicates} duplicate timestamps.")

    expected_index = pd.date_range(
        CONTROL_START,
        CONTROL_END_EXCLUSIVE - pd.Timedelta(minutes=5),
        freq="5min",
    )

    missing_times = expected_index.difference(df.index)
    if len(missing_times) > 0:
        raise ValueError(
            f"Occupancy CSV is missing {len(missing_times)} required 5-minute rows. "
            f"First missing timestamp: {missing_times[0]}"
        )

    resolved_columns: Dict[str, str] = {}
    for room, info in ROOMS.items():
        candidates = [
            f"{room.lower()}_occupant_count",
            f"{room.lower()}_occupancy_count",
            room,
        ]
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise ValueError(
                f"Missing occupancy column for {room}. Tried {candidates}. "
                f"Available columns: {list(df.columns)}"
            )

        numeric = pd.to_numeric(df[found], errors="coerce")
        if numeric.loc[expected_index].isna().any():
            raise ValueError(f"{found} contains NaN/non-numeric occupancy values.")

        if (numeric.loc[expected_index] < 0).any():
            raise ValueError(f"{found} contains negative occupancy.")

        capacity = float(info["capacity"])
        if (numeric.loc[expected_index] > capacity + 1e-9).any():
            maximum = float(numeric.loc[expected_index].max())
            raise ValueError(
                f"{found} exceeds design capacity {capacity:g}; max={maximum:g}."
            )

        df[found] = numeric.astype(float)
        resolved_columns[room] = found

    df.attrs["resolved_columns"] = resolved_columns
    return df


# ============================================================================
# IDF parsing: discover expanded FCU and supply-fan names
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


def discover_fcu_components(idf_path: Path) -> Dict[str, Dict[str, str]]:
    """Map Thermal Zone N to expanded FCU and fan names.

    For EnergyPlus 24.1 ZoneHVAC:FourPipeFanCoil:
        field 1  = object type
        field 2  = FCU name
        field 3  = availability schedule
        field 14 = supply fan object type
        field 15 = supply fan name

    List indexes below are therefore 0,1,2,...13,14.
    """
    mapping: Dict[str, Dict[str, str]] = {}

    text = idf_path.read_text(encoding="utf-8-sig", errors="replace")
    for block in split_idf_objects(text):
        fields = idf_fields(block)
        if not fields or normalize(fields[0]) != normalize("ZoneHVAC:FourPipeFanCoil"):
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
            match = re.search(r"thermal\s*zone\s*(\d+)", fcu_name, flags=re.I)
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


# ============================================================================
# API dictionary parsing
# ============================================================================

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


def api_output_keys(api_rows: Sequence[Sequence[str]], variable_name: str) -> List[str]:
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


def api_fan_candidates(api_rows: Sequence[Sequence[str]]) -> Dict[str, List[str]]:
    result = {
        "actuator": [],
        "internal_variable": [],
        "output_variable": [],
    }

    for row in api_rows:
        if not row:
            continue

        record_type = normalize(row[0])

        if (
            record_type == "actuator"
            and len(row) >= 4
            and normalize(row[1]) == "fan"
            and normalize(row[2]) == "fanairmassflowrate"
        ):
            result["actuator"].append(row[3])

        elif (
            record_type == "internalvariable"
            and len(row) >= 3
            and normalize(row[1]) == "fanmaximummassflowrate"
        ):
            result["internal_variable"].append(row[2])

        elif (
            record_type == "outputvariable"
            and len(row) >= 3
            and normalize(row[1]) == "fanairmassflowrate"
        ):
            result["output_variable"].append(row[2])

    for category in result:
        result[category] = sorted(set(x for x in result[category] if x))

    return result


def resolve_name(expected: str, zone_number: int, candidates: Sequence[str]) -> str:
    """Resolve an API key with exact match first, then conservative zone-number match."""
    if not candidates:
        return ""

    expected_norm = normalize(expected)
    for candidate in candidates:
        if normalize(candidate) == expected_norm:
            return candidate

    zone_tokens = (
        f"thermalzone{zone_number}",
        f"zone{zone_number}",
    )
    for candidate in candidates:
        norm = normalize(candidate)
        if any(token in norm for token in zone_tokens):
            return candidate

    # Do not guess if more than one unrelated candidate exists.
    return ""


def resolve_ventilation_output_key(
    api_rows: Sequence[Sequence[str]],
    variable_name: str,
    zone: str,
    vent_object: str,
    zone_number: int,
) -> str:
    keys = api_output_keys(api_rows, variable_name)
    if not keys:
        return ""

    targets = [normalize(vent_object), normalize(zone)]
    for key in keys:
        if normalize(key) in targets:
            return key

    # Conservative fallback: require the zone number in the key.
    tokens = (f"thermalzone{zone_number}", f"zone{zone_number}")
    matches = [
        key for key in keys if any(token in normalize(key) for token in tokens)
    ]
    if len(matches) == 1:
        return matches[0]

    return ""


# ============================================================================
# Report structures
# ============================================================================

@dataclass
class HandleRecord:
    room: str
    category: str
    api_type: str
    api_name: str
    key: str
    handle: int
    required: bool
    status: str


@dataclass
class PreflightState:
    handles_ready: bool = False
    api_dictionary_saved: bool = False
    callback_errors: List[str] = field(default_factory=list)
    handle_records: List[HandleRecord] = field(default_factory=list)
    trace_rows: List[Dict[str, Any]] = field(default_factory=list)
    setup_notes: List[str] = field(default_factory=list)


# ============================================================================
# Main preflight runner
# ============================================================================

class PreflightRunner:
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
        self.energyplus_output_dir.mkdir(parents=True, exist_ok=True)

        self.occupancy = load_occupancy(occupancy_path)
        self.occ_columns: Dict[str, str] = dict(
            self.occupancy.attrs["resolved_columns"]
        )

        self.fcu_components = discover_fcu_components(idf_path)

        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()

        self.report = PreflightState()
        self.coverage_stats: Dict[str, Any] = {}

        # Variable handles
        self.temp_handles: Dict[str, int] = {}
        self.rh_handles: Dict[str, int] = {}
        self.mrt_handles: Dict[str, int] = {}
        self.people_output_handles: Dict[str, int] = {}
        self.actual_heat_handles: Dict[str, int] = {}
        self.actual_cool_handles: Dict[str, int] = {}
        self.vent_current_handles: Dict[str, int] = {}
        self.vent_standard_handles: Dict[str, int] = {}

        # Actuator handles
        self.people_actuators: Dict[str, int] = {}
        self.vent_actuators: Dict[str, int] = {}
        self.heat_actuators: Dict[str, int] = {}
        self.cool_actuators: Dict[str, int] = {}
        self.availability_actuators: Dict[str, int] = {}

        # Fan / FCU handles
        self.fan_names: Dict[str, str] = {}
        self.fcu_names: Dict[str, str] = {}
        self.fan_actuators: Dict[str, int] = {}
        self.fan_max_handles: Dict[str, int] = {}
        self.fan_flow_handles: Dict[str, int] = {}
        self.fan_power_handles: Dict[str, int] = {}
        self.fcu_speed_ratio_handles: Dict[str, int] = {}
        self.fcu_plr_handles: Dict[str, int] = {}

        # Current commands for trace logging
        self.current_commands: Dict[str, Dict[str, float]] = {
            room: {
                "occupancy": 0.0,
                "vent_fraction": 0.0,
                "availability": 1.0,
                "heat_C": NORMAL_OFF_HOURS_HEAT_C,
                "cool_C": NORMAL_OFF_HOURS_COOL_C,
                "fan_ratio": float("nan"),
            }
            for room in ROOM_ORDER
        }

        self.last_fan_override_active = False

        self.request_variables()

    # ----------------------------------------------------------------------
    # Pre-run requests
    # ----------------------------------------------------------------------

    def request_variables(self) -> None:
        """Request all useful outputs before EnergyPlus starts."""
        for room, info in ROOMS.items():
            zone = info["zone"]
            vent_object = info["vent_object"]

            for variable in (
                "Zone Mean Air Temperature",
                "Zone Air Relative Humidity",
                "Zone Mean Radiant Temperature",
                "Zone People Occupant Count",
                "Zone Thermostat Heating Setpoint Temperature",
                "Zone Thermostat Cooling Setpoint Temperature",
            ):
                self.api.exchange.request_variable(self.state, variable, zone)

            # Ventilation key varies across EnergyPlus output dictionaries.
            for variable in (
                "Zone Ventilation Current Density Volume Flow Rate",
                "Zone Ventilation Standard Density Volume Flow Rate",
            ):
                self.api.exchange.request_variable(self.state, variable, zone)
                self.api.exchange.request_variable(self.state, variable, vent_object)

        # Known fan/FCU names parsed from the expanded IDF.
        for room, info in ROOMS.items():
            zone = info["zone"]
            components = self.fcu_components.get(zone, {})
            fan_name = components.get("fan_name", "")
            fcu_name = components.get("fcu_name", "")

            if fan_name:
                self.api.exchange.request_variable(
                    self.state, "Fan Air Mass Flow Rate", fan_name
                )
                self.api.exchange.request_variable(
                    self.state, "Fan Electricity Rate", fan_name
                )

            if fcu_name:
                self.api.exchange.request_variable(
                    self.state, "Fan Coil Speed Ratio", fcu_name
                )
                self.api.exchange.request_variable(
                    self.state, "Fan Coil Part Load Ratio", fcu_name
                )

    # ----------------------------------------------------------------------
    # Data access
    # ----------------------------------------------------------------------

    def occupancy_count(self, timestamp: pd.Timestamp, room: str) -> float:
        if timestamp not in self.occupancy.index:
            return 0.0
        column = self.occ_columns[room]
        return max(0.0, safe_float(self.occupancy.loc[timestamp, column], 0.0))

    def actuator_value(self, state: Any, handle: int) -> float:
        if handle == -1:
            return float("nan")
        getter = getattr(self.api.exchange, "get_actuator_value", None)
        if getter is None:
            return float("nan")
        try:
            return safe_float(getter(state, handle))
        except Exception:
            return float("nan")

    def variable_value(self, state: Any, handle: int) -> float:
        if handle == -1:
            return float("nan")
        try:
            return safe_float(self.api.exchange.get_variable_value(state, handle))
        except Exception:
            return float("nan")

    def internal_value(self, state: Any, handle: int) -> float:
        if handle == -1:
            return float("nan")
        try:
            return safe_float(
                self.api.exchange.get_internal_variable_value(state, handle)
            )
        except Exception:
            return float("nan")

    # ----------------------------------------------------------------------
    # Handle setup
    # ----------------------------------------------------------------------

    def add_handle_record(
        self,
        room: str,
        category: str,
        api_type: str,
        api_name: str,
        key: str,
        handle: int,
        required: bool,
    ) -> None:
        self.report.handle_records.append(
            HandleRecord(
                room=room,
                category=category,
                api_type=api_type,
                api_name=api_name,
                key=key,
                handle=int(handle),
                required=bool(required),
                status="PASS" if handle != -1 else ("FAIL" if required else "OPTIONAL_MISSING"),
            )
        )

    def setup_handles(self, state: Any) -> None:
        if self.report.handles_ready:
            return
        if not self.api.exchange.api_data_fully_ready(state):
            return

        print("\n" + "=" * 78)
        print("EnergyPlus API data ready - acquiring preflight handles")
        print("=" * 78)

        api_csv = self.api.exchange.list_available_api_data_csv(state).decode(
            "utf-8", errors="replace"
        )
        (self.output_dir / "available_api_data.csv").write_text(
            api_csv, encoding="utf-8"
        )
        self.report.api_dictionary_saved = True

        api_rows = parse_api_csv(api_csv)
        fan_candidates = api_fan_candidates(api_rows)

        print(
            "Fan API candidates: "
            f"actuators={len(fan_candidates['actuator'])}, "
            f"internal_vars={len(fan_candidates['internal_variable'])}, "
            f"outputs={len(fan_candidates['output_variable'])}"
        )

        for room, info in ROOMS.items():
            zone = str(info["zone"])
            zone_number = int(info["zone_number"])
            people_object = str(info["people_object"])
            vent_schedule = str(info["vent_schedule"])
            vent_object = str(info["vent_object"])
            availability_schedule = f"FCU Zone {zone_number} Availability"

            # Required state feedback
            self.temp_handles[room] = self.api.exchange.get_variable_handle(
                state, "Zone Mean Air Temperature", zone
            )
            self.rh_handles[room] = self.api.exchange.get_variable_handle(
                state, "Zone Air Relative Humidity", zone
            )
            self.mrt_handles[room] = self.api.exchange.get_variable_handle(
                state, "Zone Mean Radiant Temperature", zone
            )
            self.people_output_handles[room] = self.api.exchange.get_variable_handle(
                state, "Zone People Occupant Count", zone
            )
            self.actual_heat_handles[room] = self.api.exchange.get_variable_handle(
                state, "Zone Thermostat Heating Setpoint Temperature", zone
            )
            self.actual_cool_handles[room] = self.api.exchange.get_variable_handle(
                state, "Zone Thermostat Cooling Setpoint Temperature", zone
            )

            # Required actuators
            self.people_actuators[room] = self.api.exchange.get_actuator_handle(
                state, "People", "Number of People", people_object
            )
            self.vent_actuators[room] = self.api.exchange.get_actuator_handle(
                state, "Schedule:Constant", "Schedule Value", vent_schedule
            )
            self.heat_actuators[room] = self.api.exchange.get_actuator_handle(
                state, "Zone Temperature Control", "Heating Setpoint", zone
            )
            self.cool_actuators[room] = self.api.exchange.get_actuator_handle(
                state, "Zone Temperature Control", "Cooling Setpoint", zone
            )
            self.availability_actuators[room] = self.api.exchange.get_actuator_handle(
                state, "Schedule:Constant", "Schedule Value", availability_schedule
            )

            # Ventilation-flow output key discovery
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

            # FCU / fan discovery
            components = self.fcu_components.get(zone, {})
            parsed_fan_name = components.get("fan_name", "")
            fcu_name = components.get("fcu_name", "")

            fan_name = resolve_name(
                parsed_fan_name,
                zone_number,
                fan_candidates["actuator"],
            )
            if not fan_name:
                # If the actuator list did not match, try the output/internal keys.
                union_candidates = sorted(
                    set(
                        fan_candidates["actuator"]
                        + fan_candidates["internal_variable"]
                        + fan_candidates["output_variable"]
                    )
                )
                fan_name = resolve_name(
                    parsed_fan_name,
                    zone_number,
                    union_candidates,
                )

            self.fan_names[room] = fan_name
            self.fcu_names[room] = fcu_name

            self.fan_actuators[room] = (
                self.api.exchange.get_actuator_handle(
                    state, "Fan", "Fan Air Mass Flow Rate", fan_name
                )
                if fan_name
                else -1
            )
            self.fan_max_handles[room] = (
                self.api.exchange.get_internal_variable_handle(
                    state, "Fan Maximum Mass Flow Rate", fan_name
                )
                if fan_name
                else -1
            )
            self.fan_flow_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state, "Fan Air Mass Flow Rate", fan_name
                )
                if fan_name
                else -1
            )
            self.fan_power_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state, "Fan Electricity Rate", fan_name
                )
                if fan_name
                else -1
            )
            self.fcu_speed_ratio_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state, "Fan Coil Speed Ratio", fcu_name
                )
                if fcu_name
                else -1
            )
            self.fcu_plr_handles[room] = (
                self.api.exchange.get_variable_handle(
                    state, "Fan Coil Part Load Ratio", fcu_name
                )
                if fcu_name
                else -1
            )

            # Handle report - required
            required_items = [
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

            for category, api_type, api_name, key, handle in required_items:
                self.add_handle_record(
                    room,
                    category,
                    api_type,
                    api_name,
                    key,
                    handle,
                    required=True,
                )

            # At least one of current/standard ventilation-flow outputs is required.
            vent_feedback_handle = (
                self.vent_current_handles[room]
                if self.vent_current_handles[room] != -1
                else self.vent_standard_handles[room]
            )
            vent_feedback_key = current_key or standard_key
            vent_feedback_name = (
                "Zone Ventilation Current Density Volume Flow Rate"
                if self.vent_current_handles[room] != -1
                else "Zone Ventilation Standard Density Volume Flow Rate"
            )
            self.add_handle_record(
                room,
                "ventilation_flow_feedback",
                "OutputVariable",
                vent_feedback_name,
                vent_feedback_key,
                vent_feedback_handle,
                required=True,
            )

            # Fan control is deliberately optional until proven.
            optional_fan_items = [
                (
                    "fan_mass_flow_actuator",
                    "Actuator",
                    "Fan / Fan Air Mass Flow Rate",
                    fan_name,
                    self.fan_actuators[room],
                ),
                (
                    "fan_max_mass_flow",
                    "InternalVariable",
                    "Fan Maximum Mass Flow Rate",
                    fan_name,
                    self.fan_max_handles[room],
                ),
                (
                    "fan_mass_flow_feedback",
                    "OutputVariable",
                    "Fan Air Mass Flow Rate",
                    fan_name,
                    self.fan_flow_handles[room],
                ),
                (
                    "fan_power_feedback",
                    "OutputVariable",
                    "Fan Electricity Rate",
                    fan_name,
                    self.fan_power_handles[room],
                ),
                (
                    "fcu_speed_ratio_feedback",
                    "OutputVariable",
                    "Fan Coil Speed Ratio",
                    fcu_name,
                    self.fcu_speed_ratio_handles[room],
                ),
                (
                    "fcu_part_load_ratio_feedback",
                    "OutputVariable",
                    "Fan Coil Part Load Ratio",
                    fcu_name,
                    self.fcu_plr_handles[room],
                ),
            ]

            for category, api_type, api_name, key, handle in optional_fan_items:
                self.add_handle_record(
                    room,
                    category,
                    api_type,
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
                f"VentFlow={vent_feedback_handle}, "
                f"FanAct={self.fan_actuators[room]}, "
                f"Fan={fan_name or 'NOT_RESOLVED'}"
            )

        self.report.handles_ready = True
        self.save_handle_report()

    def save_handle_report(self) -> None:
        rows = [
            {
                "room": record.room,
                "category": record.category,
                "api_type": record.api_type,
                "api_name": record.api_name,
                "key": record.key,
                "handle": record.handle,
                "required": record.required,
                "status": record.status,
            }
            for record in self.report.handle_records
        ]
        pd.DataFrame(rows).to_csv(
            self.output_dir / "handle_report.csv", index=False
        )

    # ----------------------------------------------------------------------
    # Callback safety
    # ----------------------------------------------------------------------

    def record_callback_error(self, callback_name: str, exc: BaseException) -> None:
        message = (
            f"{callback_name}: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        self.report.callback_errors.append(message)
        if len(self.report.callback_errors) <= 3:
            print("\nCALLBACK ERROR:")
            print(message)

    # ----------------------------------------------------------------------
    # Zone-timestep loads: People + ventilation
    # ----------------------------------------------------------------------

    def apply_people_and_ventilation(self, state: Any) -> None:
        try:
            self.setup_handles(state)
            if not self.report.handles_ready:
                return
            if self.api.exchange.warmup_flag(state):
                return

            timestamp = timestamp_from_energyplus(self.api, state)
            if not in_control_period(timestamp):
                return

            for room, info in ROOMS.items():
                occupancy = self.occupancy_count(timestamp, room)
                capacity = float(info["capacity"])
                vent_fraction = clamp(occupancy / capacity, 0.0, 1.0)

                people_handle = self.people_actuators.get(room, -1)
                vent_handle = self.vent_actuators.get(room, -1)

                if people_handle != -1:
                    self.api.exchange.set_actuator_value(
                        state, people_handle, float(occupancy)
                    )
                if vent_handle != -1:
                    self.api.exchange.set_actuator_value(
                        state, vent_handle, float(vent_fraction)
                    )

                self.current_commands[room]["occupancy"] = float(occupancy)
                self.current_commands[room]["vent_fraction"] = float(vent_fraction)

        except Exception as exc:
            self.record_callback_error("apply_people_and_ventilation", exc)

    # ----------------------------------------------------------------------
    # System-timestep controls: thermostat + FCU availability
    # ----------------------------------------------------------------------

    def control_callback(self, state: Any) -> None:
        try:
            self.setup_handles(state)
            if not self.report.handles_ready:
                return
            if self.api.exchange.warmup_flag(state):
                return

            timestamp = timestamp_from_energyplus(self.api, state)
            if not in_control_period(timestamp):
                return

            phase = phase_for_timestamp(timestamp)

            for room in ROOM_ORDER:
                if phase == "thermostat_probe":
                    heat_C, cool_C = THERMOSTAT_PROBE[room]
                    availability = 1.0

                elif phase == "availability_off_probe":
                    heat_C, cool_C = PROBE_HEAT_C, PROBE_COOL_C
                    availability = 0.0

                elif phase in (
                    "availability_on_probe",
                    "availability_recovery_probe",
                    "fan_low_probe",
                    "fan_high_probe",
                ):
                    heat_C, cool_C = PROBE_HEAT_C, PROBE_COOL_C
                    availability = 1.0

                else:
                    heat_C, cool_C = fixed_setpoints(timestamp)
                    availability = 1.0

                if self.heat_actuators.get(room, -1) != -1:
                    self.api.exchange.set_actuator_value(
                        state, self.heat_actuators[room], float(heat_C)
                    )
                if self.cool_actuators.get(room, -1) != -1:
                    self.api.exchange.set_actuator_value(
                        state, self.cool_actuators[room], float(cool_C)
                    )
                if self.availability_actuators.get(room, -1) != -1:
                    self.api.exchange.set_actuator_value(
                        state,
                        self.availability_actuators[room],
                        float(availability),
                    )

                self.current_commands[room]["availability"] = availability
                self.current_commands[room]["heat_C"] = heat_C
                self.current_commands[room]["cool_C"] = cool_C

                if phase == "fan_low_probe":
                    self.current_commands[room]["fan_ratio"] = FAN_LOW_RATIO
                elif phase == "fan_high_probe":
                    self.current_commands[room]["fan_ratio"] = FAN_HIGH_RATIO
                else:
                    self.current_commands[room]["fan_ratio"] = float("nan")

        except Exception as exc:
            self.record_callback_error("control_callback", exc)

    # ----------------------------------------------------------------------
    # Inside HVAC iterations: optional direct fan probes
    # ----------------------------------------------------------------------

    def reset_fan_actuator(self, state: Any, handle: int) -> None:
        if handle == -1:
            return
        resetter = getattr(self.api.exchange, "reset_actuator", None)
        if resetter is not None:
            try:
                resetter(state, handle)
            except Exception:
                pass

    def fan_callback(self, state: Any) -> None:
        try:
            if not self.report.handles_ready:
                return
            if self.api.exchange.warmup_flag(state):
                return

            timestamp = timestamp_from_energyplus(self.api, state)
            if not in_control_period(timestamp):
                return

            phase = phase_for_timestamp(timestamp)
            fan_override = phase in ("fan_low_probe", "fan_high_probe")

            if not fan_override:
                if self.last_fan_override_active:
                    for room in ROOM_ORDER:
                        self.reset_fan_actuator(
                            state, self.fan_actuators.get(room, -1)
                        )
                self.last_fan_override_active = False
                return

            target_ratio = (
                FAN_LOW_RATIO if phase == "fan_low_probe" else FAN_HIGH_RATIO
            )

            for room in ROOM_ORDER:
                actuator = self.fan_actuators.get(room, -1)
                max_handle = self.fan_max_handles.get(room, -1)

                if actuator == -1 or max_handle == -1:
                    continue

                max_mass_flow = self.internal_value(state, max_handle)
                if not is_finite(max_mass_flow) or max_mass_flow <= 0:
                    continue

                self.api.exchange.set_actuator_value(
                    state,
                    actuator,
                    float(target_ratio * max_mass_flow),
                )

            self.last_fan_override_active = True

        except Exception as exc:
            self.record_callback_error("fan_callback", exc)

    # ----------------------------------------------------------------------
    # End-of-system-timestep logging
    # ----------------------------------------------------------------------

    def logging_callback(self, state: Any) -> None:
        try:
            if not self.report.handles_ready:
                return
            if self.api.exchange.warmup_flag(state):
                return

            interval_start, interval_end = energyplus_interval_times(
                self.api, state
            )
            if not in_control_period(interval_start):
                return

            # All data/controller semantics are interval-start aligned.
            timestamp = interval_start
            phase = phase_for_timestamp(interval_start)

            for room, info in ROOMS.items():
                command = self.current_commands[room]

                temp = self.variable_value(state, self.temp_handles.get(room, -1))
                rh = self.variable_value(state, self.rh_handles.get(room, -1))
                mrt = self.variable_value(state, self.mrt_handles.get(room, -1))
                people_actual = self.variable_value(
                    state, self.people_output_handles.get(room, -1)
                )
                heat_actual = self.variable_value(
                    state, self.actual_heat_handles.get(room, -1)
                )
                cool_actual = self.variable_value(
                    state, self.actual_cool_handles.get(room, -1)
                )
                vent_current = self.variable_value(
                    state, self.vent_current_handles.get(room, -1)
                )
                vent_standard = self.variable_value(
                    state, self.vent_standard_handles.get(room, -1)
                )

                fan_flow = self.variable_value(
                    state, self.fan_flow_handles.get(room, -1)
                )
                fan_power = self.variable_value(
                    state, self.fan_power_handles.get(room, -1)
                )
                fan_max = self.internal_value(
                    state, self.fan_max_handles.get(room, -1)
                )
                fan_ratio_actual = (
                    fan_flow / fan_max
                    if is_finite(fan_flow)
                    and is_finite(fan_max)
                    and fan_max > 0
                    else float("nan")
                )

                fcu_speed_ratio = self.variable_value(
                    state, self.fcu_speed_ratio_handles.get(room, -1)
                )
                fcu_plr = self.variable_value(
                    state, self.fcu_plr_handles.get(room, -1)
                )

                occupancy_cmd = float(command["occupancy"])
                vent_fraction_cmd = float(command["vent_fraction"])
                expected_vent_flow = (
                    VENT_FLOW_PER_PERSON_M3_S * occupancy_cmd
                )

                self.report.trace_rows.append(
                    {
                        # "timestamp" is retained as a backward-compatible alias
                        # for interval_start. New analysis should use
                        # interval_start explicitly.
                        "timestamp": interval_start,
                        "interval_start": interval_start,
                        "interval_end": interval_end,
                        "phase": phase,
                        "room": room,
                        "zone": info["zone"],
                        "occupancy_command": occupancy_cmd,
                        "people_actual": people_actual,
                        "people_tracking_error": (
                            people_actual - occupancy_cmd
                            if is_finite(people_actual)
                            else float("nan")
                        ),
                        "vent_fraction_command": vent_fraction_cmd,
                        "vent_actuator_readback": self.actuator_value(
                            state, self.vent_actuators.get(room, -1)
                        ),
                        "vent_expected_m3_s": expected_vent_flow,
                        "vent_current_density_m3_s": vent_current,
                        "vent_standard_density_m3_s": vent_standard,
                        "zone_temp_C": temp,
                        "zone_RH_percent": rh,
                        "zone_MRT_C": mrt,
                        "heating_setpoint_command_C": command["heat_C"],
                        "cooling_setpoint_command_C": command["cool_C"],
                        "actual_heating_setpoint_C": heat_actual,
                        "actual_cooling_setpoint_C": cool_actual,
                        "fcu_availability_command": command["availability"],
                        "fcu_availability_actuator_readback": self.actuator_value(
                            state, self.availability_actuators.get(room, -1)
                        ),
                        "fan_ratio_command": command["fan_ratio"],
                        "fan_name": self.fan_names.get(room, ""),
                        "fan_actuator_available": (
                            self.fan_actuators.get(room, -1) != -1
                            and self.fan_max_handles.get(room, -1) != -1
                        ),
                        "fan_mass_flow_kg_s": fan_flow,
                        "fan_max_mass_flow_kg_s": fan_max,
                        "fan_ratio_actual": fan_ratio_actual,
                        "fan_power_W": fan_power,
                        "fcu_speed_ratio": fcu_speed_ratio,
                        "fcu_part_load_ratio": fcu_plr,
                    }
                )

        except Exception as exc:
            self.record_callback_error("logging_callback", exc)

    # ----------------------------------------------------------------------
    # Execution
    # ----------------------------------------------------------------------

    def run(self) -> int:
        if not self.idf_path.exists():
            raise FileNotFoundError(f"Generated IDF not found: {self.idf_path}")
        if not self.epw_path.exists():
            raise FileNotFoundError(f"EPW not found: {self.epw_path}")

        self.api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
            self.state,
            self.apply_people_and_ventilation,
        )
        self.api.runtime.callback_begin_system_timestep_before_predictor(
            self.state,
            self.control_callback,
        )
        self.api.runtime.callback_inside_system_iteration_loop(
            self.state,
            self.fan_callback,
        )
        self.api.runtime.callback_end_system_timestep_after_hvac_reporting(
            self.state,
            self.logging_callback,
        )

        command = [
            "-w",
            str(self.epw_path),
            "-d",
            str(self.energyplus_output_dir),
            str(self.idf_path),
        ]

        print("\n" + "=" * 78)
        print("Running EnergyPlus API preflight")
        print("=" * 78)
        print(f"EnergyPlus : {self.energyplus_root}")
        print(f"IDF        : {self.idf_path}")
        print(f"EPW        : {self.epw_path}")
        print(f"Occupancy  : {self.occupancy_path}")
        print(f"Results    : {self.output_dir}")
        print("\nProbe windows:")
        for name, (start, end) in PROBE_WINDOWS.items():
            print(f"  {name:30s} {start} -> {end}")
        print()

        status = self.api.runtime.run_energyplus(self.state, command)

        # Release any optional fan overrides.
        for room in ROOM_ORDER:
            self.reset_fan_actuator(
                self.state, self.fan_actuators.get(room, -1)
            )

        print(f"\nEnergyPlus exit status: {status}")
        return int(status)

    # ----------------------------------------------------------------------
    # Analysis / PASS-FAIL reporting
    # ----------------------------------------------------------------------

    @staticmethod
    def median_numeric(frame: pd.DataFrame, column: str) -> float:
        if column not in frame.columns or frame.empty:
            return float("nan")
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(series.median()) if not series.empty else float("nan")

    @staticmethod
    def mean_abs_error(frame: pd.DataFrame, column: str) -> float:
        if column not in frame.columns or frame.empty:
            return float("nan")
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(series.abs().mean()) if not series.empty else float("nan")

    def trace_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.report.trace_rows)
        if df.empty:
            return df

        for column in ("timestamp", "interval_start", "interval_end"):
            if column in df.columns:
                df[column] = pd.to_datetime(df[column])

        # System timesteps can repeat the same 5-minute interval. Keep every
        # raw row in preflight_trace.csv; analysis later retains the final
        # reported row for each room/interval_start.
        return df

    def analysis_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return raw

        if "interval_start" not in raw.columns:
            raise RuntimeError(
                "preflight trace is missing interval_start; time alignment "
                "cannot be verified."
            )

        return (
            raw.sort_values(["interval_start", "room", "interval_end"])
            .drop_duplicates(
                subset=["interval_start", "room"],
                keep="last",
            )
            .sort_values(["interval_start", "room"])
            .reset_index(drop=True)
        )

    def make_check(
        self,
        category: str,
        room: str,
        required: bool,
        status: str,
        metric: str,
        value: Any,
        criterion: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        return {
            "category": category,
            "room": room,
            "required": required,
            "status": status,
            "metric": metric,
            "value": value,
            "criterion": criterion,
            "notes": notes,
        }

    def evaluate(self, energyplus_status: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        raw = self.trace_dataframe()
        if not raw.empty:
            raw.to_csv(self.output_dir / "preflight_trace.csv", index=False)

        frame = self.analysis_frame(raw)
        checks: List[Dict[str, Any]] = []

        # 1) Simulation and callback integrity
        checks.append(
            self.make_check(
                "simulation",
                "ALL",
                True,
                "PASS" if energyplus_status == 0 else "FAIL",
                "energyplus_exit_status",
                energyplus_status,
                "must equal 0",
            )
        )
        checks.append(
            self.make_check(
                "callbacks",
                "ALL",
                True,
                "PASS" if not self.report.callback_errors else "FAIL",
                "callback_error_count",
                len(self.report.callback_errors),
                "must equal 0",
            )
        )

        # 2) Required handle presence
        required_handle_records = [
            record for record in self.report.handle_records if record.required
        ]
        for record in required_handle_records:
            checks.append(
                self.make_check(
                    "required_handle",
                    record.room,
                    True,
                    "PASS" if record.handle != -1 else "FAIL",
                    record.category,
                    record.handle,
                    "handle != -1",
                    f"{record.api_type}: {record.api_name}; key={record.key}",
                )
            )

        # 3) Optional fan handle characterization
        for room in ROOM_ORDER:
            fan_core = (
                self.fan_actuators.get(room, -1) != -1
                and self.fan_max_handles.get(room, -1) != -1
                and self.fan_flow_handles.get(room, -1) != -1
            )
            checks.append(
                self.make_check(
                    "fan_capability",
                    room,
                    False,
                    "PASS" if fan_core else "OPTIONAL_UNAVAILABLE",
                    "direct_fan_control_core_handles",
                    int(fan_core),
                    "actuator, max-flow internal variable and flow output all available",
                    f"fan={self.fan_names.get(room, '')}",
                )
            )

        if frame.empty:
            self.coverage_stats = {
                "zone_timestep_minutes": ZONE_TIMESTEP_MINUTES,
                "expected_intervals_per_room": EXPECTED_INTERVALS_PER_ROOM,
                "expected_analysis_rows": EXPECTED_ANALYSIS_ROWS,
                "analysis_rows": 0,
                "unique_interval_starts_per_room": {
                    room: 0 for room in ROOM_ORDER
                },
                "first_interval_start": None,
                "last_interval_start": None,
                "expected_first_interval_start": str(EXPECTED_FIRST_INTERVAL_START),
                "expected_last_interval_start": str(EXPECTED_LAST_INTERVAL_START),
            }
            checks.append(
                self.make_check(
                    "trace",
                    "ALL",
                    True,
                    "FAIL",
                    "analysis_rows",
                    0,
                    "must be > 0",
                    "No RunPeriod trace rows were captured.",
                )
            )
            checks_df = pd.DataFrame(checks)
            summary = self.build_summary(
                checks_df, energyplus_status, raw_rows=0, analysis_rows=0
            )
            return checks_df, summary

        # 4) Exact 5-minute time alignment and coverage
        expected_index = pd.date_range(
            EXPECTED_FIRST_INTERVAL_START,
            EXPECTED_LAST_INTERVAL_START,
            freq=f"{ZONE_TIMESTEP_MINUTES}min",
        )

        analysis_rows_ok = len(frame) == EXPECTED_ANALYSIS_ROWS
        checks.append(
            self.make_check(
                "time_alignment",
                "ALL",
                True,
                "PASS" if analysis_rows_ok else "FAIL",
                "analysis_rows",
                int(len(frame)),
                f"must equal {EXPECTED_ANALYSIS_ROWS}",
                (
                    f"expected={EXPECTED_ANALYSIS_ROWS}; "
                    f"rooms={len(ROOM_ORDER)}; "
                    f"intervals_per_room={EXPECTED_INTERVALS_PER_ROOM}"
                ),
            )
        )

        unique_counts: Dict[str, int] = {}
        first_by_room: Dict[str, Optional[pd.Timestamp]] = {}
        last_by_room: Dict[str, Optional[pd.Timestamp]] = {}

        for room in ROOM_ORDER:
            r = frame[frame["room"] == room].copy()
            starts = (
                pd.to_datetime(r["interval_start"], errors="coerce")
                .dropna()
                .sort_values()
            )

            unique_count = int(starts.nunique())
            first_start = starts.min() if not starts.empty else None
            last_start = starts.max() if not starts.empty else None

            unique_counts[room] = unique_count
            first_by_room[room] = first_start
            last_by_room[room] = last_start

            actual_unique = pd.DatetimeIndex(starts.drop_duplicates())
            missing = expected_index.difference(actual_unique)
            extras = actual_unique.difference(expected_index)

            room_ok = (
                unique_count == EXPECTED_INTERVALS_PER_ROOM
                and first_start == EXPECTED_FIRST_INTERVAL_START
                and last_start == EXPECTED_LAST_INTERVAL_START
                and len(missing) == 0
                and len(extras) == 0
            )

            checks.append(
                self.make_check(
                    "time_alignment",
                    room,
                    True,
                    "PASS" if room_ok else "FAIL",
                    "unique_interval_starts",
                    unique_count,
                    (
                        f"must equal {EXPECTED_INTERVALS_PER_ROOM}; "
                        f"first={EXPECTED_FIRST_INTERVAL_START}; "
                        f"last={EXPECTED_LAST_INTERVAL_START}; no gaps/extras"
                    ),
                    (
                        f"first={first_start}; last={last_start}; "
                        f"missing={len(missing)}; extras={len(extras)}"
                    ),
                )
            )

        global_first = pd.to_datetime(
            frame["interval_start"], errors="coerce"
        ).min()
        global_last = pd.to_datetime(
            frame["interval_start"], errors="coerce"
        ).max()

        first_ok = global_first == EXPECTED_FIRST_INTERVAL_START
        last_ok = global_last == EXPECTED_LAST_INTERVAL_START

        checks.append(
            self.make_check(
                "time_alignment",
                "ALL",
                True,
                "PASS" if first_ok else "FAIL",
                "first_interval_start",
                str(global_first),
                f"must equal {EXPECTED_FIRST_INTERVAL_START}",
            )
        )
        checks.append(
            self.make_check(
                "time_alignment",
                "ALL",
                True,
                "PASS" if last_ok else "FAIL",
                "last_interval_start",
                str(global_last),
                f"must equal {EXPECTED_LAST_INTERVAL_START}",
            )
        )

        self.coverage_stats = {
            "zone_timestep_minutes": ZONE_TIMESTEP_MINUTES,
            "expected_intervals_per_room": EXPECTED_INTERVALS_PER_ROOM,
            "expected_analysis_rows": EXPECTED_ANALYSIS_ROWS,
            "analysis_rows": int(len(frame)),
            "unique_interval_starts_per_room": unique_counts,
            "first_interval_start_per_room": {
                room: str(value) if value is not None else None
                for room, value in first_by_room.items()
            },
            "last_interval_start_per_room": {
                room: str(value) if value is not None else None
                for room, value in last_by_room.items()
            },
            "first_interval_start": str(global_first),
            "last_interval_start": str(global_last),
            "expected_first_interval_start": str(EXPECTED_FIRST_INTERVAL_START),
            "expected_last_interval_start": str(EXPECTED_LAST_INTERVAL_START),
        }

        # 5) People tracking
        for room in ROOM_ORDER:
            r = frame[frame["room"] == room].copy()
            errors = pd.to_numeric(
                r["people_tracking_error"], errors="coerce"
            ).dropna()

            mae = float(errors.abs().mean()) if not errors.empty else float("nan")
            max_abs = (
                float(errors.abs().max()) if not errors.empty else float("nan")
            )
            passed = (
                is_finite(mae)
                and is_finite(max_abs)
                and mae <= 0.01
                and max_abs <= 0.05
            )
            checks.append(
                self.make_check(
                    "people_tracking",
                    room,
                    True,
                    "PASS" if passed else "FAIL",
                    "MAE_people",
                    mae,
                    "MAE <= 0.01 person and max abs error <= 0.05 person",
                    f"max_abs_error={max_abs}",
                )
            )

        # 6) Ventilation schedule actuator tracking
        for room in ROOM_ORDER:
            r = frame[frame["room"] == room].copy()
            readback = pd.to_numeric(
                r["vent_actuator_readback"], errors="coerce"
            )
            command = pd.to_numeric(
                r["vent_fraction_command"], errors="coerce"
            )
            valid = readback.notna() & command.notna()

            if valid.any():
                err = (readback[valid] - command[valid]).abs()
                mae = float(err.mean())
                max_abs = float(err.max())
                passed = mae <= 1e-6 and max_abs <= 1e-5
                status = "PASS" if passed else "FAIL"
                notes = f"max_abs_error={max_abs}"
            else:
                # Some pyenergyplus builds do not expose get_actuator_value.
                # Physical ventilation tracking below remains the decisive test.
                mae = float("nan")
                status = "NOT_EXPOSED"
                notes = (
                    "get_actuator_value unavailable; use physical ventilation-flow "
                    "tracking as the required verification."
                )

            checks.append(
                self.make_check(
                    "ventilation_actuator_readback",
                    room,
                    False,
                    status,
                    "MAE_schedule_fraction",
                    mae,
                    "if exposed, actuator readback should equal command",
                    notes,
                )
            )

        # 7) Physical ventilation tracking
        for room in ROOM_ORDER:
            r = frame[frame["room"] == room].copy()

            actual = pd.to_numeric(
                r["vent_current_density_m3_s"], errors="coerce"
            )
            if actual.notna().sum() == 0:
                actual = pd.to_numeric(
                    r["vent_standard_density_m3_s"], errors="coerce"
                )

            expected = pd.to_numeric(
                r["vent_expected_m3_s"], errors="coerce"
            )
            valid = actual.notna() & expected.notna()

            if not valid.any():
                checks.append(
                    self.make_check(
                        "ventilation_physical_tracking",
                        room,
                        True,
                        "FAIL",
                        "valid_rows",
                        0,
                        "must have ventilation feedback rows",
                        "No usable ventilation-flow feedback.",
                    )
                )
                continue

            a = actual[valid]
            e = expected[valid]
            abs_error = (a - e).abs()

            # Current-density volume flow can differ modestly from the nominal
            # design volume because of air density. Use a conservative combined
            # absolute/relative tolerance for the physical response check.
            tolerance = 0.001 + 0.20 * e.abs()
            pass_fraction = float((abs_error <= tolerance).mean())

            # Also require that occupied flow is larger than vacant flow when
            # both states exist; this proves the schedule affects physics.
            occupied = a[e > 1e-8]
            vacant = a[e <= 1e-8]
            occ_median = (
                float(occupied.median()) if not occupied.empty else float("nan")
            )
            vac_median = (
                float(vacant.median()) if not vacant.empty else float("nan")
            )
            response_ok = True
            if not occupied.empty and not vacant.empty:
                response_ok = occ_median > vac_median + 1e-5

            passed = pass_fraction >= 0.95 and response_ok

            checks.append(
                self.make_check(
                    "ventilation_physical_tracking",
                    room,
                    True,
                    "PASS" if passed else "FAIL",
                    "fraction_within_tolerance",
                    pass_fraction,
                    ">= 0.95 within 0.001 m3/s + 20% expected flow; occupied > vacant",
                    (
                        f"occupied_median={occ_median}; "
                        f"vacant_median={vac_median}"
                    ),
                )
            )

        # 8) Independent thermostat readback
        probe = frame[frame["phase"] == "thermostat_probe"].copy()
        for room in ROOM_ORDER:
            r = probe[probe["room"] == room]
            cmd_heat, cmd_cool = THERMOSTAT_PROBE[room]
            actual_heat = self.median_numeric(r, "actual_heating_setpoint_C")
            actual_cool = self.median_numeric(r, "actual_cooling_setpoint_C")

            heat_error = (
                abs(actual_heat - cmd_heat)
                if is_finite(actual_heat)
                else float("nan")
            )
            cool_error = (
                abs(actual_cool - cmd_cool)
                if is_finite(actual_cool)
                else float("nan")
            )
            passed = (
                is_finite(heat_error)
                and is_finite(cool_error)
                and heat_error <= 0.10
                and cool_error <= 0.10
            )

            checks.append(
                self.make_check(
                    "thermostat_independent_actuation",
                    room,
                    True,
                    "PASS" if passed else "FAIL",
                    "max_setpoint_readback_error_C",
                    max(heat_error, cool_error)
                    if is_finite(heat_error) and is_finite(cool_error)
                    else float("nan"),
                    "<= 0.10 C for both heating and cooling readback",
                    (
                        f"command=({cmd_heat},{cmd_cool}); "
                        f"actual_median=({actual_heat},{actual_cool})"
                    ),
                )
            )

        # Global uniqueness sanity check.
        median_cools = []
        for room in ROOM_ORDER:
            r = probe[probe["room"] == room]
            value = self.median_numeric(r, "actual_cooling_setpoint_C")
            if is_finite(value):
                median_cools.append(round(value, 2))

        checks.append(
            self.make_check(
                "thermostat_independence_global",
                "ALL",
                True,
                "PASS" if len(set(median_cools)) == len(ROOM_ORDER) else "FAIL",
                "unique_actual_cooling_setpoints",
                len(set(median_cools)),
                f"must equal {len(ROOM_ORDER)} during unique-setpoint probe",
                f"values={median_cools}",
            )
        )

        # 9) FCU availability physical OFF/ON response
        phase_on = frame[
            frame["phase"].isin(
                ["availability_on_probe", "availability_recovery_probe"]
            )
        ]
        phase_off = frame[frame["phase"] == "availability_off_probe"]

        for room in ROOM_ORDER:
            on = phase_on[phase_on["room"] == room]
            off = phase_off[phase_off["room"] == room]

            on_fan = self.median_numeric(on, "fan_mass_flow_kg_s")
            off_fan = self.median_numeric(off, "fan_mass_flow_kg_s")
            on_plr = self.median_numeric(on, "fcu_part_load_ratio")
            off_plr = self.median_numeric(off, "fcu_part_load_ratio")

            # Prefer fan flow; otherwise use FCU PLR.
            if is_finite(on_fan) and is_finite(off_fan):
                on_signal = on_fan
                off_signal = off_fan
                signal_name = "fan_mass_flow_kg_s"
                activity_threshold = 1e-4
            elif is_finite(on_plr) and is_finite(off_plr):
                on_signal = on_plr
                off_signal = off_plr
                signal_name = "fcu_part_load_ratio"
                activity_threshold = 1e-3
            else:
                checks.append(
                    self.make_check(
                        "fcu_availability_physical",
                        room,
                        True,
                        "FAIL",
                        "response_signal",
                        float("nan"),
                        "must have fan-flow or FCU-PLR feedback",
                        "No usable FCU response output.",
                    )
                )
                continue

            on_active = on_signal > activity_threshold
            off_near_zero = abs(off_signal) <= max(
                activity_threshold, 0.10 * abs(on_signal)
            )
            passed = on_active and off_near_zero

            checks.append(
                self.make_check(
                    "fcu_availability_physical",
                    room,
                    True,
                    "PASS" if passed else "FAIL",
                    "off_to_on_response",
                    off_signal / on_signal
                    if on_signal != 0
                    else float("nan"),
                    "forced-load ON signal > threshold and OFF <= 10% of ON",
                    (
                        f"signal={signal_name}; "
                        f"ON_median={on_signal}; OFF_median={off_signal}"
                    ),
                )
            )

        # 10) Optional fan low/high physical probe
        fan_low = frame[frame["phase"] == "fan_low_probe"]
        fan_high = frame[frame["phase"] == "fan_high_probe"]

        for room in ROOM_ORDER:
            core_available = (
                self.fan_actuators.get(room, -1) != -1
                and self.fan_max_handles.get(room, -1) != -1
                and self.fan_flow_handles.get(room, -1) != -1
            )

            if not core_available:
                checks.append(
                    self.make_check(
                        "fan_direct_control_physical",
                        room,
                        False,
                        "OPTIONAL_UNAVAILABLE",
                        "fan_control",
                        0,
                        "optional; does not fail the preflight",
                        "Direct fan mass-flow control will not be used unless verified.",
                    )
                )
                continue

            low_r = fan_low[fan_low["room"] == room]
            high_r = fan_high[fan_high["room"] == room]

            low_ratio = self.median_numeric(low_r, "fan_ratio_actual")
            high_ratio = self.median_numeric(high_r, "fan_ratio_actual")

            low_ok = is_finite(low_ratio) and abs(low_ratio - FAN_LOW_RATIO) <= 0.20
            high_ok = is_finite(high_ratio) and abs(high_ratio - FAN_HIGH_RATIO) <= 0.20
            separation_ok = (
                is_finite(low_ratio)
                and is_finite(high_ratio)
                and high_ratio - low_ratio >= 0.30
            )
            passed = low_ok and high_ok and separation_ok

            checks.append(
                self.make_check(
                    "fan_direct_control_physical",
                    room,
                    False,
                    "PASS" if passed else "OPTIONAL_FAIL",
                    "low_high_actual_ratio",
                    f"{low_ratio},{high_ratio}",
                    (
                        f"low within ±0.20 of {FAN_LOW_RATIO}; "
                        f"high within ±0.20 of {FAN_HIGH_RATIO}; separation >= 0.30"
                    ),
                    "Fan action should be excluded from the final controller if this fails.",
                )
            )

        # 11) State feedback sanity
        for room in ROOM_ORDER:
            r = frame[frame["room"] == room]
            temp = pd.to_numeric(r["zone_temp_C"], errors="coerce").dropna()
            rh = pd.to_numeric(r["zone_RH_percent"], errors="coerce").dropna()
            mrt = pd.to_numeric(r["zone_MRT_C"], errors="coerce").dropna()

            valid = (
                not temp.empty
                and not rh.empty
                and not mrt.empty
                and temp.between(5.0, 50.0).all()
                and rh.between(0.0, 100.0).all()
                and mrt.between(5.0, 60.0).all()
            )
            checks.append(
                self.make_check(
                    "state_feedback_sanity",
                    room,
                    True,
                    "PASS" if valid else "FAIL",
                    "valid_T_RH_MRT",
                    int(valid),
                    "finite and physically plausible T/RH/MRT values",
                    (
                        f"T_range=({temp.min() if not temp.empty else float('nan')},"
                        f"{temp.max() if not temp.empty else float('nan')}); "
                        f"RH_range=({rh.min() if not rh.empty else float('nan')},"
                        f"{rh.max() if not rh.empty else float('nan')}); "
                        f"MRT_range=({mrt.min() if not mrt.empty else float('nan')},"
                        f"{mrt.max() if not mrt.empty else float('nan')})"
                    ),
                )
            )

        checks_df = pd.DataFrame(checks)
        checks_df.to_csv(self.output_dir / "preflight_checks.csv", index=False)

        summary = self.build_summary(
            checks_df,
            energyplus_status,
            raw_rows=len(raw),
            analysis_rows=len(frame),
        )
        return checks_df, summary

    def build_summary(
        self,
        checks_df: pd.DataFrame,
        energyplus_status: int,
        raw_rows: int,
        analysis_rows: int,
    ) -> Dict[str, Any]:
        required = checks_df[checks_df["required"] == True]  # noqa: E712
        required_failures = required[required["status"] != "PASS"]

        fan_checks = checks_df[
            checks_df["category"] == "fan_direct_control_physical"
        ]
        fan_pass_count = int((fan_checks["status"] == "PASS").sum())

        all_required_pass = (
            energyplus_status == 0
            and len(required_failures) == 0
            and len(self.report.callback_errors) == 0
        )

        summary: Dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": SCRIPT_NAME,
            "energyplus_exit_status": int(energyplus_status),
            "project": ".",
            "idf": portable_path(self.idf_path, self.project_dir),
            "epw": portable_path(self.epw_path, self.project_dir),
            "occupancy": portable_path(self.occupancy_path, self.project_dir),
            "controlled_rooms": ROOM_ORDER,
            "raw_trace_rows": int(raw_rows),
            "analysis_rows_after_room_interval_start_dedup": int(analysis_rows),
            "time_alignment": self.coverage_stats,
            "callback_error_count": len(self.report.callback_errors),
            "required_check_count": int(len(required)),
            "required_failure_count": int(len(required_failures)),
            "required_failures": required_failures[
                ["category", "room", "metric", "value", "notes"]
            ].to_dict(orient="records"),
            "all_required_checks_pass": bool(all_required_pass),
            "fan_direct_control_verified_rooms": fan_pass_count,
            "fan_direct_control_total_rooms": len(ROOM_ORDER),
            "fan_action_recommendation": (
                "Fan direct control is verified for all six rooms and may be considered "
                "for the final action space."
                if fan_pass_count == len(ROOM_ORDER)
                else "Do NOT include direct fan stage in the final action space yet. "
                "Use thermostat + FCU availability until fan control is fully verified."
            ),
            "next_step": (
                "Proceed to 03_run_fixed_baseline.py."
                if all_required_pass
                else "Do NOT run the benchmark yet. Resolve required preflight failures."
            ),
            "callback_errors": self.report.callback_errors,
        }
        return summary

    def save_summary(
        self,
        checks_df: pd.DataFrame,
        summary: Dict[str, Any],
    ) -> None:
        (self.output_dir / "preflight_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("PREFLIGHT SUMMARY")
        print("=" * 78)

        print(
            f"Required checks: {summary['required_check_count']} | "
            f"Failures: {summary['required_failure_count']}"
        )
        print(
            "Overall required preflight: "
            + ("PASS" if summary["all_required_checks_pass"] else "FAIL")
        )
        print(
            "Direct fan control verified: "
            f"{summary['fan_direct_control_verified_rooms']}/"
            f"{summary['fan_direct_control_total_rooms']} rooms"
        )

        timing = summary.get("time_alignment", {})
        print("\nTime alignment:")
        print(
            f"  Analysis rows: "
            f"{timing.get('analysis_rows')} / "
            f"{timing.get('expected_analysis_rows')}"
        )
        unique_counts = timing.get("unique_interval_starts_per_room", {})
        if unique_counts:
            rendered = ", ".join(
                f"{room}={count}" for room, count in unique_counts.items()
            )
            print(f"  Unique interval starts per room: {rendered}")
        print(
            f"  First interval start: "
            f"{timing.get('first_interval_start')} "
            f"(expected {timing.get('expected_first_interval_start')})"
        )
        print(
            f"  Last interval start : "
            f"{timing.get('last_interval_start')} "
            f"(expected {timing.get('expected_last_interval_start')})"
        )

        if summary["required_failures"]:
            print("\nRequired failures:")
            for item in summary["required_failures"]:
                print(
                    f"  - {item['room']} | {item['category']} | "
                    f"{item['metric']}={item['value']} | {item['notes']}"
                )

        print("\nFan recommendation:")
        print(f"  {summary['fan_action_recommendation']}")
        print("\nNext step:")
        print(f"  {summary['next_step']}")
        print("\nSaved:")
        print(f"  {self.output_dir / 'handle_report.csv'}")
        print(f"  {self.output_dir / 'available_api_data.csv'}")
        print(f"  {self.output_dir / 'preflight_trace.csv'}")
        print(f"  {self.output_dir / 'preflight_checks.csv'}")
        print(f"  {self.output_dir / 'preflight_summary.json'}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict EnergyPlus runtime API preflight for the "
            "Closed-LoopAgenticLLMs project."
        )
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help=(
            "Repository root. By default it is inferred from the script location."
        ),
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
            "5-minute measured occupancy-count CSV path. Relative paths are "
            "resolved against --project."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Preflight result directory. Relative paths are resolved against "
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
        else (project / PREFLIGHT_RESULTS_SUBDIR).resolve()
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

        runner = PreflightRunner(
            EnergyPlusAPI=EnergyPlusAPI,
            energyplus_root=energyplus_root,
            project_dir=project,
            idf_path=idf_path,
            epw_path=epw_path,
            occupancy_path=occupancy_path,
            output_dir=output_dir,
        )

        status = runner.run()
        checks_df, summary = runner.evaluate(status)
        runner.save_summary(checks_df, summary)

        # Non-zero script exit when a required preflight test fails.
        return 0 if summary["all_required_checks_pass"] else 1

    except Exception as exc:
        print("\nFATAL PREFLIGHT ERROR", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
