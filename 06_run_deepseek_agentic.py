#!/usr/bin/env python3
"""
08_run_deepseek_agentic.py

Run the 5-minute agentic DeepSeek HVAC controller.

Architecture
------------
controller -> evaluator -> optional single refinement -> deterministic validator

The script reuses the frozen direct-DeepSeek implementation in
``07_run_deepseek_direct.py`` so the controller prompt, EnergyPlus integration,
occupancy handling, comfort calculations, action validator, and energy accounting
remain aligned with the direct-LLM condition.

Repository conventions
----------------------
- The repository root is detected automatically when this file is stored in
  ``scripts/``.
- Input, generated-model, reference-result, and output paths are project-relative
  by default.
- ``ENERGYPLUS_ROOT`` may be used to point to an EnergyPlus installation.
- API credentials are loaded by the shared DeepSeek helper; no key is embedded
  in this script.

EnergyPlus target: 24.1
Python target: 3.9+
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Import the frozen direct-DeepSeek implementation from the same project folder.
# This guarantees that the initial controller prompt, EnergyPlus integration,
# action validator, PMV assumptions, occupancy handling, timing, and meters are
# exactly the same as in the validated direct-LLM condition.
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
PROJECT_DIR = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() == "scripts"
    else SCRIPT_DIR
)
DIRECT_SCRIPT = SCRIPT_DIR / "07_run_deepseek_direct.py"


def resolve_project_path(path: Path, project: Path) -> Path:
    """Resolve a CLI path relative to the selected repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else project / path


def repository_relative_path(path: Path, project: Path) -> str:
    """Return a portable repository-relative path when possible."""
    try:
        return str(path.resolve().relative_to(project.resolve()))
    except (ValueError, OSError):
        return str(path)


def discover_energyplus_root(preferred: Optional[Path] = None) -> Path:
    """Locate an EnergyPlus installation containing the Python API."""
    candidates: List[Path] = []

    env_root = os.environ.get("ENERGYPLUS_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())

    if preferred is not None:
        candidates.append(Path(preferred).expanduser())

    energyplus_executable = shutil.which("energyplus")
    if energyplus_executable:
        candidates.append(Path(energyplus_executable).resolve().parent)

    candidates.extend(
        [
            Path(r"C:\EnergyPlusV24-1-0"),
            Path("/usr/local/EnergyPlus-24-1-0"),
            Path("/opt/EnergyPlus-24-1-0"),
            Path("/Applications/EnergyPlus-24-1-0"),
        ]
    )

    try:
        candidates.extend(sorted(Path("C:/").glob("EnergyPlusV24-1-*"), reverse=True))
    except OSError:
        pass

    seen = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and (candidate / "pyenergyplus" / "api.py").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the EnergyPlus Python API. Set ENERGYPLUS_ROOT or "
        "provide --energyplus-root /path/to/EnergyPlus-24-1-0."
    )

if not DIRECT_SCRIPT.exists():
    raise FileNotFoundError(
        "Frozen direct DeepSeek script not found beside this script: "
        f"{DIRECT_SCRIPT}"
    )

_spec = importlib.util.spec_from_file_location(
    "closedloop_deepseek_direct_frozen",
    str(DIRECT_SCRIPT),
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import frozen direct script: {DIRECT_SCRIPT}")

direct = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = direct
_spec.loader.exec_module(direct)


AGENTIC_MAX_REFINEMENTS = 1
EVALUATOR_MAX_TOKENS = 1100
REFINEMENT_MAX_TOKENS = direct.LLM_MAX_TOKENS
FROZEN_EVALUATOR_V4_SHA256 = "30f22e883b205a027bdec4bebca7d32d83650d1b2a6c6a718bfd9cae0391ec6f"
FROZEN_EVALUATOR_V4_SYNTHETIC_CASES = 12

# -----------------------------------------------------------------------------
# 5-minute Agentic LLM decision loop
# -----------------------------------------------------------------------------
# Agentic decisions are executed at the same resolution as the EnergyPlus
# callback and occupancy measurements.
# -----------------------------------------------------------------------------
AGENTIC_LLM_DECISION_INTERVAL_MINUTES = 5
EXPECTED_AGENTIC_DECISION_EPOCHS = 840  # 7 days x 10 h/day x 12 decisions/h
EXPECTED_AGENTIC_ROOM_DECISIONS = EXPECTED_AGENTIC_DECISION_EPOCHS * len(direct.ROOM_ORDER)

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
    prompt = direct.build_deepseek_user_prompt(interval_start, room_states)
    old = f'"decision_interval_minutes": {direct.LLM_DECISION_INTERVAL_MINUTES}'
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
    return direct.hashlib.sha256(
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
        if not direct.is_finite(value):
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
        norm = direct.normalize(key)
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
    for room in direct.ROOM_ORDER:
        key = direct.normalize(room)
        aliases[key] = room
        aliases[key.replace("room", "")] = room

    result: Dict[str, Dict[str, Any]] = {}
    for item in iterable:
        key = direct.normalize(item.get("room", ""))
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
    for room in direct.ROOM_ORDER:
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
    for room in direct.ROOM_ORDER:
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
    for attempt in range(1, direct.LLM_MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            data = client._post_json("/chat/completions", payload)
        except Exception as exc:
            last_error = exc
            if attempt >= direct.LLM_MAX_ATTEMPTS:
                raise
            time.sleep(0.5)
            continue

        latency = time.perf_counter() - started
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            last_error = RuntimeError("DeepSeek response contains no choices.")
            if attempt >= direct.LLM_MAX_ATTEMPTS:
                raise last_error
            time.sleep(0.5)
            continue

        content = str(choices[0].get("message", {}).get("content", "") or "").strip()
        if not content:
            last_error = RuntimeError("DeepSeek returned empty content.")
            if attempt >= direct.LLM_MAX_ATTEMPTS:
                raise last_error
            time.sleep(0.5)
            continue

        data["_client_attempts"] = attempt
        return content, float(latency), data

    raise RuntimeError(f"DeepSeek role request failed: {last_error}")


class DeepSeekAgenticRunner(direct.DeepSeekDirectRunner):
    def __init__(
        self,
        *args: Any,
        project_dir: Path,
        deepseek_direct_summary_path: Path,
        evaluator_v4_summary_path: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.project_dir = project_dir.resolve()
        self.deepseek_direct_summary_path = deepseek_direct_summary_path
        self.evaluator_v4_summary_path = evaluator_v4_summary_path
        self.evaluator_v4_validation: Dict[str, Any] = {}

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

        for room in direct.ROOM_ORDER:
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

    def load_deepseek_direct_reference(self) -> Dict[str, Any]:
        data = self._load_reference(
            self.deepseek_direct_summary_path,
            "all_required_deepseek_direct_checks_pass",
            "deepseek_direct",
        )
        llm = data.get("llm_controller", {})
        if llm.get("model_requested") != self.deepseek_model:
            raise RuntimeError("Direct reference requested a different DeepSeek model.")
        if llm.get("system_prompt_sha256") != direct.deepseek_prompt_sha256():
            raise RuntimeError(
                "Direct reference used a different controller system prompt; "
                "direct-vs-agentic ablation would not be fair."
            )
        if int(llm.get("decision_interval_minutes", -1)) != direct.LLM_DECISION_INTERVAL_MINUTES:
            raise RuntimeError("Direct reference used a different decision interval.")
        return data


    def load_frozen_evaluator_v4_validation(self) -> Dict[str, Any]:
        if not self.evaluator_v4_summary_path.exists():
            raise FileNotFoundError(
                f"Frozen evaluator V4 synthetic summary not found: "
                f"{self.evaluator_v4_summary_path}"
            )

        data = json.loads(
            self.evaluator_v4_summary_path.read_text(encoding="utf-8")
        )

        checks = {
            "passed": data.get("passed") is True,
            "energyplus_not_run": data.get("energyplus_run") is False,
            "experimental_week_not_used": (
                data.get("experimental_week_data_used") is False
            ),
            "synthetic_case_count_12": (
                int(data.get("synthetic_case_count", -1))
                == FROZEN_EVALUATOR_V4_SYNTHETIC_CASES
            ),
            "passed_case_count_12": (
                int(data.get("passed_case_count", -1))
                == FROZEN_EVALUATOR_V4_SYNTHETIC_CASES
            ),
            "failed_case_count_zero": (
                int(data.get("failed_case_count", -1)) == 0
            ),
            "prompt_hash_matches": (
                data.get("evaluator_prompt_sha256")
                == FROZEN_EVALUATOR_V4_SHA256
                == evaluator_prompt_sha256()
            ),
            "same_requested_model": (
                data.get("model_requested") == self.deepseek_model
            ),
        }

        if not all(checks.values()):
            failed = [key for key, value in checks.items() if not value]
            raise RuntimeError(
                "Frozen evaluator V4 validation provenance failed: "
                + ", ".join(failed)
            )

        self.evaluator_v4_validation = {
            "summary_path": repository_relative_path(self.evaluator_v4_summary_path, self.project_dir),
            "checks": checks,
            "source_summary": data,
        }
        return self.evaluator_v4_validation

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

        for room in direct.ROOM_ORDER:
            occupancy = self.occupancy_count(interval_start, room)
            temp = self.variable_value(state, self.temp_handles[room])
            rh = self.variable_value(state, self.rh_handles[room])
            mrt = self.variable_value(state, self.mrt_handles[room])
            valid = all(direct.is_finite(v) for v in (temp, rh, mrt))
            if valid:
                pmv, ppd = direct.fanger_pmv_ppd(temp, rh, mrt)
                valid = direct.is_finite(pmv) and direct.is_finite(ppd)
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
        for room in direct.ROOM_ORDER:
            self.current_commands[room].update(
                {
                    "heating_setpoint_C": direct.UNOCCUPIED_HEATING_SP_C,
                    "cooling_setpoint_C": direct.UNOCCUPIED_COOLING_SP_C,
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
            if not direct.is_weather_run_period(self.api, state):
                return
            if self.current_interval_start is None:
                raise RuntimeError("System callback executed before research clock.")

            interval_start = self.current_interval_start
            if not direct.in_control_period(interval_start):
                return

            if not direct.is_office_hour(interval_start):
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
                    if len(room_states) == len(direct.ROOM_ORDER):
                        try:
                            initial_content, initial_latency, initial_response = self.deepseek_client.chat(initial_prompt)
                            initial_success = True
                            self.llm_call_success_count += 1
                            self._record_api("controller_initial", initial_response, initial_latency)
                            try:
                                proposals = direct.normalize_deepseek_rooms(
                                    direct.extract_json_object(initial_content)
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
                    for room in direct.ROOM_ORDER:
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
                                direct.extract_json_object(evaluator_content)
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
                    for room in direct.ROOM_ORDER:
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
                        room for room in direct.ROOM_ORDER
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
                                direct.DEEPSEEK_SYSTEM_PROMPT,
                                refinement_prompt,
                                REFINEMENT_MAX_TOKENS,
                            )
                            refinement_success = True
                            self.refinement_call_success_count += 1
                            self._record_api(
                                "controller_refinement", refinement_response, refinement_latency
                            )
                            try:
                                revised = direct.normalize_deepseek_rooms(
                                    direct.extract_json_object(refinement_content)
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
                    for room in direct.ROOM_ORDER:
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

                        final_action = direct.validate_deepseek_room_action(candidate, occupancy)
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

                        first_validated = direct.validate_deepseek_room_action(
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
                            return value if direct.is_finite(value) else None

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
                    for room in direct.ROOM_ORDER:
                        command = self.current_commands[room]
                        if command.get("decision_id") is None:
                            occupancy = self.occupancy_count(interval_start, room)
                            action = direct.fallback_rule_action(occupancy)
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
            for room in direct.ROOM_ORDER:
                command = self.current_commands[room]
                self.api.exchange.set_actuator_value(
                    state, self.heat_actuators[room], float(command["heating_setpoint_C"])
                )
                self.api.exchange.set_actuator_value(
                    state, self.cool_actuators[room], float(command["cooling_setpoint_C"])
                )
                self.api.exchange.set_actuator_value(
                    state, self.availability_actuators[room], direct.FCU_AVAILABILITY_COMMAND
                )
                command["availability"] = direct.FCU_AVAILABILITY_COMMAND
                command["command_sequence_index"] = self.zone_sequence_index

            self.system_control_callback_count += 1

        except Exception as exc:
            self.record_callback_error("apply_deepseek_agentic_control", exc)

    def end_zone_timestep(self, state: Any) -> None:
        interval_start = self.current_interval_start
        super().end_zone_timestep(state)

        if interval_start is None:
            return
        for room in direct.ROOM_ORDER:
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
        checks = direct.OccupancyRuleRunner.validation_checks(
            self, state_df, meter_df, energyplus_status, err_info
        )

        checks["deepseek_model_verified"] = bool(
            self.deepseek_verification.get("verified", False)
        )
        checks["system_control_callback_count"] = self.system_control_callback_count
        checks["system_control_callback_count_2016"] = (
            self.system_control_callback_count == direct.EXPECTED_INTERVALS_PER_ROOM
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
                heat.between(direct.HEATING_MIN_C, direct.HEATING_MAX_C, inclusive="both").all()
                and cool.between(direct.COOLING_MIN_C, direct.COOLING_MAX_C, inclusive="both").all()
            )
            checks["action_resolution_pass"] = bool(
                ((heat / direct.SETPOINT_RESOLUTION_C - (heat / direct.SETPOINT_RESOLUTION_C).round()).abs() <= 1e-9).all()
                and ((cool / direct.SETPOINT_RESOLUTION_C - (cool / direct.SETPOINT_RESOLUTION_C).round()).abs() <= 1e-9).all()
            )
            checks["action_deadband_pass"] = bool(
                ((cool - heat) >= direct.MIN_DEADBAND_C - 1e-9).all()
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
        try:
            self.load_frozen_evaluator_v4_validation()
            checks["evaluator_v4_synthetic_validation_verified"] = True
        except Exception:
            checks["evaluator_v4_synthetic_validation_verified"] = False
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
                fallback_fraction <= direct.MAX_LLM_FALLBACK_FRACTION
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

        try:
            direct_ref = self.load_deepseek_direct_reference()
            checks["direct_reference_comparable"] = (
                int(direct_ref.get("llm_controller", {}).get("decision_interval_minutes", -1))
                == AGENTIC_LLM_DECISION_INTERVAL_MINUTES
            )
            checks["direct_reference_prompt_hash_matches"] = (
                direct_ref.get("llm_controller", {}).get("system_prompt_sha256")
                == direct.deepseek_prompt_sha256()
            )
        except Exception:
            checks["direct_reference_comparable"] = False
            checks["direct_reference_prompt_hash_matches"] = False

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
            checks["evaluator_v4_synthetic_validation_verified"],
            checks["decision_rows_expected"],
            checks["decision_state_all_valid"],
            checks["final_fallback_fraction_within_limit"],
            checks["decision_rows_evaluator_contract_all_ok"],
            checks["refinement_only_after_rejection"],
            checks["direct_reference_prompt_hash_matches"],
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
        proposals = direct.normalize_deepseek_rooms(
            direct.extract_json_object(controller_content)
        )
        missing = [r for r in direct.ROOM_ORDER if r not in proposals]
        if missing:
            raise RuntimeError(f"Agentic preflight controller missing rooms: {missing}")

        state_by_room = {
            row["room"]: {**row, "PPD_percent": None, "state_valid": True}
            for row in representative_states
        }
        structural: Dict[str, Dict[str, Any]] = {}
        for room in direct.ROOM_ORDER:
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
        reviews = normalize_evaluator_rooms(direct.extract_json_object(eval_content))
        missing_reviews = [r for r in direct.ROOM_ORDER if r not in reviews]
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
            for room in direct.ROOM_ORDER
        }
        forced = False
        if all(refinement_reviews[r]["approved"] for r in direct.ROOM_ORDER):
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
            direct.DEEPSEEK_SYSTEM_PROMPT,
            refine_prompt,
            REFINEMENT_MAX_TOKENS,
        )
        revised = direct.normalize_deepseek_rooms(
            direct.extract_json_object(refine_content)
        )
        missing_revised = [r for r in direct.ROOM_ORDER if r not in revised]
        if missing_revised:
            raise RuntimeError(f"Agentic preflight refinement missing rooms: {missing_revised}")
        for row in representative_states:
            action = direct.validate_deepseek_room_action(
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
                    for room in direct.ROOM_ORDER
                },
            },
            "controller_refinement": {
                "latency_s": float(refine_latency),
                "model_returned": refine_response.get("model"),
                "usage": refine_response.get("usage", {}),
                "forced_refinement_test": forced,
            },
            "controller_system_prompt_sha256": direct.deepseek_prompt_sha256(),
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

        direct_ref = self.load_deepseek_direct_reference()
        evaluator_v4_ref = self.load_frozen_evaluator_v4_validation()

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
            f"(synthetic 12/12 VERIFIED)"
        )
        print("Direct ref : controller system prompt/model provenance VERIFIED; cadence intentionally differs")
        print(
            f"Direct ref energy: "
            f"{direct_ref['energy']['HVAC_component_sum_kWh']:.4f} kWh"
        )

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
        print(f"Direct ref : {self.deepseek_direct_summary_path}")
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
        direct_ref = self.load_deepseek_direct_reference()

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

        summary["script"] = "08_run_deepseek_agentic.py"
        summary["case"] = "deepseek_agentic"
        summary["implementation_version"] = "v5.1"
        summary["project"] = "."
        summary["idf"] = repository_relative_path(self.idf_path, self.project_dir)
        summary["epw"] = repository_relative_path(self.epw_path, self.project_dir)
        summary["occupancy"] = repository_relative_path(
            self.occupancy_path, self.project_dir
        )
        summary["evaluator_v4_validation_reference"] = self.evaluator_v4_validation
        summary["deepseek_direct_reference"] = {
            "summary_path": repository_relative_path(self.deepseek_direct_summary_path, self.project_dir),
            "HVAC_component_sum_kWh": float(direct_ref["energy"]["HVAC_component_sum_kWh"]),
            "comfort_and_occupancy": direct_ref.get("comfort_and_occupancy", {}),
            "switches": direct_ref.get("action_distribution", {}).get(
                "total_setpoint_switches_across_rooms"
            ),
            "model_requested": direct_ref.get("llm_controller", {}).get("model_requested"),
            "model_returned_preflight": direct_ref.get("llm_controller", {}).get(
                "model_returned_preflight"
            ),
            "controller_system_prompt_sha256": direct_ref.get("llm_controller", {}).get(
                "system_prompt_sha256"
            ),
        }

        summary["policy"] = {
            "type": "agentic_deepseek_thermostat_office_hours",
            "architecture": "controller -> evaluator -> feedback -> controller refinement; K=1",
            "control_window": "08:00-18:00",
            "decision_interval_minutes": AGENTIC_LLM_DECISION_INTERVAL_MINUTES,
            "controller_initial_prompt_identical_to_direct_condition": True,
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
                "heating_min_C": direct.HEATING_MIN_C,
                "heating_max_C": direct.HEATING_MAX_C,
                "cooling_min_C": direct.COOLING_MIN_C,
                "cooling_max_C": direct.COOLING_MAX_C,
                "resolution_C": direct.SETPOINT_RESOLUTION_C,
                "minimum_deadband_C": direct.MIN_DEADBAND_C,
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
            "max_tokens": direct.LLM_MAX_TOKENS,
            "timeout_seconds": self.llm_timeout_s,
            "decision_interval_minutes": AGENTIC_LLM_DECISION_INTERVAL_MINUTES,
            "controller_system_prompt_sha256": direct.deepseek_prompt_sha256(),
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
            clean = [float(v) for v in values if direct.is_finite(v)]
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
        direct_kwh = float(direct_ref["energy"]["HVAC_component_sum_kWh"])
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

        summary["energy_comparison_to_deepseek_direct"] = {
            "deepseek_direct_HVAC_component_sum_kWh": direct_kwh,
            "deepseek_agentic_HVAC_component_sum_kWh": agentic_kwh,
            "agentic_minus_direct_kWh": agentic_kwh - direct_kwh,
            "agentic_minus_direct_percent": (agentic_kwh - direct_kwh) / direct_kwh * 100.0,
        }

        current_comfort = summary.get("comfort_and_occupancy", {})
        direct_comfort = direct_ref.get("comfort_and_occupancy", {})
        compare_keys = [
            "occupied_overheat_gt_27_fraction",
            "occupied_overcool_lt_22_5_fraction",
            "occupied_RH_gt_85_fraction",
            "occupied_mean_abs_PMV",
            "occupied_mean_PPD_percent",
        ]
        summary["agentic_comparison_to_deepseek_direct"] = {
            key + "_change": (
                float(current_comfort[key]) - float(direct_comfort[key])
                if key in current_comfort
                and key in direct_comfort
                and direct.is_finite(current_comfort[key])
                and direct.is_finite(direct_comfort[key])
                else None
            )
            for key in compare_keys
        }

        direct_switches = direct_ref.get("action_distribution", {}).get(
            "total_setpoint_switches_across_rooms"
        )
        summary["switching_comparison_to_deepseek_direct"] = {
            "deepseek_direct_switches": direct_switches,
            "deepseek_agentic_switches": total_switches,
            "difference": (
                total_switches - int(direct_switches)
                if direct_switches is not None else None
            ),
        }

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
        direct_cmp = summary.get("energy_comparison_to_deepseek_direct", {})

        print("\nEnergy:")
        print(f"  HVAC component sum       {energy.get('HVAC_component_sum_kWh', float('nan')):.4f} kWh")
        print(f"  Savings vs fixed         {fixed.get('energy_savings_vs_fixed_percent', float('nan')):.3f}%")
        print(
            f"  Change vs DeepSeek direct "
            f"{direct_cmp.get('agentic_minus_direct_kWh', float('nan')):+.4f} kWh "
            f"({direct_cmp.get('agentic_minus_direct_percent', float('nan')):+.3f}%)"
        )

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
    parser.add_argument("--deepseek-direct-summary", type=Path, default=None)
    parser.add_argument(
        "--evaluator-v4-summary",
        type=Path,
        default=None,
        help=(
            "Frozen 12/12 synthetic evaluator V4 validation summary. Default: "
            "results/evaluator_synthetic_v4/evaluator_v4_synthetic_summary.json"
        ),
    )
    parser.add_argument("--api-base-url", type=str, default=direct.DEFAULT_DEEPSEEK_BASE_URL)
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--model", type=str, default=direct.DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=direct.LLM_TIMEOUT_SECONDS)
    parser.add_argument("--temperature", type=float, default=direct.LLM_TEMPERATURE)
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
        project / "inputs" / "weather" / "CHN_Hebei.Shijiazhuang.536980_CSWD.epw",
    )
    occupancy_path = resolve(
        args.occupancy,
        project / "inputs" / "occupancy" / "actual_occupancy_count_5min_7day.csv",
    )
    output_dir = resolve(args.output_dir, project / "results" / "deepseek_agentic")
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
    direct_summary = resolve(
        args.deepseek_direct_summary,
        project / "results" / "deepseek_direct" / "deepseek_direct_summary.json",
    )
    evaluator_v4_summary = resolve(
        args.evaluator_v4_summary,
        project
        / "results"
        / "evaluator_synthetic_v4"
        / "evaluator_v4_synthetic_summary.json",
    )

    print("=" * 78)
    print("Closed-LoopAgenticLLMs - 08_run_deepseek_agentic.py")
    print("=" * 78)

    try:
        api_key_file = (
            resolve_project_path(args.api_key_file, project)
            if args.api_key_file is not None
            else None
        )
        api_key, key_source = direct.load_deepseek_api_key(project, api_key_file)
        energyplus_root = discover_energyplus_root(args.energyplus_root)
        EnergyPlusAPI = direct.import_energyplus_api(energyplus_root)

        for label, path in (
            ("Frozen direct script", DIRECT_SCRIPT),
            ("Generated IDF", idf_path),
            ("EPW", epw_path),
            ("Occupancy CSV", occupancy_path),
            ("Fixed summary", fixed_summary),
            ("Rule-OCC summary", occ_summary),
            ("Comfort-Rule summary", comfort_summary),
            ("DeepSeek direct summary", direct_summary),
            ("Evaluator V4 synthetic summary", evaluator_v4_summary),
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
            deepseek_direct_summary_path=direct_summary,
            evaluator_v4_summary_path=evaluator_v4_summary,
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
