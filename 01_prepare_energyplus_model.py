#!/usr/bin/env python3
r"""
01_prepare_energyplus_model.py

Prepare the EnergyPlus FCU model for the Closed-LoopAgenticLLMs project.

What this script does
---------------------
1. Reads the source IDF from ``inputs/building`` by default.

2. Creates six dedicated ventilation schedule actuators for Rooms 1, 2, 4, 5, 6, 7:
       Experimental Ventilation Fraction Room N

   The existing ZoneVentilation:DesignFlowRate objects already use Flow/Person.
   During runtime, the controller infrastructure should set each schedule to:

       ventilation_fraction = current_occupancy / design_people

   This makes the zone ventilation flow approximately:
       Flow_per_person * current_occupancy

   The schedule default is 1.0 (full design ventilation) as a fail-safe. The
   runtime/preflight script must verify that these schedules are actuated from
   measured occupancy before benchmark runs.

   Zone 3 is not one of the six experimental rooms. Its existing
   'Thermal Zone 3 Ventilation per Person' object is kept conventional and its
   Schedule Name is set to 'Office Work Occ', matching the conventional Zone 3
   occupancy schedule.

3. Removes legacy ExpandObjects preprocessing-message objects and known warning
   comment lines.

4. Removes exact duplicate Output:Variable and Output:Meter requests while
   preserving the first occurrence.

5. Ensures the core runtime diagnostics are present once, including MRT and
   zone ventilation flow.

6. Writes a clean generated IDF plus a JSON preparation manifest to
   ``generated/building`` by default.

The source IDF is never modified.

EnergyPlus target: 24.1

The repository root is detected from this script location and can be overridden
with ``--project``. All default paths are therefore portable across operating
systems and user accounts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_NAME = Path(__file__).name
SCRIPT_DIR = Path(__file__).resolve().parent

# Support the recommended repository layout (script stored in ``scripts/``)
# while also remaining usable if the file is kept temporarily in the repo root.
DEFAULT_PROJECT_DIR = (
    SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() in {"scripts", "tools"} else SCRIPT_DIR
)

INPUT_SUBDIR = Path("inputs") / "building"
GENERATED_SUBDIR = Path("generated") / "building"

DEFAULT_INPUT_NAME = "honeycomb_7zone_fcu_control_v4_experimental_people.idf"
DEFAULT_OUTPUT_NAME = "honeycomb_7zone_fcu_closed_loop_ready.idf"
DEFAULT_MANIFEST_NAME = "model_prepare_manifest.json"

CONTROLLED_ROOMS = (1, 2, 4, 5, 6, 7)

EXPECTED_PEOPLE_OBJECTS: Dict[int, str] = {
    1: "Experimental People Room 1",
    2: "Experimental People Room 2",
    4: "Experimental People Room 4",
    5: "Experimental People Room 5",
    6: "Experimental People Room 6",
    7: "Experimental People Room 7",
}

EXPECTED_VENTILATION_OBJECTS: Dict[int, str] = {
    room: f"Thermal Zone {room} Ventilation per Person"
    for room in CONTROLLED_ROOMS
}

VENT_SCHEDULES: Dict[int, str] = {
    room: f"Experimental Ventilation Fraction Room {room}"
    for room in CONTROLLED_ROOMS
}

ZONE3_VENTILATION_OBJECT = "Thermal Zone 3 Ventilation per Person"
ZONE3_VENTILATION_SCHEDULE = "Office Work Occ"

CORE_OUTPUT_VARIABLES = (
    "Zone Mean Air Temperature",
    "Zone Air Relative Humidity",
    "Zone Mean Radiant Temperature",
    "Zone Thermostat Heating Setpoint Temperature",
    "Zone Thermostat Cooling Setpoint Temperature",
    "Zone People Occupant Count",
    "Fan Air Mass Flow Rate",
    "Fan Electricity Rate",
    "Zone Ventilation Current Density Volume Flow Rate",
)

CORE_OUTPUT_METERS = (
    "Electricity:Facility",
    "Cooling:Electricity",
    "Heating:Electricity",
    "Fans:Electricity",
    "Pumps:Electricity",
    "Electricity:HVAC",
)

# Known stale text embedded by an earlier ExpandObjects run.
STALE_COMMENT_PHRASES = (
    "Cannot find Energy+.idd as specified in Energy+.ini.",
    "Since the Energy+.IDD file cannot be read no range or choice checking was performed.",
    "New objects created from ExpandObjects",
)


def split_idf_objects(text: str) -> List[str]:
    """Split IDF text into object blocks while retaining comments around objects.

    An object ends only when a semicolon occurs before the comment delimiter '!'.
    """
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
    """Return IDF object fields while preserving blank fields."""
    code_parts: List[str] = []
    for line in block.splitlines():
        code = line.split("!", 1)[0]
        if code.strip():
            code_parts.append(code)

    code = "\n".join(code_parts).replace(";", ",")
    fields = [value.strip() for value in code.split(",")]

    while fields and fields[-1] == "":
        fields.pop()

    return fields


def object_type(block: str) -> str:
    fields = idf_fields(block)
    return fields[0] if fields else ""


def object_name(block: str) -> str:
    fields = idf_fields(block)
    return fields[1] if len(fields) >= 2 else ""


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def replace_field_by_exact_comment(
    block: str,
    exact_comment_label: str,
    new_value: str,
) -> Tuple[str, bool]:
    """Replace one field selected by its exact IDF comment label.

    Example label:
        Schedule Name
    will NOT match:
        Minimum Indoor Temperature Schedule Name
    """
    out: List[str] = []
    changed = False
    target = normalize(exact_comment_label)

    for line in block.splitlines(keepends=True):
        if changed or "!" not in line:
            out.append(line)
            continue

        code, bang, comment = line.partition("!")
        label = comment.strip()
        if label.startswith("-"):
            label = label[1:].strip()

        # Remove unit text, e.g. "{m3/s}", only for matching.
        label_no_units = re.sub(r"\s*\{[^}]*\}\s*", "", label).strip()

        if normalize(label_no_units) != target:
            out.append(line)
            continue

        delimiter = ";" if ";" in code else ","
        indent = code[: len(code) - len(code.lstrip())]
        newline = "\n" if line.endswith("\n") else ""
        rendered = f"{indent}{new_value}{delimiter}".ljust(44)
        out.append(f"{rendered} !{comment.rstrip()}{newline}")
        changed = True

    return "".join(out), changed


def strip_stale_warning_comments(text: str) -> Tuple[str, int]:
    """Remove known comment-only ExpandObjects warning lines."""
    kept: List[str] = []
    removed = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("!"):
            lower = stripped.lower()
            if any(phrase.lower() in lower for phrase in STALE_COMMENT_PHRASES):
                removed += 1
                continue
        kept.append(line)

    return "".join(kept), removed


def output_object_key(block: str) -> Optional[Tuple[str, ...]]:
    """Return a semantic key for exact Output:Variable / Output:Meter duplicates."""
    fields = idf_fields(block)
    if not fields:
        return None

    obj = normalize(fields[0])

    if obj == "output:variable":
        # [type, key value, variable name, reporting frequency]
        if len(fields) >= 4:
            return tuple(normalize(x) for x in fields[:4])

    if obj == "output:meter":
        # [type, meter name, reporting frequency]
        if len(fields) >= 3:
            return tuple(normalize(x) for x in fields[:3])

    return None


def remove_duplicate_outputs(blocks: Iterable[str]) -> Tuple[List[str], int]:
    seen = set()
    result: List[str] = []
    removed = 0

    for block in blocks:
        key = output_object_key(block)
        if key is not None:
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        result.append(block)

    return result, removed


def standard_schedule_constant(name: str, value: float) -> str:
    return f"""
Schedule:Constant,
  {name},                     !- Name
  Fraction,                                 !- Schedule Type Limits Name
  {value:.6f};                               !- Hourly Value
"""


def schedule_constant_exists(blocks: Iterable[str], name: str) -> bool:
    target = normalize(name)
    for block in blocks:
        fields = idf_fields(block)
        if (
            len(fields) >= 2
            and normalize(fields[0]) == "schedule:constant"
            and normalize(fields[1]) == target
        ):
            return True
    return False


def fraction_type_exists(blocks: Iterable[str]) -> bool:
    for block in blocks:
        fields = idf_fields(block)
        if (
            len(fields) >= 2
            and normalize(fields[0]) == "scheduletypelimits"
            and normalize(fields[1]) == "fraction"
        ):
            return True
    return False


def output_variable_exists(
    blocks: Iterable[str],
    variable_name: str,
    key_value: str = "*",
    frequency: str = "Timestep",
) -> bool:
    for block in blocks:
        fields = idf_fields(block)
        if len(fields) < 4 or normalize(fields[0]) != "output:variable":
            continue
        if (
            normalize(fields[1]) == normalize(key_value)
            and normalize(fields[2]) == normalize(variable_name)
            and normalize(fields[3]) == normalize(frequency)
        ):
            return True
    return False


def output_meter_exists(
    blocks: Iterable[str],
    meter_name: str,
    frequency: str = "Timestep",
) -> bool:
    for block in blocks:
        fields = idf_fields(block)
        if len(fields) < 3 or normalize(fields[0]) != "output:meter":
            continue
        if (
            normalize(fields[1]) == normalize(meter_name)
            and normalize(fields[2]) == normalize(frequency)
        ):
            return True
    return False


def render_output_variable(variable_name: str) -> str:
    return f"""
Output:Variable,
  *,                                      !- Key Value
  {variable_name},                        !- Variable Name
  Timestep;                               !- Reporting Frequency
"""


def render_output_meter(meter_name: str) -> str:
    return f"""
Output:Meter,
  {meter_name},                           !- Key Name
  Timestep;                               !- Reporting Frequency
"""


def discover_people_capacities(blocks: Iterable[str]) -> Dict[int, float]:
    """Read the design Number of People from the six experimental People objects."""
    capacities: Dict[int, float] = {}

    for block in blocks:
        fields = idf_fields(block)
        if len(fields) < 6 or normalize(fields[0]) != "people":
            continue

        name = fields[1]
        for room, expected_name in EXPECTED_PEOPLE_OBJECTS.items():
            if normalize(name) != normalize(expected_name):
                continue

            method = normalize(fields[4])
            if method != "people":
                raise RuntimeError(
                    f"{expected_name}: expected Number of People Calculation Method "
                    f"'People', found {fields[4]!r}"
                )

            try:
                capacity = float(fields[5])
            except Exception as exc:
                raise RuntimeError(
                    f"{expected_name}: invalid design Number of People {fields[5]!r}"
                ) from exc

            if not math.isfinite(capacity) or capacity <= 0:
                raise RuntimeError(
                    f"{expected_name}: design Number of People must be > 0, "
                    f"found {capacity!r}"
                )

            capacities[room] = capacity

    missing = [room for room in CONTROLLED_ROOMS if room not in capacities]
    if missing:
        raise RuntimeError(
            "Missing experimental People objects for controlled rooms: "
            + ", ".join(map(str, missing))
        )

    return capacities


def patch_ventilation_objects(
    blocks: Iterable[str],
) -> Tuple[List[str], Dict[int, Dict[str, object]]]:
    """Patch ventilation for all seven zones.

    Rooms 1, 2, 4, 5, 6, and 7 receive dedicated runtime-actuated
    Schedule:Constant objects. Zone 3 remains conventional and is tied to
    the existing 'Office Work Occ' schedule.
    """
    result: List[str] = []
    patched: Dict[int, Dict[str, object]] = {}

    for block in blocks:
        fields = idf_fields(block)

        if not fields or normalize(fields[0]) != "zoneventilation:designflowrate":
            result.append(block)
            continue

        name = fields[1] if len(fields) > 1 else ""

        # Zone 3 is the conventional/reference zone. Keep its existing
        # Flow/Person ventilation object, but drive it with the same
        # conventional occupancy schedule used by the Zone 3 People object.
        if normalize(name) == normalize(ZONE3_VENTILATION_OBJECT):
            if len(fields) < 8:
                raise RuntimeError(
                    f"{name}: unexpected ZoneVentilation:DesignFlowRate field count."
                )

            method = normalize(fields[4])
            if method != "flow/person":
                raise RuntimeError(
                    f"{name}: expected Design Flow Rate Calculation Method 'Flow/Person', "
                    f"found {fields[4]!r}"
                )

            try:
                flow_per_person = float(fields[7])
            except Exception as exc:
                raise RuntimeError(
                    f"{name}: invalid Flow Rate per Person value {fields[7]!r}"
                ) from exc

            new_block, changed = replace_field_by_exact_comment(
                block,
                "Schedule Name",
                ZONE3_VENTILATION_SCHEDULE,
            )
            if not changed:
                new_block = replace_field_by_position(
                    block,
                    3,
                    ZONE3_VENTILATION_SCHEDULE,
                )

            result.append(new_block)
            patched[3] = {
                "ventilation_object": name,
                "schedule": ZONE3_VENTILATION_SCHEDULE,
                "flow_per_person_m3_s_person": flow_per_person,
                "control_mode": "conventional_schedule",
            }
            continue

        room_match: Optional[int] = None
        for room, expected_name in EXPECTED_VENTILATION_OBJECTS.items():
            if normalize(name) == normalize(expected_name):
                room_match = room
                break

        if room_match is None:
            result.append(block)
            continue

        if len(fields) < 8:
            raise RuntimeError(
                f"{name}: unexpected ZoneVentilation:DesignFlowRate field count."
            )

        method = normalize(fields[4])
        if method != "flow/person":
            raise RuntimeError(
                f"{name}: expected Design Flow Rate Calculation Method 'Flow/Person', "
                f"found {fields[4]!r}"
            )

        try:
            flow_per_person = float(fields[7])
        except Exception as exc:
            raise RuntimeError(
                f"{name}: invalid Flow Rate per Person value {fields[7]!r}"
            ) from exc

        schedule = VENT_SCHEDULES[room_match]
        new_block, changed = replace_field_by_exact_comment(
            block,
            "Schedule Name",
            schedule,
        )

        # Fallback for unusual comment formatting: Schedule Name is field index 3.
        if not changed:
            new_block = replace_field_by_position(block, 3, schedule)
            changed = True

        result.append(new_block)
        patched[room_match] = {
            "ventilation_object": name,
            "schedule": schedule,
            "flow_per_person_m3_s_person": flow_per_person,
        }

    missing = [room for room in CONTROLLED_ROOMS if room not in patched]
    if missing:
        raise RuntimeError(
            "Could not patch ZoneVentilation:DesignFlowRate for controlled rooms: "
            + ", ".join(map(str, missing))
        )

    if 3 not in patched:
        raise RuntimeError(
            "Could not patch 'Thermal Zone 3 Ventilation per Person' to "
            "'Office Work Occ'."
        )

    return result, patched


def replace_field_by_position(block: str, field_index: int, new_value: str) -> str:
    """Fallback field replacement.

    field_index follows idf_fields(), so:
      0 = object type
      1 = object name
      2 = zone
      3 = schedule name
    """
    current_index = -1
    out: List[str] = []

    for line in block.splitlines(keepends=True):
        code, sep, comment = line.partition("!")
        code_only = code

        # Count comma/semicolon-terminated IDF tokens on this code line.
        # The target objects in this project use one field per line.
        stripped = code_only.strip()
        if not stripped:
            out.append(line)
            continue

        if "," in code_only or ";" in code_only:
            current_index += 1

        if current_index != field_index:
            out.append(line)
            continue

        delimiter = ";" if ";" in code_only else ","
        indent = code_only[: len(code_only) - len(code_only.lstrip())]
        newline = "\n" if line.endswith("\n") else ""
        rendered = f"{indent}{new_value}{delimiter}".ljust(44)

        if sep:
            out.append(f"{rendered} !{comment.rstrip()}{newline}")
        else:
            out.append(f"{rendered}{newline}")

    return "".join(out)


def remove_preprocessor_message_objects(
    blocks: Iterable[str],
) -> Tuple[List[str], int]:
    result: List[str] = []
    removed = 0

    for block in blocks:
        if normalize(object_type(block)) == "output:preprocessormessage":
            removed += 1
            continue
        result.append(block)

    return result, removed


def ensure_fraction_type(blocks: List[str]) -> Tuple[List[str], bool]:
    if fraction_type_exists(blocks):
        return blocks, False

    blocks.append(
        """
ScheduleTypeLimits,
  Fraction,                                 !- Name
  0,                                        !- Lower Limit Value
  1,                                        !- Upper Limit Value
  Continuous;                               !- Numeric Type
"""
    )
    return blocks, True


def ensure_ventilation_schedules(
    blocks: List[str],
    default_fraction: float,
) -> Tuple[List[str], List[str]]:
    added: List[str] = []

    for room in CONTROLLED_ROOMS:
        name = VENT_SCHEDULES[room]
        if schedule_constant_exists(blocks, name):
            continue
        blocks.append(standard_schedule_constant(name, default_fraction))
        added.append(name)

    return blocks, added


def ensure_core_outputs(
    blocks: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    added_variables: List[str] = []
    added_meters: List[str] = []

    for variable in CORE_OUTPUT_VARIABLES:
        if not output_variable_exists(blocks, variable):
            blocks.append(render_output_variable(variable))
            added_variables.append(variable)

    for meter in CORE_OUTPUT_METERS:
        if not output_meter_exists(blocks, meter):
            blocks.append(render_output_meter(meter))
            added_meters.append(meter)

    return blocks, added_variables, added_meters


def find_site_location(blocks: Iterable[str]) -> Dict[str, object]:
    for block in blocks:
        fields = idf_fields(block)
        if fields and normalize(fields[0]) == "site:location":
            result: Dict[str, object] = {
                "name": fields[1] if len(fields) > 1 else None,
                "latitude": None,
                "longitude": None,
                "time_zone": None,
                "elevation_m": None,
            }
            numeric_names = (
                ("latitude", 2),
                ("longitude", 3),
                ("time_zone", 4),
                ("elevation_m", 5),
            )
            for key, index in numeric_names:
                if len(fields) > index:
                    try:
                        result[key] = float(fields[index])
                    except Exception:
                        result[key] = fields[index]
            return result
    return {}


def validate_location(site: Dict[str, object]) -> List[str]:
    warnings: List[str] = []
    if not site:
        return ["Site:Location object not found."]

    try:
        lat = float(site.get("latitude"))
        lon = float(site.get("longitude"))
        tz = float(site.get("time_zone"))
        elev = float(site.get("elevation_m"))
    except Exception:
        return [f"Could not parse Site:Location numeric fields: {site}"]

    if abs(lat - 38.03) > 0.01:
        warnings.append(f"Expected Shijiazhuang latitude about 38.03, found {lat}.")
    if abs(lon - 114.42) > 0.01:
        warnings.append(
            f"Expected Shijiazhuang longitude about 114.42, found {lon}. "
            "Fix the source IDF before publication runs."
        )
    if abs(tz - 8.0) > 0.01:
        warnings.append(f"Expected Shijiazhuang time zone +8, found {tz}.")
    if abs(elev - 81.0) > 1.0:
        warnings.append(f"Expected Shijiazhuang elevation about 81 m, found {elev}.")

    return warnings


def find_timestep(blocks: Iterable[str]) -> Optional[float]:
    for block in blocks:
        fields = idf_fields(block)
        if fields and normalize(fields[0]) == "timestep" and len(fields) >= 2:
            try:
                return float(fields[1])
            except Exception:
                return None
    return None


def find_runperiod(blocks: Iterable[str]) -> Dict[str, object]:
    for block in blocks:
        fields = idf_fields(block)
        if fields and normalize(fields[0]) == "runperiod":
            # EnergyPlus 24.1:
            # [type, name, begin month, begin day, begin year,
            #  end month, end day, end year, ...]
            result: Dict[str, object] = {
                "name": fields[1] if len(fields) > 1 else None,
            }
            labels = (
                ("begin_month", 2),
                ("begin_day", 3),
                ("begin_year", 4),
                ("end_month", 5),
                ("end_day", 6),
                ("end_year", 7),
            )
            for key, index in labels:
                if len(fields) > index:
                    result[key] = fields[index]
            return result
    return {}


def count_object_type(blocks: Iterable[str], obj_type_name: str) -> int:
    target = normalize(obj_type_name)
    return sum(1 for block in blocks if normalize(object_type(block)) == target)


def validate_no_duplicate_outputs(blocks: Iterable[str]) -> None:
    seen = set()
    duplicates = []

    for block in blocks:
        key = output_object_key(block)
        if key is None:
            continue
        if key in seen:
            duplicates.append(key)
        seen.add(key)

    if duplicates:
        raise RuntimeError(
            "Duplicate Output:Variable/Output:Meter objects remain after cleanup: "
            + repr(duplicates[:10])
        )


def portable_path(path: Path, project_dir: Path) -> str:
    """Return a repository-relative POSIX path when possible.

    This keeps generated manifests and IDF headers portable and avoids embedding
    a contributor's local absolute path in files that may be committed to Git.
    """
    resolved_path = path.resolve()
    resolved_project = project_dir.resolve()

    try:
        return resolved_path.relative_to(resolved_project).as_posix()
    except ValueError:
        # The user explicitly selected a file outside the repository. Avoid
        # publishing the full machine-specific path; retain only the file name.
        return resolved_path.name


def resolve_input(path_arg: Optional[Path], project_dir: Path) -> Path:
    if path_arg is not None:
        path = path_arg
        if not path.is_absolute():
            path = project_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    input_dir = project_dir / INPUT_SUBDIR
    preferred = input_dir / DEFAULT_INPUT_NAME
    if preferred.exists():
        return preferred

    candidates = sorted(input_dir.glob("*.idf"))
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No IDF found in {input_dir}. Expected {preferred.name}"
        )

    raise RuntimeError(
        "Multiple IDF files found in inputs/building and the preferred file name "
        f"was not found. Use --input explicitly. Candidates: {candidates}"
    )


def write_manifest(path: Path, manifest: Dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the FCU model for closed-loop EnergyPlus experiments."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help=(
            "Repository root. Defaults to the repository containing this script."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional source IDF path. The source is never modified.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional generated IDF path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path.",
    )
    parser.add_argument(
        "--vent-default-fraction",
        type=float,
        default=1.0,
        help=(
            "Initial value of the six ventilation Schedule:Constant objects. "
            "Default 1.0 is a fail-safe; runtime code must overwrite it using "
            "measured occupancy/design_people."
        ),
    )
    args = parser.parse_args()

    project = args.project.resolve()

    if not (0.0 <= args.vent_default_fraction <= 1.0):
        print(
            "ERROR: --vent-default-fraction must be between 0 and 1.",
            file=sys.stderr,
        )
        return 2

    try:
        source = resolve_input(args.input, project)
    except Exception as exc:
        print(f"ERROR resolving input IDF: {exc}", file=sys.stderr)
        return 2

    output = (
        args.output
        if args.output is not None
        else project / GENERATED_SUBDIR / DEFAULT_OUTPUT_NAME
    )
    if not output.is_absolute():
        output = project / output

    manifest_path = (
        args.manifest
        if args.manifest is not None
        else project / GENERATED_SUBDIR / DEFAULT_MANIFEST_NAME
    )
    if not manifest_path.is_absolute():
        manifest_path = project / manifest_path

    if source.resolve() == output.resolve():
        print("ERROR: output path must differ from source path.", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Closed-LoopAgenticLLMs - EnergyPlus model preparation")
    print("=" * 78)
    print(f"Project : {project}")
    print(f"Source  : {source}")
    print(f"Output  : {output}")
    print(f"Manifest: {manifest_path}")
    print()

    original = source.read_text(encoding="utf-8-sig", errors="replace")
    original, stale_comment_lines_removed = strip_stale_warning_comments(original)

    blocks = split_idf_objects(original)

    if count_object_type(blocks, "Version") != 1:
        print("WARNING: expected exactly one Version object.")

    # Verify experimental People objects before touching ventilation.
    people_capacities = discover_people_capacities(blocks)

    # Remove stale preprocessor-message objects.
    blocks, preprocessor_objects_removed = remove_preprocessor_message_objects(
        blocks
    )

    # Remove exact duplicate output requests.
    blocks, duplicate_outputs_removed = remove_duplicate_outputs(blocks)

    # Patch six ZoneVentilation objects to dedicated runtime-actuated schedules.
    blocks, ventilation_info = patch_ventilation_objects(blocks)

    # Ensure ScheduleTypeLimits:Fraction and six Schedule:Constant objects.
    blocks, fraction_type_added = ensure_fraction_type(blocks)
    blocks, ventilation_schedules_added = ensure_ventilation_schedules(
        blocks,
        args.vent_default_fraction,
    )

    # Ensure key closed-loop diagnostics.
    blocks, added_variables, added_meters = ensure_core_outputs(blocks)

    # One final duplicate cleanup after additions.
    blocks, duplicate_outputs_removed_second_pass = remove_duplicate_outputs(
        blocks
    )
    duplicate_outputs_removed += duplicate_outputs_removed_second_pass

    validate_no_duplicate_outputs(blocks)

    timestep = find_timestep(blocks)
    runperiod = find_runperiod(blocks)
    site = find_site_location(blocks)
    warnings = validate_location(site)

    if timestep != 12:
        warnings.append(
            f"Expected Timestep=12 (5-minute timestep), found {timestep!r}."
        )

    fcu_count = count_object_type(blocks, "ZoneHVAC:FourPipeFanCoil")
    if fcu_count < 7:
        raise RuntimeError(
            f"Expected at least 7 ZoneHVAC:FourPipeFanCoil objects, found {fcu_count}."
        )

    if count_object_type(blocks, "Output:PreprocessorMessage") != 0:
        raise RuntimeError("Output:PreprocessorMessage objects remain after cleanup.")

    # Verify patched ventilation schedules in the final object list.
    final_schedule_names = {
        normalize(object_name(block))
        for block in blocks
        if normalize(object_type(block)) == "schedule:constant"
    }
    for room in CONTROLLED_ROOMS:
        schedule = VENT_SCHEDULES[room]
        if normalize(schedule) not in final_schedule_names:
            raise RuntimeError(f"Missing ventilation schedule: {schedule}")

    generated_header = f"""! ============================================================================
! GENERATED FILE - DO NOT EDIT MANUALLY
! Created by: {SCRIPT_NAME}
! Created UTC: {datetime.now(timezone.utc).isoformat()}
! Source IDF: {portable_path(source, project)}
!
! Six controlled rooms: 1, 2, 4, 5, 6, 7
! Ventilation strategy:
!   Existing ZoneVentilation:DesignFlowRate Flow/Person objects are retained.
!   Each controlled room uses a dedicated Schedule:Constant.
!   Runtime code must set:
!       schedule_fraction = measured_occupancy / design_people
!   The schedule default is {args.vent_default_fraction:.3f}.
! Zone 3 ventilation:
!   Thermal Zone 3 Ventilation per Person -> Office Work Occ
! ============================================================================
!
"""

    final_text = generated_header + "".join(blocks).lstrip()
    if not final_text.endswith("\n"):
        final_text += "\n"

    output.write_text(final_text, encoding="utf-8")

    for room in CONTROLLED_ROOMS:
        ventilation_info[room]["design_people"] = people_capacities[room]
        ventilation_info[room]["runtime_fraction_formula"] = (
            f"occupancy_count / {people_capacities[room]:g}"
        )
        ventilation_info[room]["default_schedule_fraction"] = (
            args.vent_default_fraction
        )

    manifest: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT_NAME,
        "project": ".",
        "source_idf": portable_path(source, project),
        "generated_idf": portable_path(output, project),
        "energyplus_target_version": "24.1",
        "controlled_rooms": list(CONTROLLED_ROOMS),
        "site_location": site,
        "timestep_per_hour": timestep,
        "runperiod": runperiod,
        "four_pipe_fan_coil_object_count": fcu_count,
        "people_design_capacities": {
            str(room): people_capacities[room] for room in CONTROLLED_ROOMS
        },
        "ventilation": {
            "strategy": (
                "ZoneVentilation:DesignFlowRate Flow/Person with "
                "runtime-actuated per-room Schedule:Constant fraction"
            ),
            "zone3_policy": (
                "Conventional/reference ventilation using the existing "
                "'Office Work Occ' schedule"
            ),
            "zone3": ventilation_info[3],
            "rooms": {
                str(room): ventilation_info[room] for room in CONTROLLED_ROOMS
            },
        },
        "cleanup": {
            "preprocessor_message_objects_removed": preprocessor_objects_removed,
            "stale_warning_comment_lines_removed": stale_comment_lines_removed,
            "duplicate_output_objects_removed": duplicate_outputs_removed,
        },
        "added": {
            "fraction_schedule_type_limits_added": fraction_type_added,
            "ventilation_schedules_added": ventilation_schedules_added,
            "output_variables_added": added_variables,
            "output_meters_added": added_meters,
        },
        "warnings": warnings,
        "next_required_step": (
            "Run 02_preflight_energyplus_api.py and verify thermostat, People, "
            "ventilation-schedule, FCU availability, and fan handles before "
            "running any benchmark."
        ),
    }
    write_manifest(manifest_path, manifest)

    print("Preparation complete.")
    print()
    print("Experimental People capacities:")
    for room in CONTROLLED_ROOMS:
        print(
            f"  Room {room}: {people_capacities[room]:g} people; "
            f"vent schedule='{VENT_SCHEDULES[room]}'"
        )

    print(
        "  Zone 3 reference ventilation: "
        f"'{ZONE3_VENTILATION_OBJECT}' -> '{ZONE3_VENTILATION_SCHEDULE}'"
    )

    print()
    print("Cleanup:")
    print(
        f"  Output:PreprocessorMessage objects removed: "
        f"{preprocessor_objects_removed}"
    )
    print(
        f"  Stale warning comment lines removed: "
        f"{stale_comment_lines_removed}"
    )
    print(
        f"  Duplicate Output:Variable/Output:Meter objects removed: "
        f"{duplicate_outputs_removed}"
    )

    print()
    if added_variables:
        print("Output variables added:")
        for item in added_variables:
            print(f"  + {item}")
    else:
        print("Output variables added: none")

    if added_meters:
        print("Output meters added:")
        for item in added_meters:
            print(f"  + {item}")
    else:
        print("Output meters added: none")

    if warnings:
        print()
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    print()
    print(f"Generated IDF: {output}")
    print(f"Manifest     : {manifest_path}")
    print()
    print(
        "NEXT: run 02_preflight_energyplus_api.py. Do not start LLM experiments until "
        "the required EnergyPlus actuators and feedback variables are verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

