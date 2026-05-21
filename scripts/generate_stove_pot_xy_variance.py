#!/usr/bin/env python3
"""
Stove + pot variance generator for libero_90 task 45
(KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it).

Per-trial placement is sampled from two mutually-exclusive modes:
  - 10%: pot on the shelf top, stove on the table with XY variance.
  - 90%: both on the table, each with +/-10cm XY variance.

(A symmetric "stove on the shelf" mode was attempted but LIBERO's
placement code only permits :objects — not :fixtures like flat_stove_1
— to be placed on another fixture's site region. The 10% bucket that
would have held stove-on-shelf is currently absorbed into the table
mode; flipping it back on requires either re-typing the stove as an
object (which would break the Turnon predicate) or patching
``bddl_base_domain._add_placement_initializer`` to look up
``fixtures_dict`` when ``objects_dict`` misses.)

The "table with XY variance" mode shifts the relevant
``*_init_region`` and uses a frypan-handle-aware Minkowski collision
resolver to push neighbouring objects (white bowl, shelf) outward.

Collision is checked with a frypan-handle-aware bounding shape, not just
a single disc:
  - Pan body: disc of radius ``CHEFMATE_FRYPAN_BODY_R`` (~0.10m) at the
    placed centre.
  - Handle: line segment from the pan edge outward in body-frame +x for
    ``CHEFMATE_FRYPAN_HANDLE_REACH`` metres (the BDDL uses yaw=0 so
    body +x == world +x).
A neighbour collides if either the pan-body disc overlaps the
neighbour's disc, or the handle segment passes within
``HANDLE_THICKNESS`` of the neighbour's disc.

Usage as a library:
    from generate_stove_pot_xy_variance import perturb_bddl_content
    new_bddl, shifts = perturb_bddl_content(orig_bddl, random.Random(seed))

Standalone (bakes one perturbed BDDL for inspection):
    python scripts/generate_stove_pot_xy_variance.py \
        --src libero/libero/bddl_files/libero_90/KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it.bddl \
        --dst /tmp/perturbed.bddl --seed 0
"""

import argparse
import copy
import math
import random
import re
from pathlib import Path

SEED = 42

# Per-object disc radii (meters). These are conservative XY footprints
# derived from the LIBERO MJCF geometry.
#   flat_stove: base box is 0.095 half-extent => 0.135m diagonal radius.
#   chefmate_8_frypan: pan body is ~0.10m radius (geoms at +/-0.09 on
#     each axis). Handle protrudes in +x — handled below by a segment
#     check, not the disc.
OBJECT_HORIZONTAL_RADIUS = {
    "flat_stove":             0.135,
    "chefmate_8_frypan":      0.100,
    "moka_pot":               0.060,
    "white_bowl":             0.065,
    "black_bowl":             0.060,
    "akita_black_bowl":       0.060,
    "plate":                  0.090,
    "wooden_two_layer_shelf": 0.150,
}
DEFAULT_RADIUS = 0.060

# Region-name → object-type mapping for region-only references.
REGION_TYPE_HINT = {
    "flat_stove_init_region":             "flat_stove",
    "frypan_init_region":                 "chefmate_8_frypan",
    "moka_pot_init_region":               "moka_pot",
    "white_bowl_init_region":             "white_bowl",
    "black_bowl_init_region":             "black_bowl",
    "akita_black_bowl_init_region":       "akita_black_bowl",
    "plate_init_region":                  "plate",
    "wooden_two_layer_shelf_init_region": "wooden_two_layer_shelf",
}

# Chefmate frypan: pan body radius and handle reach. The handle is fixed
# in body-frame +x (BDDL sets yaw_rotation=(0,0)), with geoms at
# (0.10, ...) → (0.19, ...). HANDLE_REACH covers the furthest geom.
CHEFMATE_FRYPAN_BODY_R = 0.100
CHEFMATE_FRYPAN_HANDLE_NEAR = 0.080   # where pan body ends / handle begins
CHEFMATE_FRYPAN_HANDLE_FAR = 0.200    # handle tip (+x direction in body frame)
CHEFMATE_FRYPAN_HANDLE_THICKNESS = 0.020

# XY variance grid: +/-10cm per axis for table placement (stove + pot).
VARIANCE_HALF = 0.10

# Buffer added on top of (r_a + r_b) for the disc-vs-disc check. Keeps a
# small visual gap so neighbouring objects don't end up flush against
# each other.
COLLISION_MARGIN = 0.020

# Per-trial mode probabilities. Mutually exclusive; remainder is the
# default "both on the table" mode. STOVE_ON_SHELF_PROB is forced to 0
# because LIBERO's placement initializer can't put fixtures on a
# fixture's site region — see the module docstring.
STOVE_ON_SHELF_PROB = 0.0
POT_ON_SHELF_PROB = 0.10

# Init clauses we may replace when routing the pot or stove to the shelf.
POT_INSTANCE = "chefmate_8_frypan_1"
STOVE_INSTANCE = "flat_stove_1"
POT_TABLE_REGION = "kitchen_table_frypan_init_region"
STOVE_TABLE_REGION = "kitchen_table_flat_stove_init_region"
SHELF_TOP_REGION = "wooden_two_layer_shelf_1_top_side"

# Regions we actively perturb when on the table.
TABLE_PERTURB_REGIONS = ("flat_stove_init_region", "frypan_init_region")


def parse_regions(content: str):
    """Parse all regions with (:ranges ...) from BDDL content.

    Returns dict: region_name -> (x_min, y_min, x_max, y_max)
    """
    regions = {}
    pattern = (
        r'\((\w+_region(?:_\d+)?)\s+'
        r'\(:target\s+\w+\)\s+'
        r'\(:ranges\s*\(\s*'
        r'\(([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\)'
    )
    for m in re.finditer(pattern, content, re.S):
        name = m.group(1)
        regions[name] = (float(m.group(2)), float(m.group(3)),
                         float(m.group(4)), float(m.group(5)))
    return regions


def region_center(r):
    return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)


def shift_region(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy)


def parse_init_placements(content: str):
    """Return list of (obj_instance, region_ref) from (:init ...)."""
    placements = []
    init_match = re.search(r"\(:init\s*\n(.*?)\n\s*\)", content, re.S)
    if init_match:
        for m in re.finditer(r"\(On\s+(\w+)\s+(\w+)\)", init_match.group(1)):
            placements.append((m.group(1), m.group(2)))
    return placements


def obj_type_from_instance(instance_name: str) -> str:
    parts = instance_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return instance_name


def get_radius_for(instance_name: str, region_name: str) -> float:
    otype = obj_type_from_instance(instance_name)
    if otype in OBJECT_HORIZONTAL_RADIUS:
        return OBJECT_HORIZONTAL_RADIUS[otype]
    hint = REGION_TYPE_HINT.get(region_name)
    if hint in OBJECT_HORIZONTAL_RADIUS:
        return OBJECT_HORIZONTAL_RADIUS[hint]
    return DEFAULT_RADIUS


def _disc_disc_overlaps(c_a, r_a, c_b, r_b, margin=COLLISION_MARGIN):
    dx = c_a[0] - c_b[0]
    dy = c_a[1] - c_b[1]
    threshold = r_a + r_b + margin
    return (dx * dx + dy * dy) < threshold * threshold


def _segment_disc_overlaps(p0, p1, c, r):
    """Closest-point-on-segment to disc check."""
    vx = p1[0] - p0[0]
    vy = p1[1] - p0[1]
    wx = c[0] - p0[0]
    wy = c[1] - p0[1]
    vv = vx * vx + vy * vy
    if vv < 1e-12:
        # Degenerate segment — fall back to point-disc.
        dx, dy = c[0] - p0[0], c[1] - p0[1]
        return (dx * dx + dy * dy) < r * r
    t = max(0.0, min(1.0, (vx * wx + vy * wy) / vv))
    qx = p0[0] + t * vx
    qy = p0[1] + t * vy
    dx = c[0] - qx
    dy = c[1] - qy
    return (dx * dx + dy * dy) < r * r


def _frypan_handle_segment(frypan_center):
    """World-XY endpoints of the frypan handle (yaw=0 ⇒ +x in world)."""
    cx, cy = frypan_center
    return ((cx + CHEFMATE_FRYPAN_HANDLE_NEAR, cy),
            (cx + CHEFMATE_FRYPAN_HANDLE_FAR, cy))


def objects_collide(a, b):
    """Handle-aware collision between two objects.

    Args are object dicts with keys 'name', 'center', 'radius'.
    """
    pan_a = a["name"].startswith("chefmate_8_frypan")
    pan_b = b["name"].startswith("chefmate_8_frypan")

    # Disc-disc check for the pan body and the neighbour's bounding disc.
    if _disc_disc_overlaps(a["center"], a["radius"], b["center"], b["radius"]):
        return True

    # If either is the frypan, also check its handle line segment against
    # the other's disc.
    if pan_a:
        p0, p1 = _frypan_handle_segment(a["center"])
        if _segment_disc_overlaps(
            p0, p1, b["center"], b["radius"] + CHEFMATE_FRYPAN_HANDLE_THICKNESS
        ):
            return True
    if pan_b:
        p0, p1 = _frypan_handle_segment(b["center"])
        if _segment_disc_overlaps(
            p0, p1, a["center"], a["radius"] + CHEFMATE_FRYPAN_HANDLE_THICKNESS
        ):
            return True
    return False


class _FrozenCollision(Exception):
    pass


def _push_pair(a, b, gap):
    """Push ``b`` along the a→b axis by ``gap`` metres."""
    cx_diff = b["center"][0] - a["center"][0]
    cy_diff = b["center"][1] - a["center"][1]
    dist = math.sqrt(cx_diff * cx_diff + cy_diff * cy_diff)
    if dist < 1e-6:
        push_dx, push_dy = gap, 0.0
    else:
        nx, ny = cx_diff / dist, cy_diff / dist
        push_dx, push_dy = nx * gap, ny * gap
    b["center"] = (b["center"][0] + push_dx, b["center"][1] + push_dy)
    b["region"] = shift_region(b["region"], push_dx, push_dy)


def _required_push_gap(a, b):
    """How far to push b away from a so they no longer collide.

    Returns the smallest gap that resolves both the disc-disc overlap
    and the handle-segment overlap (if applicable). The push direction
    is computed by caller along the a→b axis, which is correct for
    disc-disc but only approximate for handle overlap — we add a small
    extra margin to compensate.
    """
    cx_diff = b["center"][0] - a["center"][0]
    cy_diff = b["center"][1] - a["center"][1]
    dist = math.sqrt(cx_diff * cx_diff + cy_diff * cy_diff)

    disc_min = a["radius"] + b["radius"] + COLLISION_MARGIN
    gap = max(0.0, disc_min - dist)

    # Handle-aware top-up: if the frypan's handle still overlaps b's disc
    # after a disc-disc push, add an extra margin so the line clears.
    extra = 0.0
    if a["name"].startswith("chefmate_8_frypan"):
        # Worst case the handle is between a and b along their axis.
        extra = CHEFMATE_FRYPAN_HANDLE_FAR - a["radius"] + \
                CHEFMATE_FRYPAN_HANDLE_THICKNESS
        gap = max(gap, max(0.0, extra - dist))
    if b["name"].startswith("chefmate_8_frypan"):
        extra = CHEFMATE_FRYPAN_HANDLE_FAR - b["radius"] + \
                CHEFMATE_FRYPAN_HANDLE_THICKNESS
        gap = max(gap, max(0.0, extra - dist))
    return gap


def resolve_collisions(objects, moved_idx, frozen_idxs, max_depth=20):
    """Push neighbours away from objects[moved_idx]; recurse into chains.

    Frozen indices may not themselves be moved; if a chain forces a
    frozen object to move, raises _FrozenCollision so the caller can
    redraw the perturbation.
    """
    if max_depth <= 0:
        return
    mover = objects[moved_idx]
    for i, other in enumerate(objects):
        if i == moved_idx:
            continue
        if not objects_collide(mover, other):
            continue
        if i in frozen_idxs:
            raise _FrozenCollision(moved_idx, i)
        gap = _required_push_gap(mover, other)
        if gap > 0.0:
            _push_pair(mover, other, gap)
        resolve_collisions(objects, i, frozen_idxs, max_depth - 1)


def _has_remaining_collisions(objects):
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects_collide(objects[i], objects[j]):
                return True
    return False


def update_region_in_content(content: str, region_name: str, new_range):
    pattern = (
        r'(\(' + re.escape(region_name) +
        r'\s+\(:target\s+\w+\)\s+\(:ranges\s*\(\s*\()'
        r'[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+'
        r'(\))'
    )
    replacement = rf'\g<1>{new_range[0]} {new_range[1]} {new_range[2]} {new_range[3]}\2'
    new_content, n = re.subn(pattern, replacement, content, count=1, flags=re.S)
    if n == 0:
        print(f"  WARNING: could not update region {region_name}")
    return new_content


def _replace_init_clause(content: str, instance: str, new_region: str) -> str:
    """Rewrite ``(On <instance> <old>)`` → ``(On <instance> <new>)``."""
    pattern = (
        r'(\(On\s+' + re.escape(instance) + r'\s+)\w+(\))'
    )
    replacement = rf'\g<1>{new_region}\g<2>'
    new_content, n = re.subn(pattern, replacement, content, count=1)
    if n == 0:
        print(f"  WARNING: could not rewrite (On {instance} ...) init clause")
    return new_content


MAX_PLACEMENT_RETRIES = 80


def perturb_bddl_content(content: str, rng: random.Random, *, table_only: bool = False):
    """Apply stove XY variance and route the pot to a randomly chosen target.

    Returns ``(new_content, shifts_dict)`` or ``(content, None)`` if the
    expected regions aren't present or all placement retries failed.
    ``shifts_dict`` keys are object instance names; values are (dx, dy).
    A pot routed to the shelf top is reported with shift (None, None).
    """
    regions = parse_regions(content)
    placements = parse_init_placements(content)

    if "flat_stove_init_region" not in regions:
        return content, None
    if "frypan_init_region" not in regions:
        return content, None

    # Build the object list. Strip fixture prefixes from region refs in
    # (On obj region) clauses to find the base region by name.
    orig_objects = []
    for obj_inst, region_ref in placements:
        rname = None
        if region_ref in regions:
            rname = region_ref
        else:
            for prefix in ("kitchen_table_", "floor_", "study_table_"):
                if region_ref.startswith(prefix):
                    candidate = region_ref[len(prefix):]
                    if candidate in regions:
                        rname = candidate
                        break
        if rname is None:
            continue
        r = regions[rname]
        orig_objects.append({
            "name": obj_inst,
            "center": region_center(r),
            "radius": get_radius_for(obj_inst, rname),
            "region": r,
            "region_name": rname,
        })

    orig_centers = {obj["name"]: obj["center"] for obj in orig_objects}

    # Sample the per-trial placement mode up front.
    #   'stove_on_shelf' : 10% — stove goes on shelf, pot stays on table
    #                      with XY variance.
    #   'pot_on_shelf'   : 10% — pot goes on shelf, stove stays on table
    #                      with XY variance.
    #   'table_both'     : 80% — both stay on the table, both get XY
    #                      variance.
    if table_only:
        # Caller (e.g. the popcorn-production bake) needs both objects
        # on the table — shelf placement would mean the agent has to
        # un-shelve before the recipe even starts.
        mode = "table_both"
    else:
        roll = rng.random()
        if roll < STOVE_ON_SHELF_PROB:
            mode = "stove_on_shelf"
        elif roll < STOVE_ON_SHELF_PROB + POT_ON_SHELF_PROB:
            mode = "pot_on_shelf"
        else:
            mode = "table_both"

    for _attempt in range(MAX_PLACEMENT_RETRIES):
        objects = copy.deepcopy(orig_objects)

        stove_idx = next(
            (i for i, o in enumerate(objects)
             if o["region_name"] == "flat_stove_init_region"), None,
        )
        frypan_idx = next(
            (i for i, o in enumerate(objects)
             if o["region_name"] == "frypan_init_region"), None,
        )
        if stove_idx is None or frypan_idx is None:
            return content, None

        # Apply XY perturbation to whichever objects stay on the table.
        # An object routed to the shelf is dropped from the table-side
        # collision graph; the shelf is far enough away that vertical
        # placement there is always safe.
        def _perturb_xy(idx):
            dx = rng.uniform(-VARIANCE_HALF, VARIANCE_HALF)
            dy = rng.uniform(-VARIANCE_HALF, VARIANCE_HALF)
            objects[idx]["center"] = (
                objects[idx]["center"][0] + dx,
                objects[idx]["center"][1] + dy,
            )
            objects[idx]["region"] = shift_region(
                objects[idx]["region"], dx, dy
            )

        def _drop(idx):
            objects.pop(idx)
            # Re-index any other tracked indices.
            return idx

        if mode == "stove_on_shelf":
            _perturb_xy(frypan_idx)
            removed = _drop(stove_idx)
            if frypan_idx > removed:
                frypan_idx -= 1
            stove_idx = None
            frozen = {frypan_idx}
        elif mode == "pot_on_shelf":
            _perturb_xy(stove_idx)
            removed = _drop(frypan_idx)
            if stove_idx > removed:
                stove_idx -= 1
            frypan_idx = None
            frozen = {stove_idx}
        else:  # table_both
            _perturb_xy(stove_idx)
            _perturb_xy(frypan_idx)
            if objects_collide(objects[stove_idx], objects[frypan_idx]):
                continue
            frozen = {stove_idx, frypan_idx}

        # Push neighbours away from frozen objects, propagating chains.
        try:
            for idx in list(frozen):
                resolve_collisions(objects, idx, frozen)
        except _FrozenCollision:
            continue

        # Final pairwise sweep for indirect collisions.
        bad = False
        for _pass in range(30):
            if not _has_remaining_collisions(objects):
                break
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    if objects_collide(objects[i], objects[j]):
                        if i in frozen and j in frozen:
                            bad = True
                            break
                        try:
                            if i in frozen:
                                resolve_collisions(objects, i, frozen, max_depth=10)
                            else:
                                resolve_collisions(objects, j, frozen, max_depth=10)
                        except _FrozenCollision:
                            bad = True
                            break
                if bad:
                    break
            if bad:
                break
        if bad or _has_remaining_collisions(objects):
            continue

        # Success — collect shifts, rewrite the BDDL.
        all_shifts = {}
        for obj in objects:
            orig_c = orig_centers.get(obj["name"])
            if orig_c is None:
                continue
            sx = obj["center"][0] - orig_c[0]
            sy = obj["center"][1] - orig_c[1]
            if abs(sx) > 1e-6 or abs(sy) > 1e-6:
                all_shifts[obj["name"]] = (sx, sy)

        new_content = content
        for obj in objects:
            new_content = update_region_in_content(
                new_content, obj["region_name"], obj["region"]
            )
        if mode == "stove_on_shelf":
            new_content = _replace_init_clause(
                new_content, STOVE_INSTANCE, SHELF_TOP_REGION
            )
            all_shifts[STOVE_INSTANCE] = ("shelf", "top")
        elif mode == "pot_on_shelf":
            new_content = _replace_init_clause(
                new_content, POT_INSTANCE, SHELF_TOP_REGION
            )
            all_shifts[POT_INSTANCE] = ("shelf", "top")
        return new_content, all_shifts

    return content, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source BDDL file")
    ap.add_argument("--dst", required=True, help="Output perturbed BDDL")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    with open(args.src) as f:
        content = f.read()

    new_content, shifts = perturb_bddl_content(content, random.Random(args.seed))
    if shifts is None:
        print("Failed to generate a valid perturbation (no regions / collisions).")
    else:
        print(f"Shifted {len(shifts)} objects:")
        for k, v in shifts.items():
            if isinstance(v[0], str):
                print(f"  {k}: routed to {v[0]}-{v[1]}")
            else:
                print(f"  {k}: ({v[0]:+.4f}, {v[1]:+.4f})")

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w") as f:
        f.write(new_content)
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
