#!/usr/bin/env python3
"""
Generate libero_object_target_xy_variance BDDL files.

For each task in libero_object/:
- Add random position variance to the target object within a 20cm x 20cm grid
- Use Minkowski sum collision detection with object bounding boxes in XY
- Recursively push colliding objects to resolve all collisions
"""

import os
import re
import copy
import random
import math
from pathlib import Path

SEED = 42

# Horizontal radius per object (meters) — from horizontal_radius_site in MuJoCo XML.
# This is the circular bounding radius in XY, rotation-invariant.
OBJECT_HORIZONTAL_RADIUS = {
    "alphabet_soup":     0.035,
    "cream_cheese":      0.042,
    "salad_dressing":    0.035,
    "bbq_sauce":         0.035,
    "ketchup":           0.035,
    "tomato_sauce":      0.035,
    "butter":            0.028,
    "milk":              0.042,
    "chocolate_pudding": 0.035,
    "orange_juice":      0.042,
    "basket":            0.10,   # large woven basket
}

# Workspace bounds (tight, matching robot-reachable area)
WS_X_MIN, WS_X_MAX = -0.20, 0.15
WS_Y_MIN, WS_Y_MAX = -0.25, 0.26

# Variance grid: 20cm x 20cm => +/-10cm
VARIANCE_HALF = 0.10

# Minimum center-to-center distance between any two objects (15cm)
MIN_OBJECT_DISTANCE = 0.15


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


def region_half_extent(r):
    return ((r[2] - r[0]) / 2, (r[3] - r[1]) / 2)


def shift_region(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy)


def obj_type_from_instance(instance_name: str) -> str:
    """e.g. 'alphabet_soup_1' -> 'alphabet_soup'"""
    return "_".join(instance_name.split("_")[:-1])


def parse_init_placements(content: str):
    """Return list of (obj_instance, region_ref) from (:init ...)"""
    placements = []
    init_match = re.search(r"\(:init\s*\n(.*?)\n\s*\)", content, re.S)
    if init_match:
        for m in re.finditer(r"\(On\s+(\w+)\s+(\w+)\)", init_match.group(1)):
            placements.append((m.group(1), m.group(2)))
    return placements


def get_obj_collision_radius(obj_instance: str, region_range=None):
    """Return effective collision radius: object horizontal_radius + region half-extent.

    The region half-extent accounts for the fact that the object centre is sampled
    randomly anywhere inside its placement region, so worst-case it sits at the
    region edge.
    """
    otype = obj_type_from_instance(obj_instance)
    obj_r = OBJECT_HORIZONTAL_RADIUS.get(otype, 0.035)
    if region_range is not None:
        # Add region half-diagonal so we cover the worst-case placement
        rhx = (region_range[2] - region_range[0]) / 2
        rhy = (region_range[3] - region_range[1]) / 2
        obj_r += math.sqrt(rhx * rhx + rhy * rhy)
    return obj_r


def minkowski_collides(center_a, radius_a, center_b, radius_b, margin=0.0):
    """Check if two circles (Minkowski sum of bounding disks) overlap."""
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    dist_sq = dx * dx + dy * dy
    min_dist = radius_a + radius_b + margin
    return dist_sq < min_dist * min_dist


def too_close(center_a, center_b):
    """Check if two object centers are closer than MIN_OBJECT_DISTANCE."""
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    return (dx * dx + dy * dy) < MIN_OBJECT_DISTANCE * MIN_OBJECT_DISTANCE


def resolve_collisions(objects, moved_idx, max_depth=20):
    """Recursively resolve collisions after moving objects[moved_idx].

    Ensures all object pairs are at least MIN_OBJECT_DISTANCE apart.
    Pushes the second object along the center-to-center line.
    """
    if max_depth <= 0:
        return

    mover = objects[moved_idx]
    for i, other in enumerate(objects):
        if i == moved_idx:
            continue
        if not too_close(mover["center"], other["center"]):
            continue

        # Too close: push 'other' away from 'mover' along center line
        cx_diff = other["center"][0] - mover["center"][0]
        cy_diff = other["center"][1] - mover["center"][1]
        dist = math.sqrt(cx_diff * cx_diff + cy_diff * cy_diff)

        if dist < 1e-6:
            push_dx = MIN_OBJECT_DISTANCE
            push_dy = 0.0
        else:
            nx, ny = cx_diff / dist, cy_diff / dist
            gap = MIN_OBJECT_DISTANCE - dist
            push_dx = nx * gap
            push_dy = ny * gap

        # Apply push
        new_cx = other["center"][0] + push_dx
        new_cy = other["center"][1] + push_dy

        # Update object
        shift_x = new_cx - other["center"][0]
        shift_y = new_cy - other["center"][1]
        other["center"] = (new_cx, new_cy)
        other["region"] = shift_region(other["region"], shift_x, shift_y)

        # Recurse on the pushed object
        resolve_collisions(objects, i, max_depth - 1)


def update_region_in_content(content: str, region_name: str, new_range):
    """Replace a region's (:ranges ...) values in the BDDL string."""
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


def _has_remaining_collisions(objects):
    """Check if any pair of objects is closer than MIN_OBJECT_DISTANCE."""
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if too_close(objects[i]["center"], objects[j]["center"]):
                return True
    return False


MAX_PLACEMENT_RETRIES = 50


def perturb_bddl_content(content: str, rng: random.Random):
    """Apply random XY variance to the target object region in BDDL content.

    Shifts the target by a random (dx, dy), then recursively pushes any
    colliding objects outward.

    Returns (new_content, all_shifts) or (content, None) if no target found.
    all_shifts is a dict: obj_instance_name -> (dx, dy) for every object that moved.
    """
    regions = parse_regions(content)
    placements = parse_init_placements(content)

    # Build original object list.
    # Use object horizontal_radius only (no region extent) for collision
    # checking — region centers are what we control, and the LIBERO sampler
    # can handle small offsets within the region box.
    orig_objects = []
    for obj_inst, region_ref in placements:
        rname = region_ref.replace("floor_", "", 1)
        if rname not in regions:
            continue
        r = regions[rname]
        otype = obj_type_from_instance(obj_inst)
        obj_r = OBJECT_HORIZONTAL_RADIUS.get(otype, 0.035)
        orig_objects.append({
            "name": obj_inst,
            "center": region_center(r),
            "radius": obj_r,
            "region": r,
            "region_name": rname,
        })

    # Find target object index
    target_idx = None
    for i, obj in enumerate(orig_objects):
        if obj["region_name"] == "target_object_region":
            target_idx = i
            break

    if target_idx is None:
        return content, None

    # Save original centers for computing shifts later
    orig_centers = {obj["name"]: obj["center"] for obj in orig_objects}

    objects = copy.deepcopy(orig_objects)
    target = objects[target_idx]

    dx = rng.uniform(-VARIANCE_HALF, VARIANCE_HALF)
    dy = rng.uniform(-VARIANCE_HALF, VARIANCE_HALF)

    new_cx = target["center"][0] + dx
    new_cy = target["center"][1] + dy

    r = target["radius"]

    actual_dx = new_cx - target["center"][0]
    actual_dy = new_cy - target["center"][1]

    target["center"] = (new_cx, new_cy)
    target["region"] = shift_region(target["region"], actual_dx, actual_dy)

    # Recursively push colliding objects away from target
    resolve_collisions(objects, target_idx)

    # Global pairwise pass: ensure ALL objects are at least 15cm apart
    for _pass in range(30):
        if not _has_remaining_collisions(objects):
            break
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if too_close(objects[i]["center"], objects[j]["center"]):
                    resolve_collisions(objects, i, max_depth=10)

    # Compute shifts for all objects that moved
    all_shifts = {}
    for obj in objects:
        orig_c = orig_centers[obj["name"]]
        sdx = obj["center"][0] - orig_c[0]
        sdy = obj["center"][1] - orig_c[1]
        if abs(sdx) > 1e-6 or abs(sdy) > 1e-6:
            all_shifts[obj["name"]] = (sdx, sdy)

    # Update content
    new_content = content
    for obj in objects:
        rname = obj["region_name"]
        new_content = update_region_in_content(new_content, rname, obj["region"])

    return new_content, all_shifts


def process_bddl_file(input_path: str, output_path: str, rng: random.Random):
    with open(input_path, "r") as f:
        content = f.read()

    new_content, all_shifts = perturb_bddl_content(content, rng)
    if all_shifts is None:
        print(f"  WARNING: no target_object_region found in {input_path}")
    else:
        print(f"  {len(all_shifts)} objects shifted: "
              + ", ".join(f"{k}=({dx:+.4f},{dy:+.4f})" for k, (dx, dy) in all_shifts.items()))

    with open(output_path, "w") as f:
        f.write(new_content)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    src_dir = base_dir / "third_party/LIBERO-PRO/libero/libero/bddl_files/libero_object"
    dst_dir = base_dir / "third_party/LIBERO-PRO/libero/libero/bddl_files/libero_object_target_xy_variance"
    dst_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    bddl_files = sorted(src_dir.glob("*.bddl"))
    print(f"Processing {len(bddl_files)} BDDL files from {src_dir}")
    print(f"Output: {dst_dir}\n")

    for fpath in bddl_files:
        print(f"Processing: {fpath.name}")
        out_path = dst_dir / fpath.name
        process_bddl_file(str(fpath), str(out_path), rng)

    # Copy tasks_info.txt if it exists
    tasks_info = src_dir / "tasks_info.txt"
    if tasks_info.exists():
        import shutil
        shutil.copy2(tasks_info, dst_dir / "tasks_info.txt")

    print(f"\nDone. Generated {len(bddl_files)} files in {dst_dir}")


if __name__ == "__main__":
    main()
