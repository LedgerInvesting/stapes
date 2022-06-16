from typing import Tuple, Union, Dict, Any, List

import numpy as np

DataCoords = Tuple[
    str,    # Name of the field
    int,    # Slice index
    int,    # Experience period index
    int,    # Development lag index
]

DataValue = Union[
    float,          # Observed scalar value,
    str,            # Placeholder for unrealized missing value
    np.ndarray,     # Realized missing value or forecasted value
]


def build_stan_data(
    train_data: Dict[DataCoords, DataValue],
    offsets: List[Tuple[int, int]]
) -> Dict[str, Any]:
    raw_index = set([coord[1:] for coord in train_data])
    core_index = _build_core_index(raw_index, offsets)
    full_index = core_index + list(raw_index - set(core_index))
    variables = set([name for name, _, _, _ in train_data])

    max_triangle_id = max([coord[0] for coord in core_index])

    stan_data = {
        "N": len(core_index),
        "T": len(full_index),
        **_build_index("TriangleId", core_index, lambda coord: coord[0]),
        **_build_index("ExpPeriodId", core_index, lambda coord: coord[1]),
        **_build_index("DevLagId", core_index, lambda coord: coord[2]),
        **_build_index("TriangleExpPeriodId", core_index,
                       lambda coord: max_triangle_id * (coord[1] - 1) + coord[0]),
        **_build_index("TriangleDevLagId", core_index,
                       lambda coord: max_triangle_id * (coord[2] - 1) + coord[0]),
        **{
            _offset_name(offset): _build_offset_lookup(core_index, full_index, offset)
            for offset in offsets
        }
    }
    for name in variables:
        stan_data = {**stan_data, **_build_variable(name, train_data, full_index)}

    return stan_data


def _build_core_index(raw_index, offsets):
    core_index = []
    for tri_id, exp_id, dev_id in raw_index:
        all_offsets_present = True
        for exp_offset, dev_offset in offsets:
            offset_coord = (tri_id, exp_id - exp_offset, dev_id - dev_offset)
            if offset_coord not in raw_index:
                all_offsets_present = False
        if all_offsets_present:
            core_index.append((tri_id, exp_id, dev_id))
    return core_index


def _offset_name(offset):
    exp_offset, dev_offset = offset
    return "Lag" + ("T" * exp_offset) + ("D" * dev_offset)


def _build_offset_lookup(core_index, full_index, offset):
    offset_lookup = []
    full_index_map = {coords: ndx+1 for ndx, coords in enumerate(full_index)}
    exp_offset, dev_offset = offset
    for tri_id, exp_id, dev_id in core_index:
        offset_lookup.append(full_index_map[(tri_id, exp_id - exp_offset, dev_id - dev_offset)])
    return offset_lookup


def _build_variable(name, train_data, full_index):
    values = []
    missing_ndxs = []
    for ndx, coord in enumerate(full_index):
        if (name, *coord) not in train_data:
            values.append(0)
            missing_ndxs.append(ndx+1)
        else:
            values.append(train_data[(name, *coord)])
    return {
        f"{name}__raw": values,
        f"{name}__num_missing": len(missing_ndxs),
        f"{name}__missing_ids": missing_ndxs
    }


def _build_index(name, core_index, index_fn):
    values = [index_fn(coord) for coord in core_index]
    return {
        name: values,
        f"{name}__count": max(values),
    }
