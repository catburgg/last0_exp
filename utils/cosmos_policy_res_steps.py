# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored minimal RES / Karras schedule helpers from cosmos-predict2.5 (see module docstring in repo).

from __future__ import annotations

from typing import List, Tuple, Union

import torch
from torch import Tensor


def get_rev_ts(
    t_min: float, t_max: float, num_steps: int, ts_order: Union[int, float], is_forward: bool = False
) -> torch.Tensor:
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max")
    if not isinstance(ts_order, (int, float)):
        raise TypeError("ts_order must be an integer or float")
    step_indices = torch.arange(num_steps + 1, dtype=torch.float64)
    time_steps = (
        t_max ** (1 / ts_order) + step_indices / num_steps * (t_min ** (1 / ts_order) - t_max ** (1 / ts_order))
    ) ** ts_order
    if is_forward:
        return time_steps.flip(dims=(0,))
    return time_steps


def common_broadcast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    ndims1 = x.ndim
    ndims2 = y.ndim
    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], "Dimensions not equal at axis {}".format(axis)
    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))
    return x, y


def batch_mul(x: Tensor, y: Tensor) -> Tensor:
    x, y = common_broadcast(x, y)
    return x * y


def phi1(t: Tensor) -> Tensor:
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return (torch.expm1(t) / t).to(dtype=input_dtype)


def phi2(t: Tensor) -> Tensor:
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return ((phi1(t) - 1.0) / t).to(dtype=input_dtype)


def res_x0_rk2_step(
    x_s: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    x0_s: torch.Tensor,
    s1: torch.Tensor,
    x0_s1: torch.Tensor,
) -> torch.Tensor:
    s = -torch.log(s)
    t = -torch.log(t)
    m = -torch.log(s1)
    dt = t - s
    assert not torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"
    assert not torch.any(torch.isclose(m - s, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"
    c2 = (m - s) / dt
    phi1_val, phi2_val = phi1(-dt), phi2(-dt)
    b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
    b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)
    return batch_mul(torch.exp(-dt), x_s) + batch_mul(dt, batch_mul(b1, x0_s) + batch_mul(b2, x0_s1))


def reg_x0_euler_step(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return batch_mul(coef_x0, x0_s) + batch_mul(coef_xs, x_s), x0_s


def order2_fn(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
    x0_preds: List,
) -> Tuple[torch.Tensor, List]:
    if x0_preds:
        x0_s1, s1 = x0_preds[0]
        x_t = res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1)
    else:
        x_t = reg_x0_euler_step(x_s, s, t, x0_s)[0]
    return x_t, [(x0_s, s)]


def policy_sigmas_like_cosmos_sampler(
    *,
    policy_num_steps: int = 35,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    num_steps = policy_num_steps
    if num_steps > 1:
        num_steps = num_steps - 1
    num_timestamps = num_steps
    return get_rev_ts(sigma_min, sigma_max, num_timestamps, rho)
