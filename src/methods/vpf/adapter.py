from __future__ import annotations

from bruce_code.methods.common import GenericPolicyAdapter
from bruce_code.methods.vpf.policy import VPFPolicy


def build_vpf_adapter(
    device_name: str,
    detection_threshold: float = 5.0,
    attractive_gain: float = 3.0,
    repulsive_gain: float = 15.0,
    max_speed: float = 5.0,
    stuck_threshold: float = 1.0,
    goal_far_threshold: float = 2.0,
    tangential_gain: float = 2.0,
):
    policy = VPFPolicy(
        device_name=device_name,
        detection_threshold=detection_threshold,
        attractive_gain=attractive_gain,
        repulsive_gain=repulsive_gain,
        max_speed=max_speed,
        stuck_threshold=stuck_threshold,
        goal_far_threshold=goal_far_threshold,
        tangential_gain=tangential_gain,
    )
    return GenericPolicyAdapter(
        name="vpf",
        policy=policy,
        device_name=device_name,
        metadata={
            "controller": "virtual_potential_field",
            "detection_threshold": detection_threshold,
            "attractive_gain": attractive_gain,
            "repulsive_gain": repulsive_gain,
            "max_speed": max_speed,
            "stuck_threshold": stuck_threshold,
            "goal_far_threshold": goal_far_threshold,
            "tangential_gain": tangential_gain,
        },
    )
