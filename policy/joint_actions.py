"""Joint targets shared by unified training and deployment.

Relative arm targets are anchored to the observation that generated the entire
chunk. The eighth value is always an absolute gripper position.
"""

JOINT_ACTION_MODES = ("absolute_joint", "relative_joint")


def joint_action_mode(mode):
    mode = {"absolute": "absolute_joint", "relative": "relative_joint"}.get(mode, mode)
    if mode not in JOINT_ACTION_MODES:
        raise ValueError(f"Unsupported joint action mode: {mode!r}; expected {JOINT_ACTION_MODES}")
    return mode


def _convert(state, actions, mode, direction):
    mode = joint_action_mode(mode)
    if state.shape[-1] != 8 or actions.shape[-1] != 8:
        raise ValueError("Joint state and action must contain seven arm joints and one gripper")
    result = actions.clone() if hasattr(actions, "clone") else actions.copy()
    if mode == "relative_joint":
        result[..., :7] += direction * state[..., None, :7]
    return result


def encode_joint_actions(state, actions, mode):
    return _convert(state, actions, mode, -1)


def decode_joint_actions(state, actions, mode):
    return _convert(state, actions, mode, 1)
