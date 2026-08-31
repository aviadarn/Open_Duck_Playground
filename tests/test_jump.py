"""Tests for the shared jump state machine."""

import numpy as np

from playground.open_duck_mini_v2.jump import (
    JUMP_COOLDOWN_STEPS,
    JUMP_MOTOR_VELOCITY,
    JUMP_WINDOW_STEPS,
    advance_jump,
)


def test_trigger_from_idle_opens_the_window():
    timer, cooldown, active = advance_jump(0, 0, 1)
    assert timer == JUMP_WINDOW_STEPS
    assert cooldown == 0
    assert active == 1.0


def test_no_trigger_stays_idle():
    timer, cooldown, active = advance_jump(0, 0, 0)
    assert timer == 0
    assert cooldown == 0
    assert active == 0.0


def test_window_counts_down():
    timer, cooldown, active = advance_jump(JUMP_WINDOW_STEPS, 0, 0)
    assert timer == JUMP_WINDOW_STEPS - 1
    assert active == 1.0


def test_window_end_starts_cooldown():
    timer, cooldown, active = advance_jump(1, 0, 0)
    assert timer == 0
    assert cooldown == JUMP_COOLDOWN_STEPS
    assert active == 0.0


def test_trigger_ignored_during_window():
    # Re-triggering mid-jump must not extend or restart the window.
    timer, _, _ = advance_jump(10, 0, 1)
    assert timer == 9


def test_trigger_ignored_during_cooldown():
    timer, cooldown, active = advance_jump(0, 10, 1)
    assert timer == 0
    assert cooldown == 9
    assert active == 0.0


def test_cooldown_expires_then_can_fire_again():
    timer, cooldown, _ = advance_jump(0, 1, 0)
    assert cooldown == 0
    timer, cooldown, active = advance_jump(timer, cooldown, 1)
    assert timer == JUMP_WINDOW_STEPS
    assert active == 1.0


def test_full_cycle_step_by_step():
    timer, cooldown = 0, 0
    timer, cooldown, active = advance_jump(timer, cooldown, 1)
    assert active == 1.0
    # Run out the window.
    for _ in range(JUMP_WINDOW_STEPS):
        timer, cooldown, active = advance_jump(timer, cooldown, 0)
    assert active == 0.0
    assert cooldown == JUMP_COOLDOWN_STEPS
    # Run out the cooldown.
    for _ in range(JUMP_COOLDOWN_STEPS):
        timer, cooldown, active = advance_jump(timer, cooldown, 0)
    assert cooldown == 0
    # Ready again.
    timer, cooldown, active = advance_jump(timer, cooldown, 1)
    assert active == 1.0


def test_works_with_numpy_arrays():
    timer = np.array(0)
    cooldown = np.array(0)
    timer, cooldown, active = advance_jump(timer, cooldown, np.array(1))
    assert int(timer) == JUMP_WINDOW_STEPS
    assert float(active) == 1.0


def test_works_under_jax_jit():
    import jax

    timer, cooldown, active = jax.jit(advance_jump)(0, 0, 1)
    assert int(timer) == JUMP_WINDOW_STEPS
    assert float(active) == 1.0


def test_jump_motor_velocity_value():
    assert JUMP_MOTOR_VELOCITY == 15.0


def test_observation_size_is_102_with_jump_command():
    from playground.open_duck_mini_v2.joystick import Joystick

    env = Joystick(task="flat_terrain")
    assert int(env.observation_size["state"][0]) == 102


def test_sample_command_returns_eight_values():
    import jax

    from playground.open_duck_mini_v2.joystick import Joystick

    env = Joystick(task="flat_terrain")
    cmd = env.sample_command(jax.random.PRNGKey(0))
    assert cmd.shape == (8,)


def test_jump_terms_are_zero_when_not_jumping():
    # The regression guard: with the jump inactive, every jump term must be
    # exactly zero so walking behaviour is unaffected.
    #
    # jump_prob is forced to 0 rather than relying on the default 1/250 not
    # firing -- otherwise this test flakes roughly once every 250 runs.
    import jax
    import jax.numpy as jp

    from playground.open_duck_mini_v2.joystick import Joystick, default_config

    cfg = default_config()
    cfg.jump_prob = 0.0
    env = Joystick(task="flat_terrain", config=cfg)

    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, jp.zeros(env.action_size))

    assert float(state.info["jump_active"]) == 0.0
    for key in ("jump_takeoff", "jump_air_time", "jump_height"):
        assert float(state.metrics[f"reward/{key}"]) == 0.0


def test_jump_flag_reaches_the_command_vector_when_triggered():
    # jump_prob forced to 1 so the trigger fires on the first step.
    import jax
    import jax.numpy as jp

    from playground.open_duck_mini_v2.joystick import Joystick, default_config

    cfg = default_config()
    cfg.jump_prob = 1.0
    env = Joystick(task="flat_terrain", config=cfg)

    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, jp.zeros(env.action_size))

    assert float(state.info["jump_active"]) == 1.0
    assert float(state.info["command"][7]) == 1.0


def test_jump_reward_scales_are_registered():
    from playground.open_duck_mini_v2.joystick import default_config

    scales = default_config().reward_config.scales
    assert scales.jump_takeoff == 30.0
    assert scales.jump_air_time == 40.0
    assert scales.jump_height == 60.0


def test_jump_reward_gate_is_nan_safe_for_non_finite_input():
    """Regression test for review finding (Task 3, fix round 1).

    A multiplicative gate (`value * jump_active`) does not guarantee exactly
    0.0 when jump_active == 0.0: jp.clip does not sanitize non-finite input
    (clip(nan, 0, inf) == nan, clip(inf, 0, inf) == inf), and IEEE-754 makes
    both absorbing under multiplication by zero (nan * 0.0 == nan and
    inf * 0.0 == nan). jump_takeoff/jump_height in joystick.py use jp.where
    instead, which selects the literal 0.0 on the inactive branch regardless
    of what the active branch evaluates to. This test exercises the two
    gating expressions directly (not via MuJoCo) and contrasts them.
    """
    import jax.numpy as jp

    jump_active = jp.float32(0.0)

    for bad_value in (jp.nan, jp.inf):
        clipped = jp.clip(jp.float32(bad_value), 0.0, jp.inf)

        # Document the bug: the multiplicative form leaks NaN instead of
        # producing exactly 0.0.
        multiplicative = clipped * jump_active
        assert jp.isnan(multiplicative), (
            "expected the multiplicative gate to demonstrate the bug "
            f"(got {multiplicative} for input {bad_value})"
        )

        # The fix actually used in joystick.py: jp.where is a branchless
        # select and is exact regardless of the other branch's value.
        gated = jp.where(jump_active > 0.0, clipped, 0.0)
        assert float(gated) == 0.0


def test_jump_reward_gate_is_nan_safe_via_real_reward_path():
    """Regression test for review finding (Task 3, fix round 2).

    The expression-level test above documents the arithmetic but never
    touches joystick.py, so it would not catch a revert of the two jp.where
    gates back to multiplicative gating. This test drives the real code
    path: it builds the env, reset()s it, corrupts the base's vertical
    velocity (inf) and height (nan) directly in the mjx.Data, forces
    jump_active = 0.0, and calls `_get_reward` directly.

    contact is forced all-True (grounded=True) rather than the more
    "natural" all-False, because jump_takeoff multiplies by `grounded`:
    with grounded=False the old buggy multiplicative form degenerates to
    `inf * 0.0 * 0.0`, which XLA/JAX evaluates to 0.0 -- accidentally
    masking the very bug this test exists to catch. With grounded=True the
    corrupted base_vz reaches the gate unmasked, so a revert to
    multiplicative gating actually fails this test (verified manually while
    writing it: reverting joystick.py's jp.where back to `* jump_active`
    made `ret["jump_takeoff"]` and `ret["jump_height"]` both come back as
    nan instead of 0.0, failing the asserts below).
    """
    import jax
    import jax.numpy as jp

    from playground.open_duck_mini_v2.joystick import Joystick, default_config

    cfg = default_config()
    cfg.jump_prob = 0.0
    env = Joystick(task="flat_terrain", config=cfg)

    state = env.reset(jax.random.PRNGKey(0))

    bad_qvel = state.data.qvel.at[env._floating_base_qvel_addr + 2].set(jp.inf)
    bad_qpos = state.data.qpos.at[env._floating_base_qpos_addr + 2].set(jp.nan)
    data = state.data.replace(qvel=bad_qvel, qpos=bad_qpos)

    info = dict(state.info)
    info["jump_active"] = jp.float32(0.0)

    contact = jp.ones(2, dtype=bool)  # grounded=True; see docstring.
    first_contact = jp.zeros(2, dtype=bool)
    done = jp.float32(0.0)

    ret = env._get_reward(
        data, jp.zeros(env.action_size), info, {}, done, first_contact, contact
    )

    assert float(ret["jump_takeoff"]) == 0.0
    assert float(ret["jump_height"]) == 0.0


def test_motor_clamp_matches_walking_limit_when_idle():
    from playground.open_duck_mini_v2.joystick import default_config

    cfg = default_config()
    jump_active = 0.0
    effective = (
        cfg.max_motor_velocity * (1.0 - jump_active)
        + cfg.jump_motor_velocity * jump_active
    )
    assert effective == cfg.max_motor_velocity


def test_motor_clamp_is_raised_during_jump():
    from playground.open_duck_mini_v2.joystick import default_config

    cfg = default_config()
    jump_active = 1.0
    effective = (
        cfg.max_motor_velocity * (1.0 - jump_active)
        + cfg.jump_motor_velocity * jump_active
    )
    assert effective == cfg.jump_motor_velocity
    assert effective > cfg.max_motor_velocity


def test_motor_clamp_enforces_jump_limit_on_real_code_path():
    """Regression test: the actual clamp in joystick.step() uses jump_active correctly.
    
    This test drives the real environment and verifies that motor_targets delta per
    step actually matches the configured bounds. The arithmetic-only tests above
    cannot catch reversions or swapped interpolation weights.
    
    With a large constant action, the raw motor target is always far outside the
    clamp band, so the achieved per-step delta becomes exactly the bound. The action
    delay (up to 3 steps) may produce small deltas early on, so we track the maximum
    delta across ~10 steps and assert it matches the jump/walking bounds.
    """
    import jax
    import jax.numpy as jp
    
    from playground.open_duck_mini_v2.joystick import Joystick, default_config
    
    # Build two environments: one that never jumps, one that jumps immediately.
    cfg_walk = default_config()
    cfg_walk.jump_prob = 0.0
    env_walk = Joystick(task="flat_terrain", config=cfg_walk)
    
    cfg_jump = default_config()
    cfg_jump.jump_prob = 1.0
    env_jump = Joystick(task="flat_terrain", config=cfg_jump)
    
    # Large constant action that saturates the clamp.
    large_action = jp.ones(env_walk.action_size) * 10.0
    
    # Step ~10 times and track max motor_targets delta.
    state_walk = env_walk.reset(jax.random.PRNGKey(0))
    max_delta_walk = 0.0
    for _ in range(10):
        prev_targets = state_walk.info["motor_targets"]
        state_walk = env_walk.step(state_walk, large_action)
        curr_targets = state_walk.info["motor_targets"]
        delta = jp.max(jp.abs(curr_targets - prev_targets))
        max_delta_walk = max(max_delta_walk, float(delta))
    
    state_jump = env_jump.reset(jax.random.PRNGKey(0))
    max_delta_jump = 0.0
    for _ in range(10):
        prev_targets = state_jump.info["motor_targets"]
        state_jump = env_jump.step(state_jump, large_action)
        curr_targets = state_jump.info["motor_targets"]
        delta = jp.max(jp.abs(curr_targets - prev_targets))
        max_delta_jump = max(max_delta_jump, float(delta))
    
    # The jump environment should have a larger max delta than the walking environment.
    assert max_delta_jump > max_delta_walk, (
        f"expected jump max_delta ({max_delta_jump}) > walk max_delta ({max_delta_walk})"
    )
    
    # Walking env's max delta should not exceed the walking bound (with small tolerance).
    walking_bound = cfg_walk.max_motor_velocity * env_walk.dt
    tolerance = 1e-5
    assert max_delta_walk <= walking_bound + tolerance, (
        f"expected walk max_delta ({max_delta_walk}) <= bound ({walking_bound})"
    )
    
    # Jump env's max delta should be close to the jump bound (with small tolerance).
    jump_bound = cfg_jump.jump_motor_velocity * env_jump.dt
    assert abs(max_delta_jump - jump_bound) < tolerance, (
        f"expected jump max_delta ({max_delta_jump}) ≈ bound ({jump_bound})"
    )


def test_inference_command_vector_is_eight_long():
    # Training and inference must agree on the command width, or the ONNX
    # policy is fed a differently-shaped observation than it trained on.
    import inspect

    from playground.open_duck_mini_v2 import mujoco_infer

    src = inspect.getsource(mujoco_infer.MjInfer.__init__)
    assert "self.commands = [0.0] * 8" in src


def test_inference_uses_shared_jump_constants():
    # Guards against the window length drifting between train and inference.
    import inspect

    from playground.open_duck_mini_v2 import mujoco_infer

    src = inspect.getsource(mujoco_infer)
    assert "from playground.open_duck_mini_v2.jump import" in src
    assert "advance_jump" in src
