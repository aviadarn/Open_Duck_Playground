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
