"""Shared jump timing for training (JAX) and inference (numpy).

The state machine is deliberately backend agnostic: it uses only arithmetic and
comparison operators, never `jax.numpy` or `numpy` calls, so one implementation
runs under JAX tracing and under plain numpy. That is the point -- if training
and inference disagreed about the window length, the policy would see a
different command profile at inference than it trained on and the jump would
silently degrade.
"""

# Control rate is 50 Hz (ctrl_dt = 0.02).
JUMP_WINDOW_STEPS = 40  # 0.8 s: crouch, flight and landing with margin
JUMP_COOLDOWN_STEPS = 50  # 1.0 s: blocks re-trigger while recovering
JUMP_MOTOR_VELOCITY = 15.0  # rad/s while the jump window is open; sim-only


def advance_jump(timer, cooldown, trigger):
    """Advance the jump state machine by one control step.

    Branchless so it can be jitted and vmapped.

    Args:
      timer: control steps left in the current jump window; 0 when idle.
      cooldown: control steps left before another jump may fire; 0 when ready.
      trigger: 1 when a jump is requested this step, 0 otherwise.

    Returns:
      ``(timer, cooldown, active)`` for the next step. ``active`` is 1.0 while
      the window is open and is what becomes ``cmd[7]``.
    """
    # A jump may only start from a full stop: no window open, no cooldown left.
    idle = (timer <= 0) * (cooldown <= 0)
    fire = idle * (trigger > 0)

    # `(timer - 1) * (timer > 0)` floors at 0 without needing a backend max().
    decayed_timer = (timer - 1) * (timer > 0)
    next_timer = fire * JUMP_WINDOW_STEPS + (1 - fire) * decayed_timer

    # The window ends on the step where the timer was about to hit zero.
    window_ended = (timer == 1) * (1 - fire)
    decayed_cooldown = (cooldown - 1) * (cooldown > 0)
    next_cooldown = (
        window_ended * JUMP_COOLDOWN_STEPS + (1 - window_ended) * decayed_cooldown
    )

    active = (next_timer > 0) * 1.0

    return next_timer, next_cooldown, active
