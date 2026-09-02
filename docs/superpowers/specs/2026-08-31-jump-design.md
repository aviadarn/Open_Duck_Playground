# Jump capability for open_duck_mini_v2

**Date:** 2026-08-31
**Status:** approved design, not yet implemented
**Scope:** sim-only, optimized for looking good in the MuJoCo viewer

## Goal

Add a one-shot triggered jump to the `open_duck_mini_v2` joystick environment.
Press a key, the duck jumps once, lands, and returns to normal walking. A single
unified policy handles both walking and jumping.

Explicitly out of scope: transfer to the physical robot. The design is free to
relax servo realism where that buys jump height.

## Feasibility

Measured from the MJCF:

| quantity | value |
|---|---|
| total mass | 2.107 kg (20.67 N) |
| standing base height | 0.15 m |
| actuator force limit | 3.23 N·m (`forcerange`) |
| position gain | kp 13.37–17.8 |
| motor velocity clamp | 5.24 rad/s |
| control rate | 50 Hz (`ctrl_dt=0.02`) |

Force is not the constraint. With a ~0.09 m effective moment arm, two legs supply
roughly 72 N against a 20.7 N body weight — about 3.5x. The binding constraint is
the **slew-rate clamp** on motor targets in `joystick.py:411`, which limits target
motion to `max_motor_velocity * ctrl_dt` = 0.105 rad per control step.

Raising that clamp during the jump is therefore the single highest-leverage knob.

## Interface change

`sample_command` returns 8 values instead of 7. The new `cmd[7]` is a jump flag in
`{0, 1}`.

```
cmd = [lin_vel_x, lin_vel_y, ang_vel_yaw,
       neck_pitch, head_pitch, head_yaw, head_roll,
       jump]
```

The full command vector is spliced into the observation, so observation size goes
**101 -> 102**:

```
3 gyro + 3 accel + 8 command + 14x6 (angles, vels, 3x last_act, motor_targets)
  + 2 contact + 2 imitation_phase = 102
```

Consequences:

- The policy network input changes, so existing checkpoints and ONNX exports
  cannot be reused. Training restarts from scratch.
- Appending at index 7 is safe for every existing `cmd[:3]` slice
  (`reward_imitation`, `cost_stand_still`). No index surgery required.

## Jump state machine

Two states plus a cooldown. Deliberately **not** a multi-phase
crouch/launch/flight/land machine: hard-coding phase boundaries in control steps
bakes in an assumption about jump timing that PPO is better placed to discover.
Rewards key off measured physics inside the window, so they cannot desync from
what the robot is actually doing.

New fields in `state.info`:

| field | meaning |
|---|---|
| `jump_active` | 1.0 during the window, 0.0 otherwise; this is `cmd[7]` |
| `jump_timer` | counts down the window, in control steps |
| `jump_cooldown` | blocks re-trigger while landing and recovering |
| `jump_peak_z` | max base height reached this jump |
| `jump_took_off` | latched once both feet leave the floor |

Timings, all config values rather than constants:

- window: 40 steps (0.8 s at 50 Hz)
- cooldown: 50 steps (1.0 s)

**Trigger.** During training, fire with probability ~1/250 per step (roughly every
5 s) when idle and off cooldown. Deliberately independent of the 500-step command
resample, because a jump is an event rather than a held command. At inference the
viewer key sets the same latch and the identical machine runs the window.

## Rewards

Two properties of the existing reward assembly constrain the design:

- Total reward is clipped non-negative (`joystick.py:448`), and `alive=20.0` sets
  the baseline. Penalties cannot pull the sum below zero.
- The jump window covers roughly 12% of steps, not a negligible slice: the average
  cycle is ~250 idle + 40 window + 50 cooldown = 340 steps, so 40/340 ~= 12%. Jump
  terms still need large scales to compete with `alive=20.0`, but less inflation
  than a 4% duty cycle would demand.

### New terms

All three are multiplied by `jump_active`, so they are identically zero during
normal walking.

| term | signal | scale |
|---|---|---|
| `jump_takeoff` | `clip(base_vz, 0, inf)` while at least one foot is in contact | 30.0 |
| `jump_air_time` | both feet off the floor | 40.0 |
| `jump_height` | `clip(base_z - jump_base_z0, 0, cap) * airborne` | 300.0 |

where `jump_base_z0` is the base height latched at the instant the jump window
opens (not an absolute datum like the 0.15 m standing height) and `cap` is
0.25 m so a single enormous outlier cannot dominate the return. Latching at
window-open rather than measuring against a fixed standing height is
necessary because a crouching robot can leave the ground without ever
exceeding standing height, so an absolute datum can read 0.00 for an entire
run even when the jump is real. `jump_height` is also gated on `airborne`
(both feet off the ground) so that holding a raised-leg pose during the
window cannot out-earn an actual ballistic jump.

`jump_takeoff` is the bootstrap term. It is dense and rewards pushing *before*
leaving the ground; without it the height reward is never discovered from a
walking prior.

Scales are starting points. Brax logs each reward term separately, so the first
run is read to rebalance rather than guessed at twice.

### Gating existing terms

| term | change | reason |
|---|---|---|
| `imitation` | `x (1 - jump_active)` | Weight 15.0 on joint positions against a walking reference; fights the jump hardest. |
| `stand_still` | `x (1 - jump_active)` | Penalizes joint motion when the velocity command is near zero, so it would actively punish a jump in place. Easy to overlook. |
| `action_rate` | `x (1 - 0.5 * jump_active)` | At -0.5 it suppresses the fast action changes a jump needs, but removing it entirely invites jitter. Soften, do not delete. |
| `tracking_lin_vel`, `tracking_ang_vel` | unchanged | Operate on horizontal velocity and gyro; a vertical jump does not enter them. |

## Raising the height ceiling

Make the slew clamp jump-aware: `max_motor_velocity` stays 5.24 for walking and
rises to `jump_motor_velocity` (~15 rad/s) while `jump_active`. This preserves the
walking gait exactly as tuned.

`forcerange` and `kp` are left alone initially. They are only raised, in a copied
MJCF, if the logs show actuator force actually saturating — no second knob on
speculation.

## Avoiding train/inference drift

The state machine exists twice: in JAX for training, in numpy for inference. If
the window and cooldown constants disagree, the policy sees a different command
profile at inference than it trained on and the jump silently degrades.

Mitigation: a single `playground/open_duck_mini_v2/jump.py` holding the constants
and a backend-agnostic timer helper, imported by both `joystick.py` and
`mujoco_infer.py`.

## Inference and viewer

`mujoco_infer.py:84` splats `command` into the observation whole, mirroring
training, so extending `self.commands` to 8 elements yields obs 102 with no index
changes.

- Space bar (keycode 32) fires the trigger.
- The existing `key_callback` zeroes `lin_vel_*` on every keypress, so the jump
  flag cannot live in the handler. The key sets a "requested" latch; the timer
  owns `cmd[7]`.
- The `max_motor_velocity` clamp at `mujoco_infer.py:220` needs the same
  jump-aware raise as training, or inference jumps land shorter than trained.

## Verification

Cheapest first:

1. **Pure-function tests** — timer transitions (fire -> window -> cooldown ->
   idle), and that every jump reward term is exactly 0 when `jump_active=0`. The
   latter is the regression guard proving walking is untouched.
2. **Observation assertion** — 102 in both training and inference, asserted in the
   same test so the two cannot diverge.
3. **Local CPU smoke run** — ~2 min, confirms training runs and ONNX exports at
   the new size.
4. **Short GPU run** — 100M steps, ~13 min, ~$0.10. Read per-term reward logs,
   confirm a jump emerged, rebalance scales.
5. **Full GPU run** — 300M steps, ~40 min, ~$0.26. Then inspect in the viewer.

Expected GPU spend: $0.35-0.50 total.

## Risks

**Walking may degrade** relative to the 300M walking-only baseline, since the
policy now learns two behaviours. At a ~12% jump duty cycle this is a real
possibility rather than a remote one, and the 100M checkpoint is the place to
check it.

Fallback requires no new code: the command is 8-dim from step 0, so training can
run with the jump scales at 0 and then resume via `--restore_checkpoint_path` with
the scales raised. Curriculum for free.

**Sparse-reward failure** — the jump never emerges at all. `jump_takeoff` is the
mitigation, being dense and gradient-bearing from the first step. If the 100M run
shows no takeoff, raising `jump_takeoff` relative to the other two is the first
lever, before touching the actuator model.

## Files touched

| file | change |
|---|---|
| `playground/open_duck_mini_v2/jump.py` | new; constants and timer helper |
| `playground/open_duck_mini_v2/joystick.py` | 8-dim command, state machine, 3 new reward terms, gating, jump-aware slew clamp |
| `playground/open_duck_mini_v2/mujoco_infer.py` | 8-element commands, space-bar latch, jump-aware clamp |
| `tests/` | new; timer and reward-gating tests |
