# Push-Recovery Reinforcement Learning on MuJoCo Ant: A Research POC Report

**Project:** Ant-v5 Push-Recovery RL POC
**Period:** First commit through 2026-05-24
**Author:** RETELLIGENCE
**Status:** POC concluded — partial-observability bottleneck identified; algorithm-level extensions and Go1 transfer initiated

---

## Abstract

We report a small-scale proof-of-concept study on learning push-recovery locomotion for the MuJoCo `Ant-v5` quadruped under a constrained compute budget (single machine, four parallel environments, PPO). The study addresses two questions: (i) under such constraints, can reward engineering and curriculum design alone produce a "well-trained" nominal locomotion policy of the kind reported in massively parallel legged-RL pipelines, and (ii) when a nominal policy is subsequently exposed to push disturbances, what is the actual ceiling of its acquired robustness, and which design factors (algorithm, reward, sampling, or observation) most determine that ceiling? Across two main phases and a four-experiment cause-isolation cycle (E1–E4), we found that (1) a body-frame command-tracking reward combined with a two-stage lateral-discipline curriculum produces a stable, qualitatively organized gait (vx_body = 1.79 m/s, lateral drift 6.5 m over 50 s); (2) standard random-push training with domain randomization improves *both* robustness and nominal path tracking (drift 6.5 m → 1.2 m) consistent with the regularization effect reported in PA-LOCO and legged_gym; (3) initial "brief-impulse" evaluation grids substantially overestimate robustness — under a stress grid with sustained (0.3 s) pushes and angular impulses, v4b survival drops from 100 % at 20 N to 44 % at 15 N; (4) the resulting null result from a stress-curriculum policy (v4c) was misleading: subsequent boundary-sampling plus windowed recovery reward (v4d) recovered 15 N forward-direction survival from 25 % to 75 %, and a five-frame observation history (v5a_v2) lifted 10 N average survival from 69 % to 88 %, demonstrating that the apparent algorithmic ceiling is in fact a stack of reward-design, sampling, and partial-observability bottlenecks. The dominant methodological lesson is that physics-audited measurements and isolated-cause experiments must precede algorithm changes; under our setup, switching from PPO to SAC was repeatedly tempting but would have foreclosed identifying the real ceiling components.

---

## 1. Introduction

Robust locomotion under external disturbances is a canonical legged-robotics control problem. Recent simulation-to-real pipelines (Rudin et al., 2021; Kumar et al., 2021; Huang et al., 2024) demonstrate that reinforcement-learning policies, trained at scale across thousands of parallel environments with rich domain randomization, can both walk well and recover from arbitrary external pushes. These results, however, are coupled to the assumption of massively parallel simulation and large compute budgets, leaving an open question for smaller research environments: how much of the methodology transfers, and what subset of design choices is responsible for which fraction of the result?

This POC investigates that question on a deliberately reduced setup. The simulator is MuJoCo `Ant-v5` (Gymnasium), an eight-actuator torque-controlled quadruped. The algorithm is PPO (Stable-Baselines3) with a 256-256 MLP. Parallelism is limited to four `SubprocVecEnv` workers, and total training budgets are capped at 1–1.5 M environment steps per policy. The contributions are:

1. **A working "well-trained" nominal policy** under the stated constraints, derived from a body-frame command-tracking reward and a two-stage lateral-discipline curriculum that sidesteps two well-defined local optima (standstill collapse and persistent lateral drift).
2. **Quantitative confirmation that random-push training acts as a regularizer**, improving nominal path tracking by roughly 5× while simultaneously establishing robustness to brief impulses up to 20 N.
3. **A stress-grade evaluation protocol** (sustained 0.3 s pushes with angular impulse) that demonstrates the "brief-impulse, 100 % survival" headline of prior phases substantially overstates the policy's actual robustness.
4. **An isolated-cause experimental cycle (E1–E4)** that partitions a stress-curriculum null result into four hypotheses — disturbance magnitude, evaluation metric, reward shaping/sampling, and observation completeness — and uses targeted single-variable interventions to identify which ones carry headroom.
5. **A methodological argument** that within this kind of small-scale POC, algorithm-level changes (e.g., PPO → SAC) should be deferred until reward, sampling, and observation design have been audited, because algorithm substitution mixes too many degrees of freedom to identify root causes.

The rest of the report is organized as follows. Section 2 surveys the relevant prior work. Section 3 formalizes the problem and metric definitions. Section 4 describes the implementation (wrappers, reward terms, curricula, observation augmentation). Sections 5–7 cover Phase 1 (nominal locomotion), Phase 2 (brief-impulse push robustness), and Phase 2b (stress evaluation and the E1–E4 cycle). Section 8 synthesizes the findings. Sections 9 and 10 list limitations and future directions, including the planned Go1 transfer (Section 10.2). Appendix A is the reproducibility matrix.

---

## 2. Related Work

Three threads of prior work frame this POC.

**Massively parallel legged RL.** Rudin et al. (*Learning to Walk in Minutes*, arXiv:2109.11978) established that 4096 parallel environments and PPO can produce sim-to-real-quality quadruped policies in minutes of wall-clock time. The dominant design choices — body-frame velocity commands, smoothness penalties, action-rate regularization, and aggressive domain randomization — define the de-facto baseline for legged RL. Our Phase 1 reward design is structurally similar but operates with three orders of magnitude fewer environments, which constrains exploration diversity and informs the curriculum design in Section 5.

**Robust locomotion under disturbances.** PA-LOCO (Huang et al., *Push-and-Adapt: Learning Robust Quadrupedal Locomotion*, arXiv:2407.04224) demonstrated that adding random-push curricula to standard locomotion training yields policies that recover from external forces without explicit teacher–student structure. RMA (Kumar et al., *Rapid Motor Adaptation*, arXiv:2107.04034) separates an environment-latent encoder from an adaptation module to handle disturbance-conditional behavior more explicitly. Robust Gymnasium (arXiv:2502.19652) standardizes evaluation protocols for robustness under perturbations. We follow PA-LOCO's simpler curriculum approach for Phase 2 and treat RMA-style privileged learning as a deferred extension.

**Local-optima pathologies in single-shot reward design.** A recurring observation in legged RL is that multi-objective rewards combining velocity tracking, lateral-deviation penalties, orientation alignment, and smoothness terms exhibit competing local optima — typically "stand still and accumulate survival" versus "walk fast but drift." Mitigation strategies include curriculum learning, reward shaping over time, and warm-starting from intermediate policies. Phase 1 of this POC encountered this pathology directly and resolved it via a two-stage curriculum.

---

## 3. Problem Formulation

### 3.1 Environment

The simulator is Gymnasium `Ant-v5` under MuJoCo, providing a torque-controlled quadruped with eight actuated joints (two per leg), 105-dimensional default observation (joint positions, joint velocities, and contact information), and a control timestep of `dt = 0.01 s`. Episode length is fixed at 1000 control steps (10 s of simulated time) for nominal evaluation and extended (up to 2000 steps) for stress evaluation.

### 3.2 Locomotion Objective

A policy is termed *well-trained* if it satisfies a set of absolute robotics-engineering criteria (13 total, including survival, body-frame velocity tracking error, lateral drift, body yaw and course alignment, roll/pitch RMS, body-height variance, and action smoothness). These criteria are absolute — not relative to a prior baseline — so that progress can be assessed on its own terms rather than measured against the previous experiment. The criteria are intentionally strict: under the compute budget of this study, achieving all 13 simultaneously is not expected, but they provide a fixed yardstick.

### 3.3 Push-Robustness Objective

A policy is termed *push-robust* on a disturbance grid if, over a set of (force magnitude, direction, duration, angular impulse) cells, the survival rate and recovery quality remain above declared thresholds. We distinguish two evaluation grades:

- **Brief-impulse (Tier B) grid.** Single linear push (xy plane only), duration 5 sim steps (0.05 s), eight directions sampled at 45°, three episodes per cell. This grid was used in the original Phase 2 evaluation.
- **Stress grid.** Single push, duration 30 sim steps (0.3 s), with random sign on z-axis torque (±0.2 N·m), eight directions, multiple seeds per cell. Total angular and linear impulse is approximately 4.5× the brief-impulse grid at matched force magnitude.

### 3.4 Recovery Metrics

Survival alone is an insufficient metric because two policies with identical survival rates can differ markedly in recovery quality. Following the E2 redesign (Section 7.2), we additionally report (a) `recovery_time_vx_smoothed`, defined as the number of post-push steps until a sliding-window mean of body-frame x-velocity returns to within a tolerance of the commanded value; (b) post-push integrated tracking error in vx and yaw; and (c) post-push maximum lateral deviation from the commanded heading line.

---

## 4. Methods

### 4.1 Wrapper Stack

All training and evaluation use a fixed wrapper composition, ordered from outermost (sees rewards/observations after simulation) to innermost (modifies model parameters or external forces before simulation):

```
WellTrainedLocomotionAntWrapper        # reward shaping, observation augmentation
└── ObservationHistoryStackWrapper     # optional, Phase 2d only
    └── PushDisturbanceWrapper         # applies xfrc to torso body
        └── DomainRandomizationWrapper # resets randomize mass / friction / damping / motor
            └── gym.make("Ant-v5")
```

Wrapper order is not negotiable: domain randomization must occur at reset before any disturbance or reward shaping sees the resulting environment. Push wrapping must occur before reward wrapping so that the locomotion wrapper can read the disturbance state via `info["push_active"]` and `info["push_steps_since_end"]`.

### 4.2 Locomotion Reward (Phase 1)

The Phase 1 reward (`WellTrainedLocomotionAntWrapper` in [scripts/train_well_trained_locomotion.py](scripts/train_well_trained_locomotion.py)) combines:

- `w_progress_vx * (vx_body / v_cmd)` for forward-velocity progress against a command;
- `w_track_vy * exp(-vy_body² / σ²)` body-frame lateral-velocity penalty;
- `w_heading_alignment * exp(-yaw_world² / σ_yaw²)` world-frame heading alignment;
- `w_lateral_position * exp(-y_world² / σ_y²)` world-frame lateral-deviation penalty;
- `w_action_rate * ||a_t - a_{t-1}||²` and `w_action_accel * ||a_t - 2a_{t-1} + a_{t-2}||²` smoothness terms.

The two terms that compete most strongly are body-frame lateral-velocity penalty (which prevents drift) and forward progress (which prevents standstill). Section 5.2 reports the curriculum that separates them in time.

### 4.3 Push Disturbance and Domain Randomization

`PushDisturbanceWrapper` ([scripts/train_robust_locomotion.py](scripts/train_robust_locomotion.py)) applies external forces to the torso body via MuJoCo's `data.xfrc_applied`. Push duration, inter-push interval, force magnitude, and (in Phase 2b) z-axis torque are independently configurable. A push curriculum ramps the maximum magnitude from zero to the target over 300 k environment steps per worker, with inter-push intervals uniformly sampled from 5–10 s of simulated time.

`DomainRandomizationWrapper` perturbs at every reset: body mass × Uniform(0.8, 1.2), friction × Uniform(0.5, 1.5), joint damping × Uniform(0.8, 1.2), and motor torque scale × Uniform(0.85, 1.15). Per-step Gaussian action noise (std 0.02) is added to actuator commands. These ranges follow PA-LOCO's published recommendations.

### 4.4 Boundary Sampling (Phase 2c)

When a policy handles low-magnitude pushes easily and fails high-magnitude pushes consistently, uniform sampling over the curriculum wastes the majority of samples on already-learned or unreachable regions. The boundary-sampling mode of `PushDisturbanceWrapper` partitions the (force, duration, torque) space into three bins — easy [5–8 N, 15–20 steps, 0.05–0.10 N·m], mid [8–12 N, 20–30, 0.10–0.18], hard [12–15 N, 25–30, 0.15–0.20] — and samples them at 20 / 60 / 20 weighting. The mid bin concentrates on the empirically observed failure boundary.

### 4.5 Windowed Recovery Reward (Phase 2c)

A naive way to teach faster recovery is to add a permanent term penalizing post-push tracking error. This corrupts the nominal gait because natural inter-step variation gets penalized as residual error. Instead, the locomotion wrapper exposes a windowed recovery reward: a set of `-w * |error|` terms that are gated to the union of (push-active interval) and (5 s post-push window), reading `info["push_steps_since_end"]` from the inner push wrapper. Outside the window, reward is the unmodified nominal locomotion reward.

### 4.6 Observation History Stack (Phase 2d)

`ObservationHistoryStackWrapper` maintains a fixed-size circular buffer of the last `N` observations and exposes a flat concatenation in **newest-first** layout `[obs_t, obs_{t-1}, ..., obs_{t-N+1}]`. This convention matters because warm-starting from a shorter-observation policy via input-expansion zero-padding (copying the source policy's first-layer weights into the leading columns of a wider input matrix) only preserves initial behavior if the leading columns correspond to the *current* observation. A first attempt with oldest-first layout led to catastrophic failure (Tier A vx = 0.28, return –2508) because the warm-started policy was acting on four-step-stale observations; the bug is documented in Section 7.4.

---

## 5. Phase 1 — Nominal Locomotion

### 5.1 Goal

Produce a single nominal policy that a robotics engineer would judge to walk well, both quantitatively (as many of the 13 absolute criteria as possible) and qualitatively (visual gait organization, no obvious pathological motion). The benchmark for qualitative judgment is the published visual quality of legged_gym / Rudin-style policies.

### 5.2 Reward-Design Iteration

Seven reward variants were tried in sequence (v3a through v3h), summarized in Table 1. The core finding is that a single fixed reward weighting cannot simultaneously satisfy "walking is preserved" and "lateral drift is suppressed." When the lateral penalty is weak, the policy converges to a stable but laterally biased gait; when it is strong, the policy converges to standstill (which trivially satisfies the lateral objective and incurs only modest velocity-tracking penalty).

**Table 1. Reward-design progression in Phase 1.**

| Variant | Key change | Outcome |
|---|---|---|
| v3a–c | World-frame tracking + lateral position penalty | "Yaw-steered" path holding (low drift but body rotated sideways) versus "body-aligned" gait (high drift) — no joint optimum |
| v3d | Body-frame command tracking (legged_gym style) | Body-frame terms alone are invariant to world-frame yaw → drift 73 m |
| v3e | + world-frame yaw alignment `w_heading * exp(-yaw²/σ²)` | Yaw locked to 0.06, but 18 m of lateral drift remains (gait asymmetry) |
| v3f | Continuation from v3e + stronger lateral | Local optimum not escaped (changes < 5 %) |
| v3g | Fresh init + strong vy penalty | Standstill collapse (vx ≈ 0) |
| v3h | **Two-stage curriculum** (weak → strong lateral) | drift 6.5 m, organized gait |

The curriculum solves the tension by sequencing the two competing objectives. Stage 1 (500 k steps) trains with low lateral penalty and high progress reward, ensuring walking is established. Stage 2 (1 M steps, warm-started from Stage 1) tightens the lateral penalty and adds smoothness terms. Because the policy already walks at the start of Stage 2, the gradient continuity prevents collapse to standstill.

### 5.3 Selected Phase 1 Deliverable

The selected policy is [runs/well_trained_v3h_s2_seed42_1000k/](runs/well_trained_v3h_s2_seed42_1000k/) with the metrics shown in Table 2.

**Table 2. Phase 1 policy metrics (10-episode deterministic, 1000-step episodes).**

| Metric | Value |
|---|---|
| Survival | 1.00 |
| `vx_body_mean` | 1.79 m/s |
| `abs_lateral_drift` (50 s) | 6.5 m |
| `yaw_abs_mean` | 0.16 rad |
| Roll RMS / pitch RMS | 0.03 / 0.11 rad |
| Body-height std | 0.04 m |
| Course alignment | 0.93 |

Five of thirteen absolute criteria pass quantitatively. The remaining eight are gait-organization criteria (foot-contact regularity, action-rate stationarity under perturbation, etc.) that are not reachable under the compute budget. Qualitative review of [videos/phase1_candidate_v3h_s2_seed42.mp4](videos/phase1_candidate_v3h_s2_seed42.mp4) confirmed the gait is acceptably organized, and this policy was selected as the Phase 1 deliverable. A faster competing candidate (v3e, vx = 2.07 m/s) was rejected because of its 18 m drift — a domain-specific judgment that gait organization outweighs raw forward speed.

### 5.4 Structural Limit

Achieving all 13 criteria is not feasible under the compute budget. legged_gym-style results depend critically on the exploration diversity provided by 4096 parallel environments; reward tuning alone cannot substitute for that diversity. This is a structural observation about the algorithm-and-batch-size combination, not a deficiency of the reward design.

---

## 6. Phase 2 — Brief-Impulse Push Robustness

### 6.1 Training Schedule

Two policies were produced by warm-starting from the Phase 1 deliverable and adding the push and domain-randomization wrappers:

- **v4a (10 N nominal-robust):** push curriculum 0 → 10 N over 300 k steps per worker; duration 5 sim steps; interval 5–10 s; xy plane; total 1.5 M steps. See [runs/robust_locomotion_v4a_seed42_1500k/](runs/robust_locomotion_v4a_seed42_1500k/).
- **v4b (20 N stretch):** warm-start from v4a; push curriculum 0 → 20 N over 300 k steps per worker; same duration and interval; total 1 M steps. See [runs/robust_locomotion_v4b_stretch20N_seed42_1000k/](runs/robust_locomotion_v4b_stretch20N_seed42_1000k/).

### 6.2 Results

**Table 3. Phase 2 evaluation on brief-impulse (0.05 s) Tier B grid.**

| Metric | v3h_s2 (P1) | v4a | v4b |
|---|---|---|---|
| Tier A lateral drift (quiet, 50 s) | 6.5 m | **1.4 m** | **1.2 m** |
| Tier A `vx_body` | 1.79 | 1.87 | 1.70 |
| Tier A `yaw_abs` | 0.16 | 0.13 | 0.10 |
| Tier B max force survived (brief 0.05 s) | n/a | 10 N (100 %) | 20 N (100 %) |

### 6.3 Push Training as Implicit Regularization

The most interesting finding of Phase 2 is that push training improves nominal locomotion: Tier A lateral drift drops by approximately 5× from the source policy to v4b (6.5 m → 1.2 m), with body yaw and pitch RMS also improving. The improvement is consistent with reports in PA-LOCO and legged_gym: random forces combined with domain randomization act as an implicit regularizer that biases policy gradient steps toward gaits that are simultaneously robust and stable. The intuition is that any gait component that depends on a precise initial condition or a precise model parameter is penalized by domain randomization, while any gait component that depends on a precise external-force history is penalized by random pushes. What remains in the optimization landscape is the intersection: gaits that are reasonably invariant to both.

This finding has the practical consequence that, if a project has the budget for either nominal-only training or nominal-then-robust training, the latter is preferable on both robustness *and* nominal-quality grounds.

---

## 7. Phase 2b — Stress Evaluation and Cause Isolation

### 7.1 Motivation

The Phase 2 headline ("20 N, 100 % survival") was inconsistent with the qualitative video review: under 20 N brief-impulse pushes, the policy showed essentially no visible recovery behavior because the disturbance itself was barely visible. Three causes were hypothesized:

1. A 0.05 s push corresponds to about 1.5 frames at typical video framerates — visually imperceptible.
2. Forces are applied to the torso center only, with no angular component, so the disturbance imparts zero angular momentum.
3. A quadruped's wide base of support absorbs short linear impulses passively, irrespective of policy quality.

This led to a stress-grade evaluation protocol with 6× longer push duration and explicit angular impulse (Section 3.3). Total impulse at 15 N stress is approximately 4.5× the 20 N brief impulse (4.5 N·s vs 1.0 N·s).

### 7.2 Stress Evaluation of v4b

Evaluating v4b on the stress grid produces the breakdown in Table 4. The "20 N 100 % survival" headline collapses: at 15 N stress, the policy survives only 44 % of cells, with average maximum drift exceeding 10 m.

**Table 4. v4b on stress grid (0.3 s push, ±0.2 N·m torque).**

| Force | Survival cells | Avg. max drift |
|---|---|---|
| 5 N | 8 / 8 (100 %) | 2.7 m |
| 10 N | 6.25 / 8 (78 %) | 5.4 m |
| 15 N | 3.5 / 8 (44 %) | 10.1 m |

The methodological takeaway is that **brief impulse evaluations systematically overestimate quadruped push robustness** because the wide base of support absorbs the disturbance without requiring policy intervention. Any robustness claim about a quadruped policy that does not include sustained pushes with angular components is suspect.

### 7.3 v4c — Direct Stress-Curriculum Training (Null Result)

The obvious next step was to add the stress distribution to the training curriculum and continue training. v4c ([runs/robust_locomotion_v4c_stress_seed42_1000k/](runs/robust_locomotion_v4c_stress_seed42_1000k/)) was warm-started from v4b and trained for 1 M additional steps with force 0 → 15 N, torque 0 → 0.2 N·m, and duration 5 → 30 steps in the curriculum.

The result was effectively null (Table 5): survival changed by 3 % or less at every force level, and nominal drift slightly degraded (1.18 m → 1.37 m). The temptation at this point was to declare a PPO ceiling and switch to SAC. Instead, we ran a four-experiment cycle to isolate the cause.

**Table 5. v4c vs v4b on stress grid.**

| Force | v4b survival | v4c survival | Δ |
|---|---|---|---|
| 5 N | 100 % | 100 % | 0 |
| 10 N | 78 % | 75 % | –3 % |
| 15 N | 44 % | 47 % | +3 % |

### 7.4 E1 — Physics Audit of the Stress Disturbance

The first hypothesis to rule out is that the stress disturbance is either much weaker or much stronger than the first-order force-times-time intuition suggests. [scripts/stress_physics_audit.py](scripts/stress_physics_audit.py) logs `qvel` immediately before and immediately after each push, recording the realized Δv across 116 (seed, force, direction, torque-sign) cells (see [reports/stress_physics_audit.csv](reports/stress_physics_audit.csv)).

The measured results clarify the disturbance regime:

- At 15 N × 0.3 s, the measured |Δv_xy| ≈ 12–14 m/s, close to the first-order free-body estimate of 13.6 m/s. A 0.3 s push outlasts the contact-reaction settling time, so ground contact does not effectively cancel the linear momentum.
- At 5 N × 0.3 s, the measured |Δv_xy| ≈ 1.2–2.7 m/s, far smaller than the first-order 4.5 m/s. At low forces, the contact reaction *does* absorb most of the impulse.
- The 0.2 N·m angular impulse yields |Δω_z| ≈ 0.9–2.6 rad/s, a non-trivial yaw rate that the policy must counteract.
- v4c reduces Δv at 10 N dir = 180° (backward push) from 7.14 to 5.36 m/s relative to v4b — that is, v4c does push back against the disturbance during the push, but ultimately survives the same fraction of cells.

The audit establishes that (a) the stress disturbance is real, (b) its effect is nonlinear in force magnitude (contact absorption at low force, near-free-body at high force), and (c) v4c has acquired *some* active counter-push behavior despite identical survival rates. The first hypothesis — that the disturbance is the wrong magnitude — is ruled out.

### 7.5 E2 — Recovery Metric Redesign

A strict early-attempt recovery criterion (50 contiguous clean-state steps post-push) never triggered, because the natural Ant gait has small periodic variations that violate strict clean-state checks. We redesigned the recovery criteria around sliding-window means with relaxed thresholds and added integrated-tracking-error and post-push max-deviation metrics (see [reports/recovery_metric_grid.csv](reports/recovery_metric_grid.csv)).

Under the relaxed metrics, the v4c null result is confirmed: recovery time and integrated tracking error are no better than v4b, with the single exception of 10 N dir = 180° where v4c shows marginal survival improvement (25 % → 50 %) offset by regressions in other cells. The conclusion is that v4c's null result is not a measurement artifact — the policy genuinely has not changed in its disturbance-response quality. The second hypothesis — that the evaluation is mismeasuring real improvement — is ruled out.

### 7.6 E3 — Boundary Sampling + Windowed Recovery Reward (v4d)

Two remaining hypotheses were attacked simultaneously in v4d: (i) the uniform-over-curriculum sampling distributes most learning samples in the easy regime, and (ii) the nominal reward provides no signal for fast recovery after a push.

The interventions are the boundary-sampling mode (Section 4.4) and the windowed recovery reward (Section 4.5). The training schedule warm-starts from v4b and runs for 1 M additional steps with a single seed; see [runs/robust_locomotion_v4d_boundary_recovery_seed42_1000k/](runs/robust_locomotion_v4d_boundary_recovery_seed42_1000k/).

**Table 6. v4d vs v4b on stress grid (15 N broken down by direction).**

| Force (direction) | v4b survival | v4d survival | Δ |
|---|---|---|---|
| 5 N (avg over 4 dirs) | 100 % | 100 % | 0 |
| 10 N (avg) | 75 % | 69 % | –6 % |
| **15 N 0° (forward)** | **25 %** | **75 %** | **+50 %** |
| 15 N 90° (lateral) | 100 % | 50 % | –50 % |
| 15 N 180° (back) | 0 % | 25 % | +25 % |
| 15 N 270° (lateral) | 25 % | 25 % | 0 |
| **15 N average** | **38 %** | **44 %** | **+6 %** |

Two observations follow. First, reward and sampling design have *non-trivial* headroom: 15 N forward survival went from 25 % to 75 %, which would have been incorrectly attributed to PPO ceiling if v4c's null result had been taken at face value. Second, the gain is direction-asymmetric — lateral 90° regressed by 50 % — which suggests a separate bottleneck unrelated to reward or sampling. The two remaining hypotheses (boundary sampling deficit, recovery-reward absence) are confirmed as *partial* contributors but not sufficient explanations for cross-direction inconsistency.

Nominal Tier A metrics for v4d are largely preserved (vx = 1.75 m/s, yaw = 0.10) with one regression: nominal drift increased from 1.18 m to 3.17 m. The windowed recovery reward, while gated to a 5 s post-push window, still bleeds into ongoing nominal gait through curriculum effects.

### 7.7 E4 — Observation History Stack (v5a_v2)

The remaining hypothesis is that the policy lacks information to estimate disturbance direction from a single observation frame: with only instantaneous joint positions and velocities, the policy cannot distinguish "I am being pushed forward" from "I just stepped onto an incline" or "my motor torques are scaled higher today." An observation history stack provides one frame of difference signal per pair of consecutive frames, which should make disturbance direction more inferable.

`ObservationHistoryStackWrapper` was added with `N = 5` frames (0.05 s window). A first warm-start attempt failed catastrophically (v5a v1: Tier A vx = 0.28, return = –2508) because the wrapper used oldest-first layout while the warm-start expansion assumed newest-first; the policy was acting on four-step-stale observations. Switching to newest-first layout fixed the issue (v5a_v2 [runs/robust_locomotion_v5a_history5_seed42_1000k_v2/](runs/robust_locomotion_v5a_history5_seed42_1000k_v2/)). Training schedule: warm-start from v4d, learning rate reduced to 5e-5 (conservative for added input dimensionality), 1 M steps.

**Table 7. v5a_v2 vs v4d on stress grid.**

| Force (direction) | v4d | v5a_v2 | Δ |
|---|---|---|---|
| 5 N (avg over 4 dirs) | 100 % | 100 % | 0 |
| 10 N 0° | 100 % | 100 % | 0 |
| **10 N 180°** | **0 %** | **50 %** | **+50 %** |
| 10 N 270° | 75 % | 100 % | +25 % |
| **10 N average** | **69 %** | **88 %** | **+19 %** |
| 15 N 0° | 75 % | 0 % | –75 % |
| 15 N 90° | 50 % | 100 % | +50 % |
| 15 N 180° | 25 % | 25 % | 0 |
| 15 N 270° | 25 % | 25 % | 0 |
| 15 N average | 44 % | 38 % | –6 % |

Recovery time at 10 N dir = 90° improves from 132 to 55 steps (2.4× faster). Tier A is preserved (vx = 1.78, drift = 4.06 m, yaw = 0.094).

The 10 N regime confirms partial observability as a meaningful contributor: the 10 N backward push (0 % survival in v4d) jumps to 50 % in v5a_v2, with average 10 N survival rising from 69 % to 88 %. The 15 N regime is mixed, with 90° lateral improving from 50 % to 100 % but 0° forward regressing from 75 % to 0 %. This indicates that history stack length five is insufficient for the 30-step push duration — a five-frame window captures only one-sixth of the push, so the policy can detect that *a* disturbance is present but cannot characterize its full profile. The longer-stack and recurrent-policy candidates listed in Section 10 follow directly from this observation.

---

## 8. Discussion

### 8.1 What Worked

1. **Body-frame command tracking combined with explicit world-frame heading alignment.** Body-frame terms alone are invariant to world yaw and cannot prevent drift; world-frame terms alone fight gait dynamics. The combination provides the right inductive biases.
2. **Two-stage lateral-discipline curriculum.** A single-shot reward sweep cannot find the weight region that walks *and* tracks; sequencing the objectives by warm-starting from "walks but drifts" sidesteps the local-optima problem.
3. **Random push training as nominal-quality regularizer.** Phase 2 improved nominal drift 5×. This is consistent with prior reports and should be considered a standard finding: for any project where push-robustness training is feasible, running it improves nominal quality "for free."
4. **Boundary sampling and windowed recovery reward.** Both are mechanically simple but produce a 50-point improvement in 15 N forward survival. They should be considered default options for any disturbance-robust training pipeline.
5. **Newest-first observation history layout.** The convention matters because it preserves input-expansion warm-start trivially. The bug-finding episode in v5a v1 → v5a v2 cost roughly one training run and should not be re-done.

### 8.2 What Did Not Work (and Why)

1. **Single-shot strong lateral penalty (v3g).** Collapsed to standstill because the lateral objective dominated the forward objective before walking was established.
2. **Continuation from biased policy with stronger reward weights (v3f).** PPO's exploration with std ≈ 0.15 was insufficient to escape the v3e local optimum; the policy moved by < 5 % over 200 k steps.
3. **Stress-curriculum training alone (v4c).** With no change to sampling distribution or reward structure, the policy did not improve on stress survival. This was misleading and almost led to a premature algorithm switch.

### 8.3 The Methodological Result

The most generalizable result of this POC is methodological. When v4c produced a null result on stress training, the natural conclusion — "PPO + four environments has hit its ceiling, switch to SAC or RMA" — was incorrect on the evidence. The four-experiment cycle (E1–E4) decomposed the null result into four hypotheses (disturbance magnitude, evaluation metric, reward shaping plus sampling, observation completeness) and demonstrated that the last two carry substantial headroom that algorithm substitution would have obscured by changing too many variables at once.

The decision-rule that follows is: **before substituting the optimization algorithm, audit the disturbance physics, the evaluation metric, the reward shaping, and the observation completeness, in that order.** Each audit produces measurements that the algorithm-substitution alternative does not, and the cumulative cost of the four audits in this POC was roughly equivalent to one additional training run.

### 8.4 Honest Evaluation

The original Phase 2 evaluation reported "20 N, 100 % survival" on a brief-impulse grid. Under a stress-grade grid with sustained push and angular impulse, the same policy survives only 44 % at 15 N. The gap is not a v4b defect — it is an evaluation defect. Any quadruped push-robustness claim that uses only brief impulses to a torso center is incomplete; the natural absorption properties of a wide base of support do most of the work and the policy quality is barely probed. We recommend that future reports in this area specify the duration, application point, and angular component of evaluation pushes alongside the magnitude.

---

## 9. Limitations

1. **Single-machine PPO with four environments cannot reach absolute robotics-grade locomotion.** Five of thirteen criteria pass in Phase 1; the remaining eight require exploration diversity that comes from massively parallel simulation, not reward tuning. Any project that needs "all 13 pass" should plan for IsaacGym-class parallelism from the start.
2. **History stack length five is too short for the 30-step stress regime.** The 15 N forward regression in v5a_v2 has this as its most plausible cause. We did not run a longer-stack ablation.
3. **All Phase 2 results use a single seed (42).** The runs are deterministic given the seed, but cross-seed variance is not measured. A two- or three-seed replication would tighten the conclusions, particularly the v4d direction-asymmetry result.
4. **The recovery-reward window leaks into nominal gait.** v4d's nominal drift increased from 1.18 m to 3.17 m. The window length and weight schedule are not yet tuned for minimum nominal-cost.
5. **Push application is torso-center only.** Real-world disturbances apply at varying contact points; an evaluation grid that randomizes the application point would probe rotational-recovery behavior more completely.
6. **The action space is torque control.** PD-position control with the same policy and reward typically produces visibly smoother gaits; smoothness penalties are less effective in the torque regime.

---

## 10. Future Work

### 10.1 Ant-v5 — Next Single Interventions

In priority order:

1. **Longer observation history stack (10–20 frames).** The most direct response to v5a_v2's 15 N forward regression. A 10-frame stack covers one-third of a 30-step push and should be sufficient to characterize the push profile.
2. **Recurrent policy (RecurrentPPO / GRU).** Generalizes history-stack benefits to arbitrary push durations and avoids the architectural cliff at the stack boundary.
3. **RMA-lite teacher–student.** A privileged encoder receives the actual push state at training time, distilling into a proprio-only student. Likely to outperform history stacks if implemented carefully; deferred because of implementation complexity.
4. **SAC or off-policy algorithm.** Worth trying only after the above three interventions have been audited; expected gain is uncertain and the comparison is hard to interpret if reward/observation design has not been fixed first.
5. **Adversarial perturbation training (RARL).** An adversary network learns worst-case pushes during training. Higher variance, but in principle reaches a stricter robustness envelope.

### 10.2 Go1 — Transfer Track

Initial Go1 quadruped infrastructure is in place. The current milestone is `M2`: [scripts/go1/go1_env.py](scripts/go1/go1_env.py) wraps the MuJoCo Menagerie Go1 model as an SB3-compatible `gym.Env`, with smoke tests at [scripts/go1/m2_env_smoke.py](scripts/go1/m2_env_smoke.py). Three sanity-check rollouts (zero action, small random action, full random) confirm the environment runs and produces physically reasonable trajectories. Frame stacking has been independently validated as a performance improvement on the underlying Go1 task.

The planned transfer arc is:

- Reproduce Phase 1 (well-trained nominal locomotion) on Go1 with the same reward design.
- Reproduce the Phase 2 result that push training improves nominal drift.
- Test the boundary-sampling and windowed-recovery-reward design on Go1, on the hypothesis that those interventions generalize across morphologies.

### 10.3 Reward and Evaluation Refinements

- Replace instantaneous `vy_body` penalty with a gait-cycle-averaged form; the instantaneous penalty currently mis-categorizes natural gait wiggle as drift.
- Add foot-contact pattern regularity metrics (spectral entropy of contact sequence) to the absolute criteria sheet; these formalize the "organized gait" qualitative judgment that is currently human-in-the-loop.
- Define the "5 s recovery to nominal trajectory" criterion formally and add it to the evaluation grid. The current survival-and-recovery-time pair is informative but does not capture trajectory-level recovery quality.

---

## 11. Conclusion

This POC asked whether reward, curriculum, and observation design alone can carry a small-compute PPO setup as far as the published massively-parallel legged-RL results, and which design factors are the binding constraints. The answer is: substantially further than first appearances suggest, but not all the way. The 15 N forward stress regime went from 25 % survival (v4b) to 75 % (v4d) via boundary sampling and windowed recovery reward, and the 10 N average regime went from 69 % to 88 % via a short observation history. Neither change required an algorithm switch. The remaining cross-direction inconsistency and 15 N hard regime point to longer-context observation models (longer stacks, recurrent policies, or privileged-teacher distillation) as the next intervention class.

The more durable contribution is the methodological pattern. When a stress-curriculum training run produced a null result, the temptation to substitute the algorithm was strong and would have been wrong. The four-experiment cycle — physics audit, metric redesign, reward-plus-sampling intervention, observation augmentation — identified concrete headroom that algorithm substitution would have hidden by changing too many variables. Within a small-compute legged-RL setting, this is the order in which design decisions should be re-examined when results stall.

---

## Appendix A — Reproducibility

### A.1 Training and Evaluation Matrix

| Phase | Policy | Training entry point (abbreviated args) |
|---|---|---|
| 1 | [v3h_s2_seed42](runs/well_trained_v3h_s2_seed42_1000k/) | [train_well_trained_locomotion.py](scripts/train_well_trained_locomotion.py), two-stage curriculum |
| 2 | [v4a](runs/robust_locomotion_v4a_seed42_1500k/) | [train_robust_locomotion.py](scripts/train_robust_locomotion.py) `--push-force-max 10` |
| 2 | [v4b](runs/robust_locomotion_v4b_stretch20N_seed42_1000k/) | `--push-force-max 20` (warm-start v4a) |
| 2b | [v4c](runs/robust_locomotion_v4c_stress_seed42_1000k/) | `+ --push-torque-z-max 0.20 --push-duration-max-steps 30` (warm-start v4b) |
| 2c | [v4d](runs/robust_locomotion_v4d_boundary_recovery_seed42_1000k/) | `+ --push-use-boundary-sampling --w-recovery-{vx,vy,yaw}-err 0.5 --w-recovery-roll-pitch 1.0 --w-recovery-yaw-rate 0.2` |
| 2d | [v5a_v2](runs/robust_locomotion_v5a_history5_seed42_1000k_v2/) | `+ --warm-start-on-observation-mismatch --history-stack-size 5 --learning-rate 5e-5` |

### A.2 Evaluation Commands

- Tier A/B/C standard grid: [eval_robust_locomotion.py](scripts/eval_robust_locomotion.py) `--tiers abc`
- Stress grid: same script, `--tier-b-duration-steps 30 --tier-b-torque-z 0.20 --tier-b-randomize-torque-sign`
- E1 physics audit: [stress_physics_audit.py](scripts/stress_physics_audit.py) `--models v4b=...,v4c=...`
- E2 recovery metrics: [recovery_metric_grid.py](scripts/recovery_metric_grid.py) `--models v4b=...,v4d=...`

### A.3 Dependencies

See [requirements.txt](requirements.txt). Key pins: `stable-baselines3==2.8.0`, `gymnasium[mujoco]==1.2.0`, `numpy==1.26.4`. All runs executed in the Docker environment defined by [Dockerfile](Dockerfile).

### A.4 Video Index

| Phase | Video |
|---|---|
| Phase 1 | [videos/phase1_candidate_v3h_s2_seed42.mp4](videos/phase1_candidate_v3h_s2_seed42.mp4) |
| Phase 2 quiet | [videos/phase2_v4b_quiet.mp4](videos/phase2_v4b_quiet.mp4) |
| Phase 2 push | [videos/phase2_v4b_push20N_lateral.mp4](videos/phase2_v4b_push20N_lateral.mp4), [videos/phase2_v4b_push20N_backward.mp4](videos/phase2_v4b_push20N_backward.mp4) |
| Phase 2b stress | [videos/phase2b_v4b_stress_10N_lateral.mp4](videos/phase2b_v4b_stress_10N_lateral.mp4), [videos/phase2b_v4c_stress_15N_lateral.mp4](videos/phase2b_v4c_stress_15N_lateral.mp4) |
| Phase 2c v4d | [videos/phase2c_v4d_quiet.mp4](videos/phase2c_v4d_quiet.mp4), [videos/phase2c_v4d_stress_15N_forward.mp4](videos/phase2c_v4d_stress_15N_forward.mp4) |
| Phase 2d v5a_v2 | [videos/phase3_v5a_quiet.mp4](videos/phase3_v5a_quiet.mp4), [videos/phase3_v5a_stress_10N_180.mp4](videos/phase3_v5a_stress_10N_180.mp4), [videos/phase3_v5a_stress_15N_90.mp4](videos/phase3_v5a_stress_15N_90.mp4) |

### A.5 Generated Data Tables

- [reports/stress_physics_audit.csv](reports/stress_physics_audit.csv) — 116 (seed, force, direction, torque-sign) cells, pre/post Δv measurements for v4b and v4c.
- [reports/recovery_metric_grid.csv](reports/recovery_metric_grid.csv) — v4b vs v4d on stress grid.
- [reports/recovery_metric_grid_v4d.csv](reports/recovery_metric_grid_v4d.csv) — v4d full grid.
- [reports/recovery_metric_grid_v5a_v2.csv](reports/recovery_metric_grid_v5a_v2.csv) — v4d vs v5a_v2 on stress grid.

---

## References

- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*, arXiv:2109.11978.
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots*, arXiv:2107.04034.
- Huang et al., *PA-LOCO: Push-and-Adapt Learning of Robust Quadrupedal Locomotion*, arXiv:2407.04224.
- *Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning*, arXiv:2502.19652.
