# Ant Push-Recovery RL — Project Report

## 1. 프로젝트 개요

MuJoCo `Ant-v5` 환경에서 (a) robotics-engineer 관점에서 "well-trained"라 부를 만한 nominal locomotion 정책과 (b) 외부 push disturbance에 robust한 정책을 학습하는 것이 목표. 기존 baseline (vanilla Ant-v5 + PPO)은 forward reward에만 의존하여 path가 일정치 않고 자세가 흔들리는 한계가 있었음.

연구 흐름은 두 단계로 구성:
- **Phase 1**: command-tracking objective 설계 + curriculum 학습으로 안정적 nominal gait 확보.
- **Phase 2 / 2b**: nominal 정책에 push curriculum + domain randomization 추가, 점진적으로 더 가혹한 stress 환경까지 평가.

## 2. 실험 환경과 제약

- **Sim**: MuJoCo Ant-v5 (Gymnasium), 토크 액추에이터 8개, dt=0.01s.
- **Algorithm**: PPO (Stable-Baselines3), MLP 256-256 policy/value.
- **Compute**: 단일 머신 + Docker, `n_envs=4` SubprocVecEnv. 이 점이 결과를 해석할 때 중요한 제약 (legged_gym/PA-LOCO 류는 보통 1k~4k env 병렬을 가정).
- **Budget**: 단계당 1~1.5M timesteps × 1~3 seeds.

참고 문헌: Rudin et al. *Learning to Walk in Minutes* (arXiv:2109.11978), Kumar et al. *RMA* (arXiv:2107.04034), PA-LOCO (arXiv:2407.04224), Robust Gymnasium (arXiv:2502.19652).

## 3. Phase 1 — Well-Trained Nominal Locomotion

### 3.1 문제 정의

"Well-trained"의 절대 기준을 robotics 관점에서 13개로 명세 (survival, vx tracking error, lateral drift, body yaw/course alignment, roll/pitch RMS, height std, action smoothness 등). 단순히 직전 실험 대비 개선이 아니라 절대 수치로 평가.

### 3.2 Objective 설계의 evolution

| 버전 | 핵심 변경 | 발견 |
|---|---|---|
| v3a–c | World-frame tracking + lateral position penalty | "yaw로 steer해서 path 유지" (drift 적지만 body가 옆으로 yaw됨) 와 "body 정렬 + 큰 drift" 사이에서 둘 다 만족 불가 |
| v3d | Body-frame command tracking (legged_gym 형식) | Body-frame 항만으로는 yaw rotation 자체에 invariant → drift 73m |
| v3e | + `w_heading_alignment * exp(-yaw²/σ²)` | yaw=0.06으로 잡힘, drift 18m 잔존 (gait asymmetry) |
| v3f | continuation + 강한 lateral | local optimum 못 깸 (변화 거의 없음) |
| v3g | fresh + 강한 vy penalty | standstill collapse (`vx≈0`, reward maximization at no-motion) |
| v3h | **2-stage curriculum** (weak lateral → strong lateral) | drift 6.5m, 안정적 정리된 gait |

핵심 insight: 단일 reward weight set으로는 "walking을 안 깨면서 lateral drift도 잡는" 영역을 찾기 어려움. **vy penalty가 약하면 drift bias 잔존, 강하면 standstill로 붕괴**. Curriculum (weak → strong) 으로 "이미 걷는 정책"을 warm-start 함으로써 두 attractor를 모두 회피.

### 3.3 Phase 1 deliverable

[runs/well_trained_v3h_s2_seed42_1000k/](runs/well_trained_v3h_s2_seed42_1000k/) — `WellTrainedLocomotionAntWrapper` body-frame command tracking objective, 2-stage curriculum.

| metric | 값 |
|---|---|
| survival | 1.0 |
| vx_body_mean | 1.79 |
| abs_lateral_drift (50s) | 6.5m |
| yaw_abs_mean | 0.16 |
| roll/pitch RMS | 0.03 / 0.11 |
| height_std | 0.04 |

13/13 절대 기준 중 5개만 정량적으로 통과. 그러나 영상 정성 평가에서 "robotics 엔지니어가 보기에 안정적인 보행"으로 분류됨 (사용자 검증). 빠른 속도(vx=2.07, v3e)보다 안정성(v3h_s2 vx=1.79)이 우선되는 도메인 가치 판단.

### 3.4 구조적 한계

PPO + 4 env로는 13/13 도달 불가. legged_gym 류 결과는 4096 env의 batch diversity가 핵심 — 같은 hyperparam으로 100배 batch라 exploration이 다름. 단일 머신 환경에서 reward tuning 만으로 메우는 데는 천장이 있음.

## 4. Phase 2 — Push Robustness

### 4.1 접근

PA-LOCO / Rudin 류 표준 베이스라인 채택: **random push curriculum + domain randomization, reward는 변경 없음**. Teacher-student / RMA 류 privileged learning은 phase 3 후보로 보류.

구현 wrapper:
- `PushDisturbanceWrapper`: torso body에 `data.xfrc_applied`로 외력 적용. push duration, interval, magnitude curriculum 지원.
- `DomainRandomizationWrapper`: 리셋 시 mass (0.8–1.2×), friction (0.5–1.5×), joint damping (0.8–1.2×), motor strength (0.85–1.15×) 랜덤화 + 매 step Gaussian action noise.

Wrapper stack (outermost → innermost): WellTrainedLocomotion → Push → DR → Ant-v5.

### 4.2 Training schedule

- v4a (nominal robust): warm-start from v3h_s2_seed42. Push curriculum 0 → 10N over 300k env-steps, duration 0.05s, interval 5–10s, xy 평면, 1.5M total.
- v4b (stretch): warm-start from v4a. Push 0 → 20N (same duration), 1M total.

### 4.3 결과

| metric | v3h_s2 (P1) | v4a | v4b |
|---|---|---|---|
| Tier A drift | 6.5m | **1.4m** | **1.2m** |
| Tier A vx_body | 1.79 | 1.87 | 1.70 |
| Tier A yaw | 0.16 | 0.13 | 0.10 |
| Tier B max force survived (brief 0.05s) | n/a | 10N (100%) | 20N (100%) |

**핵심 관찰**: push training이 nominal locomotion을 *개선* (drift 6.5m → 1.2m). DR + push perturbation이 단순 robustness 학습을 넘어 implicit regularization으로 작용해 더 정확한 path tracking gait가 형성됨. 이는 legged_gym/PA-LOCO 논문의 표준 관찰과 일치.

## 5. Phase 2b — 정직한 Robustness 평가

### 5.1 동기

Phase 2의 "20N까지 100% survival" 헤드라인이 정성적 영상에서는 push 효과가 거의 안 보였음. 원인 분석:
- Push duration 0.05s = 영상 1.5 프레임, 시각적으로 인지 불가.
- xy 평면만 + torso center → 각운동량 disturbance 0, quadruped wide-base가 본질적으로 흡수.
- 인간이 미는 직관 (~0.3s sustained + 회전) 과 mismatch.

### 5.2 Stress test 설계

- Duration 0.05s → **0.3s** (현실적 push 시간).
- **각 impulse 추가**: torso z축 ±0.2 N·m torque.
- Linear force 5–15N. 절대값은 낮으나 총 impulse는 v4b 대비 약 4.5× (15N × 0.3s = 4.5 N·s vs 20N × 0.05s = 1.0 N·s).

PushDisturbanceWrapper에 `push_torque_z_max`, `push_duration_max_steps` 추가하여 동일 curriculum progress로 force / torque / duration 함께 ramp.

### 5.3 v4b on stress grid (baseline)

| force | survival 통과 셀 | 평균 max drift |
|---|---|---|
| 5N | 8/8 (100%) | 2.7m |
| 10N | 6.25/8 (78%) | 5.4m |
| 15N | 3.5/8 (**44%**) | 10.1m |

→ 원래 평가의 "20N 100% survival"은 실제 정책의 robustness가 아니라 disturbance가 너무 약했기 때문.

### 5.4 v4c — stress curriculum 학습

v4b에서 warm-start, stress curriculum (force 0→15N, torque 0→0.2 N·m, duration 5→30 steps) 으로 1M step 추가 학습.

| force | v4b survival | v4c survival | Δ |
|---|---|---|---|
| 5N | 100% | 100% | 0 |
| 10N | 78% | 75% | -3% |
| 15N | 44% | 47% | +3% |

**Null result**: stress 환경에서 1M 추가 학습했음에도 survival 향상 거의 없음. v4c의 nominal drift도 v4b 대비 약간 악화 (1.18m → 1.37m).

### 5.5 해석

PPO + 4 env 구조 한계와 일치. 15N × 0.3s 만큼의 impulse는 Ant 0.33kg에 대해 이론상 δv ≈ 13.6 m/s 의 lateral velocity change에 해당 (지면 friction 일부 흡수 가정해도 매우 큼). 이 정도 disturbance에 대해서는 reward signal과 학습 budget이 충분하지 않은 듯하며, 알고리즘 / 병렬화 수준의 변경이 필요할 것으로 추정.

**솔직한 robustness ceiling**: ~10N sustained 0.3s + 0.2 N·m angular impulse 가 현재 학습 setup의 한계점.

## 6. Research Takeaways

### 6.1 작동한 design choice

1. **Body-frame command tracking + 명시적 world-frame yaw alignment 동시 사용**. 둘 중 하나만으로는 부족. body-frame은 walking gait dynamics 모델링에 자연스럽고, yaw alignment는 world-frame 목표 방향을 anchor.

2. **Curriculum (weak → strong lateral discipline)** 으로 attractor 사이 통과. Single-shot tuning은 "standstill" 혹은 "biased gait" 둘 중 하나로 수렴.

3. **Push training이 nominal performance를 개선**. DR + perturbation이 regularizer로 작용하는 패턴은 legged_gym 문헌과 일치하며, robust 학습을 nominal 학습 후에 별도로 두는 것이 nominal 성능까지 끌어올림.

### 6.2 작동하지 않은 시도

1. **단일-shot 강한 lateral penalty**: standstill local optimum으로 정책 붕괴.
2. **Continuation from biased policy + 강화된 reward**: PPO `std=0.15` 정도의 낮은 entropy로는 local optimum 탈출 불가.
3. **Stress curriculum 추가 학습**: 동일 algorithm + 동일 batch size 영역 안에서는 disturbance 한계를 못 넘김.

### 6.3 정직한 평가의 중요성

**원래 Tier B (brief 0.05s torso-only impulse)는 quadruped 본질적 안정성에 흡수되어 정책 robustness를 거의 probe 하지 못 함.** 진짜 도전적 평가는 (a) 충분한 duration, (b) 각 impulse, (c) 가능하면 random body part 까지 포함해야 함. 짧은 impulse만 평가하는 push robustness 보고는 실제 성능을 과대평가할 위험이 있음.

## 7. Limitations & Future Directions

### 7.1 단일 머신 PPO 환경 한계

1.5M timesteps × 4 env로는 nominal/robust 모두 절대 robotics-grade에 도달 불가. 같은 reward 설계라도 4096 env로 학습하면 다를 가능성 큼. Robotics-quality result가 필요하면 IsaacGym/leggedrobotics-style 병렬화로 이동하는 것이 효율적.

### 7.2 알고리즘 후보

- **SAC** (off-policy, sample-efficient): 작은 batch에서도 PPO보다 explorative하다는 보고.
- **RMA / privileged learning**: 환경 latent encoder + adaptation module 분리. push disturbance estimation을 명시화하여 robustness 천장 높이는 데 유리.
- **Adversarial RL** (RARL, SA2RT): adversary가 worst-case perturbation을 찾도록 학습. 다만 학습 안정성이 PPO 베이스라인보다 낮음.

### 7.3 Reward 측면

- vy_body가 항상 0이 되도록 강제하기보다, "gait cycle 평균 vy = 0" 같은 time-averaged 형태의 reward가 더 자연스러울 수 있음. 현재 instantaneous penalty가 gait wiggle을 standstill로 오인.
- Action smoothness (action_rate, action_accel) 항은 torque actuator에서 효과 제한적. PD-position control 환경에서는 같은 weight로 훨씬 부드러운 gait 가능.

### 7.4 Evaluation 측면

- 현재 "survival to 1000 steps" 기준이 weak. "5초 내 nominal trajectory 복귀" 같은 stricter recovery metric이 필요.
- Gait period regularity (foot contact pattern의 spectral entropy 등) 같은 지표가 정성적 "정리된 gait" 판단을 정량화하는 데 도움이 됨.

## 8. Reproducibility

| Phase | 정책 | 학습 명령 (요약) |
|---|---|---|
| 1 | [v3h_s2_seed42](runs/well_trained_v3h_s2_seed42_1000k/) | `train_well_trained_locomotion.py` 2-stage curriculum |
| 2 | [v4a](runs/robust_locomotion_v4a_seed42_1500k/) | `train_robust_locomotion.py` --push-force-max 10 |
| 2 | [v4b](runs/robust_locomotion_v4b_stretch20N_seed42_1000k/) | `train_robust_locomotion.py` --push-force-max 20 |
| 2b | [v4c](runs/robust_locomotion_v4c_stress_seed42_1000k/) | + `--push-torque-z-max 0.20 --push-duration-max-steps 30` |

평가:
- Tier A/B/C standard grid: `eval_robust_locomotion.py --tiers abc`
- Stress grid: `--tier-b-duration-steps 30 --tier-b-torque-z 0.20 --tier-b-randomize-torque-sign`

영상 (Phase별 정성 비교):
- Phase 1: `videos/phase1_candidate_v3h_s2_seed42.mp4`
- Phase 2: `videos/phase2_v4b_quiet.mp4`, `videos/phase2_v4b_push20N_*.mp4`
- Phase 2b: `videos/phase2b_v4b_stress_10N_lateral.mp4`, `videos/phase2b_v4c_stress_*.mp4`

## 9. Closing

이 프로젝트의 가장 큰 결과는 단일 robust policy 자체가 아니라, **3단계 (nominal → robust → stress eval) 의 정직한 cycle을 통해 PPO + 4 env 셋업에서 가능한 영역과 한계를 명확히 그어냈다는 것**. 추가 작업은 알고리즘 변경 또는 병렬화 수준 변경이 ROI가 높음.
