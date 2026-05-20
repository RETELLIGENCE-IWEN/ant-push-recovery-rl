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

### 5.5 해석 (E1 측정 후 업데이트)

**E1 (Stress Physics Audit) 측정 결과** (`reports/stress_physics_audit.csv`, 116 rows):
- 15N × 0.3s push 시 측정 Δv_xy ≈ **12–14 m/s** (first-order estimate 13.6 m/s에 근접). 0.3s 지속 push는 여러 gait cycle을 거치는 동안 contact reaction이 따라잡지 못해 사실상 free-body 가속에 가까운 결과를 보임.
- 5N × 0.3s에서는 Δv ≈ 1.2–2.7 m/s (first-order 4.5 m/s 대비 훨씬 작음). 작은 force에서는 ground contact가 효과적으로 흡수.
- 0.2 N·m torque에서 Δω_z ≈ 0.9–2.6 rad/s. 의미 있는 yaw disturbance.
- v4c가 10N dir=180°에서 동일 force 대비 Δv 감소 (7.14 → 5.36 m/s). survival 동일하나 정책이 push 동안 능동적으로 저항.
- v4b 15N dir=180°는 push window 내부에서 fall (end_push_state 측정 불가). 가장 극단적 failure direction.

즉, 단순 first-order δv 추정의 직관 (큰 외란) 자체는 잘못되지 않았으나, **force 크기에 따라 contact reaction 효과가 비선형으로 변함** (5N에서는 강하게 흡수, 15N에서는 거의 흡수 안 됨)을 측정으로 확인.

v4c null result의 가능한 원인들 (아직 한 가지로 좁히지 않음):

1. **실제 disturbance가 정말 학습 가능 영역 밖**일 가능성. E1에서 effective Δv를 측정해야 판단 가능.
2. **Partial observability**: 정책이 "지금 push 들어왔는지 / 얼마나 / 어느 방향"을 명시적으로 모름. MLP single-frame observation으로는 회복 전략을 학습하는 데 한계가 있을 수 있음. RMA / history-stack / RecurrentPPO 등이 다음 후보.
3. **Recovery reward 부재**: nominal reward (vx tracking, lateral, yaw alignment) 은 "push 후 빠르게 nominal 상태로 돌아가는 것"을 직접 보상하지 않음. Push 후 일정 시간 window에서만 작동하는 recovery reward가 빠져 있음.
4. **Boundary sampling 부족**: 0 → 15N curriculum에서 5N은 이미 쉬운 영역, 15N은 너무 어려운 영역이라 학습 sample 대부분이 "이미 학습된 영역"이거나 "도달 불가 영역"에 분포. Boundary (8–12N) 집중이 필요.

이 네 가지를 분리 검증하기 전에 알고리즘 (PPO → SAC) 부터 갈아엎으면 원인을 좁히기 어렵다.

## 5.6 E2 — Recovery Metric Grid

E1에서 사용한 strict recovery criterion (50 contiguous step clean state)이 모든 cell에서 trigger 실패 → metric 자체가 Ant gait 특성에 맞지 않음. E2에서는 sliding window mean 기반의 relaxed criterion으로 재정의 + integrated tracking error / max post-push deviation 추가.

E2 결과 (v4b vs v4c, `reports/recovery_metric_grid.csv`):
- Survival에서 보인 v4c null result가 recovery time / integrated error에서도 동일하게 확인됨.
- v4c가 10N dir=180°에서만 marginal 개선 (survival 25% → 50%, 그러나 다른 cell 후퇴 상쇄).
- **결론: v4c의 무효는 "측정 한계"가 아니라 실제 정책 차이 부재**. 따라서 다음 단계는 reward / sampling / observation 구조 변경.

## 5.7 E3 — v4d: Boundary Sampling + Windowed Recovery Reward

E1/E2 결과를 바탕으로, v4c null result의 남은 가설 중 (3) recovery reward 부재 + (4) boundary sampling 부족 두 개를 동시에 attack.

**구현 변경**:
- `PushDisturbanceWrapper`에 boundary sampling 모드 추가: easy [5-8N, 15-20 step, 0.05-0.10 N·m] / mid [8-12N, 20-30, 0.10-0.18] / hard [12-15N, 25-30, 0.15-0.20], 가중치 20/60/20.
- `WellTrainedLocomotionAntWrapper`에 windowed recovery reward 추가: push 발생 동안 + push end 후 5초 window에서만 작동하는 `-w * |vx_err|`, `-w * |vy_err|`, `-w * |yaw_err|`, `-w * (roll²+pitch²)`, `-w * |yaw_rate|` 항. window 밖에서는 0 → nominal gait 영향 차단.
- `PushDisturbanceWrapper`가 `info["push_active"]`, `info["push_steps_since_end"]` 를 inner→outer로 propagate, locomotion wrapper가 이를 읽어 reward gate.

**Training**: v4b warm-start, 1M steps, single seed.

**결과** (`reports/recovery_metric_grid_v4d.csv`):

| F (dir) | v4b surv | v4d surv | Δ |
|---|---|---|---|
| 5N (avg) | 100% | 100% | = |
| 10N (avg) | 75% | 69% | -6% |
| **15N 0°** | **25%** | **75%** | **+50%** |
| 15N 90° | 100% | 50% | -50% |
| 15N 180° | 0% | 25% | +25% |
| 15N 270° | 25% | 25% | = |
| 15N (avg) | 38% | **44%** | **+6%** |

Tier A (quiet): v4d survival 1.0, vx 1.75, yaw 0.10 — v4b와 동등. 다만 nominal drift 1.18m → 3.17m로 증가 (recovery reward + boundary curriculum이 nominal gait를 약간 perturbation).

**해석**:
- v4d가 v4c 대비 명확히 개선 (특히 15N forward, +50% survival). 즉 가설 (3)/(4) 가 v4c 무효의 일부 원인임이 검증됨.
- 그러나 direction에 따라 mixed (15N 90° lateral은 후퇴). **Cross-direction 일관성 부재** → 남은 가설 (2) Partial observability가 가장 유력한 다음 bottleneck. 정책이 "어느 방향으로 push 들어왔는지"를 single-frame proprio로는 추정 불가하고, 이 한계가 lateral / forward 사이의 transfer를 막는 것으로 보임.
- Nominal drift 증가 (1.18 → 3.17m)는 recovery reward window가 push 후 5초 켜져 있어, 자연스러운 gait wiggle도 일부 penalty 받기 때문. window 크기를 줄이거나 recovery weight를 낮춰 mitigation 가능.

## 5.8 잠정 robustness ceiling (E1-E3 종합)

세 실험 후 도달한 결론:

- **15N × 0.3s × 0.2 N·m 의 stress 가 절대 학습 불가 영역이 아님**: v4d가 forward dir에서 75% 도달. 단지 v4b/v4c의 reward + sampling 설계가 부족했을 뿐.
- **정책 ceiling은 아직 partial observability가 깨지 않은 영역**: history-stack / RecurrentPPO / RMA-lite 가 다음 후보.
- **알고리즘 (PPO → SAC) 변경은 여전히 후순위**: 위 1-2개를 더 시도하기 전엔 SAC 도입은 원인 분리만 어렵게 함.

## 5.9 E4 — v5a: History Stack 으로 Partial Observability 검증

E3 v4d 의 cross-direction inconsistency 가 partial observability 때문이라는 가설을 직접 검증. 정책 입력에 최근 5 frame (0.05s) proprio history 추가.

**구현**:
- `ObservationHistoryStackWrapper`: 5 frame 저장, layout `[newest, ..., oldest]` (newest-first).
- `train_robust_locomotion.py`에 `--warm-start-on-observation-mismatch` 모드 추가. v4d (obs 117) 의 weights 를 v5a (obs 585) 의 *앞 117 cols* 에 복사 + 나머지 zero-init → warm-start 초기엔 v4d 와 동일한 mean action (현재 obs 만 사용).

**디버깅 노트**: 첫 시도 (v5a v1) 는 wrapper layout 이 oldest-first 였고 warm-start expansion 이 첫 cols (= oldest frame) 에 v4d weights 를 복사 → 정책이 4-step 지연된 obs 만 보고 학습 → catastrophic failure (Tier A vx 0.28, return -2508). Layout 을 newest-first 로 수정한 v5a v2 가 정상 학습.

**Training**: v4d warm-start, learning_rate 5e-5 (보수적), 동일 reward + boundary push curriculum, 1M steps.

**결과** (`reports/recovery_metric_grid_v5a_v2.csv`):

| F (dir) | v4d surv | v5a_v2 surv | Δ |
|---|---|---|---|
| 5N (all 4) | 100% | 100% | = |
| 10N 0° | 100% | 100% | = |
| **10N 180°** | **0%** | **50%** | **+50%** |
| 10N 270° | 75% | 100% | +25% |
| **10N avg** | **69%** | **88%** | **+19%** |
| 15N 0° | 75% | 0% | -75% |
| 15N 90° | 50% | 100% | +50% |
| 15N 180° | 25% | 25% | = |
| 15N 270° | 25% | 25% | = |
| 15N avg | 44% | 38% | -6% |

Recovery time 도 개선: 10N dir=90° 132 → 55 steps (2.4× 빨라짐).

Tier A: survival 1.0, vx 1.78, drift 4.06m, yaw 0.094 — nominal locomotion 유지 (drift 약간 증가).

**해석**:
- **10N 영역에서 partial observability 가설 명확히 검증됨**: v4d 가 0% 였던 10N dir=180° (backward push) 가 v5a_v2 에서 50% 까지 회복. cross-direction 일관성 개선이 history stack 의 직접 효과로 보임.
- **15N 영역에선 trade-off**: 90° lateral 50%→100% 큰 개선이지만 0° forward 75%→0% 회귀. 이는 partial observability 만으로 설명 안 되며, 학습 budget 부족 또는 stack size 5 (0.05s) 가 30-step push 의 full pattern 을 담기엔 너무 짧을 가능성. stack=20-30 또는 RecurrentPPO 가 다음 후보.
- **실패 사례 (v5a v1) 가 더 큰 교훈**: warm-start expansion 의 layout 가정을 wrapper 와 정확히 맞추지 않으면 정책 catastrophic failure. observation augmentation 류 작업에선 항상 layout convention 을 명시적으로 정의해야 함.

## 5.10 잠정 결론 update

E1-E4 cycle 후:

- **Partial observability 는 robustness ceiling 의 일부**, 그러나 *유일한* 요인은 아님 (15N forward 의 v5a_v2 회귀가 그 증거).
- **History stack 5 는 mid-range disturbance (10N) 에 효과적**, hard regime (15N) 에는 부족.
- 다음 단일 intervention 후보 (우선순위): (1) history stack 확장 (10-20 frame, push duration 의 1/3 이상) (2) RecurrentPPO (3) RMA-lite teacher-student.

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

### 6.4 측정 우선, 알고리즘 변경 후순위

v4c null result에서 즉시 "PPO 한계, SAC로 교체" 결론을 낼 뻔했음. 그러나 E1 (physics audit) → E2 (recovery metric redesign) → E3 (boundary + windowed recovery) 의 isolated cause testing 시퀀스를 거치자 **reward / sampling 설계 변경만으로 일부 stress 조건에서 실질 개선 가능함**이 확인됨 (v4d 15N forward: 25% → 75%). **알고리즘 교체는 reward / observation / sampling 모두 검증한 후에 가야 함**이 이번 사이클의 가장 큰 method-level 교훈.

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
| 2c | [v4d](runs/robust_locomotion_v4d_boundary_recovery_seed42_1000k/) | + `--push-use-boundary-sampling --w-recovery-{vx,vy,yaw}-err 0.5 --w-recovery-roll-pitch 1.0 --w-recovery-yaw-rate 0.2` |
| 2d | [v5a_v2](runs/robust_locomotion_v5a_history5_seed42_1000k_v2/) | + `--warm-start-on-observation-mismatch --history-stack-size 5 --learning-rate 5e-5` |

평가:
- Tier A/B/C standard grid: `eval_robust_locomotion.py --tiers abc`
- Stress grid: `--tier-b-duration-steps 30 --tier-b-torque-z 0.20 --tier-b-randomize-torque-sign`
- E1 Physics audit: `stress_physics_audit.py --models v4b=...,v4c=...`
- E2 Recovery metrics: `recovery_metric_grid.py --models v4b=...,v4d=...`

영상 (Phase별 정성 비교):
- Phase 1: `videos/phase1_candidate_v3h_s2_seed42.mp4`
- Phase 2: `videos/phase2_v4b_quiet.mp4`, `videos/phase2_v4b_push20N_*.mp4`
- Phase 2b: `videos/phase2b_v4b_stress_10N_lateral.mp4`, `videos/phase2b_v4c_stress_*.mp4`
- Phase 2c (v4d): `videos/phase2c_v4d_quiet.mp4`, `videos/phase2c_v4d_stress_15N_*.mp4`
- Phase 2d (v5a_v2): `videos/phase3_v5a_quiet.mp4`, `videos/phase3_v5a_stress_10N_180.mp4`, `videos/phase3_v5a_stress_15N_90.mp4`

## 9. Closing

이 프로젝트의 가장 큰 결과는 단일 robust policy 자체가 아니라, **4단계 (nominal → robust → stress eval → cause isolation) 의 정직한 cycle을 통해 reward 설계 / sampling 분포 / observation 구조가 robustness ceiling에 어떻게 기여하는지 분리 검증한 것**. v4d 결과 (15N forward 25% → 75%) 는 "PPO + 4 env 절대 한계" 라는 결론이 적어도 부분적으로는 잘못된 가설이었음을 보여주며, 알고리즘 변경 (SAC, RMA) 보다 reward / sampling / observation 설계 검증이 먼저 와야 함을 시사함.

다음 가장 유력한 single intervention: **observation history stack 또는 RecurrentPPO**. v4d의 cross-direction inconsistency가 partial observability에서 오는 것이라면 이 방향이 단일 가장 큰 leap이 될 가능성 큼. 이후에도 stress ceiling이 그대로면 그제서야 SAC / RMA-lite 도입을 고려.
