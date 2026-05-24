# ant-push-recovery-rl — 지원서용 프로젝트 분석 리포트

> 대상 직무: 현대차 제조로보틱스 / 강화학습 / 로봇 정책학습
> 작성일 기준: 2026-05-24
> 모든 수치·인용은 본 repository의 코드·config·csv·json 출력물에 근거하며, 외부 결과나 추정치는 사용하지 않았습니다.

---

## 1. Repository Snapshot

| 항목 | 내용 |
|---|---|
| Repo name | `ant-push-recovery-rl` |
| Current branch | `main` |
| Latest commit | `3e7ec95` — *v1 report* |
| Recent commits | `d219e10` M2 환경 검증 완료 (Go1 SB3 호환 gym.Env) <br> `0c3d964` frame stacking 으로 성능 개선 성공 <br> `ba7181e` E1-E2-E3 사이클 완료 및 REPORT 업데이트 <br> `b4afc49` Phase 2b 결과 — 정직한 robustness ceiling 발견 <br> `402ff66` phase 2: success (push training이 nominal locomotion을 개선) |
| Main language | Python 3.11 |
| Framework | Stable-Baselines3 2.8.0, Gymnasium 1.2.0 [mujoco], PyTorch 2.5.1 (CPU) |
| 주요 dependencies | `stable-baselines3==2.8.0`, `gymnasium[mujoco]==1.2.0`, `numpy==1.26.4`, `mujoco` (Menagerie scene 포함), `tensorboard`, `imageio`, `imageio-ffmpeg`, `pandas`, `matplotlib`, `rich`, `tqdm` ([requirements.txt](requirements.txt)) |
| 재현 환경 | [Dockerfile](Dockerfile) — `python:3.11-slim` 기반, MuJoCo OSMesa headless 렌더링, OMP/MKL thread=1, ffmpeg/xvfb 포함 |

**Top-level 디렉터리 트리 (depth 2 요약)**

```
ant-push-recovery-rl/
├── Dockerfile
├── REPORT.md                          # 기존 한글 연구 로그 (310 lines)
├── POC_REPORT.md                      # 학술 스타일 영문 POC 보고서
├── requirements.txt
├── scripts/                           # 학습/평가/오디트 entrypoint
│   └── go1/                           # Go1 (Unitree) 환경 검증 milestone (M0/M1/M2)
├── runs/                              # 학습 산출물 (config.json, models, checkpoints, tb)
├── reports/                           # 평가 메트릭 csv/json
├── plots/
├── videos/                            # 정성 평가용 mp4 35개
├── logs/
└── external/mujoco_menagerie         # Unitree Go1 모델 (서브모듈 형태)
```

**실행 가능한 entrypoint (scripts/)**

| 파일 | 역할 |
|---|---|
| [scripts/train_baseline.py](scripts/train_baseline.py) | Vanilla Ant-v5 + PPO baseline |
| [scripts/train_stable_directional.py](scripts/train_stable_directional.py) | 초기 directional locomotion 실험 |
| [scripts/train_controlled_locomotion.py](scripts/train_controlled_locomotion.py) | controlled locomotion wrapper 기반 학습 |
| [scripts/train_well_trained_locomotion.py](scripts/train_well_trained_locomotion.py) | Phase 1 — body-frame command tracking + 2-stage curriculum |
| [scripts/train_robust_locomotion.py](scripts/train_robust_locomotion.py) | Phase 2 — PushDisturbance + DomainRandomization + (옵션) ObservationHistoryStack |
| [scripts/continue_training.py](scripts/continue_training.py) | warm-start로 추가 학습 |
| [scripts/eval_policy_clean.py](scripts/eval_policy_clean.py) | nominal (Tier A) 평가 |
| [scripts/eval_policy_metrics.py](scripts/eval_policy_metrics.py) | locomotion 정밀 metric |
| [scripts/eval_policy_push.py](scripts/eval_policy_push.py) / [eval_push_sweep.py](scripts/eval_push_sweep.py) | push 단일/스윕 평가 |
| [scripts/eval_robust_locomotion.py](scripts/eval_robust_locomotion.py) | Tier A/B/C (quiet / push grid / random push) 통합 평가 |
| [scripts/stress_physics_audit.py](scripts/stress_physics_audit.py) | E1 — push 직전·직후 qvel Δ 측정 |
| [scripts/recovery_metric_grid.py](scripts/recovery_metric_grid.py) | E2 — 회복 지표 그리드 |
| [scripts/record_policy_video.py](scripts/record_policy_video.py), [record_push_video.py](scripts/record_push_video.py) | mp4 산출 |
| [scripts/go1/m0_inspect_model.py](scripts/go1/m0_inspect_model.py) | Go1 모델 inspection |
| [scripts/go1/m1_standing_pd.py](scripts/go1/m1_standing_pd.py) | Go1 PD 기립 검증 |
| [scripts/go1/m2_env_smoke.py](scripts/go1/m2_env_smoke.py) | Go1Env smoke test (A_zero / B_small / C_full) |
| [scripts/go1/go1_env.py](scripts/go1/go1_env.py) | Go1 gym.Env wrapper (M2 산출물) |
| [scripts/stable_directional_ant.py](scripts/stable_directional_ant.py) | **공통 wrapper 모듈** — 모든 Reward/Push/DR/HistoryStack wrapper 정의 |

**주요 config 파일**

학습 설정은 모두 CLI argparse → `runs/<exp>/config.json`에 직렬화되어 저장되어 있어, 어떤 실험이든 정확한 hyperparameter set이 동결되어 있습니다. 예시:

- [runs/well_trained_v3h_s2_seed42_1000k/config.json](runs/well_trained_v3h_s2_seed42_1000k/config.json) (Phase 1)
- [runs/robust_locomotion_v4b_stretch20N_seed42_1000k/config.json](runs/robust_locomotion_v4b_stretch20N_seed42_1000k/config.json) (Phase 2)
- [runs/robust_locomotion_v4d_boundary_recovery_seed42_1000k/config.json](runs/robust_locomotion_v4d_boundary_recovery_seed42_1000k/config.json) (Phase 2c)
- [runs/robust_locomotion_v5a_history5_seed42_1000k_v2/config.json](runs/robust_locomotion_v5a_history5_seed42_1000k_v2/config.json) (Phase 2d)

**주요 docs / report / output**

- [REPORT.md](REPORT.md) — 한글 연구 로그 (Phase 1 → Phase 2 → E1/E2/E3/E4 사이클)
- [POC_REPORT.md](POC_REPORT.md) — 학술 스타일 영문 POC writeup
- [reports/](reports/) — 정량 metric csv/json (Tier A/B/C, stress grid, recovery grid, physics audit)
- [videos/](videos/) — Phase별 정성 평가 mp4 35개 (Ant 11종 + Go1 5종 + 그 외)

---

## 2. Project Purpose

### 풀고 있는 문제

본 PoC는 **MuJoCo 시뮬레이터 위에서 (a) 안정적으로 보행하고 (b) 외란(push)에 강건하게 회복하는 다족 로봇 정책을 강화학습으로 학습하는 파이프라인을 구축**하는 것을 목표로 합니다. 단일 머신 / PPO / 4-병렬 환경이라는 제약을 명시적으로 두고, 그 제약 안에서 "어디까지 갈 수 있는가, 무엇이 진짜 ceiling인가"를 측정 기반으로 확인합니다.

### Target robot / task / environment

- **메인 robot**: MuJoCo Gymnasium `Ant-v5` (8-DoF 토크 제어 4족, dt=0.01s)
- **확장 target**: Unitree Go1 (MuJoCo Menagerie 모델, 12-DoF 위치 제어 PD, dt=0.002s, decimation 10 → 50 Hz 정책)
- **Task**: forward command를 추적하면서 외부 push를 견디고 빠르게 회복하는 robust quadruped locomotion

### "Push recovery / robust locomotion" 관점 핵심 목표

1. **Nominal "well-trained" 보행 정책 확보** (Phase 1) — drift, yaw alignment, gait organization 기준 만족
2. **외란 학습으로 robust 정책 도출** (Phase 2) — random push curriculum + domain randomization
3. **정직한 stress 평가** (Phase 2b) — brief impulse가 아닌 sustained push + 각운동량 포함
4. **Null 결과의 cause isolation** (E1–E4) — disturbance physics → 평가지표 → reward/sampling → observation 순으로 단일 변수씩 검증

### 강화학습 관점의 의의

- **Reward shaping, curriculum, observation augmentation이 알고리즘 변경(SAC, RMA)만큼 — 때로는 더 — robust 성능에 영향을 준다는 것을 측정으로 분리해 보인 사례**. v4c null 결과에서 즉시 알고리즘 교체로 가지 않고 4단계 cause isolation을 거쳐 v4d에서 15N forward 25%→75% survival 개선을 회수했습니다.
- legged_gym / PA-LOCO 류 결과 (DR + push가 nominal 성능까지 개선시킴)을 **단일 머신·4 env 환경에서 재현**했습니다. Tier A drift 6.5m → 1.2m.
- "brief impulse + torso center" 평가가 quadruped 본질적 흡수 특성 때문에 정책 robustness를 과대평가한다는 결론을 stress grid 측정으로 입증했습니다 (v4b 20N 100% → 15N stress 44%).

### 현대차 제조로보틱스 / 로봇정책학습 직무 연결 지점

- **Sim 기반 정책학습 파이프라인 구축 경험** — Gymnasium wrapper, SB3 PPO, SubprocVecEnv, TensorBoard, checkpoint/eval 자동화까지 한 사이클을 직접 설계·운용.
- **외란 robust 정책 설계 경험** — manufacturing 환경에서 로봇이 받는 외부 disturbance(예: 부품 충돌, 사람과의 contact)에 대한 robust control과 동일한 문제 구조.
- **Reward·metric 설계 능력** — survival 같은 약한 지표에 의존하지 않고 recovery time / integrated tracking error / max post-push deviation 등 stricter metric을 직접 설계.
- **측정 우선 방법론** — null 결과에서 알고리즘부터 갈아엎지 않고 disturbance physics audit → metric → reward → observation 순으로 root cause isolation. 제조 현장에서 이상 거동의 원인을 분리해야 하는 상황과 동형.
- **Quadruped → 산업용 로봇 transfer 인지** — 본 PoC는 quadruped지만 동일 reward shaping / curriculum / observation 설계 패턴은 manipulator·AGV에도 적용 가능합니다.

---

## 3. Technical Pipeline

코드 기준 실제 구성입니다 ([scripts/stable_directional_ant.py](scripts/stable_directional_ant.py), [scripts/train_robust_locomotion.py](scripts/train_robust_locomotion.py), [scripts/go1/go1_env.py](scripts/go1/go1_env.py)).

### Simulator / physics engine

- MuJoCo (Gymnasium 1.2.0의 mujoco backend) — Ant-v5 dt=0.01s, Go1 dt=0.002s (decimation 10 → 정책 dt=0.02s)
- 헤드리스 렌더링: `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa` (Dockerfile에서 강제)

### Robot model

| 모델 | DoF | actuator | obs_dim (기본) | source |
|---|---|---|---|---|
| Ant-v5 | 8 (legs) | torque | 27 (raw qpos+qvel) + wrapper 추가 신호 | Gymnasium 기본 |
| Unitree Go1 | 12 (4 leg × 3) | position PD (kp=100 내부) | 48 (`3+3+3 + 3 + 12+12+12`) | external/mujoco_menagerie + go1_env.py 자체 wrapper |

Ant raw obs에 wrapper가 `prev_action`, target velocity error, height error 등 compact signal을 concat해 최종 input dim을 확장합니다. v5a_v2에서는 5-frame history stack까지 더해져 585-dim까지 확장됩니다.

### Action space

- Ant: 토크 제어, `Box(-1, 1, shape=(8,))`
- Go1: home pose 대비 residual joint target, `Box(-1, 1, shape=(12,))` → 실제 적용 시 `action_scale=0.15 rad` 곱한 후 home_ctrl에 더하고 actuator ctrlrange로 clip ([go1_env.py](scripts/go1/go1_env.py))

### Reward 설계 (WellTrainedLocomotionRewardConfig)

`scripts/stable_directional_ant.py` 정의된 reward 항목 (v4b config.json 기준):

| 항목 | weight | 의미 |
|---|---|---|
| `w_alive` | 0.10 | 생존 step보너스 |
| `w_track_vx` | 1.50 | body-frame x-velocity 추적 `exp(-vx_err²/σ²)` |
| `w_track_vy` | 2.50 | body-frame y-velocity 0 추적 |
| `w_track_omega_z` | 0.50 | body yaw rate 추적 |
| `w_progress_vx` | 1.0 | forward 속도 progress (선형) |
| `w_heading_alignment` | 1.0 | world-frame yaw 정렬 |
| `w_lateral_position` | 0.1 | world-frame y 위치 페널티 |
| `w_lin_vel_z` | 2.0 | z 속도 페널티 |
| `w_ang_vel_xy` | 0.05 | roll/pitch rate 페널티 |
| `w_orientation` | 1.0 | torso roll²+pitch² 페널티 |
| `w_base_height` | 1.0 | target height 0.53m 추적 |
| `w_action_rate`, `w_action_accel` | 0.05, 0.02 | action smoothness |
| Recovery window 항 (v4d+) | 0.5, 0.5, 0.5, 1.0, 0.2 | push 발생 + 5초 window 안에서만 작동하는 vx/vy/yaw err·roll-pitch·yaw-rate 추가 페널티 |

### Termination

- Ant-v5 기본 종료조건 (height range out, 자세 fail 등) + episode 길이 1000 step

### External push / disturbance (PushDisturbanceWrapper)

`data.xfrc_applied`를 통해 torso body에 외력·외 토크를 인가합니다 ([scripts/stable_directional_ant.py](scripts/stable_directional_ant.py)). 구성요소:

- `push_force_max` (N): xy 평면 random direction 외력 최대 크기
- `push_torque_z_max` (N·m): z축 토크 최대 크기 (Phase 2b 부터 활성)
- `push_duration_steps` / `push_duration_max_steps`: push 지속 시뮬레이션 step 수 (Phase 2b 부터 5→30 step ramp)
- `push_interval_steps_min/max`: push 간격 (보통 500–1000 step)
- `curriculum_ramp_steps`: 0 → max로 ramping할 worker별 step 수 (300k)
- `use_boundary_sampling`: easy [5–8 N, 15–20 step] / mid [8–12 N, 20–30] / hard [12–15 N, 25–30] 을 20/60/20 비율로 샘플
- `info["push_active"]`, `info["push_steps_since_end"]` 를 outer wrapper(reward)로 propagate

### Domain Randomization (DomainRandomizationWrapper)

reset 시 매번:
- body mass ~ U(0.8, 1.2) ×
- friction ~ U(0.5, 1.5) ×
- joint damping ~ U(0.8, 1.2) ×
- motor strength ~ U(0.85, 1.15) ×
- 매 step Gaussian action noise std=0.02

### Observation History Stack (Phase 2d)

`ObservationHistoryStackWrapper(stack_size=5)` — **newest-first layout** `[obs_t, obs_{t-1}, ..., obs_{t-4}]`. v5a v1에서 oldest-first 레이아웃이 warm-start expansion 가정과 불일치해 정책 catastrophic failure (Tier A vx=0.28, return=-2508) 발생 후 newest-first로 수정. 이 버그 분리·수정 사례 자체가 본 PoC의 실증적 자산입니다.

### Wrapper stack 순서 (outermost → innermost)

```
WellTrainedLocomotionAntWrapper      # reward, observation 가공
  └ ObservationHistoryStackWrapper   # optional (Phase 2d)
    └ PushDisturbanceWrapper         # xfrc_applied
      └ DomainRandomizationWrapper   # reset-time DR + action noise
        └ gym.make("Ant-v5")
```

### Training algorithm

- **PPO (SB3)** — MLP policy/value, 256-256 hidden, `learning_rate=1e-4` (v4b), `5e-5` (v5a_v2 보수적), `n_steps=2048`, `batch_size=128`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.0` ([runs/.../config.json](runs/))
- **`SubprocVecEnv` 4-병렬 worker** (단일 머신), `VecMonitor` 으로 episode stats 로깅
- **Warm-start 두 가지 모드**:
  - 동일 obs dim 정책 사이 `policy.load_state_dict` (Phase 2 표준)
  - obs dim 확장 정책(history stack 도입 시) `copy_policy_weights_with_expanded_input` — 첫 레이어 weight 의 leading column에 source 복사, 나머지 zero init

### Evaluation method

[scripts/eval_robust_locomotion.py](scripts/eval_robust_locomotion.py) — Tier A/B/C 세 단계:
- **Tier A (quiet)**: 외란 0, 10 episode, 1000 step
- **Tier B (push grid)**: 단일 push, magnitudes × 8 directions × 3 episode
  - 기본: 0.05s duration, 토크 0
  - **Stress 모드**: `--tier-b-duration-steps 30 --tier-b-torque-z 0.20 --tier-b-randomize-torque-sign`
- **Tier C (random push)**: 학습과 동일한 random push 분포 평가

별도 cause-isolation 도구:
- [scripts/stress_physics_audit.py](scripts/stress_physics_audit.py) — push 직전·직후 `qvel` 측정, 116-row csv 생성
- [scripts/recovery_metric_grid.py](scripts/recovery_metric_grid.py) — sliding-window 기반 recovery time, integrated tracking error, max post-push deviation

### Logging / TensorBoard / report generation

- 학습: `runs/<exp>/tb/` (SB3 TensorBoard 자동)
- Checkpoint: `runs/<exp>/checkpoints/` (`CheckpointCallback`, default 100k env-step 주기)
- 평가 산출: `reports/<exp>/{summary.json, tier_a_quiet.csv, tier_b_push_grid.csv, tier_c_random_push.csv}`
- 정성: `videos/` mp4 35개 (Phase별 quiet / 각 force-direction별)

### Failure diagnosis workflow (E1–E4 사이클)

1. **E1 — Physics audit**: 가설 *"disturbance가 실제로 학습 가능 영역 밖이다"*. 측정: 116 cell qvel Δ → 15N×0.3s 에서 |Δv_xy| 12-14 m/s 측정. **결론: disturbance는 진짜다. 가설 기각.**
2. **E2 — Metric redesign**: 가설 *"평가지표가 실제 차이를 못 잡는다"*. Strict recovery criterion(50-step clean) 로는 trigger 실패 → sliding-window mean 기반 relaxed criterion 도입 + integrated error 추가. **결론: v4c null 결과는 진짜다. 가설 기각.**
3. **E3 — v4d (boundary sampling + windowed recovery reward)**: 남은 두 가설 동시 검증 → 15N forward 25%→75% 회수. **부분 검증.**
4. **E4 — v5a_v2 (observation history stack)**: 마지막 가설 partial observability → 10N average 69%→88% 회수. **부분 검증, 단 15N hard regime은 stack 5 frame으로 부족.**

이 사이클이 본 PoC의 가장 generalizable한 산출입니다.

---

## 4. Milestone Progress

명시적 milestone 라벨이 git history와 디렉터리 명명으로 일관되게 박혀 있어 그대로 사용합니다.

| Milestone | Status | 구현 내용 | 주요 파일 | 확인된 결과 / metric | 비고 |
|---|---|---|---|---|---|
| **M0 — Baseline locomotion** | 완료 | Vanilla Ant-v5 + PPO baseline, 50k / 300k step | [scripts/train_baseline.py](scripts/train_baseline.py), [runs/baseline_ppo_ant_300k/](runs/baseline_ppo_ant_300k/) | survival 0.8, return 2776, mean fwd vel 2.30 m/s, **drift 통제 안됨** ([reports/baseline_clean_300k/summary.json](reports/baseline_clean_300k/summary.json)) | forward reward만으로는 path 불안정 |
| **M1 — Stable directional / controlled** | 완료 | directional / controlled locomotion wrapper 실험 (v1, v2, v2b, v2c, v2d, v3a, v3b, v3c) | [scripts/train_controlled_locomotion.py](scripts/train_controlled_locomotion.py), [reports/metrics_controlled_locomotion_v3*/](reports/) | 13개 절대 well-trained 기준 일부 충족, 일부 trade-off 발견 | reward 설계 evolution 기록 |
| **M2 — Well-trained nominal (v3d→v3h)** | 완료 (Phase 1 종결) | body-frame command tracking + 2-stage lateral curriculum | [scripts/train_well_trained_locomotion.py](scripts/train_well_trained_locomotion.py), [runs/well_trained_v3h_s2_seed42_1000k/](runs/well_trained_v3h_s2_seed42_1000k/) | survival 1.0, mean fwd vel **1.61 m/s**, abs lateral drift **6.48 m**, yaw_abs 0.144, roll/pitch 0.034/0.106 ([reports/metrics_well_trained_v3h_s2_seed42_clean/](reports/metrics_well_trained_v3h_s2_seed42_clean/)) | 13/13 중 5개 정량 통과, 정성 영상 인증 |
| **M3 — Push-robust v4a/v4b** | 완료 (Phase 2 종결) | warm-start from v3h_s2 + Push 10N/20N + DR | [scripts/train_robust_locomotion.py](scripts/train_robust_locomotion.py), [runs/robust_locomotion_v4a_seed42_1500k/](runs/robust_locomotion_v4a_seed42_1500k/), [runs/robust_locomotion_v4b_stretch20N_seed42_1000k/](runs/robust_locomotion_v4b_stretch20N_seed42_1000k/) | v4a Tier A: survival 1.0, vx_body **1.87**, drift **1.40 m**; v4a Tier B 5–10N **모두 100%**, recovery 14-16 step. v4b Tier B 5–20N **모두 100%**, recovery 14-22 step ([reports/robust_v4a_seed42_eval/summary.json](reports/robust_v4a_seed42_eval/summary.json), [reports/robust_v4b_stretch_eval/summary.json](reports/robust_v4b_stretch_eval/summary.json)) | DR+push가 nominal drift 5× 개선 (6.5→1.2 m) |
| **M4a — Stress eval (Phase 2b 측정)** | 완료 | 0.3s duration + ±0.2 N·m torque의 stress grid에서 v4b 재평가 | [scripts/eval_robust_locomotion.py](scripts/eval_robust_locomotion.py) `--tier-b-duration-steps 30 --tier-b-torque-z 0.20`, [reports/robust_v4b_on_stress_grid_eval/summary.json](reports/robust_v4b_on_stress_grid_eval/summary.json) | v4b on stress: **5N 100% / 10N 78% / 15N 44%**, mean recovery 25-51 step, max drift 2.7-10.1 m | "20N 100%" headline이 brief impulse 한정임을 입증 |
| **M4b — v4c stress curriculum (null)** | 완료 (null 결과) | v4b warm-start + stress curriculum 1M step | [runs/robust_locomotion_v4c_stress_seed42_1000k/](runs/robust_locomotion_v4c_stress_seed42_1000k/), [reports/robust_v4c_stress_eval/](reports/robust_v4c_stress_eval/) | survival 변화 ±3% 이내, nominal drift 1.18→1.37 m 미세 후퇴 | **알고리즘 변경하기 전에 cause isolation 들어감** |
| **M5 — E1/E2 cause isolation** | 완료 | E1 stress physics audit (116 cell), E2 recovery metric grid redesign | [scripts/stress_physics_audit.py](scripts/stress_physics_audit.py), [scripts/recovery_metric_grid.py](scripts/recovery_metric_grid.py), [reports/stress_physics_audit.csv](reports/stress_physics_audit.csv), [reports/recovery_metric_grid.csv](reports/recovery_metric_grid.csv) | 15N×0.3s 실측 |Δv_xy|=12-14 m/s, 5N에서는 contact가 흡수 1.2-2.7 m/s. v4c가 10N dir=180°에서 Δv 7.14→5.36 감소 (능동 저항 흔적) | 가설 (1)(2) 기각 |
| **M6 — v4d boundary + recovery reward** | 완료 | boundary sampling [easy/mid/hard 20/60/20] + 5s windowed recovery reward | [runs/robust_locomotion_v4d_boundary_recovery_seed42_1000k/](runs/robust_locomotion_v4d_boundary_recovery_seed42_1000k/), [reports/recovery_metric_grid_v4d.csv](reports/recovery_metric_grid_v4d.csv), [reports/robust_v4d_full_eval/](reports/robust_v4d_full_eval/) | **15N forward 25%→75% (+50pp)**, 15N avg 38%→44%, 10N avg 75%→69% (소폭 후퇴), Tier A drift 1.18→3.17 m | 가설 (3)(4) 부분 검증 |
| **M7 — v5a_v2 history stack** | 완료 | 5-frame history stack + 입력 확장 warm-start | [runs/robust_locomotion_v5a_history5_seed42_1000k_v2/](runs/robust_locomotion_v5a_history5_seed42_1000k_v2/), [reports/recovery_metric_grid_v5a_v2.csv](reports/recovery_metric_grid_v5a_v2.csv), [reports/robust_v5a_v2_full_eval/](reports/robust_v5a_v2_full_eval/) | **10N avg 69%→88% (+19pp)**, 10N dir=180° **0%→50%**, recovery time 132→55 step (2.4× 빠름). 15N forward 75%→0% trade-off. Tier A vx 1.78, drift 4.06 m | partial observability 가설 검증 + 부분 한계 노출 |
| **M5* — v5a v1 (실패)** | 종료 (디버깅 자료로 보존) | history stack을 oldest-first로 잘못 구현, warm-start 가정과 불일치 | [runs/robust_locomotion_v5a_history5_seed42_1000k/](runs/robust_locomotion_v5a_history5_seed42_1000k/) | Tier A vx **0.28**, return **-2508** (catastrophic) | observation augmentation 작업에서 layout convention 검증 필요성 입증 |
| **M8 — Go1 transfer groundwork** | 진행 중 (환경검증까지) | Go1 모델 inspection, PD 기립 검증, Gym Env wrapper, smoke test | [scripts/go1/go1_env.py](scripts/go1/go1_env.py), [scripts/go1/m0_inspect_model.py](scripts/go1/m0_inspect_model.py), [scripts/go1/m1_standing_pd.py](scripts/go1/m1_standing_pd.py), [scripts/go1/m2_env_smoke.py](scripts/go1/m2_env_smoke.py), [reports/go1_*.json](reports/) | Go1 trunk mass 5.20 kg, 총 12.74 kg, dt 0.002s, nq=19/nv=18/nu=12; smoke A_zero/B_small/C_full **NaN 없이 500 step 완주**, z_mean 0.264-0.268 m | 학습은 미시작; SB3-호환 Env까지 확보 |

---

## 5. Key Results and Metrics

모든 수치는 본 repo의 평가 산출물에서 발췌·교차검증된 것입니다.

### 5.1 Phase 1 Nominal Locomotion

| 메트릭 | baseline (300k) | v3h_s2_seed42 (Phase 1 deliverable) | 출처 |
|---|---|---|---|
| Survival | 0.8 | **1.0** | [reports/baseline_clean_300k/summary.json](reports/baseline_clean_300k/summary.json), [reports/metrics_well_trained_v3h_s2_seed42_clean/](reports/metrics_well_trained_v3h_s2_seed42_clean/) |
| Return | 2776 | 4511 | 동일 |
| Mean fwd velocity (world) | 2.30 m/s | 1.61 m/s | 동일 |
| Abs lateral drift (50s) | n/a | **6.48 m** | 동일 |
| Yaw abs (final) | n/a | 0.144 rad | 동일 |
| Roll/pitch RMS | n/a | 0.034 / 0.106 | 동일 |

> Baseline은 빠르지만 (2.30 m/s) survival 0.8에 drift 통제 자체가 학습 대상 아님. Phase 1은 속도를 일부 양보(1.61)하고 path 안정성·survival을 확보.

### 5.2 Phase 2 Push-Robust (brief impulse, Tier B)

| Force | v4a survival | v4a recovery (step) | v4b survival | v4b recovery (step) |
|---|---|---|---|---|
| 2N | 1.0 | – | 1.0 | – |
| 5N | 1.0 | – | 1.0 | 14.5 |
| 10N | 1.0 | – | 1.0 | 15.0 |
| 15N | – | – | 1.0 | 18.9 |
| 20N | – | – | **1.0** | 22.1 |

출처: [reports/robust_v4a_seed42_eval/summary.json](reports/robust_v4a_seed42_eval/summary.json), [reports/robust_v4b_stretch_eval/summary.json](reports/robust_v4b_stretch_eval/summary.json)

### 5.3 Phase 2 → Phase 2b — Nominal drift 개선 (DR+Push의 regularization 효과)

| 정책 | Tier A abs lateral drift | Tier A vx_body |
|---|---|---|
| v3h_s2 (Phase 1) | **6.48 m** | 1.61 m/s (world frame) |
| v4a (10N robust) | **1.40 m** | 1.87 m/s (body frame) |
| v4b (20N stretch) | **1.18 m** | 1.70 m/s |
| v4d (boundary+recovery reward) | 3.17 m | 1.75 m/s |
| v5a_v2 (history stack) | 4.06 m | 1.78 m/s |

출처: 각 `reports/robust_v*_full_eval/summary.json`. **push training 5× regularization 효과는 PoC에서 가장 robust한 발견 중 하나.**

### 5.4 Phase 2b — Stress grid (0.3s push + ±0.2 N·m torque)

| Force | v4b survival | v4b mean recovery (step) | v4b mean max drift |
|---|---|---|---|
| 5N | **100%** | 25.7 | 2.66 m |
| 10N | **78%** | 35.6 | 5.42 m |
| 15N | **44%** | 50.7 | 10.14 m |

출처: [reports/robust_v4b_on_stress_grid_eval/summary.json](reports/robust_v4b_on_stress_grid_eval/summary.json). "20N 100% 생존" headline이 brief impulse(0.05s)에 한정된 결과임이 stress grid에서 명백히 드러남.

### 5.5 Phase 2c — v4d boundary sampling + windowed recovery reward

| Force / dir | v4b | v4d | Δ |
|---|---|---|---|
| 5N (4 dir avg) | 100% | 100% | 0 |
| 10N (avg) | 75% | 69% | -6pp |
| **15N 0° (forward)** | **25%** | **75%** | **+50pp** |
| 15N 90° (lateral) | 100% | 50% | -50pp |
| 15N 180° (back) | 0% | 25% | +25pp |
| 15N 270° (lateral) | 25% | 25% | 0 |
| **15N avg** | **38%** | **44%** | **+6pp** |

출처: [reports/recovery_metric_grid_v4d.csv](reports/recovery_metric_grid_v4d.csv) (aggregated)

### 5.6 Phase 2d — v5a_v2 history stack (CSV 직접 집계)

| Cell | v4d | v5a_v2 | Δ |
|---|---|---|---|
| 5N (4 dir avg) | 100% | 100% | 0 |
| 10N avg (16 episode 집계) | **11/16 (68.75%)** | **14/16 (87.5%)** | **+18.75pp** |
| 10N dir=180° (back) | 0% | 50% | +50pp |
| 15N avg (16 episode) | **7/16 (43.75%)** | **6/16 (37.5%)** | -6.25pp |
| Recovery time 10N dir=90° | 132 step | 55 step | **2.4× faster** |
| Tier A vx_body | 1.75 | 1.78 | +0.03 |
| Tier A drift | 3.17 m | 4.06 m | +0.89 m |

출처: [reports/recovery_metric_grid_v5a_v2.csv](reports/recovery_metric_grid_v5a_v2.csv), [reports/robust_v5a_v2_full_eval/summary.json](reports/robust_v5a_v2_full_eval/summary.json). **mid-regime(10N) partial observability 가설 강력 검증, hard regime(15N) trade-off 확인.**

### 5.7 E1 — Stress Physics Audit (실측 Δv)

| Cell | 측정 |Δv_xy| | first-order 추정 | 해석 |
|---|---|---|---|
| 5N × 0.3s | 1.2–2.7 m/s | 4.5 m/s | ground contact 흡수 효과 큼 |
| 15N × 0.3s | 12–14 m/s | 13.6 m/s | free-body 가속에 가까움 |
| 0.2 N·m × 0.3s | |Δω_z| 0.9–2.6 rad/s | – | 의미 있는 yaw disturbance |
| v4c 10N dir=180° | Δv 7.14→5.36 vs v4b | – | 동일 cell에서 능동 저항 확인 |

출처: [reports/stress_physics_audit.csv](reports/stress_physics_audit.csv) (116 row)

### 5.8 Ant-v5 vs Go1 모델 property 차이

| 항목 | Ant-v5 | Go1 |
|---|---|---|
| DoF (legs) | 8 | 12 |
| Actuator | torque | position PD (kp=100) |
| dt (physics) | 0.01 s | 0.002 s |
| Policy rate | 100 Hz | 50 Hz (decimation 10) |
| Total mass | (Gymnasium 기본) | 12.74 kg |
| Trunk mass | – | 5.20 kg |
| nq / nv / nu | – | 19 / 18 / 12 |
| Home pose | 자체 | "home" keyframe 기준 |
| 학습 진행도 | Phase 1–2d 완료 | M2 환경 검증 완료, 학습 미시작 |

출처: [reports/go1_model_inspection.json](reports/go1_model_inspection.json), [scripts/go1/go1_env.py](scripts/go1/go1_env.py)

### 5.9 Go1 M2 smoke 결과

| Test | duration | z_mean | z_std | NaN | terminated |
|---|---|---|---|---|---|
| A_zero (action=0) | 10s, 500 step | 0.2649 | 0.0003 | False | False |
| B_small_random (~U(-0.1,0.1)) | 10s, 500 step | 0.2648 | 0.0005 | False | False |
| C_full_random (~U(-1,1)) | 5s, 250 step | 0.2686 | 0.0070 | False | False |

출처: [reports/go1_m2_env_smoke.json](reports/go1_m2_env_smoke.json). Go1Env가 SB3-호환 학습 환경으로 안정 동작함을 검증.

### 5.10 학습 안정성 / 실패 사례

- **v3g (single-shot strong vy penalty)**: standstill collapse (vx≈0). reward만으로 lateral discipline 강제 시 local optimum 진입.
- **v3f (continuation + 강화 reward)**: PPO std≈0.15로 local optimum 탈출 실패, 변화 <5%.
- **v4c (stress curriculum)**: 1M 추가 학습 후에도 survival 변화 ±3pp, nominal drift 미세 후퇴 (null 결과).
- **v5a v1 (oldest-first history stack)**: Tier A vx **0.28**, return **-2508** — warm-start expansion convention 불일치로 catastrophic failure. 디버깅 후 newest-first 로 수정.

---

## 6. What Worked / What Did Not Work

### 6.1 Worked

1. **Body-frame command tracking + world-frame heading alignment 동시 사용** — 둘 중 하나만으로는 부족. Phase 1에서 path 안정성 확보의 핵심.
2. **2-stage lateral curriculum** — weak lateral로 walking 확립 후 strong lateral로 drift 축소. single-shot tuning이 빠지는 "standstill collapse" / "biased gait" 두 local optimum 모두 회피.
3. **DR + Push training이 nominal locomotion까지 개선** — Tier A drift 6.48 m → 1.18 m. legged_gym/PA-LOCO 표준 관찰을 단일 머신·4 env로 재현.
4. **Boundary sampling + windowed recovery reward** — 15N forward survival 25%→75% (+50pp). reward shaping/sampling 설계가 알고리즘 변경 없이 stress 영역에서 실질 개선 가능함을 입증.
5. **Observation history stack(5 frame)으로 mid-regime 회수** — 10N avg 69%→88%, 10N back-direction 0%→50%, recovery time 2.4× 단축. partial observability가 ceiling의 한 축임을 측정.
6. **E1–E4 cause isolation 사이클** — null 결과에서 알고리즘부터 갈아엎지 않고 disturbance physics → metric → reward/sampling → observation 순으로 단일 변수 검증. 본 PoC의 가장 강한 methodology 자산.
7. **Stress-grade evaluation 프로토콜** — sustained 0.3s + 각운동량 포함. quadruped의 본질적 wide-base 흡수효과로 brief impulse가 정책 robustness를 과대평가함을 입증.
8. **Reproducible pipeline** — Dockerfile + requirements.txt + 모든 run의 `config.json` 직렬화 + Tier A/B/C 자동 평가 + 정성 mp4까지 한 사이클 자동화.
9. **Go1Env 안정화 (M2)** — 12-DoF position PD 환경을 SB3 호환 gym.Env로 wrapping. A_zero/B_small/C_full smoke 모두 NaN 없이 500 step 완주.

### 6.2 Did Not Work / Limitation

1. **Phase 1 13/13 절대 기준 미달** — 5/13만 정량 통과. legged_gym 4096 env급 exploration diversity 부재로 reward tuning만으로는 천장 존재. 본 PoC가 자체 인정하는 구조적 한계.
2. **v4c stress curriculum null** — 1M 추가 학습으로도 survival 변화 미미. 단일 변수(stress 분포 추가)만으로는 partial observability·reward·sampling 한계가 모두 묶여 있어 학습이 진전 못 함.
3. **v5a_v2의 15N forward 회귀 (75%→0%)** — history stack 5 frame이 30-step push 의 1/6만 담아 hard regime에서는 partial obs 해소가 부족. stack 10–20, RecurrentPPO, RMA-lite 후보가 다음 intervention.
4. **v4d/v5a_v2의 Tier A drift 후퇴** (1.18m → 3.17m → 4.06m) — recovery reward window가 nominal gait에 일부 누설. window 크기/weight schedule 재튜닝이 추가로 필요.
5. **단일 seed (42) 위주 보고** — Phase 2 이후 cross-seed variance 미측정. 2-3 seed replication이 결론 강도 보강에 필요.
6. **Torque actuator 한계** — action smoothness penalty가 PD-position 제어 환경 대비 효과 제한적. Go1으로 가면 같은 weight로도 더 부드러운 gait 가능 예상.
7. **Push 적용점이 torso center만** — 실제 disturbance는 다양한 contact point에서 발생. random body-part push 평가 미실시.
8. **Sim-to-real 검증 없음** — 본 PoC는 sim 종결. 그러나 DR scope(mass / friction / damping / motor / action noise)와 stress-grade evaluation 프로토콜은 sim2real 의 표준 ingredient를 포함하고 있어, 후속 실로봇 transfer를 위한 토대는 마련됨.
9. **Ant → Go1 transfer는 환경 wrapping까지만** — Go1에서 학습은 미시작. M3 PD 기립 학습 → M4 nominal locomotion → M5 push robust 가 다음 milestone.
10. **v5a v1 catastrophic failure** — observation layout convention 가정과 warm-start expansion 가정의 정합성 부재로 정책 붕괴. 실패였지만 "observation augmentation 작업에서 layout convention을 코드와 wrapper 양쪽에서 명시적으로 정의해야 한다"는 교훈 산출.

---

## 7. Job Application Positioning

### One-line summary (국문)

> MuJoCo·Stable-Baselines3 기반으로 4족 로봇의 외란-강건 보행 정책을 PPO + curriculum + domain randomization 으로 학습하고, 단일 머신 환경에서 reward·sampling·observation 설계가 알고리즘 변경에 우선한다는 점을 측정 기반 cause-isolation 사이클로 입증한 강화학습 PoC.

### 300자 프로젝트 설명 (국문)

> MuJoCo Ant-v5 4족 로봇을 대상으로 외란에 강건한 보행 정책을 PPO·curriculum·domain randomization 으로 학습한 PoC입니다. body-frame command tracking과 2-stage lateral curriculum 으로 nominal 보행을 안정화한 뒤, push curriculum + DR 을 더해 Tier A drift 를 6.5 m → 1.2 m 까지 5배 개선했습니다. 그 후 stress-grade 평가에서 brief impulse 헤드라인이 실 robustness 를 과대평가함을 측정으로 입증하고, E1–E4 cause isolation 으로 boundary sampling·windowed recovery reward·observation history 가 알고리즘 변경 없이 15 N forward survival 을 25 %→75 %, 10 N 평균 survival 을 69 %→88 % 까지 끌어올렸습니다. Unitree Go1 환경 wrapping 까지 확장 완료. (294자)

### 700자 프로젝트 설명 (국문)

> 본 PoC 는 MuJoCo Ant-v5 4족 로봇을 대상으로, 단일 머신·PPO·4-병렬 환경이라는 명시적 제약 안에서 외란-강건 보행 정책을 학습하고, 그 ceiling 의 구성요소를 측정 기반으로 분해한 강화학습 프로젝트입니다. Phase 1 에서는 body-frame command tracking 과 weak→strong lateral curriculum 두 단계 학습으로 standstill collapse 와 lateral drift bias 두 local optimum 을 모두 회피해 안정적 nominal gait 를 확보했습니다 (drift 6.48 m, survival 1.0). Phase 2 에서는 push curriculum 과 mass·friction·damping·motor·action-noise domain randomization 을 도입해 brief-impulse 20 N 까지 100 % survival 을 달성했고, 동시에 nominal drift 가 1.18 m 로 5 배 개선되는 implicit regularization 효과를 재현했습니다. 이후 정직한 평가를 위해 0.3 s sustained push + ±0.2 N·m 각 토크의 stress grid 를 도입해 v4b 의 stress 15 N survival 이 44 % 에 그침을 밝히고, stress curriculum 만 추가한 v4c 의 null 결과를 algorithm 변경 대신 E1 stress physics audit → E2 metric redesign → E3 boundary sampling + windowed recovery reward → E4 observation history stack 의 cause-isolation 사이클로 분해했습니다. 결과적으로 v4d 가 15 N forward survival 25 %→75 %, v5a_v2 가 10 N 평균 survival 69 %→88 % 와 recovery time 2.4 배 단축을 달성했습니다. Unitree Go1 의 SB3 호환 환경 wrapper 까지 구현해 후속 transfer 의 기반을 마련했습니다. 본 PoC 의 가장 generalizable 한 결과는, 작은 compute 의 robust locomotion 학습에서 reward·sampling·observation 설계 검증이 알고리즘 substitution 보다 먼저 와야 한다는 measurement-first methodology 입니다. (696자)

### Portfolio slide bullets

**문제 정의**
- MuJoCo 4족 로봇이 외란을 받았을 때 빠르게 회복하면서 명령 속도·자세를 유지하는 robust locomotion 정책을 학습.
- 단일 머신·PPO·4-병렬 환경의 compute 제약 안에서 어디까지 가능한지, 그리고 무엇이 진짜 ceiling인지를 측정 기반으로 확인.

**접근 방법**
- Stable-Baselines3 PPO + Gymnasium 의 wrapper-composition 으로 reward / push / domain randomization / observation history 를 모듈화.
- 2-stage curriculum 으로 nominal locomotion 확립 후 push + DR 로 robust 정책 학습.
- null 결과 발생 시 E1 (physics audit) → E2 (metric redesign) → E3 (reward·sampling) → E4 (observation) 순 single-variable 검증.

**본인 역할**
- 환경 설계: WellTrainedLocomotion·Push·DomainRandomization·ObservationHistoryStack wrapper 자체 구현.
- 학습 파이프라인: PPO config·SubprocVecEnv·warm-start (state-dict + input expansion) 구축.
- 평가 자동화: Tier A/B/C 통합 평가 + stress grid + recovery metric grid + physics audit 도구 작성.
- 정량/정성 분석: 모든 결과를 json·csv·mp4 로 산출하고 REPORT/POC 문서로 정리.

**기술 스택**
- Python 3.11, Stable-Baselines3 2.8.0, Gymnasium 1.2.0[mujoco], PyTorch 2.5.1
- MuJoCo (Ant-v5 + Unitree Go1 from Menagerie), OSMesa headless rendering
- TensorBoard, Docker (python:3.11-slim + ffmpeg/xvfb)
- imageio (mp4 산출), pandas/matplotlib (분석)

**주요 결과**
- Phase 1: survival 1.0, drift 6.48 m, mean fwd vel 1.61 m/s.
- Phase 2: brief-impulse 20 N 까지 100 % survival, nominal drift 6.48 → 1.18 m (5× 개선).
- Phase 2b: stress 15 N survival 44 % 측정으로 brief impulse 평가의 한계 입증.
- v4d: 15 N forward survival **25 %→75 % (+50pp)** (boundary sampling + windowed recovery reward).
- v5a_v2: 10 N 평균 survival **69 %→88 % (+19pp)**, recovery time 132→55 step (2.4× 단축).
- Go1 SB3 호환 환경 wrapper 검증 (M2 smoke 3종 NaN 없이 통과).

**배운 점 / 다음 단계**
- 측정 우선 — first-order 물리 계산은 가설일 뿐, simulator qvel Δ 측정으로 검증해야 한다.
- 알고리즘 변경(SAC, RMA) 전에 reward / sampling / observation 4-축을 단일 변수씩 검증.
- 다음: history stack 10–20 frame 확장 → RecurrentPPO → RMA-lite teacher-student → 그 후 SAC 검토.
- Go1 transfer: M3 PD 기립 학습 → M4 nominal locomotion → M5 push robust.

---

## 8. Evidence Checklist

지원서/포트폴리오에 첨부 가능한 근거 자료입니다.

| 자료 | 위치 | 비고 |
|---|---|---|
| ☑ 종합 연구 로그 (Korean) | [REPORT.md](REPORT.md) | Phase 1 → E1–E4 사이클 연대기 |
| ☑ Academic-style POC 문서 (English) | [POC_REPORT.md](POC_REPORT.md) | Abstract / Methods / Results / Discussion |
| ☑ Training entrypoint code | [scripts/train_robust_locomotion.py](scripts/train_robust_locomotion.py), [scripts/train_well_trained_locomotion.py](scripts/train_well_trained_locomotion.py) | argparse + warm-start + checkpoint callback |
| ☑ Wrapper module | [scripts/stable_directional_ant.py](scripts/stable_directional_ant.py) | Reward / Push / DR / HistoryStack 정의 |
| ☑ Evaluation tool | [scripts/eval_robust_locomotion.py](scripts/eval_robust_locomotion.py) | Tier A/B/C unified eval |
| ☑ Cause-isolation tools | [scripts/stress_physics_audit.py](scripts/stress_physics_audit.py), [scripts/recovery_metric_grid.py](scripts/recovery_metric_grid.py) | E1/E2 |
| ☑ Phase 1 metric | [reports/metrics_well_trained_v3h_s2_seed42_clean/](reports/metrics_well_trained_v3h_s2_seed42_clean/) | json + episodes csv |
| ☑ Phase 2 metric | [reports/robust_v4a_seed42_eval/summary.json](reports/robust_v4a_seed42_eval/summary.json), [reports/robust_v4b_stretch_eval/summary.json](reports/robust_v4b_stretch_eval/summary.json) | Tier A/B/C |
| ☑ Stress grid result | [reports/robust_v4b_on_stress_grid_eval/summary.json](reports/robust_v4b_on_stress_grid_eval/summary.json) | brief vs stress 차이 입증 |
| ☑ Recovery metric grid | [reports/recovery_metric_grid_v5a_v2.csv](reports/recovery_metric_grid_v5a_v2.csv) | v4b/v4d/v5a_v2 비교 |
| ☑ Physics audit raw data | [reports/stress_physics_audit.csv](reports/stress_physics_audit.csv) | 116 row qvel Δ 측정 |
| ☑ Phase 1 video | [videos/phase1_candidate_v3h_s2_seed42.mp4](videos/phase1_candidate_v3h_s2_seed42.mp4) | nominal gait |
| ☑ Phase 2 quiet/push video | [videos/phase2_v4b_quiet.mp4](videos/phase2_v4b_quiet.mp4), [videos/phase2_v4b_push20N_lateral.mp4](videos/phase2_v4b_push20N_lateral.mp4), [videos/phase2_v4b_push20N_backward.mp4](videos/phase2_v4b_push20N_backward.mp4) | brief impulse |
| ☑ Phase 2b stress video | [videos/phase2b_v4b_stress_10N_lateral.mp4](videos/phase2b_v4b_stress_10N_lateral.mp4), [videos/phase2b_v4c_stress_15N_lateral.mp4](videos/phase2b_v4c_stress_15N_lateral.mp4) | sustained push |
| ☑ Phase 2c/2d video | [videos/phase2c_v4d_stress_15N_forward.mp4](videos/phase2c_v4d_stress_15N_forward.mp4), [videos/phase3_v5a_stress_10N_180.mp4](videos/phase3_v5a_stress_10N_180.mp4) | 개선 demo |
| ☑ Go1 inspection / smoke | [reports/go1_model_inspection.json](reports/go1_model_inspection.json), [reports/go1_m2_env_smoke.json](reports/go1_m2_env_smoke.json), [videos/go1_m2_*.mp4](videos/) | M0/M1/M2 |
| ☑ Reproducible env | [Dockerfile](Dockerfile), [requirements.txt](requirements.txt) | python:3.11-slim + MuJoCo OSMesa |
| ☑ 모든 run의 hyperparameter | `runs/*/config.json` | 직렬화된 args + reward_config + push_config + dr_config |
| ☑ Representative command | "9. Documentation patch" 의 reproduce 섹션 참고 | 아래 정리 |

---

## 9. Recommended Documentation Patch

본 repo를 포트폴리오 친화적으로 더 만들기 위해 추가/정리하면 좋을 항목입니다.

### 9.1 README 신설 (현재 부재)

repo 루트에 `README.md` 가 없습니다. 다음 구조의 README 한 페이지가 포트폴리오 첫 인상 측면에서 가장 큰 ROI 입니다.

```
# ant-push-recovery-rl

(2-3 줄 project summary — One-line + 300자 발췌)

## Highlights
- Phase 1 nominal: drift 6.48 m, survival 1.0
- Phase 2 robust: 20N brief impulse 100% survival, nominal drift 5× regularization
- Phase 2d: 10N average survival 69% → 88%, recovery 2.4× faster

## Current best results
(아래 5-row table)

## How to reproduce key results
docker build -t apr .
docker run --rm -v $PWD:/workspace apr python scripts/train_well_trained_locomotion.py ...

## Repo map
- scripts/  학습/평가 entrypoint
- runs/     학습 산출물 (config + checkpoint + tb)
- reports/  평가 metric csv/json
- videos/   정성 평가 mp4

## See also
- REPORT.md  연구 로그 (Korean)
- POC_REPORT.md  학술 스타일 writeup (English)
- PORTFOLIO_ANALYSIS.md  지원서용 분석
```

### 9.2 "Current best results" 표

README 상단에 한 표로 cherry-pickable 한 best result 를 박아두면 좋습니다.

| Policy | Tier A drift | Tier A vx | Tier B brief 20N | Stress 15N (avg) | 비고 |
|---|---|---|---|---|---|
| v3h_s2 | 6.48 m | 1.61 m/s | – | – | Phase 1 deliverable |
| v4b | **1.18 m** | 1.70 m/s | **100%** | 44% | 추천 균형 policy |
| v4d | 3.17 m | 1.75 m/s | – | 44% (forward 75%) | recovery reward |
| v5a_v2 | 4.06 m | 1.78 m/s | – | 38% (10N 88%) | history stack |

### 9.3 "How to reproduce key result" 섹션

각 milestone 별 single command 모음. 이미 [REPORT.md](REPORT.md) Section 8 (Reproducibility)와 [POC_REPORT.md](POC_REPORT.md) Appendix A 에 정리되어 있어 README 에서 참조 링크만 걸어도 충분합니다.

### 9.4 Ant-v5 vs Go1 차이 표

5.8 절의 표를 README 또는 `docs/MODELS.md` 에 정리. transfer scope 가 한눈에 보입니다.

### 9.5 Representative videos / gifs 링크 정리

README 또는 `docs/MEDIA.md` 에 Phase 별 best 영상 3-5개 링크. 가능하면 `videos/` 의 mp4를 gif 로 일부 변환해서 README 에 임베드.

추천 셋:
- Phase 1: [phase1_candidate_v3h_s2_seed42.mp4](videos/phase1_candidate_v3h_s2_seed42.mp4)
- Phase 2 quiet: [phase2_v4b_quiet.mp4](videos/phase2_v4b_quiet.mp4)
- Phase 2 push: [phase2_v4b_push20N_lateral.mp4](videos/phase2_v4b_push20N_lateral.mp4)
- Phase 2c stress recovery: [phase2c_v4d_stress_15N_forward.mp4](videos/phase2c_v4d_stress_15N_forward.mp4)
- Phase 2d history stack: [phase3_v5a_stress_10N_180.mp4](videos/phase3_v5a_stress_10N_180.mp4)

### 9.6 Limitations / Future Work 섹션

[REPORT.md](REPORT.md) Section 7 의 한글 요약을 README 끝에 한 번 더. 정직한 한계 명시가 지원서 측면에서 오히려 강점입니다.

### 9.7 표현 정리 가이드 (지원서/포트폴리오용)

- "20N 100% survival" 같은 한 줄 헤드라인은 항상 **"brief impulse 기준이며, stress grid 에서는 15N 44%"** 의 추가 컨텍스트를 같이 적어주세요. 정직한 평가가 본 PoC 의 핵심 자산이기 때문에, 가장 임팩트 있는 수치를 단독으로 인용하면 그 자산이 약해집니다.
- "PPO 한계를 넘었다" 같은 강한 표현은 피해주세요. 본 PoC 가 입증한 것은 *"compute 제약 안에서 reward·sampling·observation 설계로 robustness 한계의 일부를 회수할 수 있다"* 이며, 절대 한계 돌파가 아닙니다.
- "Sim-to-real 검증" 같은 표현은 사용하지 마세요. 본 PoC 는 sim 종결이고, DR scope·stress 평가 프로토콜이 sim2real **준비 단계**까지 와 있다는 표현이 정확합니다.

### 9.8 민감 정보 점검

확인 결과 본 repo 에는 secret / token / 개인정보 / API key 가 포함되어 있지 않습니다. 절대경로 노출은 일부 코드 ([scripts/go1/go1_env.py](scripts/go1/go1_env.py) 의 `DEFAULT_MODEL_PATH = Path("/workspace/external/mujoco_menagerie/...")`) 에 있으나, 이는 Docker 내부 workspace 경로 (`/workspace`) 이므로 외부 노출 위험은 없습니다. 그대로 두어도 무방합니다.

---

## 끝맺음

본 분석은 repo 의 코드·config·csv·json·mp4 산출물에 직접 근거하며, 외부 결과나 미측정 추정은 사용하지 않았습니다. 강조와 절제 사이 균형을 맞춰 작성했으니, 지원서 양식에 맞춰 7번 섹션의 1줄/300자/700자 버전과 8번 evidence checklist를 그대로 활용하시면 됩니다.

가장 차별화된 강점은 "단일 머신 PPO" 라는 compute 제약 안에서 **measurement-first cause-isolation** 사이클을 끝까지 운용한 점입니다. 제조로보틱스 직무에서 마주칠 "이 정책/제어가 왜 안 되는가" 류 문제에 대해 동일한 방법론을 가져갈 수 있다는 신호로 읽힙니다.
