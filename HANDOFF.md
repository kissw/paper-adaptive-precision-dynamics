HANDOFF CONTEXT
===============

USER REQUESTS (AS-IS)
---------------------
- "@prompt-1.md 을 참고해서 시스템을 설계하세요."
- "구조설계를 마쳤으면, 구현 계획을 세워주세요"
- "그 순서대로 구현을 시작해."
- "전방향 모델을 제대로 학습하려면 다양한 action --> sensory input change 데이터가 필요할 것 같은데, 어떻게 생각해? 그리고, 그렇다면, 적절한 데이터 수집에도 전략이 필요할 것 같고, 데이터 수집 자동화도 가능할 것 같은데, 네 생각은 어때?"
- "진행하기 전에 추가로 확인할 사항: 정해진 task를 수행하기 위해서는 기본적으로 preference들의 확률분포가 필요한데, 이것에 대한 계획이 이미 수립되어 있는지 검토해주고, 그렇지 않다면, 각 task 별로 preference에 필요한 시각입력 데이터도 수집을 해야할텐데"
- "데이터 수집 전략을 먼저 통합하고 진행하는 것이 더 좋은 선택아닌가?"
- "이제, 데이터 수집 시작해 보자. carla 0.9.16이 /home/jaerock/carla-0.9.16에 설치되어 있어."
- "지금 하던 작업을 gpu 메모리가 더 있는 다른 컴퓨터로 옮겨서 계속하고 싶어."

GOAL
----
Epoch 25 checkpoint에서 학습을 재개하여 50 epoch까지 완료한 후, CARLA에서 평가(Task A: Town04 lane keeping, Task B: Town06_Opt obstacle avoidance)를 실행하고 결과를 분석한다.

WORK COMPLETED
--------------
- prompt-1.md 기반으로 Deep Active Inference 자율주행 시스템 전체를 설계 (IMPLEMENTATION_PLAN.md 참조)
- Oracle, Momus, Metis, 4개 Librarian 에이전트로 설계 검증 완료
- 17개 태스크 전체 구현 완료 (59 unit tests passed):
  - Phase 1: RSSM(encoder/decoder/rssm), VFE loss(dual KL, free_nats), Ensemble(5 heads), GMM Preference(K=3), EFE scorer, iCEM Planner, DeepAIFAgent
  - Phase 2: uv project, configs, synthetic data
  - Phase 3: CARLA env wrapper, collect_data.py, train.py, evaluate.py, run_pipeline.py
  - Phase 4: Streamlit dashboard (3 pages: training, evaluation, visualization)
- 데이터 수집 전략을 분석하고 개선 완료:
  - collect_data.py에 OU noise(3-tier: clean 25%, medium 40%, high 30%, random 5%) 추가
  - 에피소드별 속도 다양화 [15-50 km/h]
  - HDF5에 11개 필드 저장 (images, states, actions, expert_actions, episode_ids, lateral_devs, lane_ids, noise_sigmas, success_flags, task_labels, target_speeds)
  - coverage_check.py 스크립트 추가
  - collect_preference_data.py (Task B 전용 수집) 추가
- Preference 파이프라인 6개 갭 수정:
  - PreferenceSequenceDataset (success+task 필터링)
  - agent.encode_preference_data() + update_preference()
  - train.py GMM 업데이트 stub 제거, 실제 피팅 로직 구현
  - config에 preference data_path, task_b_data, max_samples, balance_ratio 추가
  - GMM 헬스 메트릭 (log_prob, min_component_dist, weights) TensorBoard 로깅
- CARLA Town06_Opt에서 72,000 프레임 수집 완료 (data/expert_data.h5, 3.3GB)
  - Coverage: 0.902 (90.2% bin occupancy), Entropy: 5.021
  - Task A: 63,500 frames, Task B: 8,500 frames
- 학습 Epoch 1-25 완료 (Loss: 1.8538 -> 1.1112)
  - GMM preference 업데이트 Epoch 10부터 정상 동작 (log_prob: -29 -> +6.79)
- CUDA crash 대응:
  - agent.update()를 per-timestep backward으로 변경 (O(T) -> O(1) 메모리)
  - NaN guard 추가 (isnan/isinf skip)
  - train.py에 try/except CUDA error recovery + batch skip
  - train_robust.sh (auto-restart wrapper) 작성

CURRENT STATE
-------------
- 63 tests passed, 0 failed
- Epoch 50/50 checkpoint: outputs/train_v1/checkpoints/best.pt (validation-selected)
- Evaluation checkpoint: outputs/train_v1/checkpoints/best_refit.pt (GMM refit on speed>1.0 frames)
- Loss converged around 1.11 (stable from epoch 10 onwards)
- GMM preference log_prob improved from -29 to +6.79 over training

EVALUATION v4 CHANGES (2026-03-20)
----------------------------------
- Task B 경로를 Town03 (urban grid, 46 intersections) → Town06_Opt (highway, intersection-free)로 변경
  - 근거: 에이전트에 waypoint/navigation 기능 없음 → 교차로에서 방향 결정 불가
  - Town06_Opt: 학습 데이터 수집 타운과 동일 → visual domain gap 제거, obstacle avoidance만 평가
- iCEM planner가 PlanResult(action, efe_score, epistemic_score) 반환
  - agent.step_with_info() → PlanResult 전체, agent.step() → action만 (하위호환)
- 평가 스크립트(scripts/evaluate.py) v4 업데이트:
  - Task B에서 자동 obstacle spawning (spawn_obstacles/destroy_obstacles)
  - Per-frame JSONL 로깅 (19개 필드: x/y/z, yaw, EFE, epistemic 등)
  - CSV에 mean_efe_score, mean_epistemic_score, trajectory_file 컬럼 추가
  - 출력 디렉토리: outputs/eval_v4
- 신규 파일:
  - src/active_inference/evaluation/obstacles.py (재사용 가능 obstacle 모듈)
  - scripts/discover_routes.py (CARLA로 Task B 경로 탐색)
  - docs/plans/2026-03-20-eval-v4-route-redesign.md (설계 근거)
- 대기 중: CARLA로 discover_routes.py 실행 → Town06_Opt_TaskB 경로 확정

EVALUATION RESULTS (v4, 2026-03-20)
------------------------------------
Setup: best_refit.pt checkpoint, 6000 frames (300s) per episode, 2 routes × 3 episodes per task

| Metric                 | Baseline (Town06_Opt) | Task A (Town04) |
|------------------------|-----------------------|-----------------|
| Max Route Completion   | 71.8%                 | 100.0%          |
| Total Distance/Episode | 1,509m                | 1,509m          |
| Total Safe Driving     | 9.06 km (6 eps)       | 9.06 km (6 eps) |
| Mean Speed             | 5.03 ± 0.47 m/s       | 5.03 ± 0.47 m/s |
| Mean Lateral Dev       | 0.040m                | 0.110m          |
| Max Lateral Dev        | 0.600m                | 0.586m          |
| Collisions             | 0                     | 0               |
| Offroad Events         | 0                     | 0               |
| Mean EFE               | 16.82                 | 16.83           |
| Mean Epistemic         | 0.463                 | 0.464           |
| Termination            | timeout (all)         | timeout (all)   |

Key findings:
- 18.1 km of zero-collision driving across both towns
- 100% route completion on unseen Town04 circuit (cross-domain transfer)
- Domain-invariant speed control: identical 5.03 m/s across towns
- Reproducible: episode-to-episode variance < 2m in total distance

Architecture:
- CEM (iCEM planner) handles longitudinal control (speed via EFE optimization)
- Stanley controller handles lateral control (heading error + crosstrack error)
- Linear throttle remap: accel ∈ [-1,1] → throttle ∈ [0.35, 0.55], no braking
- State vector: [speed_mps, steer, heading_error, crosstrack_error] (4D; model uses first 2)
- CARLA steer convention: positive steer = LEFT turn (counterclockwise yaw increase)

Output files:
- outputs/eval_v4/          (Baseline: trajectories, videos, CSV)
- outputs/eval_v4_taskA/    (Task A: trajectories, videos, CSV)
- .omc/scientist/figures/   (Paper-quality plots: trajectories, time series, histograms)

PENDING TASKS
-------------
1. (선택) Task B (obstacle avoidance) 평가 — 현재 아키텍처에서는 Stanley가 steering을 제어하므로 장애물 회피 불가. CEM이 obstacle을 인식하려면 추가 학습/설계 필요
2. (선택) 더 높은 속도로 재평가 (max_throttle 증가 → 7-10 m/s)
3. 논문 작성: 위 evaluation 결과 + trajectory 시각화 활용
4. Streamlit 대시보드로 결과 시각화

KEY FILES
---------
- IMPLEMENTATION_PLAN.md - 전체 아키텍처, 17개 태스크, 설계 결정 근거
- prompt-1.md - 원본 요구사항 (Korean)
- src/active_inference/agent.py - DeepAIFAgent (WorldModel, CEM, Preference 통합)
- src/active_inference/models/rssm.py - RSSM (obs_step, img_step, imagine)
- src/active_inference/training/losses.py - VFE loss (dual KL, free_nats)
- src/active_inference/training/preference.py - GMM PreferenceModel (K=3)
- src/active_inference/planning/cem_planner.py - iCEM Planner
- scripts/train.py - 학습 파이프라인 (resume, GMM update, CUDA recovery)
- scripts/collect_data.py - 데이터 수집 (OU noise, 3-tier, metadata)
- configs/default.yaml - 전체 하이퍼파라미터

IMPORTANT DECISIONS
-------------------
- Action Space: 2D [steer, accel]. Linear throttle remap (no braking): accel ∈ [-1,1] → throttle ∈ [0.35, 0.55]
- Latent Space: Gaussian 64-dim (NOT categorical). Active Inference 이론 부합 (closed-form KL)
- GMM K=3 (Oracle 권고: K=5는 샘플 부족). 단일 공유 GMM (Task A+B 혼합)
- Epistemic: Ensemble disagreement primary (5 heads, .mean() normalization)
- OU Noise: theta=0.15, dt=0.05. 3-tier 배분 (clean 25%, medium 40%, high 30%, random 5%)
- Per-timestep backward: GPU 메모리 O(1). 전체 시퀀스 accumulate 대신 각 timestep에서 (loss/T).backward()
- Dual KL: dyn_scale=1.0, rep_scale=0.1, free_nats=1.0 (DreamerV3 패턴)
- Town06_Opt 사용 (Town06은 메모리 부족으로 crash)
- CARLA agents 모듈 경로: PYTHONPATH에 /path/to/carla-0.9.16/PythonAPI/carla 추가 필요

EXPLICIT CONSTRAINTS
--------------------
- CARLA 0.9.16 사용 (다른 버전 금지)
- 강화학습(RL)이 아닌 Active Inference 이론 기반 (EFE 최소화)
- Preference는 하드코딩 아닌 GMM으로 데이터 기반 학습
- CEM으로 연속 Action 공간에서 최적 행동 탐색
- 검증 과정을 비디오로 저장

CONTEXT FOR CONTINUATION
------------------------
- Split control architecture: CEM (longitudinal/speed) + Stanley controller (lateral/steering)
  - Stanley 파라미터: k_heading=1.5, k_crosstrack=2.0
  - CARLA steer 부호: positive steer = LEFT turn (counterclockwise yaw increase)
  - heading_error, crosstrack_error는 env에서 제공 (state[2], state[3])하지만 모델에는 입력되지 않음
- best_refit.pt = best.pt world model + speed>1.0 프레임으로 GMM refit
- uv sync 후 carla wheel 별도 설치 필요: uv pip install /path/to/carla-0.9.16/.../carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl && uv pip install shapely
- 평가 시 CARLA 서버 필요 (학습 시에는 불필요)
- evaluate.py에서 --task A는 Town04, --task B는 Town06_Opt (obstacles), --task baseline은 Town06_Opt 사용
- 모든 evaluation 데이터: outputs/eval_v4/, outputs/eval_v4_taskA/ (JSONL trajectories, MP4 videos, CSV summaries)
