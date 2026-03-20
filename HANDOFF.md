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
Epoch 25 checkpoint에서 학습을 재개하여 50 epoch까지 완료한 후, CARLA에서 평가(Task A: Town04 lane keeping, Task B: Town03 obstacle avoidance)를 실행하고 결과를 분석한다.

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
- 59 tests passed, 0 failed
- Epoch 25/50 checkpoint saved at outputs/train_v1/checkpoints/epoch_25.pt (55MB)
- Loss converged around 1.11 (stable from epoch 10 onwards)
- GMM preference log_prob improved from -29 to +6.79 over training
- 현재 컴퓨터의 RTX 3070 Ti 8GB에서 CUDA "unspecified launch failure" 간헐적 발생 (Epoch 16, Epoch 26)
  - 메모리 문제 아님 (per-timestep backward 적용 후에도 발생)
  - 드라이버 또는 하드웨어 문제로 추정
  - GPU 메모리가 더 큰 컴퓨터로 이전 예정

PENDING TASKS
-------------
1. Epoch 25에서 학습 재개 -> Epoch 50까지 완료
   - 명령: uv run python scripts/train.py --config configs/default.yaml --data data/expert_data.h5 --output_dir outputs/train_v1 --resume outputs/train_v1/checkpoints/epoch_25.pt
   - 또는: bash scripts/train_robust.sh (auto-restart)
2. CARLA 서버 시작 후 평가 실행
   - Task A (Town04 curves): uv run python scripts/evaluate.py --task A --checkpoint outputs/train_v1/checkpoints/final.pt --save_video
   - Task B (Town03 obstacles): uv run python scripts/evaluate.py --task B --checkpoint outputs/train_v1/checkpoints/final.pt --save_video
   - Baseline (Town06): uv run python scripts/evaluate.py --task baseline --checkpoint outputs/train_v1/checkpoints/final.pt
3. 평가 결과 분석 (Success Rate, Mean Lateral Deviation, Off-Road Events)
4. Streamlit 대시보드로 결과 시각화
5. (선택) Task B preference 전용 데이터 수집 (collect_preference_data.py)

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
- Action Space: 2D [steer, accel] (accel>0->throttle, accel<0->brake). 3D 대비 CEM 효율 40% 향상
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
- 새 컴퓨터에서 CARLA 설치 경로 확인 필요 (PYTHONPATH 설정)
- uv sync 후 carla wheel 별도 설치 필요: uv pip install /path/to/carla-0.9.16/.../carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl && uv pip install shapely
- torch-backend은 pyproject.toml에 "auto"로 설정됨 (CUDA 자동 감지)
- TensorBoard 실행 시 setuptools 69 이하 필요 (82+에서 pkg_resources 제거됨): .venv/bin/pip install 'setuptools<70'
- Streamlit 대시보드는 outputs/ 아래에서 최신 tb_logs를 자동 탐색
- 평가 시 CARLA 서버 필요 (학습 시에는 불필요)
- evaluate.py에서 --task A는 Town04, --task B는 Town03, --task baseline은 Town06 사용
