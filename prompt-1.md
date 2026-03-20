**[System Role]**
당신은 자율주행 및 체화 인공지능(Embodied AI) 분야의 고급 딥러닝 엔지니어입니다. PyTorch와 CARLA 시뮬레이터를 활용하여 Karl Friston의 능동추론(Active Inference) 이론에 기반한 이동형 로봇(자율주행 차량)의 인지 제어 시스템을 구현해야 합니다.

**[Task Description]**
CARLA 시뮬레이터 환경에서 연속 동작(Continuous Action: Steer, Throttle, Brake)을 제어하는 '확률적 심층 능동추론(Deep Active Inference) 에이전트'를 구현해 주세요. 단순한 RL(강화학습)이 아닌, 환경의 생성 모델(Generative Model)을 학습하고 기대 자유 에너지(Expected Free Energy, EFE)를 최소화하는 방향으로 행동을 추론해야 합니다.

**1. 환경 및 관측 데이터 (Sensorimotor Loop)**

* **학습 환경:** CARLA Town06 (샘플 개수는 generalization을 위해 충분히 큰 값을 사용해주세요). The car must be driven in a reasonable speed. The velocity unit must be consistant through out the pipeline. Clearly declare the unit if it is m/s or km/h.
* **테스트 환경:** Town04 (곡선 주행) 및 Town03 (장애물 회피)
* **Observation ($o_t$):** 전방 카메라 RGB 이미지 (차원 축소 적용) 및 차량의 현재 속도/조향각.
* **Action ($a_t$):** Steering (-1~1), Throttle (0~1), Brake (0~1)의 연속 공간.

**2. 확률적 세계 모델 (Generative Model) 아키텍처**
전방 모델(Forward Model)은 VAE(Variational Autoencoder) 또는 RSSM(Recurrent State Space Model) 구조를 띠어야 합니다.
세계 모델의 단순화를 위해 필요하다면 카메라 입력에서 주행에 중요하다고 필요한 영역만 사용하는 것도 허락됩니다.

* **Representation Model:** $q(s_t | o_t, s_{t-1}, a_{t-1})$ -> 잠재 상태의 가우시안 분포(Mean, Variance) 출력.
* **Transition (Dynamics) Model:** $p(s_t | s_{t-1}, a_{t-1})$ -> 다음 상태 예측.
* **Observation Model (Decoder):** $p(o_t | s_t)$ -> 관측값 복원.
* **업데이트 로직 (VFE 최소화):** 모델은 다음의 변분 자유 에너지(Variational Free Energy)를 최소화하도록 학습되어야 합니다.
$VFE = D_{KL}[q(s_t | o_t) || p(s_t | s_{t-1}, a_{t-1})] - \mathbb{E}_{q}[\ln p(o_t | s_t)]$
* **전방 모델 데이터 특성:** 전방 모델의 효과적 학습을 위해서는 최대한 다양한 형태의 행동에 따른 환경변화를 학습 데이터에 수집할 수 있는 전략이 필요. 

**3. 행동 선택 (Action Inference) 및 기대 자유 에너지 (EFE)**
에이전트는 기대 자유 에너지(EFE, $G$)를 최소화하는 행동 시퀀스를 선택해야 합니다. 이를 위해 목표 분포인 선호도(Preference) $p(s_{pref})$를 다음과 같이 정의하고 계산하는 로직을 명시적으로 구현해 주세요.

* **선호도 $p(s_{pref})$의 정의 및 데이터 기반 학습 (Preference Learning):**
* 사용자가 단일 이미지를 하드코딩하는 방식 대신, **학습 데이터셋(Expert Demonstrations)의 통계적 분포를 활용하여 $p(s_{pref})$를 동적으로 구축**하는 로직을 구현해 주세요.
* **구현 지침:**
1. 세계 모델(Generative Model) 학습 루프 내에, 특정 주기에 맞춰 `update_preference_distribution()` 메서드를 호출하도록 합니다.
2. 이 메서드는 사전에 필터링된 '성공적인 Task A(Lane Keeping)' 및 'Task B(Lane Change)' 관측 샘플 배치를 현재 학습 중인 **Encoder**에 통과시킵니다.
3. 산출된 잠재 벡터들을 바탕으로 단일 가우시안이 아닌 **가우시안 혼합 모델(Gaussian Mixture Model, GMM) 기반의 $p(s_{pref})$**를 피팅(Fitting)하여 저장합니다. (PyTorch의 경우 `torch.distributions.MixtureSameFamily` 등을 활용).


* **Action Selection 시의 반영:** EFE($G$)를 계산할 때, Instrumental value인 $D_{KL}[q(s_{\tau}|a) || p(s_{pref})]$ 계산 시 이 GMM 분포와의 거리를 계산하도록 하여, 에이전트가 상황에 따라 차선 유지 모드 또는 회피 모드의 잠재 공간(Attractor) 중 더 가까운 쪽으로 자연스럽게 끌려가도록(Attracted) 설계해 주세요.


* **EFE 수식 ($G$):** $G(\pi) \approx \underbrace{D_{KL}[q(s_{\tau}|a) || p(s_{pref})]}_{\text{Instrumental (목표 도달)}} - \underbrace{\mathbb{E}_{q(s_{\tau}|a)}[\mathcal{H}(p(o_{\tau}|s_{\tau}))]}_{\text{Epistemic (정보 이득 / 불확실성 해소)}}$
* **탐색 최적화(Action Selection) 요구사항:**
* Epistemic term을 구하기 위해 디코더(Decoder)의 예측 분산(Variance)을 활용하거나, Ensemble Dynamics Model의 분산을 정보 이득으로 근사(Approximation)하는 방식을 적용해 주세요.
* 연속된 Action 공간(-1~1, 0~1)에서 최적의 조향과 가속을 찾기 위해, 생성 모델을 이용해 미래를 상상(Imagination)하고 EFE를 최소화하는 **Cross-Entropy Method (CEM)** 기반의 MPC(Model Predictive Control) 롤아웃 로직을 작성해 주세요.

**4. 테스트 태스크 및 평가 지표**

* The car must move to evaluate the model in a task. The model must not choose not to move to get a better metrics (success rate, less deviation from the center). 이를 위해 충분히 긴 구간과 다양한 곡률을 가진 구간을 골라서, 시작점과 목표지점을 정해 놓아야 합니다. 성공률은 목표지점 도달여부입니다. 어떤 타운의 어느 구간을 이용했는지 명확히 문서화해 주세요.
* **Task B:** 도로에 정차된 차량을 피하기 위한 Lane Change (Object Avoidance - Epistemic 탐험이 장애물 회피에 기여하는지 확인).
* **Task A:** 정해진 경로의 Lane Keeping.   
* **출력 지표:** 매 롤아웃/테스트 세션 종료 시 Success Rate (SR), Mean Lateral Deviation (MLD), Off-Road Events 횟수를 로깅해 주세요.
* 결과의 질적 검증을 위해 검증하는 과정을 비디오로 저장해주세요.


**[Action Items]**

* **[1단계 작업]**
위 명세서를 바탕으로, 전체 프로젝트의 뼈대가 될 1) PyTorch 기반의 Generative Model (Representation, Transition, Observation) 클래스와 2) EFE(Instrumental + Epistemic)를 계산하는 Loss 함수 및 CEM Action Planner 부분의 코드를 우선적으로 작성해 주세요. CARLA 연동 코드는 추후 작성할 예정이니 알고리즘의 핵심 수학적 로직 구현에 집중해 주세요.

* **[2단계 작업]** 1) uv나 poetry 같은 패키지 관리 도구를 사용해 프로젝트를 구성해 주세요. 2) 패키지 구조는 data, scripts, config, notebook 등으로 체계적으로 구성해 주세요.

* **[3단계 작업]** 1) CARLA 연동 코도를 작성해주세요.
* 데이터 수집 -> 훈련 -> 검증 작업이 이루어지도록 파이프라인을 구성해 주세요.

* **[4단계 작업]** 1) 이전 단계의 작업들이 모두 잘 이루어졌는지 확인하고 나면, 2) 이 작업의 진행 상황을 웹기반의 사용자 인터페이스를 통해 실시간으로 확인할 수 있도록 해주세요.

