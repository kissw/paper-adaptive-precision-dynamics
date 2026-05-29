# 세션 요약

---

## 2026-05-29

### 작업 범위
Position-normalised Shared Token GMM 구현 완료 (이전 세션에서 시작한 작업 이어받음).

### 완료된 작업

1. **`src/active_inference/training/token_preference.py` 업데이트**
   - `compute_position_stats(clean_tokens, eps)` 추가: clean 토큰에서 per-position mean/std 계산, shape `(N, D)`
   - `apply_position_normalization(tokens, mu_pos, std_pos)` 추가: 각 위치별 정규화 적용
   - 모듈 docstring에 `pos_norm` 모드 설명 추가

2. **`scripts/fit_token_contrastive_preference.py` 재작성**
   - `--mode shared` (기존 동작, position correction 없음)
   - `--mode pos_norm` (default, 제안 방법): clean 토큰 통계로 위치 편향 제거
   - `--pos_norm_eps`, `--topk_default` 인자 추가
   - pos_norm 체크포인트에 `position_stats` (`mu_pos`, `std_pos`, `eps`) 저장

3. **`tests/test_token_preference.py` 업데이트**
   - 8개 테스트 추가 (기존 13개 → 총 21개):
     - `compute_position_stats` shape/positivity/clamping 3개
     - `apply_position_normalization` shape/zero-mean-unit-std 2개
     - pos_norm GMM 위치 편향 데이터에서 gap > 0 확인 1개
     - pos_norm/shared 체크포인트 구조 검증 2개
   - 전체 21/21 통과

4. **드라이런 검증**
   - shared 모드: 정상 실행, checkpoint key `token_contrastive_preference` 생성 확인
   - pos_norm 모드: 정상 실행, `position_stats` (mu_pos `(64,320)`, std_pos `(64,320)`) 저장 확인

5. **커밋**: `6e488e7` — `feature/token-vit-world-model` 브랜치

### 주요 설계 결정
- position stats는 clean 토큰에서만 계산 (obstacle 데이터로부터 독립)
- shared 모드 체크포인트에는 `position_stats` 키 없음 (not None, absent)
- pos_norm이 default mode (연구 제안 방법)

### 다음 단계 (미완)
- Phase B (transition latent alignment) 훈련 준비
- TokenViT EFE 통합 검증 (visual surprise path `decode_from_feat`)

---

## 2026-05-29 (2차)

### 작업 범위
Token VampPrior-like position-conditional preference 구현.

기존 GMM과의 차이:
- 기존 token GMM: `concat(deter, stoch)` 320D feature → GMM density
- 이번 방식: `token_mean`, `token_std` (64D posterior distribution) → per-position posterior mixture prior

### 완료된 작업

1. **`src/active_inference/training/token_vampprior_preference.py` 신규 생성**
   - `extract_token_posterior_stats(state)`: `token_mean/token_std` 추출, 2D state reject
   - `fit_position_posterior_mixture(mu, std, K, seed, min_std)`: 위치별 KMeans → K-component mixture, 빈 cluster fallback 처리
   - `log_prob_posterior_under_mixture(q_mean, q_std, pm, pls, pl)`: q-integrated density — `log N(q_mean; mu_k, sqrt(sigma_k^2 + q_std^2))` via logsumexp
   - `log_prob_mean_only(...)`: ablation용 point estimate
   - `TokenVampPriorPreference` class: `token_scores`, `score_frames(direction="clean"/"obstacle")`, `state_dict/from_state_dict`
   - `topk_mean_score`: top-k mean aggregation

2. **`scripts/fit_token_vampprior_preference.py` 신규 생성**
   - `encode_token_posteriors()`: HDF5 → `post.token_mean/token_std` 추출 (get_feat/concat 사용 안 함)
   - clean/avoid per-position mixture fitting
   - per-token + top-k clean-preference + top-k obstacle-energy 3종 diagnostic
   - checkpoint key `token_vampprior_preference` 저장 (기존 key 보존)

3. **`tests/test_token_vampprior_preference.py` 신규 생성**
   - 17개 테스트 전체 통과

4. **드라이런 검증**
   - clean posterior avg score: +1.29, avoid: -1.58, gap: +2.87
   - clean_mean shape: `(64, 3, 64)`, avoid_mean shape: `(64, 3, 64)`

5. **전체 test suite**: 124/124 통과

6. **커밋**: `d8b6fc0` — `feature/token-vit-world-model`

### 주요 설계 결정
- `score_mode=q_integrated` (default): posterior uncertainty를 반영 (`q_std^2` 를 effective variance에 더함)
- `score_mode=mean_only`: ablation용, posterior spread 무시
- `direction="obstacle"`: obstacle-energy = -score → 높을수록 obstacle-like

### 다음 단계 (미완)
- `token_vampprior_preference`를 EFE/evaluate.py에 연결할지 판단
- Phase B (transition latent alignment) 훈련 준비

---

## 2026-05-30

### 작업 범위
Token VampPrior-like preference에 `q_std_scale` / `proto_std_scale` precision scaling 추가.

### 배경
q_integrated effective_var = proto_std² + q_std² 구조에서, 실제 checkpoint의 token_std가 크면 clean/avoid log-prob 차이가 희석된다. scale parameter를 추가해 posterior uncertainty와 prototype sharpness를 각각 조절 가능하게 한다.

### 완료된 작업

1. **`src/active_inference/training/token_vampprior_preference.py` 수정**
   - `log_prob_posterior_under_mixture`: `q_std_scale`, `proto_std_scale`, `min_var` 인자 추가
     - `effective_var = (proto_std_scale * proto_std)² + (q_std_scale * q_std)²`
     - `min_var` clamp으로 수치 안정성 보장
   - `log_prob_mean_only`: `proto_std_scale` 인자 추가 (q_std_scale은 mean_only에서 무시)
   - `TokenVampPriorPreference.__init__`: `q_std_scale`, `proto_std_scale` field 추가
   - `_log_prob_clean/_log_prob_avoid`: scale 값을 함수에 전달
   - `state_dict/from_state_dict`: 두 scale 저장/복원, 구 checkpoint는 1.0 default

2. **`scripts/fit_token_vampprior_preference.py` 수정**
   - `--q_std_scale` (default=1.0), `--proto_std_scale` (default=1.0) CLI 인자 추가
   - 로그 출력에 두 값 포함
   - `TokenVampPriorPreference` 생성 및 checkpoint 저장 dict에 포함

3. **`tests/test_token_vampprior_preference.py` 수정**
   - 8개 테스트 추가 (총 25개):
     - default scale이 원래 수식과 동일한지 manual 계산으로 검증
     - q_std_scale=0.0이 q_std_scale=1.0과 다름 (large q_std에서)
     - proto_std_scale=0.5가 1.0과 다름
     - mean_only에서도 proto_std_scale 반영
     - state_dict save/load + 구 checkpoint backward compat + checkpoint dict 구조

4. **드라이런 결과 (random-weight checkpoint, 15 frames, N=64 positions)**

   | q_std_scale | proto_std_scale | per-token gap | k=4 clean-pref gap | k=4 obstacle-energy gap |
   |-------------|-----------------|--------------|-------------------|------------------------|
   | 1.0 | 1.0 | +2.87 | +3.28 | +3.20 |
   | 0.25 | 1.0 | +4.84 | +5.60 | +5.38 |
   | 0.1 | 1.0 | +5.03 | +5.83 | +5.60 |
   | 0.25 | 0.5 | +17.12 | +19.98 | +18.75 |

5. **전체 test suite**: 132/132 통과

6. **커밋**: `274e873` — `feature/token-vit-world-model`

### 설계 결정
- `q_std_scale=0.0`은 mean_only와 동일하게 수렴 (posterior variance 무시)
- `proto_std_scale=0.5` + `q_std_scale=0.25` 조합이 가장 큰 gap을 보임 (real checkpoint에서는 ablation 필요)
- 기존 동작: 두 scale 모두 1.0이면 이전 버전과 수치적으로 동일

### 다음 단계 (미완)
- `token_vampprior_preference`를 EFE/evaluate.py에 연결할지 판단
- 실제 학습된 checkpoint에서 scale ablation 실험

---

## 2026-05-30 (3차)

### 작업 범위
RSSM vs TokenViT transition prior rollout 비교 시각화 script 구현.

### 완료된 작업

1. **`scripts/compare_rssm_token_vit_rollout.py` 신규 생성**
   - context 구간: `obs_step` (GT observation 사용, posterior filtering)
   - rollout 구간: `img_step` only — future GT image 미사용 (pure prior rollout)
   - decode path:
     - RSSM: `obs_decoder(rssm.get_feat(state))` via `WorldModel.decode_obs()`
     - TokenViT: `obs_decoder(state.deter, state.stoch)` — direct 3D token decode
   - 5-row grid (GT | RSSM pred | RSSM L1 err | TokenViT pred | TokenViT L1 err)
   - Horizon-wise MSE/PSNR, SSIM은 skimage soft-optional (uv env에 없어 NaN)
   - Multi-case (`--num_cases`, `--stride`), metrics CSV 저장

2. **Smoke run 결과**

   Obstacle data (expert_data_v4, start=1000):
   | step | RSSM PSNR | TokenViT PSNR |
   |------|-----------|---------------|
   | +1 | 23.15 | **26.39** |
   | +3 | 23.31 | **25.92** |
   | +5 | 23.00 | **23.33** |

   Clean data (expert_data_town04, start=1000):
   | step | RSSM PSNR | TokenViT PSNR |
   |------|-----------|---------------|
   | +1 | 23.22 | **28.19** |
   | +3 | 23.76 | **26.97** |
   | +5 | 24.16 | **25.91** |

   → TokenViT가 모든 horizon에서 PSNR 우위. 격차는 초반 step에서 더 큼.

3. **커밋**: `997e731` — `feature/token-vit-world-model`

### 제한사항
- SSIM: uv env에 skimage 없어 NaN — 별도 설치 또는 system python으로 실행 시 동작
- Image range assumption: min < -0.1이면 [-0.5,0.5]로 자동 판정
- Action indexing: actions[i]가 frame i → i+1 전환에 해당한다는 convention

### 다음 단계 (미완)
- 더 많은 case (--num_cases)로 전체 dataset rollout 품질 분석
- EFE/evaluate.py에 token preference 연결 여부 판단

---

## 2026-05-30 (4차)

### 작업 범위
`compare_rssm_token_vit_rollout.py`에 crop_road 지원 추가.

### 배경
RSSM 설정 (task_b_v5.yaml)과 TokenViT 설정 (token_vit.yaml) 모두 `encoder.crop_road: true`. 인코더가 내부적으로 crop_road를 적용하므로 GT 표시도 같은 변환을 거쳐야 공정한 시각적 비교가 가능하다.

### 완료된 작업

1. **`scripts/compare_rssm_token_vit_rollout.py` 수정** — 커밋 `51967b0`

   신규 함수:
   - `get_crop_road_from_config(cfg_path)`: YAML 설정에서 `encoder.crop_road` 읽기 (없으면 False)
   - `apply_crop_road(img)`: `_crop_road_transform` 래퍼
   - `prepare_gt_images(images, start, context_len, horizon, crop_road)`: crop_road 조건부로 GT 이미지 리스트 반환

   변경된 함수:
   - `make_grid`: `gt_rssm_imgs` / `gt_tvit_imgs` 분리 인자 + `crop_road_rssm` / `crop_road_tvit` 플래그 추가
     - crop_road가 어느 한 모델에서 활성화된 경우 6행 레이아웃 (RSSM GT + RSSM rollout + RSSM error + TVit GT + TVit rollout + TVit error)
     - 둘 다 같은 경우 5행 (GT 공유)
   - `run_case`: `crop_road_rssm` / `crop_road_tvit` 파라미터 추가
     - 모델별 GT 생성 (`gt_rssm_list`, `gt_tvit_list`)
     - 에러맵과 메트릭을 각 모델에 맞는 GT에 대해 계산
   - `main`: `--crop_road` CLI 인자 추가 (양쪽 강제 적용)
     - 자동 읽기: `crop_road_rssm = args.crop_road or get_crop_road_from_config(args.rssm_config)`

2. **스모크 런 결과**
   - RSSM crop_road: True (task_b_v5.yaml), TokenViT crop_road: True (token_vit.yaml) — 둘 다 crop_road 적용
   - 정상 실행, 그리드 저장 확인
   - 5행 레이아웃 (crop_road 동일하므로)

### 설계 결정
- GT는 인코더가 실제로 처리하는 입력과 동일하게 표시 (공정한 오차 계산)
- crop_road가 두 모델에서 다를 때 6행으로 확장해 차이를 명시적으로 표시
- `--crop_road` 플래그는 설정 자동 읽기보다 우선 (override)

### 다음 단계 (미완)
- 더 많은 case (--num_cases)로 전체 dataset rollout 품질 분석
- EFE/evaluate.py에 token preference 연결 여부 판단

---

## 2026-05-30 (5차)

### 작업 범위
WorldModel-level image preprocessing 일관성 수정 (encoder/target mismatch 버그 픽스).

### 발견된 버그
- `ConvEncoder.forward()`에만 `crop_road`가 적용되고, reconstruction target은 full image → encoder input과 decoder target이 다른 image space
- TokenViT encoder는 `_crop_road` 파라미터 자체가 없어서 `cfg.encoder.crop_road: true`가 사실상 무시됨
- preference fitting, rollout visualization, visual surprise 모두 raw full image를 기준으로 계산

### 완료된 작업

1. **`src/active_inference/agent.py`** 수정 — 커밋 `cfd790e`
   - `self._crop_road` 저장, `ConvEncoder`에 `crop_road=False` 전달 (double crop 방지)
   - `WorldModel.preprocess_image(img)` 신규
   - `WorldModel.encode_obs(img, state)` 신규 — 외부 호출용 진입점
   - `update()`: `img_t = preprocess_image(img_raw_t)`, VFE target을 `img_t`로 수정
   - `step_with_info()`: `img_model` 기준 visual surprise & `ref_image`
   - `encode_preference_data()`: `encode_obs()` 사용

2. **스크립트 전체 `.encoder(` → `.encode_obs(` 교체**
   - `fit_contrastive_preference.py`, `fit_token_contrastive_preference.py`, `fit_token_vampprior_preference.py`, `analyze_5d_results.py`, `refit_preference.py`, `diagnose_state_decoder.py`

3. **`compare_rssm_token_vit_rollout.py`** 리팩터
   - 이전 crop 관련 헬퍼 (`get_crop_road_from_config`, `apply_crop_road`, `prepare_gt_images`) 제거
   - `preprocess_for_display(wm, img_chw, device)` 신규 (`wm.preprocess_image` 기반)
   - `rollout_model`: `encode_obs` 사용
   - `make_grid`: 단일 GT 5행으로 단순화, `gt_label` 파라미터 추가
   - `run_case`: RSSM preprocess_for_display GT 기준 단일 메트릭
   - `main`: `--crop_road` 인자 제거, `wm._crop_road` 자동 읽기, 불일치 warning

4. **`tests/test_image_preprocessing_consistency.py`** 신규 — 6/6 통과

5. **전체 test suite**: 137/138 (CARLA env 제외)

6. **검증 결과**
   - `wm._crop_road=True`, `wm.encoder._crop_road=False` ✓
   - `preprocess_image` = manual `crop_road()` (diff = 0.00e+00) ✓
   - 기존 checkpoint: MSE(recon, full)=0.0065 < MSE(recon, crop)=0.0261 (기존 버그로 학습된 결과, 예상됨)

### 설계 결정
- `preprocess_image()`가 모든 image preprocessing의 단일 진입점
- `encode_obs(img, state)` = external API, raw image → preprocess → encode
- `encoder(img, state)` = internal only (preprocessed image 받는 것을 암묵적으로 가정)
- `token_vit.yaml`: 이미 `crop_road: true` 존재 — 별도 config 생성 불필요

### 수동 실행 커맨드 (학습 실행 금지 — 참고용)

RSSM 64x64 corrected training:
```bash
RUN=runs/rssm_64_cropfix_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/.local/bin/uv run python -u scripts/train.py \
    --config configs/experiment/task_b_v5.yaml \
    --data data/expert_data_mixed.h5 \
    --output_dir "$RUN" 2>&1 | tee "$RUN/train.log"
```

TokenViT 64x64 corrected training:
```bash
RUN=runs/token_vit_64_cropfix_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/.local/bin/uv run python -u scripts/train.py \
    --config configs/experiment/token_vit.yaml \
    --data data/expert_data_mixed.h5 \
    --output_dir "$RUN" 2>&1 | tee "$RUN/train.log"
```

### 다음 단계 (미완)
- RSSM과 TokenViT 재학습 (corrected preprocessing target으로)
- 재학습 후 compare_rssm_token_vit_rollout 재실행으로 공정한 비교

---

## 2026-05-30 (6차)

### 작업 범위
`scripts/train.py` checkpoint에 metadata 추가.

### 완료된 작업

1. **`scripts/train.py`** 수정 — 커밋 `3aad724`

   신규 helper 함수 3개:
   - `_safe_config_to_dict(cfg)`: `dataclasses.asdict(cfg)` 기반, 실패 시 `{}` fallback
   - `_get_git_info()`: `git rev-parse` 호출, 실패 시 `(None, None)` 반환
   - `build_checkpoint(agent, cfg, args, epoch, global_step, train_loss, best_epoch, best_loss, checkpoint_type, is_best)`: 기존 key + metadata 포함 dict 반환

   저장 변경:
   - `best.pt`: `build_checkpoint(..., checkpoint_type="best", is_best=True)`
   - `epoch_N.pt`: `build_checkpoint(..., checkpoint_type="epoch")`
   - `final.pt`: `build_checkpoint(..., checkpoint_type="final", is_best=False)`
   - `crash_epoch_*.pt`: 기존 `agent.save_checkpoint()` 유지 (lightweight)

   `best_epoch` 변수 추가로 best 갱신 시점 epoch 추적.
   `epoch_losses`, `mean_loss` loop 전에 초기화 (edge case 방어).

2. **metadata keys 목록**:
   | key | 설명 |
   |-----|------|
   | `epoch` | 저장 시점 epoch |
   | `global_step` | 누적 update step 수 |
   | `best_epoch` | best 갱신된 epoch |
   | `best_metric` / `best_loss` | best loss 값 (동일) |
   | `train_loss` | 해당 epoch mean loss |
   | `config` | Config dataclass → plain dict (JSON-serialisable) |
   | `world_model_type` | `cfg.model.world_model_type` |
   | `crop_road` | `cfg.encoder.crop_road` |
   | `image_size` | `cfg.encoder.image_size` |
   | `data_path` | CLI `args.data` |
   | `output_dir` | CLI `args.output_dir` |
   | `checkpoint_type` | `"best"`, `"epoch"`, `"final"` |
   | `is_best` | bool |
   | `timestamp` | ISO 8601 |
   | `git_commit` | short hash |
   | `git_branch` | branch name |

3. **`tests/test_checkpoint_metadata.py`** 신규 — 20/20 통과

4. **전체 test suite**: 157/158 (CARLA env 제외)

### 설계 결정
- 기존 key (`world_model`, `optimizer`, `preference`) 위치/구조 완전 유지
- metadata는 top-level에 추가 key로만 삽입 → downstream scripts 호환
- `crash_epoch_*.pt`는 CUDA 오류 복구 path이므로 lightweight `agent.save_checkpoint` 유지
- `config` snapshot은 JSON-serialisable plain dict (OmegaConf-free)

### 수동 확인 명령 (재학습 후)
```bash
~/.local/bin/uv run python - <<'PY'
import torch, glob
paths = sorted(glob.glob("runs/*cropfix*/checkpoints/*.pt"))
if not paths:
    print("No cropfix checkpoints found yet.")
for p in paths:
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    print("=" * 100)
    print(p)
    for k in ["epoch","global_step","best_epoch","best_loss","train_loss",
              "checkpoint_type","is_best","world_model_type","crop_road",
              "image_size","git_branch","git_commit","timestamp"]:
        print(k, "=", ckpt.get(k, None))
PY
```

### 다음 단계 (미완)
- RSSM과 TokenViT 재학습 (corrected preprocessing + metadata checkpoint으로)
- 재학습 후 compare_rssm_token_vit_rollout 재실행으로 공정한 비교
