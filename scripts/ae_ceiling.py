"""
Pure autoencoder ceiling experiment.

목적:
  world model 의 encoder/decoder 구조를, KL/transition/stochastic 없이
  순수 autoencoder 로 학습해서 "이 구조가 obstacle 을 얼마나 복원할 수 있는가"
  (= decoder 의 원리적 상한선) 를 측정한다.

  - AE 가 obstacle 을 선명히 복원하면:
      decoder 구조는 obstacle 복원 능력이 있다.
      → world model 에서 흐릿한 건 KL/transition 제약(latent 압축) 때문.
  - AE 도 obstacle 못 그리면:
      decoder 구조/해상도(64x64)/용량 자체의 한계.
      → 해상도/decoder 용량을 키워야 (예: 이전 PAActInf 128px).

두 경로 모두 측정:
  rssm_ae : ConvEncoder -> (embed) -> ObsDecoder
  vit_ae  : TokenViTEncoder -> (token grid) -> TokenImageDecoder

비교 지표:
  - full-frame recon MSE
  - obstacle-region recon MSE (obstacle_bbox 영역) <- 핵심
  - 정성: 복원 그리드 PNG (GT vs rssm_ae vs vit_ae)

실행 예:
  uv run python scripts/ae_ceiling.py \
      --data data/expert_obstacle_v5.h5 \
      --epochs 30 --batch 64 --max_frames 8000 \
      --device cuda --output_dir outputs/ae_ceiling
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from active_inference.models.encoder import ConvEncoder
from active_inference.models.decoder import ObsDecoder
from active_inference.models.token_vit import TokenViTEncoder, TokenImageDecoder
from active_inference.utils.transforms import crop_road as _crop_road


# ---------------------------------------------------------------------------
# AE wrappers (no KL, no transition, no sampling — deterministic embed only)
# ---------------------------------------------------------------------------
class RSSMAutoEncoder(nn.Module):
    """ConvEncoder embed (256) -> pad to feat_dim(320) -> ObsDecoder."""

    def __init__(self, state_dim, embed_dim=256, feat_dim=320,
                 image_channels=3, crop_road=False):
        super().__init__()
        self.encoder = ConvEncoder(
            image_channels=image_channels, state_dim=state_dim,
            embed_dim=embed_dim, crop_road=crop_road,
        )
        # embed(256) -> feat(320) so ObsDecoder(feat_dim=320) unchanged
        self.bridge = nn.Linear(embed_dim, feat_dim)
        self.decoder = ObsDecoder(feat_dim=feat_dim, image_channels=image_channels)

    def forward(self, img, state):
        z = self.encoder(img, state)        # (B, embed)
        feat = self.bridge(z)               # (B, feat)
        return self.decoder(feat)           # (B, 3, 64, 64)


class ViTAutoEncoder(nn.Module):
    """TokenViTEncoder (B,N,E) -> deter/stoch split -> TokenImageDecoder.

    TokenImageDecoder.forward(deter, stoch) 를 그대로 쓰기 위해,
    encoder token embed(E) 를 deter(256)+stoch(64) 로 쪼개 공급한다.
    (stochastic sampling 없음 — deterministic split)
    """

    def __init__(self, state_dim, image_size=64, patch_size=8,
                 embed_dim=320, deter_dim=256, stoch_dim=64,
                 image_channels=3, crop_road=False):
        super().__init__()
        self._crop = crop_road
        # embed_dim must be deter+stoch so we can split
        assert embed_dim == deter_dim + stoch_dim
        self.encoder = TokenViTEncoder(
            image_size=image_size, patch_size=patch_size,
            embed_dim=embed_dim, state_dim=state_dim,
        )
        self.decoder = TokenImageDecoder(
            num_tokens=(image_size // patch_size) ** 2,
            deter_dim=deter_dim, stoch_dim=stoch_dim,
            image_channels=image_channels,
        )
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim

    def forward(self, img, state):
        if self._crop:
            img = _crop_road(img)
        tok = self.encoder(img, state)          # (B, N, E=320)
        deter = tok[..., :self._deter_dim]      # (B, N, 256)
        stoch = tok[..., self._deter_dim:]      # (B, N, 64)
        return self.decoder(deter, stoch)       # (B, 3, 64, 64)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_frames(path, max_frames, seed=0):
    with h5py.File(path, "r") as f:
        n_total = len(f["images"])
        n = min(max_frames, n_total)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=n, replace=False))
        images = f["images"][idx]                         # (n,3,64,64) raw
        states = f["states"][idx].astype(np.float32)
        ov = (f["obstacle_visible"][idx].astype(bool)
              if "obstacle_visible" in f else np.zeros(n, bool))
        bbox = (f["obstacle_bbox"][idx].astype(np.float32)
                if "obstacle_bbox" in f else None)         # (n,4) x,y,w,h? 형식 확인
    return images, states, ov, bbox, idx


def normalize(img_u8):
    # train.py 와 동일: /255 - 0.5
    return torch.tensor(img_u8, dtype=torch.float32) / 255.0 - 0.5


def bbox_mask(bbox_row, H=64, W=64):
    """bbox_row: (4,) -> (1,H,W) mask. 형식이 [x0,y0,x1,y1] 또는 [x,y,w,h] 일 수 있어
    안전하게 처리: 값이 0~1 정규화면 *H/W, 픽셀이면 그대로. NaN 이면 None."""
    if bbox_row is None or np.any(np.isnan(bbox_row)):
        return None
    b = bbox_row.astype(np.float32)
    # 정규화(0~1) 추정: 최대값 <= 1.5 이면 정규화로 간주
    norm = b.max() <= 1.5
    if norm:
        b = b * np.array([W, H, W, H], dtype=np.float32)
    x0, y0, a, c = b
    # [x,y,w,h] 와 [x0,y0,x1,y1] 모두 허용: a,c 가 x0보다 작으면 w,h 로 해석
    if a <= x0 or c <= y0:
        x1, y1 = x0 + a, y0 + c
    else:
        x1, y1 = a, c
    x0, y0, x1, y1 = [int(max(0, v)) for v in (x0, y0, x1, y1)]
    x1, y1 = min(W, x1), min(H, y1)
    m = torch.zeros(1, H, W)
    if x1 > x0 and y1 > y0:
        m[:, y0:y1, x0:x1] = 1.0
    return m


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------
def train_ae(model, imgs, states, dev, epochs, batch, lr=3e-4, tag=""):
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = imgs.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, batch):
            bi = perm[s:s + batch]
            img = imgs[bi].to(dev)
            st = states[bi].to(dev)
            recon = model(img, st)
            # crop_road 를 encoder 가 적용하면 target 도 동일 crop 필요.
            # 여기서는 crop_road=False 로 학습(상한선은 전체 프레임 기준)
            loss = F.mse_loss(recon, img)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        if ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"  [{tag}] epoch {ep:3d}  recon_mse={tot / n:.5f}")
    return model


@torch.no_grad()
def eval_ae(model, imgs, states, ov, bboxes, dev):
    model.eval()
    n = imgs.shape[0]
    full_err, obs_err, obs_cnt = 0.0, 0.0, 0
    for s in range(0, n, 128):
        img = imgs[s:s + 128].to(dev)
        st = states[s:s + 128].to(dev)
        recon = model(img, st)
        err = (recon - img) ** 2                       # (b,3,64,64)
        full_err += err.mean(dim=(1, 2, 3)).sum().item()
        # obstacle-region (visible frame 만, bbox 있으면)
        for j in range(img.shape[0]):
            gi = s + j
            if not ov[gi] or bboxes is None:
                continue
            m = bbox_mask(bboxes[gi])
            if m is None:
                continue
            m = m.to(dev)
            denom = m.sum() * img.shape[1]
            if denom > 0:
                obs_err += (err[j] * m).sum().item() / denom.item()
                obs_cnt += 1
    full_mse = full_err / n
    obs_mse = obs_err / max(1, obs_cnt)
    return full_mse, obs_mse, obs_cnt


def save_grid(models, imgs, states, ov, dev, out_png, n_show=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # obstacle visible frame 우선 선택
    vis_idx = np.where(ov)[0]
    sel = vis_idx[:n_show] if len(vis_idx) >= n_show else np.arange(n_show)
    rows = 1 + len(models)
    fig, ax = plt.subplots(rows, n_show, figsize=(2 * n_show, 2 * rows))
    def show(a, t):
        im = (t.detach().cpu().numpy().transpose(1, 2, 0) + 0.5).clip(0, 1)
        a.imshow(im); a.axis("off")
    for c, gi in enumerate(sel):
        show(ax[0, c], imgs[gi])
        if c == 0:
            ax[0, c].set_ylabel("GT", rotation=0, labelpad=30, fontsize=11)
    for r, (name, m) in enumerate(models.items(), start=1):
        m.eval()
        with torch.no_grad():
            for c, gi in enumerate(sel):
                rec = m(imgs[gi:gi+1].to(dev), states[gi:gi+1].to(dev))[0]
                show(ax[r, c], rec)
                if c == 0:
                    ax[r, c].set_title(name, fontsize=10, loc="left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"  grid -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_obstacle_v5.h5")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_frames", type=int, default=8000)
    ap.add_argument("--state_dim", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="outputs/ae_ceiling")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.max_frames} frames from {args.data} …")
    imgs_u8, states_np, ov, bbox, idx = load_frames(args.data, args.max_frames)
    imgs = normalize(imgs_u8)
    states = torch.tensor(states_np, dtype=torch.float32)
    print(f"  {imgs.shape[0]} frames, obstacle_visible={int(ov.sum())}, "
          f"bbox={'yes' if bbox is not None else 'no'}")

    # train/eval split (간단히 80/20, frame-level — AE 상한선이라 누수 영향 적음)
    n = imgs.shape[0]; ntr = int(n * 0.8)
    perm = torch.randperm(n)
    tr, te = perm[:ntr], perm[ntr:]
    ov_te = ov[te.numpy()]
    bbox_te = bbox[te.numpy()] if bbox is not None else None

    results = {}
    models = {}

    print("\n=== RSSM-path AE ===")
    rssm_ae = RSSMAutoEncoder(state_dim=args.state_dim, crop_road=False)
    rssm_ae = train_ae(rssm_ae, imgs[tr], states[tr], dev,
                       args.epochs, args.batch, tag="rssm_ae")
    fm, om, oc = eval_ae(rssm_ae, imgs[te], states[te], ov_te, bbox_te, dev)
    results["rssm_ae"] = {"full_mse": fm, "obs_mse": om, "obs_frames": oc}
    models["rssm_ae"] = rssm_ae
    print(f"  EVAL full_mse={fm:.5f}  obs_mse={om:.5f}  (obs frames {oc})")

    print("\n=== ViT-path AE ===")
    vit_ae = ViTAutoEncoder(state_dim=args.state_dim, crop_road=False)
    vit_ae = train_ae(vit_ae, imgs[tr], states[tr], dev,
                      args.epochs, args.batch, tag="vit_ae")
    fm, om, oc = eval_ae(vit_ae, imgs[te], states[te], ov_te, bbox_te, dev)
    results["vit_ae"] = {"full_mse": fm, "obs_mse": om, "obs_frames": oc}
    models["vit_ae"] = vit_ae
    print(f"  EVAL full_mse={fm:.5f}  obs_mse={om:.5f}  (obs frames {oc})")

    # 정성 그리드
    save_grid(models, imgs[te], states[te], ov_te, dev, str(out / "ae_recon_grid.png"))

    import json
    json.dump(results, open(out / "ae_ceiling_results.json", "w"), indent=2)
    print(f"\nresults -> {out / 'ae_ceiling_results.json'}")
    print("\n=== Verdict hints ===")
    print("  AE obs_mse 가 world model 의 obs recon_mse(~0.002~0.006) 보다 "
          "크게 낮으면:")
    print("    -> decoder 구조는 obstacle 복원 능력 있음. world model 의 흐릿함은")
    print("       KL/transition 제약 탓. (KL 조정으로 개선 여지)")
    print("  AE obs_mse 도 비슷하게 높으면:")
    print("    -> 64x64 해상도/decoder 용량의 구조적 한계. 해상도 확대 필요.")


if __name__ == "__main__":
    main()