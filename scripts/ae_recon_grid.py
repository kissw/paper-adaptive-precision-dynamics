"""
AE reconstruction grid.

stage=ae 로 학습한 world model 의 encoder-decoder 복원 품질을 눈으로 확인.
update_ae 와 동일한 forward 경로(obs_step -> _det_state -> decode_obs)를 미러한다.

train / valid 각각에서 무작위 N개 프레임을 뽑아
  행: [train GT, train recon, valid GT, valid recon]
  열: 5개 샘플
그리드로 저장.

실행:
  uv run python scripts/ae_recon_grid.py \
      --checkpoint outputs/v6_vit_ae_klrep0.0/checkpoints/best.pt \
      --config configs/experiment/token_vit.yaml \
      --train_data data/wm_train_mixed_v6_train.h5 \
      --valid_data data/wm_train_mixed_v6_valid.h5 \
      --n 5 --output outputs/v6_vit_ae_klrep0.0/ae_recon_grid.png
"""
import argparse
import numpy as np
import torch

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.data.dataset import SequenceDataset


@torch.no_grad()
def reconstruct(agent, images_seq, states_seq, actions_seq):
    """update_ae 와 동일 경로로 시퀀스를 복원. 마지막 timestep 복원 반환.

    images_seq: (T, C, H, W) raw (정규화 전)
    return: (gt_norm (C,H,W), recon (C,H,W)) for the chosen timestep
    """
    wm = agent.world_model
    dev = agent._device
    T = images_seq.shape[0]

    prev_state = wm.rssm.initial(1, dev)
    last = None
    for t in range(T):
        img_t = wm.preprocess_image(images_seq[t:t+1].to(dev))
        st_t = states_seq[t:t+1].to(dev)
        act_t = actions_seq[t:t+1].to(dev)
        embed = wm.encoder(img_t, st_t)
        post, _ = wm.rssm.obs_step(prev_state, act_t, embed)
        det_post = agent._det_state(post)
        recon = wm.decode_obs(det_post)
        prev_state = post
        last = (img_t[0].detach().cpu(), recon[0].detach().cpu())
    return last  # 마지막 timestep (context 가 가장 많이 쌓인 시점)


def to_img(t):
    """[-0.5,0.5] tensor (C,H,W) -> (H,W,C) 0~1."""
    a = t.numpy().transpose(1, 2, 0) + 0.5
    return np.clip(a, 0, 1)


def pick_samples(dataset, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/experiment/token_vit.yaml")
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--valid_data", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="ae_recon_grid.png")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.training.stage = "ae"
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)
    agent.world_model.eval()

    tr = SequenceDataset(args.train_data, seq_len=args.seq_len)
    va = SequenceDataset(args.valid_data, seq_len=args.seq_len)
    tr_idx = pick_samples(tr, args.n, args.seed)
    va_idx = pick_samples(va, args.n, args.seed + 1)

    def collect(ds, idxs):
        gts, recons = [], []
        for i in idxs:
            sample = ds[int(i)]
            # SequenceDataset 반환 형식: (images, states, actions, [labels])
            images, states, actions = sample[0], sample[1], sample[2]
            gt, rec = reconstruct(agent, images, states, actions)
            gts.append(to_img(gt)); recons.append(to_img(rec))
        return gts, recons

    print("reconstructing train samples ...")
    tr_gt, tr_rc = collect(tr, tr_idx)
    print("reconstructing valid samples ...")
    va_gt, va_rc = collect(va, va_idx)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = args.n
    rows = 4  # train GT, train recon, valid GT, valid recon
    fig, ax = plt.subplots(rows, n, figsize=(2.2 * n, 2.2 * rows))
    row_data = [("train GT", tr_gt), ("train recon", tr_rc),
                ("valid GT", va_gt), ("valid recon", va_rc)]
    for r, (label, imgs) in enumerate(row_data):
        for c in range(n):
            ax[r, c].imshow(imgs[c]); ax[r, c].axis("off")
            if c == 0:
                ax[r, c].set_ylabel(label, rotation=0, ha="right",
                                    va="center", fontsize=11)
        # 행 라벨을 첫 칸 title 로도
        ax[r, 0].set_title(label if c == 0 else "", fontsize=10, loc="left")
    plt.tight_layout()
    plt.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.output}")

    # 정량: 각 행의 복원 MSE 평균
    def mse(gts, rcs):
        return float(np.mean([(np.array(g) - np.array(r)) ** 2
                              for g, r in zip(gts, rcs)]))
    print(f"train recon MSE (display space): {mse(tr_gt, tr_rc):.5f}")
    print(f"valid recon MSE (display space): {mse(va_gt, va_rc):.5f}")


if __name__ == "__main__":
    main()