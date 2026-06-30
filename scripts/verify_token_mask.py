"""
Token mask 시각 검증.

crop_road 적용된(encoder 가 실제 보는) 이미지 위에, _bbox_token_mask 가
선택한 obstacle token 영역을 겹쳐 그려서, 장애물 위치와 일치하는지 눈으로 확인.

각 샘플마다 3개 패널:
  1) 원본 이미지 + 원본 bbox (빨강 박스)
  2) crop 이미지 (encoder 가 보는 것) + 장애물 위치
  3) crop 이미지 + 선택된 token grid 영역 (반투명 노랑)
token 영역이 crop 이미지의 장애물 위에 겹치면 정상.

실행:
  uv run python scripts/verify_token_mask.py \
      --data data/expert_obstacle_v5.h5 \
      --n 6 --output outputs/token_mask_check.png
"""
import argparse
import h5py
import numpy as np
import torch

from active_inference.agent import DeepAIFAgent
from active_inference.utils.transforms import crop_road, normalize_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_obstacle_v5.h5")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--keep_bottom_frac", type=float, default=0.6)
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--output", default="outputs/token_mask_check.png")
    args = ap.parse_args()

    G = args.image_size // args.patch_size  # 8
    P = args.patch_size

    with h5py.File(args.data, "r") as f:
        ov = f["obstacle_visible"][:].astype(bool)
        vis_idx = np.where(ov)[0]
        # 다양한 위치의 장애물을 보기 위해 균등 샘플
        sel = vis_idx[np.linspace(0, len(vis_idx) - 1, args.n).astype(int)]
        images = f["images"][sel]            # (n,3,64,64) raw uint8-ish
        bboxes = f["obstacle_bbox"][sel]     # (n,4) xyxy pixel (원본 좌표)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = args.n
    fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 8))

    def to_disp(img_chw):
        # raw -> [-0.5,0.5] -> 0~1 display
        t = normalize_image(torch.tensor(img_chw, dtype=torch.float32))
        return (t.numpy().transpose(1, 2, 0) + 0.5).clip(0, 1)

    for c in range(n):
        img = images[c]                                  # (3,64,64)
        bbox = bboxes[c].astype(np.float32)              # [x1,y1,x2,y2] 원본
        orig_disp = to_disp(img)

        # crop 이미지 (encoder 가 보는 것)
        t = normalize_image(torch.tensor(img, dtype=torch.float32)).unsqueeze(0)
        cropped = crop_road(t, keep_bottom_frac=args.keep_bottom_frac)[0]
        crop_disp = (cropped.numpy().transpose(1, 2, 0) + 0.5).clip(0, 1)

        # token mask (crop 보정 적용)
        bb = torch.tensor(bbox).unsqueeze(0)
        mask = DeepAIFAgent._bbox_token_mask(
            bb, G * G, args.image_size, P, "cpu",
            crop_road=True, keep_bottom_frac=args.keep_bottom_frac,
        )[0]  # (64,)

        # 패널 1: 원본 + 원본 bbox
        ax[0, c].imshow(orig_disp)
        x1, y1, x2, y2 = bbox
        ax[0, c].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, edgecolor="red", lw=2))
        ax[0, c].axis("off")
        if c == 0:
            ax[0, c].set_title("orig + bbox", loc="left", fontsize=10)

        # 패널 2: crop 이미지 (장애물이 어디로 갔는지)
        ax[1, c].imshow(crop_disp)
        ax[1, c].axis("off")
        if c == 0:
            ax[1, c].set_title("cropped (encoder view)", loc="left", fontsize=10)

        # 패널 3: crop 이미지 + 선택된 token 영역 오버레이
        ax[2, c].imshow(crop_disp)
        overlay = np.zeros((args.image_size, args.image_size, 4))
        for i in range(G * G):
            if mask[i] > 0:
                r, col = i // G, i % G
                overlay[r * P:(r + 1) * P, col * P:(col + 1) * P] = [1, 1, 0, 0.4]
        ax[2, c].imshow(overlay)
        # token grid 선
        for k in range(G + 1):
            ax[2, c].axhline(k * P - 0.5, color="white", lw=0.3, alpha=0.5)
            ax[2, c].axvline(k * P - 0.5, color="white", lw=0.3, alpha=0.5)
        ax[2, c].axis("off")
        ntok = int(mask.sum())
        if c == 0:
            ax[2, c].set_title("cropped + token mask", loc="left", fontsize=10)
        ax[2, c].text(1, 6, f"{ntok} tok", color="yellow", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.output}")
    print("패널3의 노란 token 영역이 패널2 crop 이미지의 장애물 위에 겹치면 정상.")


if __name__ == "__main__":
    main()