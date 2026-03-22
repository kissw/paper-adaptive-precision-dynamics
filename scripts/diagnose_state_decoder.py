"""Diagnose state decoder predictions during CEM imagination.

Checks whether the state decoder produces meaningful differentiation
between different steering actions during open-loop RSSM rollout.
"""
import torch
import numpy as np
from active_inference.config import Config
from active_inference.agent import DeepAIFAgent

cfg = Config.from_yaml("configs/default.yaml")
agent = DeepAIFAgent(cfg)
agent.load_checkpoint("outputs/train_v4/checkpoints/best.pt")
agent.reset()

device = agent._device
wm = agent.world_model
wm.eval()

# Create a realistic initial state from zeros (cold start)
state = wm.rssm.initial(1, device)
action = torch.zeros(1, 2, device=device)

# Simulate a few observation steps with dummy data to get a warm state
# Use training data statistics: speed~3m/s, steer~0, heading~0, crosstrack~0
dummy_img = torch.zeros(1, 3, 64, 64, device=device)
dummy_state = torch.tensor([[3.0, 0.0, 0.0, 0.0]], device=device)

embed = wm.encoder(dummy_img, dummy_state)
post, prior = wm.rssm.obs_step(state, action, embed)

print("=== State decoder on posterior (observation-informed) ===")
feat_post = wm.rssm.get_feat(post)
decoded_post = wm.state_decoder(feat_post)
print(f"Decoded: speed={decoded_post[0,0]:.4f}, steer={decoded_post[0,1]:.4f}, "
      f"heading_err={decoded_post[0,2]:.4f}, crosstrack_err={decoded_post[0,3]:.4f}")

print("\n=== State decoder on prior (imagination-only) ===")
feat_prior = wm.rssm.get_feat(prior)
decoded_prior = wm.state_decoder(feat_prior)
print(f"Decoded: speed={decoded_prior[0,0]:.4f}, steer={decoded_prior[0,1]:.4f}, "
      f"heading_err={decoded_prior[0,2]:.4f}, crosstrack_err={decoded_prior[0,3]:.4f}")

# Now test imagination with different steering actions
print("\n=== Imagination rollout (10 steps) with different steering ===")
n_actions = 5
steers = [-0.5, -0.2, 0.0, 0.2, 0.5]

for steer_val in steers:
    # Roll out 10 imagination steps
    expanded = type(post)(*[x.expand(1, -1) for x in post])
    rollout_states = []
    s = expanded
    for t in range(10):
        a = torch.tensor([[steer_val, 0.3]], device=device)  # fixed steer, moderate accel
        s = wm.rssm.img_step(s, a)
        feat = wm.rssm.get_feat(s)
        decoded = wm.state_decoder(feat)
        rollout_states.append(decoded[0].detach().cpu().numpy())

    rollout_states = np.array(rollout_states)
    print(f"\nSteer={steer_val:+.1f}:")
    print(f"  speed:     {rollout_states[:,0].mean():.4f} ± {rollout_states[:,0].std():.4f}")
    print(f"  steer:     {rollout_states[:,1].mean():.4f} ± {rollout_states[:,1].std():.4f}")
    print(f"  heading:   {rollout_states[:,2].mean():.4f} ± {rollout_states[:,2].std():.4f}")
    print(f"  crosstrack:{rollout_states[:,3].mean():.4f} ± {rollout_states[:,3].std():.4f}")
    print(f"  state_penalty (h²+c²): {(rollout_states[:,2]**2 + rollout_states[:,3]**2).mean():.6f}")

# CEM-like test: score 500 random trajectories
print("\n=== CEM-like scoring: 500 random trajectories ===")
N = 500
expanded = type(post)(*[x.expand(N, -1) for x in post])

# Random actions
actions = torch.randn(10, N, 2, device=device) * 0.5
actions = actions.clamp(-1, 1)

# Imagine
trajectory = wm.rssm.imagine(expanded, actions)

# Decode states at each step
for t in [0, 4, 9]:
    feat = wm.rssm.get_feat(trajectory[t])
    decoded = wm.state_decoder(feat)
    he = decoded[:, 2]
    ce = decoded[:, 3]
    penalty = he.pow(2) + ce.pow(2)
    print(f"\nStep {t}:")
    print(f"  heading_err: mean={he.mean():.4f} std={he.std():.4f} range=[{he.min():.4f}, {he.max():.4f}]")
    print(f"  crosstrack:  mean={ce.mean():.4f} std={ce.std():.4f} range=[{ce.min():.4f}, {ce.max():.4f}]")
    print(f"  penalty:     mean={penalty.mean():.6f} std={penalty.std():.6f}")

    # Correlation with steering action at this step
    steer_t = actions[t, :, 0].cpu().numpy()
    he_np = he.detach().cpu().numpy()
    ce_np = ce.detach().cpu().numpy()
    corr_he = np.corrcoef(steer_t, he_np)[0,1]
    corr_ce = np.corrcoef(steer_t, ce_np)[0,1]
    print(f"  corr(action_steer, heading_err): {corr_he:.4f}")
    print(f"  corr(action_steer, crosstrack):  {corr_ce:.4f}")
