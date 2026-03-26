# Open Questions

## remove-stanley-pure-efe - 2026-03-21

- [ ] **Horizon vs. real-time constraint:** At horizon=25, n_samples=500, the CEM planning loop runs 500x25 RSSM imagine steps per action. On the target GPU, does this complete within the 50ms frame budget (20 FPS)? If not, what is the maximum horizon/samples combination that fits? -- Determines whether the plan is deployable in real-time CARLA.

- [ ] **Preference data filtering strategy:** Should the preference model be fitted only on frames with `lateral_dev < 0.3m` (strictly centered), or on all `success_only=True` frames (which may include gentle off-center driving)? -- Stricter filtering gives sharper preferences but fewer samples; looser filtering gives more data but blurrier preferences.

- [ ] **Action smoothness penalty:** Should an explicit temporal smoothness term be added to the EFE score (penalizing large action changes between consecutive horizon steps)? This would reduce steering jitter but adds a non-EFE term to the objective. -- Theoretical purity vs. practical smoothness tradeoff.

- [ ] **Braking support:** The current throttle mapping is `accel in [-1,1] -> throttle in [0.35, 0.55]` with zero braking. For obstacle avoidance (Task B), the agent may need to brake. Should the action-to-CARLA mapping be expanded to include braking (e.g., `accel < -0.5 -> brake`)? -- Required for Task B success but increases action space complexity.

- [ ] **Old checkpoint migration:** After state_dim changes from 2 to 4, old checkpoints will fail to load (encoder linear layer shape mismatch). Should we add a migration utility, or is clean retrain acceptable? -- Affects workflow for anyone with existing trained models.

- [ ] **Heading error sign convention verification:** The code computes `heading_error = road_yaw - vehicle_yaw` (wrapped). When the road curves left, is heading_error positive or negative? The sign must be consistent with CARLA's steer convention (positive = left turn). -- Incorrect sign convention would invert the steering gradient in EFE.
