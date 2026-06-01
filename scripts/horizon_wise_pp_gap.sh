cd ~/av/paper-adaptive-precision-dynamics

EVAL_DIR=$(ls -td runs/eval_cropfix_* eval_cropfix_* 2>/dev/null | head -n 1)

for d in \
  pp_gap_rssm_pooled \
  pp_gap_tokenvit_pooled \
  pp_gap_tokenvit_shared_topk4 \
  pp_gap_tokenvit_posnorm_topk4 \
  pp_gap_tokenvit_vampprior_topk4
do
  echo "================================================================================"
  echo "$d"
  cat "$EVAL_DIR/$d/posterior_prior_gap_summary.txt" | grep -E "h=|overall|delta_post|delta_prior|posterior_prior_gap|prior_retention_ratio"
done