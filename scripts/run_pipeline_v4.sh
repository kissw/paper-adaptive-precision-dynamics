#!/bin/bash
#
# Automated Evaluation Pipeline V4
# ==================================
# Phases: diagnostic → config sweep → data collection → train → eval → plot → iterate
# Designed for 24-hour unattended operation.
#
# Usage:
#   bash scripts/run_pipeline_v4.sh
#
set -uo pipefail

CARLA_ROOT="/data/jaerock/carla-0.9.16"
PROJECT_ROOT="/data/jaerock/projects/active-inference-omc"
CARLA_PORT=2000
MAX_ITERATIONS=5

# Starting checkpoint and config
CHECKPOINT="$PROJECT_ROOT/outputs/train_v1/checkpoints/final.pt"
CONFIG="$PROJECT_ROOT/configs/default.yaml"
OUTPUT_BASE="$PROJECT_ROOT/outputs/eval_v4"

export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/pipeline.log"
CARLA_PID=""
ACTIVE_CONFIG="$CONFIG"

# ── Logging ──────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ── CARLA management ────────────────────────────────────────────
start_carla() {
    # Check if CARLA already running on our port
    if pgrep -f "carla-port=$CARLA_PORT" >/dev/null 2>&1; then
        CARLA_PID=$(pgrep -f "CarlaUE4-Linux.*port=$CARLA_PORT" | head -1)
        log "CARLA already running on port $CARLA_PORT (PID=$CARLA_PID)"
        return 0
    fi

    log "Starting CARLA on port $CARLA_PORT..."
    "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -benchmark -fps=20 \
        -carla-port=$CARLA_PORT &>/tmp/carla_pipeline_v4.log &
    CARLA_PID=$!
    log "Waiting 40s for CARLA to initialize (PID=$CARLA_PID)..."
    sleep 40

    if ! kill -0 $CARLA_PID 2>/dev/null; then
        log "ERROR: CARLA failed to start. Check /tmp/carla_pipeline_v4.log"
        return 1
    fi
    log "CARLA started successfully"
}

stop_carla() {
    if [ -n "${CARLA_PID:-}" ] && kill -0 "$CARLA_PID" 2>/dev/null; then
        log "Stopping CARLA (PID=$CARLA_PID)..."
        kill "$CARLA_PID" 2>/dev/null || true
        sleep 5
        # Force kill if still alive
        kill -9 "$CARLA_PID" 2>/dev/null || true
        log "CARLA stopped"
    fi
    CARLA_PID=""
}

cleanup() {
    stop_carla
    log "Pipeline terminated (cleanup)"
}
trap cleanup EXIT

# ── Config generation ────────────────────────────────────────────
create_config() {
    local name=$1
    local beta_i=$2
    local beta_e=$3
    local n_samples=$4
    local horizon=$5
    local cfg_dir="$OUTPUT_BASE/configs"
    mkdir -p "$cfg_dir"
    local cfg_path="$cfg_dir/${name}.yaml"

    uv run python -c "
from omegaconf import OmegaConf
from active_inference.config import Config

schema = OmegaConf.structured(Config)
raw = OmegaConf.load('$CONFIG')
merged = OmegaConf.merge(schema, raw)

merged.efe.beta_instrumental = $beta_i
merged.efe.beta_epistemic = $beta_e
merged.cem.n_samples = $n_samples
merged.cem.horizon = $horizon

OmegaConf.save(merged, '$cfg_path')
print('Created config: $cfg_path')
print(f'  EFE: beta_i={$beta_i}, beta_e={$beta_e}')
print(f'  CEM: samples={$n_samples}, horizon={$horizon}')
" 2>&1
    echo "$cfg_path"
}

# ── Phase: Diagnostic ───────────────────────────────────────────
run_diagnostic() {
    local tag=$1
    local config=$2
    local diag_dir="$OUTPUT_BASE/diagnostic_${tag}"

    log "── Diagnostic [$tag] with config=$(basename $config) ──"
    if uv run python scripts/diagnose_agent.py \
        --checkpoint "$CHECKPOINT" \
        --config "$config" \
        --output_dir "$diag_dir" \
        --port $CARLA_PORT 2>&1 | tee -a "$LOG_FILE"; then
        return 0
    else
        return 1
    fi
}

# ── Phase: Config sweep ─────────────────────────────────────────
run_config_sweep() {
    log "── Config Sweep: trying alternative EFE/CEM parameters ──"

    # Sweep configs: name, beta_i, beta_e, n_samples, horizon
    local configs=(
        "high_epistemic:1.0:5.0:200:12"
        "very_high_epistemic:0.5:10.0:200:12"
        "low_instrumental:0.1:1.0:200:12"
        "more_samples:1.0:1.0:500:16"
        "balanced:0.5:2.0:300:12"
        "explore_heavy:0.1:5.0:300:16"
    )

    for entry in "${configs[@]}"; do
        IFS=: read -r name beta_i beta_e samples horizon <<< "$entry"
        log "  Trying: $name (beta_i=$beta_i, beta_e=$beta_e, samples=$samples, H=$horizon)"

        local cfg_path
        cfg_path=$(create_config "$name" "$beta_i" "$beta_e" "$samples" "$horizon")

        if run_diagnostic "sweep_${name}" "$cfg_path"; then
            log "  *** Config '$name' produces functional agent! ***"
            ACTIVE_CONFIG="$cfg_path"
            return 0
        fi
        log "  Config '$name' failed"
    done

    log "  All sweep configs failed"
    return 1
}

# ── Phase: Collect Task B preference data ────────────────────────
collect_preference_data() {
    local pref_file="$PROJECT_ROOT/data/preference_task_b.h5"
    if [ -f "$pref_file" ]; then
        log "── Task B preference data exists, skipping ──"
        return 0
    fi

    log "── Collecting Task B preference data (3000 samples) ──"
    uv run python scripts/collect_preference_data.py \
        --town Town06_Opt \
        --num_samples 3000 \
        --output "$pref_file" \
        --port $CARLA_PORT \
        --num_obstacles 3 2>&1 | tee -a "$LOG_FILE"

    if [ -f "$pref_file" ]; then
        log "Task B preference data collected: $pref_file"
    else
        log "WARNING: Task B preference data collection may have failed"
    fi
}

# ── Phase: Retrain ───────────────────────────────────────────────
retrain() {
    local iter=$1
    local extra_epochs=${2:-10}
    local train_dir="$PROJECT_ROOT/outputs/train_v4_iter${iter}"

    log "── Retraining (iter=$iter, +${extra_epochs} epochs) ──"
    log "  Base checkpoint: $CHECKPOINT"
    log "  Config: $ACTIVE_CONFIG"
    log "  Output: $train_dir"

    local total_epochs
    total_epochs=$(uv run python -c "
from active_inference.config import Config
cfg = Config.from_yaml('$ACTIVE_CONFIG')
print(cfg.training.epochs + $extra_epochs)
" 2>/dev/null)

    uv run python scripts/train.py \
        --config "$ACTIVE_CONFIG" \
        --data "$PROJECT_ROOT/data/expert_data.h5" \
        --resume "$CHECKPOINT" \
        --epochs "$total_epochs" \
        --output_dir "$train_dir" 2>&1 | tee -a "$LOG_FILE"

    if [ -f "$train_dir/checkpoints/final.pt" ]; then
        CHECKPOINT="$train_dir/checkpoints/final.pt"
        log "New checkpoint: $CHECKPOINT"
    else
        log "WARNING: Retraining may have failed, keeping original checkpoint"
    fi
}

# ── Phase: Full evaluation ───────────────────────────────────────
evaluate_all() {
    local iter=$1
    local eval_dir="$OUTPUT_BASE/iter${iter}"

    log "── Full Evaluation (iter=$iter) ──"
    log "  Checkpoint: $CHECKPOINT"
    log "  Config: $ACTIVE_CONFIG"
    log "  Output: $eval_dir"

    for task in baseline A B; do
        log "  Evaluating Task $task..."
        uv run python scripts/evaluate.py \
            --task "$task" \
            --checkpoint "$CHECKPOINT" \
            --config "$ACTIVE_CONFIG" \
            --episodes 3 \
            --max_frames 2000 \
            --save_video \
            --output_dir "$eval_dir" \
            --port $CARLA_PORT 2>&1 | tee -a "$LOG_FILE" || {
            log "  WARNING: Task $task evaluation had errors"
        }
    done

    log "Evaluation complete for iter $iter"
}

# ── Phase: Generate plots ───────────────────────────────────────
generate_plots() {
    local iter=$1
    local eval_dir="$OUTPUT_BASE/iter${iter}"

    if [ ! -d "$eval_dir/trajectories" ]; then
        log "No trajectory directory found for iter $iter, skipping plots"
        return 0
    fi

    log "── Generating Trajectory Plots (iter=$iter) ──"
    uv run python scripts/plot_trajectory.py \
        --input_dir "$eval_dir/trajectories" \
        --output_dir "$eval_dir/plots" 2>&1 | tee -a "$LOG_FILE"
}

# ── Phase: Analyze results ──────────────────────────────────────
analyze_results() {
    local iter=$1
    local eval_dir="$OUTPUT_BASE/iter${iter}"
    local csv_path="$eval_dir/eval_results.csv"

    if [ ! -f "$csv_path" ]; then
        log "No eval results CSV for iter $iter"
        echo "false"
        return 0
    fi

    log "── Analysis (iter=$iter) ──"

    uv run python -c "
import csv, json, sys
import numpy as np
from pathlib import Path

results = []
with open('$csv_path') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

if not results:
    report = {'iteration': $iter, 'acceptable': False, 'reason': 'no results'}
    Path('$eval_dir/analysis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('ACCEPTABLE=false')
    sys.exit(0)

tasks = {}
for r in results:
    task = r['task']
    if task not in tasks:
        tasks[task] = []
    tasks[task].append(r)

analysis = {}
for task, rows in tasks.items():
    sr = sum(int(r['success']) for r in rows) / len(rows)
    comp = np.mean([float(r['route_completion_pct']) for r in rows])
    mld = np.mean([float(r['mean_lateral_dev']) for r in rows])
    efe = np.mean([float(r.get('mean_efe_score', 0)) for r in rows])
    epi = np.mean([float(r.get('mean_epistemic_score', 0)) for r in rows])
    frames = np.mean([int(r['frames']) for r in rows])
    analysis[task] = {
        'success_rate': round(sr, 3),
        'mean_completion': round(float(comp), 1),
        'mean_lateral_dev': round(float(mld), 3),
        'mean_efe': round(float(efe), 4),
        'mean_epistemic': round(float(epi), 4),
        'mean_frames': round(float(frames), 0),
        'episodes': len(rows),
    }

# Acceptability: baseline must show at least 15% completion and < 5m lateral dev
baseline = analysis.get('baseline', {})
acceptable = baseline.get('mean_completion', 0) > 15 and baseline.get('mean_lateral_dev', 99) < 5

report = {
    'iteration': $iter,
    'tasks': analysis,
    'acceptable': acceptable,
    'config': '$ACTIVE_CONFIG',
    'checkpoint': '$CHECKPOINT',
}

Path('$eval_dir/analysis.json').write_text(json.dumps(report, indent=2))

for task, info in analysis.items():
    print(f'  Task {task}: SR={info[\"success_rate\"]:.0%} Comp={info[\"mean_completion\"]:.1f}% '
          f'MLD={info[\"mean_lateral_dev\"]:.3f} EFE={info[\"mean_efe\"]:.4f} '
          f'Epistemic={info[\"mean_epistemic\"]:.4f}')

if acceptable:
    print('ACCEPTABLE=true')
else:
    print('ACCEPTABLE=false')
    if baseline.get('mean_completion', 0) < 5:
        print('ISSUE=agent_barely_moves')
    elif baseline.get('mean_lateral_dev', 99) > 5:
        print('ISSUE=high_lateral_deviation')
    else:
        print('ISSUE=low_completion')
" 2>&1 | tee -a "$LOG_FILE"
}

# ── Generate final summary report ────────────────────────────────
generate_report() {
    log "── Generating Final Report ──"

    uv run python -c "
import json
from pathlib import Path

output_base = Path('$OUTPUT_BASE')
report_lines = ['# Evaluation V4 Pipeline Report', '']

# Collect all iteration analyses
for iter_dir in sorted(output_base.glob('iter*')):
    analysis_path = iter_dir / 'analysis.json'
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        iter_num = analysis.get('iteration', '?')
        report_lines.append(f'## Iteration {iter_num}')
        report_lines.append(f'- Config: {analysis.get(\"config\", \"N/A\")}')
        report_lines.append(f'- Checkpoint: {analysis.get(\"checkpoint\", \"N/A\")}')
        report_lines.append(f'- Acceptable: {analysis.get(\"acceptable\", False)}')
        report_lines.append('')
        for task, info in analysis.get('tasks', {}).items():
            report_lines.append(f'### Task {task}')
            report_lines.append(f'| Metric | Value |')
            report_lines.append(f'|--------|-------|')
            for k, v in info.items():
                report_lines.append(f'| {k} | {v} |')
            report_lines.append('')

# Diagnostic summaries
for diag_dir in sorted(output_base.glob('diagnostic_*')):
    report_path = diag_dir / 'diagnostic_report.json'
    if report_path.exists():
        diag = json.loads(report_path.read_text())
        tag = diag_dir.name.replace('diagnostic_', '')
        report_lines.append(f'## Diagnostic: {tag}')
        report_lines.append(f'- Functional: {diag.get(\"functional\", False)}')
        report_lines.append(f'- Speed: {diag.get(\"post_warmup_speed\", 0):.3f} m/s')
        report_lines.append(f'- Distance: {diag.get(\"total_distance_m\", 0):.1f} m')
        report_lines.append(f'- Mean EFE: {diag.get(\"mean_efe\", 0):.6f}')
        report_lines.append(f'- Config: {diag.get(\"config\", \"N/A\")}')
        report_lines.append('')

report_text = '\n'.join(report_lines)
(output_base / 'pipeline_report.md').write_text(report_text)
print(report_text)
" 2>&1 | tee -a "$LOG_FILE"
}

# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
main() {
    log "═══════════════════════════════════════════════════"
    log "  Evaluation Pipeline V4 — Automated"
    log "  Max iterations: $MAX_ITERATIONS"
    log "  Checkpoint: $CHECKPOINT"
    log "  Config: $CONFIG"
    log "  Output: $OUTPUT_BASE"
    log "═══════════════════════════════════════════════════"

    start_carla

    for iter in $(seq 0 $((MAX_ITERATIONS - 1))); do
        log ""
        log "═══════════════════════════════════════════════════"
        log "  ITERATION $iter"
        log "═══════════════════════════════════════════════════"

        # ── Step 1: Diagnostic with current config ──
        if run_diagnostic "iter${iter}" "$ACTIVE_CONFIG"; then
            log "Agent is functional with current config"
        else
            log "Agent NOT functional with current config"

            # ── Step 1b: Config sweep ──
            if run_config_sweep; then
                log "Found working config via sweep: $ACTIVE_CONFIG"
            else
                log "No working config found in sweep"

                # ── Step 1c: Retrain ──
                collect_preference_data
                retrain "$iter" 10
                # After retraining, re-check
                if ! run_diagnostic "iter${iter}_retrained" "$ACTIVE_CONFIG"; then
                    log "Agent still non-functional after retraining"
                    log "Trying sweep again with retrained model..."
                    if ! run_config_sweep; then
                        log "CRITICAL: Cannot make agent functional at iter $iter"
                        log "Continuing to next iteration with more training..."
                        retrain "$iter" 20
                        continue
                    fi
                fi
            fi
        fi

        # ── Step 2: Collect preference data ──
        collect_preference_data

        # ── Step 3: Full evaluation ──
        evaluate_all "$iter"

        # ── Step 4: Generate plots ──
        generate_plots "$iter"

        # ── Step 5: Analyze results ──
        analysis_output=$(analyze_results "$iter")

        if echo "$analysis_output" | grep -q "ACCEPTABLE=true"; then
            log "*** Performance ACCEPTABLE at iteration $iter ***"
            break
        fi

        # Determine issue and adjust
        if echo "$analysis_output" | grep -q "ISSUE=agent_barely_moves"; then
            log "Issue: agent barely moves → will retrain with more epochs"
            retrain "$iter" 20
        elif echo "$analysis_output" | grep -q "ISSUE=high_lateral_deviation"; then
            log "Issue: high lateral deviation → adjusting CEM/EFE params"
            ACTIVE_CONFIG=$(create_config "iter${iter}_fix" 0.5 3.0 400 12)
        else
            log "Issue: low completion → retraining with additional epochs"
            retrain "$iter" 10
        fi
    done

    # ── Final report ──
    generate_report

    log ""
    log "═══════════════════════════════════════════════════"
    log "  Pipeline V4 COMPLETE"
    log "  Results: $OUTPUT_BASE"
    log "  Report: $OUTPUT_BASE/pipeline_report.md"
    log "═══════════════════════════════════════════════════"
}

main 2>&1 | tee -a "$LOG_FILE"
