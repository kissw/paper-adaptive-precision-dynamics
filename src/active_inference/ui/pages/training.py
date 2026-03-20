import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir):
    """Load scalar data from TensorBoard logs."""
    if not os.path.exists(log_dir):
        return None

    # Find event files
    event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
    if not event_files:
        return None

    # Load the most recent event file
    event_file = sorted(event_files, key=os.path.getmtime)[-1]

    ea = EventAccumulator(str(event_file))
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame(
            [(e.step, e.value) for e in events], columns=["step", "value"]
        ).set_index("step")

    return data


def render():
    st.header("Training Curves")

    default_log_dir = "outputs/train/tb_logs"
    candidates = (
        sorted(Path("outputs").glob("*/tb_logs"), key=os.path.getmtime, reverse=True)
        if Path("outputs").exists()
        else []
    )
    if candidates:
        default_log_dir = str(candidates[0])

    col1, col2 = st.columns([3, 1])
    with col1:
        log_dir = st.text_input("TensorBoard Log Directory", value=default_log_dir)
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (5s)")

    data = load_tensorboard_data(log_dir)

    if data is None:
        st.info("No training logs found. Run `python scripts/train.py` first.")
    else:
        metrics = [
            "train/total_loss",
            "train/img_loss",
            "train/kl_dyn",
            "train/kl_rep",
            "train/epoch_loss",
        ]

        for metric in metrics:
            if metric in data:
                st.subheader(metric)
                st.line_chart(data[metric])
            else:
                st.warning(f"Metric '{metric}' not found in logs.")

    if auto_refresh:
        time.sleep(5)
        st.rerun()
