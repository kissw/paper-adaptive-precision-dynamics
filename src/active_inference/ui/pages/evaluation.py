import os

import pandas as pd
import streamlit as st


def render():
    st.header("Evaluation Metrics")

    from pathlib import Path

    default_csv = "outputs/eval/eval_results.csv"
    candidates = (
        sorted(Path("outputs").glob("*/eval_results.csv"), key=os.path.getmtime, reverse=True)
        if Path("outputs").exists()
        else []
    )
    if candidates:
        default_csv = str(candidates[0])

    csv_path = st.text_input("Evaluation Results CSV", value=default_csv)

    if not os.path.exists(csv_path):
        st.info("No evaluation results. Run `python scripts/evaluate.py` first.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    if df.empty:
        st.warning("CSV is empty.")
        return

    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("Summary Metrics")

    # Overall metrics
    overall_sr = df["success"].mean()
    overall_mld = df["mean_lateral_dev"].mean()
    overall_offroad = df["offroad_events"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Success Rate", f"{overall_sr:.2%}")
    col2.metric("Overall Mean Lateral Dev", f"{overall_mld:.4f}")
    col3.metric("Total Offroad Events", int(overall_offroad))

    st.subheader("Metrics per Task")

    # Group by task
    task_stats = (
        df.groupby("task")
        .agg(
            success_rate=("success", "mean"),
            mean_lateral_dev=("mean_lateral_dev", "mean"),
            offroad_events=("offroad_events", "sum"),
            episodes=("episode", "count"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Success Rate per Task**")
        st.bar_chart(task_stats.set_index("task")["success_rate"])

    with col2:
        st.write("**Mean Lateral Deviation per Task**")
        st.bar_chart(task_stats.set_index("task")["mean_lateral_dev"])
