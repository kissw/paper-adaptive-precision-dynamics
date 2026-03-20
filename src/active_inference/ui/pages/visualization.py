import os

import streamlit as st


def render():
    st.header("Visualization")

    st.write(
        "Load a trained model checkpoint to visualize its internal representations and planning."
    )

    ckpt_path = st.text_input("Model Checkpoint Path", value="outputs/train/checkpoints/best.pt")

    if not os.path.exists(ckpt_path):
        st.warning(f"Checkpoint not found at {ckpt_path}. Please train a model first.")
        return

    st.success(f"Found checkpoint at {ckpt_path}")

    st.divider()

    st.subheader("Imagination Rollout")
    st.write(
        "This section will decode the Cross-Entropy Method (CEM) planning states into images. "
        "It allows you to see what the agent 'imagines' will happen for different action sequences."
    )
    st.info("Placeholder: Imagination rollout visualization will be implemented here.")

    st.divider()

    st.subheader("Preference t-SNE")
    st.write(
        "This section will show a t-SNE projection of the latent space with the Gaussian Mixture Model (GMM) "
        "preference distribution overlaid. It helps visualize how the agent clusters different driving situations."
    )
    st.info("Placeholder: Preference t-SNE visualization will be implemented here.")
