"""
Streamlit Web Application: Rice Crop Disease Detection & Diagnosis System.
Interactive prototype combining:
- Handcrafted Multi-Level Feature Extraction (MCLBP + Color Correlogram)
- Machine Learning models from ML ALL.pdf
- Agronomic treatment recommendations
"""

import os
import sys
import glob
# pyrefly: ignore[missing-import]
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local modules are accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DATASETS, SAVED_MODELS_DIR, REPORTS_DIR, ML_LEVELS
from src.preprocessing import preprocess_image, logarithmic_enhancement, resize_image
from src.feature_extraction import (
    extract_fused_features,
    compute_lbp_channel,
    extract_color_correlogram
)
from predict import predict_disease, find_default_model, DISEASE_ADVISORY
from src.dataset_loader import load_dataset
from src.models import get_all_models, get_model
from src.evaluation import evaluate_predictions

# Streamlit page configuration
st.set_page_config(
    page_title="Rice Crop Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-left: 5px solid #43a047;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .diagnosis-card {
        background-color: #e8f5e9;
        border: 2px solid #2e7d32;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_sample_images():
    """Gathers sample images from each dataset for quick testing in UI."""
    samples = {}
    for ds_key, info in DATASETS.items():
        base_path = info.get('path') or info.get('train_path')
        if base_path and os.path.exists(base_path):
            subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')])
            for s in subdirs:
                sp = os.path.join(base_path, s)
                imgs = [os.path.join(sp, f) for f in os.listdir(sp) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    label = f"[{ds_key}] {s}: {os.path.basename(imgs[0])}"
                    samples[label] = imgs[0]
    return samples


def main():
    st.markdown('<div class="main-title">🌾 Rice Crop Disease Detection & Diagnosis Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Powered by Multi-Level Handcrafted Feature Extraction (MCLBP + Color Correlogram) & ML Models '
        'from Nature Scientific Reports (Alsakar et al., 2024) and ML ALL Foundations.</div>',
        unsafe_allow_html=True
    )

    # Sidebar: Model Selector & Info
    st.sidebar.header("⚙️ Model Configuration")
    
    # List saved models
    saved_model_files = []
    if os.path.exists(SAVED_MODELS_DIR):
        saved_model_files = [
            os.path.join(SAVED_MODELS_DIR, f) for f in os.listdir(SAVED_MODELS_DIR) if f.endswith('.joblib')
        ]
    
    selected_model_path = None
    if saved_model_files:
        model_names = [os.path.basename(f) for f in saved_model_files]
        chosen_idx = st.sidebar.selectbox("Active Saved Model", range(len(model_names)), format_func=lambda i: model_names[i])
        selected_model_path = saved_model_files[chosen_idx]
    else:
        st.sidebar.warning("No saved model found yet. Please run training or use the Retrain tab.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Datasets in Workspace:")
    st.sidebar.markdown("- **DS1**: 4 Classes (BLB, Blast, Brownspot, Tungro)")
    st.sidebar.markdown("- **DS2**: 5 Classes (Blight, Blast, Brownspot, Tungro, Sheath Blight)")
    st.sidebar.markdown("- **DS3**: 3 Classes (BLB, Brownspot, Leaf Smut)")

    # Tabs
    tab_diagnose, tab_benchmark, tab_methodology, tab_retrain = st.tabs([
        "🔍 Disease Diagnosis",
        "📊 ML Model Benchmark",
        "📖 Research Methodology",
        "⚡ Train / Retrain Model"
    ])

    # =========================================================================
    # TAB 1: DISEASE DIAGNOSIS
    # =========================================================================
    with tab_diagnose:
        col_input, col_results = st.columns([1, 1.2])

        with col_input:
            st.subheader("1. Input Rice Leaf Image")
            input_mode = st.radio("Choose Input Method:", ["Select Benchmark Sample", "Upload Custom Image"], horizontal=True)

            img_path_to_process = None
            uploaded_file = None

            if input_mode == "Select Benchmark Sample":
                samples_dict = get_sample_images()
                if samples_dict:
                    selected_sample_label = st.selectbox("Choose a sample leaf image:", list(samples_dict.keys()))
                    img_path_to_process = samples_dict[selected_sample_label]
                    raw_img = cv2.imread(img_path_to_process)
                    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                    st.image(raw_img, caption="Original Input Leaf Image", use_container_width=True)
                else:
                    st.error("No sample images found in dataset folders.")
            else:
                uploaded_file = st.file_uploader("Upload leaf image (.jpg, .jpeg, .png)", type=['jpg', 'jpeg', 'png'])
                if uploaded_file is not None:
                    # Save temporary file
                    temp_path = os.path.join(SAVED_MODELS_DIR, "temp_upload.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    img_path_to_process = temp_path
                    raw_img = cv2.imread(temp_path)
                    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                    st.image(raw_img, caption="Uploaded Leaf Image", use_container_width=True)

        with col_results:
            st.subheader("2. Diagnosis & Agronomic Report")
            if img_path_to_process is not None:
                if selected_model_path is None:
                    st.info("💡 A model needs to be trained first before prediction. Go to '⚡ Train / Retrain Model' tab to train with 1 click!")
                else:
                    try:
                        with st.spinner("Analyzing image via Logarithmic Enhancement & Handcrafted Fusion..."):
                            result = predict_disease(img_path_to_process, selected_model_path)
                        
                        adv = result['advisory']
                        conf = result['confidence (%)']

                        st.markdown(f"""
                        <div class="diagnosis-card">
                            <h3 style="color:#2e7d32; margin-top:0;">🌱 Detected Disease: {adv['display_name']}</h3>
                            <h4 style="color:#1565c0;">Confidence Score: <b>{conf:.2f}%</b></h4>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability Distribution Chart
                        st.markdown("**Probability Distribution across Classes:**")
                        df_probs = pd.DataFrame(
                            list(result['probabilities'].items()),
                            columns=['Disease Class', 'Probability (%)']
                        ).sort_values(by='Probability (%)', ascending=True)

                        fig, ax = plt.subplots(figsize=(6, 2.5))
                        ax.barh(df_probs['Disease Class'], df_probs['Probability (%)'], color='#43a047')
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Probability (%)')
                        st.pyplot(fig)

                        # Agronomic Treatment Advice
                        with st.expander("🌾 Clinical Symptoms & Agronomic Management", expanded=True):
                            st.markdown(f"**Biological Symptoms:**\n{adv['symptoms']}")
                            st.markdown(f"**Pathogen & Causes:**\n{adv['causes']}")
                            st.markdown(f"**Recommended Agronomic Action:**\n{adv['treatment']}")

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

        # Inspection of Image Preprocessing and LBP Feature Maps
        if img_path_to_process is not None:
            st.markdown("---")
            st.subheader("🔬 Image Processing & Feature Map Inspection")
            st.write("Visualizing the transformations defined in Research.pdf Equations (7, 8, 11, 12):")

            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            # Preprocess
            prep_img = preprocess_image(img_path_to_process, enhance=True, augment=False)
            raw_bgr = cv2.imread(img_path_to_process)
            if raw_bgr is not None:
                norm_img = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                norm_img = resize_image(norm_img)
            else:
                norm_img = prep_img

            with col_p1:
                st.image(norm_img, caption="1. Resized (224x224)", use_container_width=True)
            with col_p2:
                st.image(prep_img, caption="2. Log-Enhanced (c=2.0)", use_container_width=True)

            # Compute LBP channels
            lbp_r = compute_lbp_channel(prep_img[:, :, 0])
            lbp_g = compute_lbp_channel(prep_img[:, :, 1])
            with col_p3:
                st.image(lbp_r, caption="3. LBP Red Channel Map", use_container_width=True, clamp=True)
            with col_p4:
                st.image(lbp_g, caption="4. LBP Green Channel Map", use_container_width=True, clamp=True)

    # =========================================================================
    # TAB 2: MODEL BENCHMARK
    # =========================================================================
    with tab_benchmark:
        st.subheader("📊 Comparative Evaluation of ML Models (from ML ALL.pdf & Research.pdf Table 11)")
        st.write(
            "Evaluation metrics formulated in Research.pdf Equations (17)-(22): "
            "Accuracy, Precision, Recall/Sensitivity, F1-Score, Specificity, and Matthews Correlation Coefficient (MCC)."
        )

        # Check existing CSV reports
        report_files = [os.path.join(REPORTS_DIR, f) for f in os.listdir(REPORTS_DIR) if f.endswith('.csv')]
        if report_files:
            rep_names = [os.path.basename(f) for f in report_files]
            sel_rep = st.selectbox("Select Benchmark Report:", rep_names)
            df_rep = pd.read_csv(os.path.join(REPORTS_DIR, sel_rep))
            st.dataframe(df_rep.style.highlight_max(axis=0, color='#c8e6c9'), use_container_width=True)

            # Bar plot of accuracy
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(df_rep['Model'], df_rep['Accuracy (%)'], color='#1e88e5')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Model Accuracy Comparison')
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig)
        else:
            st.info("No benchmark reports generated yet. Run `python train.py --model all` or train in the 'Train / Retrain Model' tab.")

    # =========================================================================
    # TAB 3: RESEARCH METHODOLOGY
    # =========================================================================
    with tab_methodology:
        st.subheader("📖 Theoretical Pipeline: Research.pdf & ML ALL.pdf")
        
        st.markdown("""
        ### 1. Image Preprocessing (Research.pdf Section 3.2)
        - **Resizing (Eq. 2)**: All images scaled to $224 \\times 224$ pixels.
        - **Logarithmic Enhancement (Eq. 7-8)**: Overcomes illumination and lighting variances:
          $$D_{tNorm} = \\frac{D_t}{255}, \\quad D_{tLog} = c \\cdot \\ln(1 + D_{tNorm}), \\quad c = 2.0$$
        - **Data Augmentation (Eq. 3-6)**: Translation ($-5$ to $+15$ px), Flipping (horizontal/vertical), Rotation ($-10^\\circ$ to $+10^\\circ$).

        ### 2. Feature Extraction (Research.pdf Section 3.3)
        - **Multi-Level Multi-Channel LBP (ML-MCLBP, Eq. 10-12)**:
          - Image is divided into $L \\times L = N$ spatial blocks (Levels 1 to 5).
          - For each block and each color channel (Red, Green, Blue), LBP with $P=8, R=1$ is computed:
            $$LBP(x_c, y_c) = \\sum_{p=0}^7 2^p \\cdot g(B_p - B(x_c, y_c))$$
          - Histogram of 256 bins is normalized for each channel.
        - **Spatial Color Correlogram (CC, Eq. 9)**:
          - Color space quantized into $4 \\times 4 \\times 4 = 64$ discrete bins.
          - Measures probability $\\gamma^{(D)}_{c_i, c_j}$ of finding matching color at distances $D \\in \\{1, 3, 5, 7\\}$.
        - **Feature Fusion (Eq. 13)**:
          $$FE = [\\gamma^{(D)}_{c_i, c_j}, \\ LBP(RGB)]$$
          Concatenates global chromatic correlation with localized multi-scale structural leaf textures.

        ### 3. Machine Learning Models (ML ALL.pdf & Research.pdf Table 11)
        - **Linear Support Vector Machine (Proposed Winner)**: Minimizes margin soft loss with One-Against-All strategy (Eq. 14, 16).
        - **Non-Linear SVMs**: Gaussian RBF Kernel and Polynomial Kernel.
        - **Decision Trees (CART)**: Gini index and Information Gain splits.
        - **Ensembles**: Random Forest Bagging and AdaBoost.
        - **K-Nearest Neighbors (KNN)**: Euclidean distance local proximity search.
        - **Multi-Layer Perceptron (MLP)**: Backpropagation and Adam optimizer.
        """)

    # =========================================================================
    # TAB 4: RETRAIN MODEL
    # =========================================================================
    with tab_retrain:
        st.subheader("⚡ Train or Retrain ML Models")
        st.write("Train any model on any of the 3 datasets directly from this interface:")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            sel_ds = st.selectbox("Select Dataset:", ["DS1", "DS2", "DS3"])
            sel_model = st.selectbox(
                "Select Model Architecture:",
                ["all", "extra_trees", "svm_linear", "svm_rbf", "svm_poly", "random_forest", "decision_tree", "knn", "mlp", "softmax"]
            )
        with col_t2:
            sel_samples = st.number_input("Max Samples per Class (0 = Full Dataset):", min_value=0, max_value=2000, value=80, step=20)
            sel_level = st.slider("ML Spatial Levels (L):", min_value=1, max_value=5, value=3)

        if st.button("🚀 Start Training Pipeline", use_container_width=True):
            sample_val = None if sel_samples == 0 else int(sel_samples)
            with st.spinner(f"Extracting features and training {sel_model} on {sel_ds}..."):
                try:
                    X, y, class_names, paths = load_dataset(
                        dataset_key=sel_ds,
                        max_samples_per_class=sample_val,
                        level=sel_level,
                        use_cache=True
                    )
                    from src.dataset_loader import split_dataset_3way
                    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_3way(
                        X, y, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, random_state=42
                    )

                    if sel_model == 'all':
                        models_to_run = get_all_models()
                    else:
                        models_to_run = {sel_model: get_model(sel_model)}

                    rows = []
                    best_acc = -1
                    best_tr = -1
                    best_val = -1
                    best_p = None
                    best_m = None

                    for name, pipe in models_to_run.items():
                        pipe.fit(X_train, y_train)
                        preds = pipe.predict(X_test)
                        eval_res = evaluate_predictions(y_test, preds, class_names)
                        macro = eval_res['macro_metrics']
                        tr_acc = float(pipe.score(X_train, y_train) * 100.0)
                        val_acc = float(pipe.score(X_val, y_val) * 100.0)
                        rows.append({
                            'Model': name,
                            'Train Acc (70%)': tr_acc,
                            'Val Acc (10%)': val_acc,
                            'Test Acc (20%)': macro['Accuracy (%)'],
                            'Precision (%)': macro['Precision (%)'],
                            'Recall (%)': macro['Recall (%)'],
                            'F1-Score (%)': macro['F1-Score (%)'],
                            'Specificity (%)': macro['Specificity (%)'],
                            'MCC (%)': macro['MCC (%)']
                        })
                        if macro['Accuracy (%)'] > best_acc:
                            best_acc = macro['Accuracy (%)']
                            best_tr = tr_acc
                            best_val = val_acc
                            best_p = pipe
                            best_m = name

                    df_res = pd.DataFrame(rows).sort_values(by='Test Acc (20%)', ascending=False)
                    st.success(f"Training Complete! Top model: **{best_m}** (Test: **{best_acc:.2f}%**, Val: **{best_val:.2f}%**).")
                    st.dataframe(df_res.style.highlight_max(axis=0, color='#c8e6c9'), use_container_width=True)

                    # Save model
                    import joblib
                    save_path = os.path.join(SAVED_MODELS_DIR, f"best_model_{sel_ds.lower()}.joblib")
                    payload = {
                        'model_name': best_m,
                        'pipeline': best_p,
                        'class_names': class_names,
                        'levels': sel_level,
                        'dataset': sel_ds,
                        'split': {'train': 0.70, 'val': 0.10, 'test': 0.20},
                        'train_accuracy': best_tr,
                        'val_accuracy': best_val,
                        'test_accuracy': best_acc
                    }
                    joblib.dump(payload, save_path)
                    st.info(f"Model saved to `{save_path}`. You can now use it in the 'Disease Diagnosis' tab!")
                    st.info(f"Model saved to `{save_path}`. You can now use it in the 'Disease Diagnosis' tab!")

                except Exception as e:
                    st.error(f"Training failed: {e}")


if __name__ == '__main__':
    main()
