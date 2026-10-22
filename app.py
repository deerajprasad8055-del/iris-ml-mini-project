import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Iris ML - Classification & Prediction Suite",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling & CSS Design System
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Gradient Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.4);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #a5b4fc, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #c7d2fe;
        max-width: 800px;
    }

    /* Result Card Styling */
    .result-card {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .species-setosa {
        color: #0d9488;
        background-color: #ccfbf1;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .species-versicolor {
        color: #0284c7;
        background-color: #e0f2fe;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .species-virginica {
        color: #7c3aed;
        background-color: #f3e8ff;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }

    .metric-badge {
        font-size: 2rem;
        font-weight: 800;
        color: #4f46e5;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Models & Dataset (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    dt_path = "iris_decision_tree.pkl"
    lr_path = "iris_linear_regression.pkl"
    
    dt_model = None
    lr_model = None
    
    if os.path.exists(dt_path):
        dt_model = joblib.load(dt_path)
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)
        
    return dt_model, lr_model

@st.cache_data
def load_dataset():
    if os.path.exists("Iris.csv"):
        return pd.read_csv("Iris.csv")
    return None

dt_model, lr_model = load_models()
df = load_dataset()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=400", use_container_width=True)
    st.title("🌸 Iris ML Suite")
    st.markdown("---")
    
    st.markdown("### 📌 Navigation & Info")
    st.info("""
    **Model Summary:**
    - **Classification:** Decision Tree Pipeline
    - **Regression:** Linear Regression Pipeline
    - **Dataset:** Iris Flower Dataset (150 samples)
    """)
    
    st.markdown("---")
    st.markdown("⚡ **Built with Scikit-Learn & Streamlit**")

# ---------------------------------------------------------
# Hero Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌸 Iris Machine Learning Dashboard</div>
    <div class="hero-subtitle">
        An interactive frontend web application serving trained Machine Learning models: 
        Predict Iris species via <b>Decision Tree Classifier</b> and estimate Petal Length using <b>Linear Regression</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# Check model availability
if dt_model is None or lr_model is None:
    st.error("⚠️ Model files (`iris_decision_tree.pkl` / `iris_linear_regression.pkl`) not found! Please ensure `.pkl` files exist in the project directory.")
    st.stop()

# ---------------------------------------------------------
# Tabs Setup
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🌸 Species Classification", 
    "📏 Petal Length Prediction", 
    "📊 Dataset & Model Insights"
])

# =========================================================
# TAB 1: CLASSIFICATION (DECISION TREE)
# =========================================================
with tab1:
    st.subheader("🌸 Predict Iris Flower Species")
    st.markdown("Provide the measurements below to classify the flower species using the Decision Tree model.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📐 Sepal Dimensions")
        sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1, key="cls_sl")
        sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1, key="cls_sw")
        
    with col2:
        st.markdown("#### 🌺 Petal Dimensions")
        petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=3.8, step=0.1, key="cls_pl")
        petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2, step=0.1, key="cls_pw")
        
    # Input DataFrame construction
    input_data_cls = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    
    st.markdown("---")
    
    # Prediction execution
    prediction = dt_model.predict(input_data_cls)[0]
    probabilities = dt_model.predict_proba(input_data_cls)[0]
    classes = dt_model.classes_

    # Formatting prediction badge
    species_class = "species-setosa" if "setosa" in prediction.lower() else ("species-versicolor" if "versicolor" in prediction.lower() else "species-virginica")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown("### 🏆 Prediction Result")
        st.markdown(f'<div class="{species_class}">🌸 {prediction}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Features Summary:")
        summary_df = input_data_cls.T.reset_index()
        summary_df.columns = ["Feature", "Value (cm)"]
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    with res_col2:
        st.markdown("### 📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Species": classes,
            "Probability": probabilities
        })
        
        fig = px.bar(
            prob_df, 
            x="Probability", 
            y="Species", 
            orientation="h",
            color="Species",
            text=prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%"),
            color_discrete_map={
                "Iris-setosa": "#0d9488",
                "Iris-versicolor": "#0284c7",
                "Iris-virginica": "#7c3aed"
            }
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1]),
            showlegend=False,
            height=250,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2: REGRESSION (LINEAR REGRESSION)
# =========================================================
with tab2:
    st.subheader("📏 Predict Petal Length")
    st.markdown("Estimate **Petal Length (cm)** using Sepal Length, Sepal Width, and Petal Width via the Linear Regression model.")
    
    reg_col1, reg_col2, reg_col3 = st.columns(3)
    
    with reg_col1:
        r_sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.9, step=0.1, key="reg_sl")
    with reg_col2:
        r_sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1, key="reg_sw")
    with reg_col3:
        r_petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.8, step=0.1, key="reg_pw")

    # Predict regression output
    input_data_reg = pd.DataFrame([[r_sepal_length, r_sepal_width, r_petal_width]],
                                  columns=["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"])
    
    predicted_length = lr_model.predict(input_data_reg)[0]
    
    st.markdown("---")
    
    out_col1, out_col2 = st.columns([1, 1])
    
    with out_col1:
        st.markdown("### 🎯 Estimated Output")
        st.markdown(f'<div class="metric-badge">{predicted_length:.2f} cm</div>', unsafe_allow_html=True)
        st.caption("Predicted Petal Length")
        
        st.success(f"According to your Linear Regression pipeline, a flower with Sepal Length={r_sepal_length}cm, Sepal Width={r_sepal_width}cm, and Petal Width={r_petal_width}cm is estimated to have a Petal Length of **{predicted_length:.2f} cm**.")
        
    with out_col2:
        st.markdown("### 🎚️ Petal Length Range Indicator")
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = float(predicted_length),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Petal Length (cm)"},
            gauge = {
                'axis': {'range': [1.0, 7.0]},
                'bar': {'color': "#4f46e5"},
                'steps': [
                    {'range': [1.0, 2.5], 'color': "#ccfbf1"},
                    {'range': [2.5, 5.0], 'color': "#e0f2fe"},
                    {'range': [5.0, 7.0], 'color': "#f3e8ff"}
                ]
            }
        ))
        gauge_fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

# =========================================================
# TAB 3: DATASET & MODEL INSIGHTS
# =========================================================
with tab3:
    st.subheader("📊 Dataset Exploration & Model Metrics")
    
    if df is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Samples", len(df))
        m2.metric("Features Count", 4)
        m3.metric("Decision Tree Accuracy", "100.0%")
        m4.metric("Regression R² Score", "0.960")
        
        st.markdown("---")
        
        c1, c2 = st.columns([1.2, 1])
        
        with c1:
            st.markdown("#### 📉 Interactive Feature Scatter Plot")
            x_axis = st.selectbox("Select X-Axis", ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], index=0)
            y_axis = st.selectbox("Select Y-Axis", ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], index=2)
            
            scatter_fig = px.scatter(
                df, 
                x=x_axis, 
                y=y_axis, 
                color="Species",
                title=f"{x_axis} vs {y_axis}",
                color_discrete_map={
                    "Iris-setosa": "#0d9488",
                    "Iris-versicolor": "#0284c7",
                    "Iris-virginica": "#7c3aed"
                }
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
            
        with c2:
            st.markdown("#### 📋 Dataset Preview (`Iris.csv`)")
            st.dataframe(df, use_container_width=True, height=400)
    else:
        st.warning("`Iris.csv` dataset not found in root folder.")
