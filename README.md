# 🌸 Iris Classification & Petal Length Prediction ML Suite

An end-to-end Machine Learning project featuring data analysis, classification, regression modeling, and an interactive **Streamlit** web application frontend.

---

## 🚀 Features

- **🌸 Species Classification:** Classify Iris flowers into *Setosa*, *Versicolor*, or *Virginica* using a trained **Decision Tree Classifier** pipeline with live probability charts.
- **📏 Petal Length Prediction:** Estimate petal length (cm) based on sepal length, sepal width, and petal width using a **Linear Regression** pipeline.
- **📊 Interactive Data Exploration:** Visualize dataset features, distributions, correlation scatters, and summary metrics.

---

## 📁 Repository Structure

```text
.
├── Iris.csv                      # Iris flower dataset
├── dataml.ipynb                  # Data preprocessing, EDA, model training notebook
├── iris_decision_tree.pkl        # Saved Decision Tree classification model pipeline
├── iris_linear_regression.pkl    # Saved Linear Regression prediction model pipeline
├── app.py                        # Streamlit web application frontend
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Local Setup & Running the Frontend

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application

**Option A: Double-Click Batch File (Windows)**
Simply double-click [`start.bat`](file:///d:/ml/start.bat) in the project folder!

**Option B: Terminal Command**
Start the Streamlit web app server manually:
```bash
python -m streamlit run app.py
```
*(or `streamlit run app.py` if Streamlit is in your system PATH)*

This will automatically open the application in your web browser at `http://localhost:8501`.

---

## 📤 Pushing Changes to GitHub

To push the new frontend code (`app.py`, `requirements.txt`, `README.md`) to your GitHub repository:

```bash
# 1. Check status of changed files
git status

# 2. Add all new and modified files
git add .

# 3. Commit your changes
git commit -m "Add Streamlit frontend web app for classification and prediction models"

# 4. Push to GitHub
git push origin main
```

---

## 🌐 Free Cloud Deployment (Streamlit Community Cloud)

You can host your app live on the internet for free in 3 simple steps:

1. Sign up at [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **"New App"** and select your GitHub repository (`Iris-ML-Mini-Project`).
3. Set **Main file path** to `app.py` and click **Deploy**!

---

## 📊 Model Performance Summary

| Model Task | Algorithm | Key Metric | Score |
| :--- | :--- | :--- | :--- |
| **Classification** | Decision Tree Classifier (`max_depth=4`) | Accuracy | **100%** |
| **Regression** | Linear Regression | R² Score | **0.960** |
