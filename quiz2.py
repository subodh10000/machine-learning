import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_blobs, make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# --- Page Config ---
st.set_page_config(page_title="Easy ML Learner", layout="wide")

# --- Helper Function to Plot Decision Boundaries ---
def plot_decision_boundary(model, X, y, title):
    h = .02  # Step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    ax.set_title(title)
    return fig

# --- App Sidebar ---
st.sidebar.title("🎓 Easy ML Learner")
st.sidebar.write("Choose a lesson below:")
lesson = st.sidebar.radio(
    "",["1. Linear Models", 
     "2. KNN & Naïve Bayes", 
     "3. Decision Trees & SVM"]
)

# ==========================================
# LESSON 1: LINEAR MODELS
# ==========================================
if lesson == "1. Linear Models":
    st.title("📈 Supervised Learning: Linear Models")
    st.write("Linear models try to find a straight line that best represents the data. They are simple, fast, and easy to understand.")
    
    tab1, tab2 = st.tabs(["Linear Regression", "Logistic Regression"])
    
    with tab1:
        st.header("Linear Regression (Predicting a Number)")
        st.info("**The Analogy:** Imagine trying to draw a straight line through a scatter of stars so that the line is as close to all the stars as possible.")
        
        noise = st.slider("Add Noise (Randomness) to Data", 5.0, 50.0, 15.0)
        X, y = make_regression(n_samples=100, n_features=1, noise=noise, random_state=42)
        
        model = LinearRegression()
        model.fit(X, y)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='blue', alpha=0.6, label="Actual Data")
        ax.plot(X, model.predict(X), color='red', linewidth=3, label="Best Fit Line (Prediction)")
        ax.legend()
        st.pyplot(fig)
        
    with tab2:
        st.header("Logistic Regression (Predicting a Category)")
        st.info("**The Analogy:** Instead of drawing a line *through* the data, we draw a line to *separate* the data into two buckets (e.g., Spam vs. Not Spam).")
        
        X_log, y_log = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42, cluster_std=2.0)
        model_log = LogisticRegression()
        model_log.fit(X_log, y_log)
        
        fig_log = plot_decision_boundary(model_log, X_log, y_log, "Logistic Regression Boundary")
        st.pyplot(fig_log)

# ==========================================
# LESSON 2: KNN & NAÏVE BAYES
# ==========================================
elif lesson == "2. KNN & Naïve Bayes":
    st.title("🏘️ KNN & Naïve Bayes Classifiers")
    st.write("These algorithms predict categories based on proximity or probability.")
    
    # We use a curved dataset here to show how non-linear models work
    X_moon, y_moon = make_moons(n_samples=200, noise=0.25, random_state=42)
    
    tab1, tab2 = st.tabs(["K-Nearest Neighbors (KNN)", "Naïve Bayes"])
    
    with tab1:
        st.header("K-Nearest Neighbors (KNN)")
        st.info("**The Analogy:** 'Birds of a feather flock together.' If you move to a new town and want to know if a neighborhood is safe, you ask your 'K' closest neighbors. If the majority say yes, you assume it's safe.")
        
        k = st.slider("Number of Neighbors (K)", 1, 30, 5)
        st.write("Notice how a low K makes the boundaries very 'choppy' (overthinking), while a high K makes it smooth.")
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_moon, y_moon)
        
        fig_knn = plot_decision_boundary(knn, X_moon, y_moon, f"KNN (K={k})")
        st.pyplot(fig_knn)
        
    with tab2:
        st.header("Naïve Bayes")
        st.info("**The Analogy:** Like a doctor diagnosing a patient using past statistics. If you have a fever and a cough, past statistics heavily suggest you have a cold. It assumes all symptoms act independently (which is why it's called 'naïve').")
        
        nb = GaussianNB()
        nb.fit(X_moon, y_moon)
        
        fig_nb = plot_decision_boundary(nb, X_moon, y_moon, "Naïve Bayes Decision Boundary")
        st.pyplot(fig_nb)

# ==========================================
# LESSON 3: DECISION TREES & SVM
# ==========================================
elif lesson == "3. Decision Trees & SVM":
    st.title("🌳 Decision Trees & Support Vector Machines (SVM)")
    
    X_moon, y_moon = make_moons(n_samples=200, noise=0.2, random_state=42)
    
    tab1, tab2 = st.tabs(["Decision Trees", "SVM"])
    
    with tab1:
        st.header("Decision Trees")
        st.info("**The Analogy:** Playing '20 Questions'. The algorithm asks a series of Yes/No questions to split the data (e.g., 'Is X > 0.5?' -> 'Is Y < -0.2?').")
        
        depth = st.slider("Max Questions Allowed (Max Depth)", 1, 15, 3)
        st.write("Notice: High depths result in 'blocky' boundaries that map exactly to the training dots (Overfitting).")
        
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_moon, y_moon)
        
        fig_tree = plot_decision_boundary(tree, X_moon, y_moon, f"Decision Tree (Depth={depth})")
        st.pyplot(fig_tree)
        
    with tab2:
        st.header("Support Vector Machines (SVM)")
        st.info("**The Analogy:** Imagine trying to build the widest possible multi-lane highway between two different cities. The wider the highway without hitting a house (data point), the better.")
        
        kernel = st.radio("Choose the Highway Shape (Kernel)", ["linear", "rbf (curved)"])
        st.write("Linear tries to draw a straight street. RBF curves the street to wrap around neighborhoods.")
        
        svm = SVC(kernel=kernel.split()[0], C=1.0)
        svm.fit(X_moon, y_moon)
        
        fig_svm = plot_decision_boundary(svm, X_moon, y_moon, f"SVM with {kernel} kernel")
        st.pyplot(fig_svm)

st.sidebar.markdown("---")
st.sidebar.success("💡 **Tip:** Play around with the sliders and see how the background colors (the model's brain) change based on your inputs!")