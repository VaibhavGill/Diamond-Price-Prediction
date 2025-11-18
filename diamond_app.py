import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Define luxury color palette
colors = {
    'primary': {
        'merlot': '#730000',
        'plum': '#701C1C',
        'bordeaux': '#50222D',
        'charcoal': '#36454F'
    },
    'accent': {
        'gold': '#C5A880',
        'ivory': '#F0EAD6',
        'silver': '#C4C3D0',
        'cream': '#F5DEB3'
    },
    'graph': {
        'primary': ['#730000', '#701C1C', '#50222D', '#36454F'],
        'secondary': ['#C5A880', '#F0EAD6', '#C4C3D0', '#F5DEB3'],
        'highlight': '#C5A880'
    }
}

# Set Matplotlib style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['figure.facecolor'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = colors['primary']['charcoal']
plt.rcParams['axes.labelcolor'] = colors['primary']['charcoal']
plt.rcParams['xtick.color'] = colors['primary']['charcoal']
plt.rcParams['ytick.color'] = colors['primary']['charcoal']
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# Set page configuration
st.set_page_config(
    page_title="Diamond Price Analysis",
    page_icon="💎",
    layout="wide"
)

# Custom styling
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {colors['primary']['bordeaux']};
        text-align: center;
        font-weight: 600;
    }}
    .sub-header {{
        font-size: 1.5rem;
        color: {colors['primary']['plum']};
        font-weight: 500;
    }}
    .metric-container {{
        background-color: {colors['accent']['ivory']};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid {colors['primary']['bordeaux']};
    }}
    .algorithm-card {{
        background-color: {colors['accent']['ivory']};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid {colors['primary']['merlot']};
    }}
    .selected-algorithm {{
        border-left: 4px solid {colors['accent']['gold']};
        background-color: {colors['accent']['ivory']};
    }}
    .accuracy-card {{
        background-color: {colors['accent']['ivory']};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid {colors['accent']['gold']};
    }}
    .stApp {{
        background-color: #FFFFFF;
    }}
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 5px;
    }}
    div[data-testid="stMetric"] > div:first-child {{
        color: {colors['primary']['bordeaux']};
    }}
    div[data-testid="stMetric"] > div:nth-child(2) {{
        color: {colors['primary']['charcoal']};
    }}
    .stDataFrame {{
        border: 1px solid {colors['accent']['silver']};
        border-radius: 5px;
    }}
    .stButton > button {{
        background-color: {colors['primary']['bordeaux']};
        color: white;
        border: none;
    }}
    .stButton > button:hover {{
        background-color: {colors['primary']['merlot']};
    }}
    .stSelectbox > div > div {{
        background-color: {colors['accent']['ivory']};
    }}
    .stSlider > div > div > div {{
        background-color: {colors['primary']['bordeaux']};
    }}
    .stProgress > div > div > div {{
        background-color: {colors['primary']['bordeaux']};
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>💎 Diamond Price Analysis & Prediction</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/diamond.png", width=100)
st.sidebar.title("Navigation")
st.sidebar.markdown(f"<div style='background-color: {colors['accent']['ivory']}; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Price vs Attributes", 
                                  "Distribution Analysis", "Model Building", "Model Evaluation", 
                                  "Feature Correlation", "Error Analysis", "Price Prediction"])
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Initialize session state if not already done
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Load data
@st.cache_data
@st.cache_data
def load_data():
    df = pd.read_csv('https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/diamonds.csv')
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    df = pd.read_csv("diamonds.csv")
    return df


df = load_data()

# Data Overview
if page == "Data Overview":
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Sample Data")
        st.dataframe(df.head())

    with col2:
        st.write("### Dataset Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write("### Missing Values")
        st.dataframe(df.isna().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

        st.write("### Summary Statistics")
        st.dataframe(df.describe())

# Exploratory Analysis
elif page == "Exploratory Analysis":
    st.markdown("<h2 class='sub-header'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    st.write("### Categorical Features vs Price")

    feature = st.selectbox("Select Feature", ["cut", "color", "clarity"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[feature], y=df['price'], ax=ax, palette=[colors['primary']['bordeaux'], colors['primary']['plum'], 
                                                             colors['primary']['merlot'], colors['primary']['charcoal']])
    plt.title(f'Boxplot between "{feature.upper()}" and "PRICE"', fontsize=14, color=colors['primary']['bordeaux'])
    ax.set_xlabel(feature.capitalize(), fontsize=12, color=colors['primary']['charcoal'])
    ax.set_ylabel('Price ($)', fontsize=12, color=colors['primary']['charcoal'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write(f"### Insights for {feature.upper()} vs Price")
    if feature == "cut":
        st.write("The boxplot shows how diamond prices vary across different cut qualities. Ideal cuts generally command higher prices due to better light reflection and brilliance.")
    elif feature == "color":
        st.write("Diamond color grades from D (colorless) to J (slight color). Colorless diamonds (D-F) typically have higher prices than those with visible color.")
    else:  # clarity
        st.write("Clarity measures the absence of inclusions and blemishes. Higher clarity grades (IF, VVS1, VVS2) generally result in higher prices.")

# Price vs Attributes
elif page == "Price vs Attributes":
    st.markdown("<h2 class='sub-header'>Price vs Numerical Attributes</h2>", unsafe_allow_html=True)

    attributes = ["carat", "depth", "table", "x", "y", "z"]
    attribute = st.selectbox("Select Attribute", attributes)

    attribute_colors = {
        "carat": colors['primary']['merlot'],
        "depth": colors['primary']['plum'],
        "table": colors['primary']['bordeaux'],
        "x": colors['primary']['charcoal'],
        "y": colors['accent']['gold'],
        "z": colors['primary']['bordeaux']
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(df[attribute], df['price'], color=attribute_colors[attribute], alpha=0.6)
    plt.title(f'Relationship between {attribute.upper()} and PRICE', fontsize=14, color=colors['primary']['bordeaux'])
    ax.set_xlabel(attribute.capitalize(), fontsize=12, color=colors['primary']['charcoal'])
    ax.set_ylabel('Price ($)', fontsize=12, color=colors['primary']['charcoal'])
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Add correlation coefficient
    corr = df[attribute].corr(df['price'])
    st.markdown(f"<div style='background-color: {colors['accent']['ivory']}; padding: 15px; border-radius: 10px; border-left: 4px solid {colors['primary']['bordeaux']}'>", unsafe_allow_html=True)
    st.write(f"### Correlation with Price: {corr:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    if attribute == "carat":
        st.write("Carat weight has the strongest positive correlation with price. As carat weight increases, price increases exponentially.")
    elif attribute in ["x", "y", "z"]:
        st.write("Diamond dimensions (x, y, z) are highly correlated with carat weight and consequently with price.")

# Distribution Analysis
elif page == "Distribution Analysis":
    st.markdown("<h2 class='sub-header'>Price Distribution Analysis</h2>", unsafe_allow_html=True)

    plot_type = st.radio("Select Plot Type", ["Distribution Plot", "Probability Density Function"])

    if plot_type == "Distribution Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(df['price'], color=colors['primary']['bordeaux'], fill=True, alpha=0.3, ax=ax)
        plt.axvline(df['price'].mean(), label=f'Mean price = {df["price"].mean():.3f}', color=colors['primary']['charcoal'], linestyle='--')
        plt.title('Price Distribution', fontsize=14, color=colors['primary']['bordeaux'])
        ax.set_xlabel('Price ($)', fontsize=12, color=colors['primary']['charcoal'])
        ax.set_ylabel('Density', fontsize=12, color=colors['primary']['charcoal'])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    else:
        # Create a probability density function for plotting the normal distribution
        def prob(arr, mean, std):
            coeff = 1/(std*np.sqrt(2*np.pi))
            power = np.exp(-((arr-mean)**2/(2*(std**2))))
            return coeff*power

        # Plot the normal distribution curve using plt.scatter()
        rho = prob(df["price"].sort_values(), df["price"].mean(), df["price"].std())
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(df["price"].sort_values(), rho, color=colors['primary']['bordeaux'], alpha=0.6, s=10)
        plt.axvline(x=df["price"].mean(), label="Mean price", color=colors['primary']['charcoal'], linestyle='--')
        plt.title('Normal Distribution Curve', fontsize=14, color=colors['primary']['bordeaux'])
        ax.set_xlabel('Price ($)', fontsize=12, color=colors['primary']['charcoal'])
        ax.set_ylabel('Probability Density', fontsize=12, color=colors['primary']['charcoal'])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    st.write("### Price Distribution Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"${df['price'].mean():.2f}")
    col2.metric("Median", f"${df['price'].median():.2f}")
    col3.metric("Std Dev", f"${df['price'].std():.2f}")
    col4.metric("Range", f"${df['price'].max() - df['price'].min():.2f}")

# Model Building
elif page == "Model Building":
    st.markdown("<h2 class='sub-header'>Advanced Model Building</h2>", unsafe_allow_html=True)

    # Create a copy of the dataframe for encoding
    df_encoded = df.copy()

    with st.expander("View Feature Encoding"):
        st.write("### Encoding Categorical Features")

        st.code("""
# Replace values of 'cut' column
df["cut"].replace({"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}, inplace=True)

# Replace values of 'color' column
df["color"].replace({"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}, inplace=True)

# Replace values of 'clarity' column
df["clarity"].replace({"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}, inplace=True)
        """)

        # Encode categorical features
        df_encoded["cut"].replace({"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}, inplace=True)
        df_encoded["color"].replace({"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}, inplace=True)
        df_encoded["clarity"].replace({"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}, inplace=True)

        st.dataframe(df_encoded.head())

    # Data preprocessing options
    st.markdown(f"<div style='background-color: {colors['accent']['ivory']}; padding: 15px; border-radius: 10px;'>", unsafe_allow_html=True)
    st.write("### Data Preprocessing Options")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.radio("Feature Selection", ["Full Model (All Features)", "Reduced Model (Without x, y, z)"])

        if model_type == "Full Model (All Features)":
            features = list(df_encoded.columns)
            features.remove('price')
        else:
            # Drop features highly correlated with 'carat'
            df_encoded_reduced = df_encoded.drop(columns=['x', 'y', 'z'])
            features = list(df_encoded_reduced.columns)
            features.remove('price')
            df_encoded = df_encoded_reduced

    with col2:
        scale_features = st.checkbox("Scale Features (Recommended for SVM and some algorithms)", value=True)
        use_cross_validation = st.checkbox("Use Cross-Validation", value=True)
        if use_cross_validation:
            cv_folds = st.slider("Number of CV Folds", 3, 10, 5)

    st.markdown("</div>", unsafe_allow_html=True)

    # Split data
    X = df_encoded[features]
    Y = df_encoded['price']

    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.33, 0.01)
    random_state = st.number_input("Random State", 0, 100, 42)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Scale features if selected
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Store scaler for later use
        st.session_state.scaler = scaler
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        st.session_state.scaler = None

    # Store data in session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.features = features

    st.write(f"### Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Algorithm selection
    st.markdown(f"<div style='background-color: {colors['accent']['ivory']}; padding: 15px; border-radius: 10px;'>", unsafe_allow_html=True)
    st.write("### Select Algorithms to Train")

    # Define available algorithms with their parameters
    algorithms = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {}
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "max_depth": st.slider("Max depth", 2, 30, 10, key="dt_depth"),
                "min_samples_split": st.slider("Min samples split", 2, 20, 2, key="dt_split"),
                "min_samples_leaf": st.slider("Min samples leaf", 1, 20, 1, key="dt_leaf")
            }
        },
        "Random Forest": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": st.slider("Number of trees", 10, 200, 100, 10, key="rf_trees"),
                "max_depth": st.slider("Max depth", 2, 30, 10, key="rf_depth"),
                "min_samples_split": st.slider("Min samples split", 2, 20, 2, key="rf_split")
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(),
            "params": {
                "n_estimators": st.slider("Number of estimators", 10, 200, 100, 10, key="gb_trees"),
                "learning_rate": st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01, key="gb_lr"),
                "max_depth": st.slider("Max depth", 2, 10, 3, key="gb_depth")
            }
        },
        "Support Vector Machine": {
            "model": SVR(),
            "params": {
                "C": st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1, key="svm_c"),
                "kernel": st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key="svm_kernel"),
                "gamma": st.selectbox("Gamma", ["scale", "auto"], key="svm_gamma")
            }
        }
    }

    # Let user select which algorithms to train
    selected_algorithms = []

    col1, col2 = st.columns(2)

    with col1:
        if st.checkbox("Linear Regression", value=True):
            selected_algorithms.append("Linear Regression")
        if st.checkbox("Decision Tree"):
            selected_algorithms.append("Decision Tree")
        if st.checkbox("Random Forest"):
            selected_algorithms.append("Random Forest")

    with col2:
        if st.checkbox("Gradient Boosting"):
            selected_algorithms.append("Gradient Boosting")
        if st.checkbox("Support Vector Machine"):
            selected_algorithms.append("Support Vector Machine")

    st.markdown("</div>", unsafe_allow_html=True)

    # Train models
    if st.button("Train Selected Models"):
        if not selected_algorithms:
            st.error("Please select at least one algorithm to train.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Clear previous models if any
            st.session_state.models = {}

            for i, algo_name in enumerate(selected_algorithms):
                status_text.text(f"Training {algo_name}...")

                # Get algorithm and parameters
                algo_info = algorithms[algo_name]
                model = algo_info["model"]
                params = algo_info["params"]

                # Set parameters
                model.set_params(**params)

                # Train model
                model.fit(X_train_scaled, y_train)

                # Make predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)

                # Calculate metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

                # Cross-validation if selected
                cv_scores = None
                if use_cross_validation:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')

                # Store model and metrics
                st.session_state.models[algo_name] = {
                    "model": model,
                    "params": params,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "cv_scores": cv_scores
                }

                # Update progress
                progress_bar.progress((i + 1) / len(selected_algorithms))

            status_text.text("All models trained successfully!")
            st.session_state.current_model = selected_algorithms[0]

            # Show success message
            st.success(f"Successfully trained {len(selected_algorithms)} models!")

    # Display trained models if any
    if st.session_state.models:
        st.write("### Trained Models Performance")

        # Create a dataframe to compare models
        model_comparison = []

        for algo_name, model_info in st.session_state.models.items():
            model_data = {
                "Algorithm": algo_name,
                "Train R²": f"{model_info['train_r2']:.4f}",
                "Test R²": f"{model_info['test_r2']:.4f}",
                "Train RMSE": f"${model_info['train_rmse']:.2f}",
                "Test RMSE": f"${model_info['test_rmse']:.2f}"
            }

            if model_info['cv_scores'] is not None:
                model_data["CV R² (mean)"] = f"{model_info['cv_scores'].mean():.4f}"
                model_data["CV R² (std)"] = f"{model_info['cv_scores'].std():.4f}"

            model_comparison.append(model_data)

        # Display comparison table
        st.dataframe(pd.DataFrame(model_comparison).set_index("Algorithm"))

        # Plot model comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        model_names = list(st.session_state.models.keys())
        train_scores = [st.session_state.models[name]["train_r2"] for name in model_names]
        test_scores = [st.session_state.models[name]["test_r2"] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        ax.bar(x - width/2, train_scores, width, label='Train R²', color=colors['primary']['bordeaux'], alpha=0.8)
        ax.bar(x + width/2, test_scores, width, label='Test R²', color=colors['accent']['gold'], alpha=0.8)

        ax.set_ylabel('R² Score', fontsize=12, color=colors['primary']['charcoal'])
        ax.set_title('Model Performance Comparison', fontsize=14, color=colors['primary']['bordeaux'])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)

        # Feature importance for tree-based models
        tree_models = ["Decision Tree", "Random Forest", "Gradient Boosting"]
        available_tree_models = [m for m in tree_models if m in st.session_state.models]

        if available_tree_models:
            st.write("### Feature Importance")

            selected_model = st.selectbox("Select model for feature importance", available_tree_models)

            if selected_model in st.session_state.models:
                model_info = st.session_state.models[selected_model]
                model = model_info["model"]

                # Get feature importance
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.bar(range(X_train.shape[1]), importances[indices], align='center', 
                       color=[colors['primary']['bordeaux'], colors['primary']['plum'], colors['accent']['gold'], 
                             colors['primary']['charcoal']])
                plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
                plt.title(f'Feature Importance - {selected_model}', fontsize=14, color=colors['primary']['bordeaux'])
                ax.set_xlabel('Features', fontsize=12, color=colors['primary']['charcoal'])
                ax.set_ylabel('Importance', fontsize=12, color=colors['primary']['charcoal'])
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)

# Model Evaluation
elif page == "Model Evaluation":
    st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)

    if not st.session_state.models:
        st.warning("Please train at least one model first in the 'Model Building' section.")
    else:
        # Let user select which model to evaluate
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model to Evaluate", model_names, 
                                     index=model_names.index(st.session_state.current_model) if st.session_state.current_model in model_names else 0)

        # Update current model
        st.session_state.current_model = selected_model

        # Get model info
        model_info = st.session_state.models[selected_model]
        model = model_info["model"]

        # Get data
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        # Apply scaling if needed
        if st.session_state.scaler is not None:
            X_train_eval = st.session_state.scaler.transform(X_train)
            X_test_eval = st.session_state.scaler.transform(X_test)
        else:
            X_train_eval = X_train
            X_test_eval = X_test

        # Make predictions
        y_train_pred = model.predict(X_train_eval)
        y_test_pred = model.predict(X_test_eval)

        # Display metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.write("### Training Set Metrics")
            st.metric("R-squared", f"{model_info['train_r2']:.3f}")
            st.metric("Root Mean Squared Error", f"{model_info['train_rmse']:.3f}")
            st.metric("Mean Absolute Error", f"{mean_absolute_error(y_train, y_train_pred):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.write("### Test Set Metrics")
            st.metric("R-squared", f"{model_info['test_r2']:.3f}")
            st.metric("Root Mean Squared Error", f"{model_info['test_rmse']:.3f}")
            st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_test_pred):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Cross-validation results if available
        if model_info['cv_scores'] is not None:
            st.write("### Cross-Validation Results")
            cv_scores = model_info['cv_scores']

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean R²", f"{cv_scores.mean():.3f}")
            col2.metric("Std Dev", f"{cv_scores.std():.3f}")
            col3.metric("Min/Max", f"{cv_scores.min():.3f} / {cv_scores.max():.3f}")

            # Plot CV scores
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(cv_scores) + 1), cv_scores, 'o-', color=colors['primary']['bordeaux'], label='R² score')
            ax.axhline(y=cv_scores.mean(), color=colors['primary']['charcoal'], linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
            ax.set_xlabel('Fold', fontsize=12, color=colors['primary']['charcoal'])
            ax.set_ylabel('R² Score', fontsize=12, color=colors['primary']['charcoal'])
            ax.set_title('Cross-Validation Scores', fontsize=14, color=colors['primary']['bordeaux'])
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

        # Actual vs Predicted Plot
        st.write("### Actual vs Predicted Prices")

        dataset = st.radio("Select Dataset", ["Training Set", "Test Set"])

        if dataset == "Training Set":
            actual = y_train
            predicted = y_train_pred
        else:
            actual = y_test
            predicted = y_test_pred

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.6, color=colors['primary']['bordeaux'], s=30)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], '--', color=colors['primary']['charcoal'])
        plt.xlabel('Actual Price', fontsize=12, color=colors['primary']['charcoal'])
        plt.ylabel('Predicted Price', fontsize=12, color=colors['primary']['charcoal'])
        plt.title(f'Actual vs Predicted Prices ({dataset})', fontsize=14, color=colors['primary']['bordeaux'])
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Residual plot
        st.write("### Residual Analysis")

        residuals = actual - predicted

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Residual scatter plot
        ax1.scatter(predicted, residuals, alpha=0.6, color=colors['primary']['bordeaux'], s=30)
        ax1.axhline(y=0, color=colors['primary']['charcoal'], linestyle='--')
        ax1.set_xlabel('Predicted Price', fontsize=12, color=colors['primary']['charcoal'])
        ax1.set_ylabel('Residuals', fontsize=12, color=colors['primary']['charcoal'])
        ax1.set_title('Residuals vs Predicted', fontsize=14, color=colors['primary']['bordeaux'])
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Residual histogram
        ax2.hist(residuals, bins=30, alpha=0.7, color=colors['accent']['gold'])
        ax2.axvline(x=0, color=colors['primary']['charcoal'], linestyle='--')
        ax2.set_xlabel('Residual Value', fontsize=12, color=colors['primary']['charcoal'])
        ax2.set_ylabel('Frequency', fontsize=12, color=colors['primary']['charcoal'])
        ax2.set_title('Residual Distribution', fontsize=14, color=colors['primary']['bordeaux'])
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)

# Feature Correlation
elif page == "Feature Correlation":
    st.markdown("<h2 class='sub-header'>Feature Correlation Analysis</h2>", unsafe_allow_html=True)

    # Create a copy of the dataframe for encoding
    df_encoded = df.copy()

    # Encode categorical features
    df_encoded["cut"].replace({"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}, inplace=True)
    df_encoded["color"].replace({"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}, inplace=True)
    df_encoded["clarity"].replace({"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}, inplace=True)

    # Correlation heatmap
    st.write("### Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(df_encoded.corr(), annot=True, cmap=cmap, center=0, ax=ax)
    plt.title('Feature Correlation Heatmap', fontsize=14, color=colors['primary']['bordeaux'])
    st.pyplot(fig)

    # VIF calculation
    if st.session_state.X_train is not None:
        st.write("### Variance Inflation Factor (VIF)")

        X_train = st.session_state.X_train

        # Add a constant to feature variables
        X_train_sm = sm.add_constant(X_train)

        # Create a dataframe that will contain the names of the feature variables and their respective VIFs
        VIF = pd.DataFrame()
        VIF['features'] = X_train_sm.columns
        VIF['vif'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.values.shape[1])]
        VIF['vif'] = round(VIF['vif'], 2)
        VIF = VIF.sort_values(by='vif', ascending=False)

        st.dataframe(VIF)

        st.markdown(f"<div style='background-color: {colors['accent']['ivory']}; padding: 15px; border-radius: 10px; border-left: 4px solid {colors['primary']['bordeaux']}'>", unsafe_allow_html=True)
        st.write("""
        ### VIF Interpretation
        - VIF > 10: High multicollinearity
        - 5 < VIF < 10: Moderate multicollinearity
        - VIF < 5: Low multicollinearity
        
        Features with high VIF values may be redundant and could potentially be removed to simplify the model.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Error Analysis
elif page == "Error Analysis":
    st.markdown("<h2 class='sub-header'>Error Analysis</h2>", unsafe_allow_html=True)

    if not st.session_state.models:
        st.warning("Please train at least one model first in the 'Model Building' section.")
    else:
        # Let user select which model to analyze
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model to Analyze", model_names, 
                                     index=model_names.index(st.session_state.current_model) if st.session_state.current_model in model_names else 0)

        # Update current model
        st.session_state.current_model = selected_model

        # Get model info
        model_info = st.session_state.models[selected_model]
        model = model_info["model"]

        # Get data
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        # Apply scaling if needed
        if st.session_state.scaler is not None:
            X_train_eval = st.session_state.scaler.transform(X_train)
            X_test_eval = st.session_state.scaler.transform(X_test)
        else:
            X_train_eval = X_train
            X_test_eval = X_test

        # Make predictions
        y_train_pred = model.predict(X_train_eval)
        y_test_pred = model.predict(X_test_eval)

        # Calculate errors
        train_errors = y_train - y_train_pred
        test_errors = y_test - y_test_pred

        # Error distribution
        dataset = st.radio("Select Dataset", ["Training Set", "Test Set"], key="error_dataset")

        if dataset == "Training Set":
            errors = train_errors
            y_actual = y_train
            predictions = y_train_pred
        else:
            errors = test_errors
            y_actual = y_test
            predictions = y_test_pred

        st.write(f"### Error Distribution ({dataset})")

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.hist(errors, bins='sturges', color=colors['primary']['bordeaux'], alpha=0.7)
        plt.axvline(errors.mean(), color=colors['primary']['charcoal'], linestyle='--', label=f'Mean Error: {errors.mean():.2f}')
        plt.title(f'Error Distribution in {dataset}', fontsize=14, color=colors['primary']['bordeaux'])
        plt.xlabel('Error Value', fontsize=12, color=colors['primary']['charcoal'])
        plt.ylabel('Frequency', fontsize=12, color=colors['primary']['charcoal'])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.write(f"### Error vs Actual Price ({dataset})")

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(errors, y_actual, color=colors['primary']['bordeaux'], alpha=0.6, s=30)
        plt.axhline(y=0, color=colors['primary']['charcoal'], linestyle='-', alpha=0.3)
        plt.title(f'Errors vs Actual Price ({dataset})', fontsize=14, color=colors['primary']['bordeaux'])
        plt.xlabel('Errors', fontsize=12, color=colors['primary']['charcoal'])
        plt.ylabel('Actual Price', fontsize=12, color=colors['primary']['charcoal'])
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Error statistics
        st.write(f"### Error Statistics ({dataset})")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Error", f"{errors.mean():.2f}")
        col2.metric("Median Error", f"{np.median(errors):.2f}")
        col3.metric("Std Dev", f"{errors.std():.2f}")
        col4.metric("Max Abs Error", f"{np.abs(errors).max():.2f}")

        # Error by price range
        st.write("### Error Analysis by Price Range")

        # Create price bins
        price_bins = [0, 1000, 2000, 5000, 10000, 20000]
        price_labels = ['0-1K', '1K-2K', '2K-5K', '5K-10K', '10K+']

        # Assign bins
        y_binned = pd.cut(y_actual, bins=price_bins, labels=price_labels, right=False)

        # Calculate mean absolute error by bin
        error_by_bin = pd.DataFrame({
            'Actual': y_actual,
            'Predicted': predictions,
            'Error': errors,
            'Abs_Error': np.abs(errors),
            'Price_Bin': y_binned
        })

        bin_stats = error_by_bin.groupby('Price_Bin').agg({
            'Abs_Error': ['mean', 'median', 'std', 'count'],
            'Error': ['mean']
        })

        bin_stats.columns = ['MAE', 'Median AE', 'Std Dev', 'Count', 'Mean Error']

        st.dataframe(bin_stats)

        # Plot MAE by price bin
        fig, ax = plt.subplots(figsize=(10, 6))
        bin_stats['MAE'].plot(kind='bar', ax=ax, color=colors['primary']['bordeaux'])
        plt.title('Mean Absolute Error by Price Range', fontsize=14, color=colors['primary']['bordeaux'])
        plt.ylabel('Mean Absolute Error ($)', fontsize=12, color=colors['primary']['charcoal'])
        plt.xlabel('Price Range', fontsize=12, color=colors['primary']['charcoal'])
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

# Price Prediction
elif page == "Price Prediction":
    st.markdown("<h2 class='sub-header'>Diamond Price Prediction</h2>", unsafe_allow_html=True)

    if not st.session_state.models:
        st.warning("Please train at least one model first in the 'Model Building' section.")
    else:
        # Let user select which model to use for prediction
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model for Prediction", model_names, 
                                     index=model_names.index(st.session_state.current_model) if st.session_state.current_model in model_names else 0)

        # Update current model
        st.session_state.current_model = selected_model

        # Get model info
        model_info = st.session_state.models[selected_model]
        model = model_info["model"]
        features = st.session_state.features

        st.markdown(f"<div style='background-color: {colors['accent']['ivory']}; padding: 15px; border-radius: 10px; border-left: 4px solid {colors['primary']['bordeaux']}'>", unsafe_allow_html=True)
        st.write(f"### Using {selected_model} for Prediction")
        st.write(f"Test R² Score: {model_info['test_r2']:.4f} | Test RMSE: ${model_info['test_rmse']:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("### Enter Diamond Characteristics")

        col1, col2 = st.columns(2)

        input_data = {}

        with col1:
            if "carat" in features:
                carat = st.number_input("Carat Weight", 0.2, 5.0, 1.0, 0.1)
                input_data["carat"] = carat

            if "cut" in features:
                cut_options = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
                cut = st.selectbox("Cut Quality", list(cut_options.keys()))
                input_data["cut"] = cut_options[cut]

            if "color" in features:
                color_options = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
                color = st.selectbox("Color Grade", list(color_options.keys()))
                input_data["color"] = color_options[color]

            if "clarity" in features:
                clarity_options = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
                clarity = st.selectbox("Clarity Grade", list(clarity_options.keys()))
                input_data["clarity"] = clarity_options[clarity]

        with col2:
            if "depth" in features:
                depth = st.number_input("Depth Percentage", 50.0, 80.0, 61.5, 0.1)
                input_data["depth"] = depth

            if "table" in features:
                table = st.number_input("Table Percentage", 50.0, 80.0, 57.0, 0.1)
                input_data["table"] = table

            if "x" in features:
                x = st.number_input("Length (mm)", 0.0, 10.0, 5.0, 0.1)
                input_data["x"] = x

            if "y" in features:
                y = st.number_input("Width (mm)", 0.0, 10.0, 5.0, 0.1)
                input_data["y"] = y

            if "z" in features:
                z = st.number_input("Depth (mm)", 0.0, 10.0, 3.0, 0.1)
                input_data["z"] = z

        # Create input data for prediction
        input_df = pd.DataFrame([input_data])

        # Ensure all features are present in input_df
        for feature in features:
            if feature not in input_df.columns:
                st.error(f"Missing feature: {feature}. Please check your input data.")

        if st.button("Predict Price"):
            # Apply scaling if needed
            if st.session_state.scaler is not None:
                input_scaled = st.session_state.scaler.transform(input_df)
            else:
                input_scaled = input_df

            # Make prediction with selected model
            predicted_price = model.predict(input_scaled)[0]

            # Find similar diamonds for accuracy estimation
            similar_diamonds = None
            accuracy_estimate = None

            if "carat" in df.columns:
                # Find diamonds with similar characteristics
                similar_diamonds = df.copy()

                # Filter by carat (within 10%)
                similar_diamonds = similar_diamonds[(similar_diamonds['carat'] >= carat*0.9) & 
                                                   (similar_diamonds['carat'] <= carat*1.1)]

                # Filter by cut if possible
                if "cut" in similar_diamonds.columns and cut in ["Fair", "Good", "Very Good", "Premium", "Ideal"]:
                    similar_diamonds = similar_diamonds[similar_diamonds['cut'] == cut]

                # Filter by color if possible
                if "color" in similar_diamonds.columns and color in ["D", "E", "F", "G", "H", "I", "J"]:
                    similar_diamonds = similar_diamonds[similar_diamonds['color'] == color]

                # Filter by clarity if possible
                if "clarity" in similar_diamonds.columns:
                    similar_diamonds = similar_diamonds[similar_diamonds['clarity'] == clarity]

                # Calculate accuracy metrics if we have similar diamonds
                if not similar_diamonds.empty and len(similar_diamonds) >= 5:
                    # Calculate mean and std of prices for similar diamonds
                    mean_price = similar_diamonds['price'].mean()
                    std_price = similar_diamonds['price'].std()

                    # Calculate prediction error percentage
                    error_percent = abs(predicted_price - mean_price) / mean_price * 100

                    # Estimate accuracy based on how close our prediction is to the mean of similar diamonds
                    if error_percent <= 5:
                        accuracy_estimate = "Very High (±5%)"
                    elif error_percent <= 10:
                        accuracy_estimate = "High (±10%)"
                    elif error_percent <= 15:
                        accuracy_estimate = "Moderate (±15%)"
                    elif error_percent <= 25:
                        accuracy_estimate = "Fair (±25%)"
                    else:
                        accuracy_estimate = "Low (>25%)"

            # Display prediction with accuracy information
            st.markdown(f"""
            <div style="background-color: {colors['accent']['ivory']}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px; border: 1px solid {colors['primary']['bordeaux']}">
                <h2 style="color: {colors['primary']['bordeaux']};">Predicted Diamond Price</h2>
                <h1 style="color: {colors['primary']['bordeaux']}; font-size: 3rem;">${predicted_price:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Display accuracy information
            st.markdown(f"<div class='accuracy-card'>", unsafe_allow_html=True)
            st.write("### Prediction Accuracy Information")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Model R² Score", f"{model_info['test_r2']:.4f}")
                st.metric("Model RMSE", f"${model_info['test_rmse']:.2f}")

                if model_info['cv_scores'] is not None:
                    st.metric("Cross-Validation R²", f"{model_info['cv_scores'].mean():.4f} ± {model_info['cv_scores'].std():.4f}")

            with col2:
                # Display confidence based on model performance and similar diamonds
                if accuracy_estimate:
                    st.metric("Estimated Accuracy", accuracy_estimate)

                    # Calculate 95% confidence interval based on similar diamonds
                    mean_price = similar_diamonds['price'].mean()
                    std_price = similar_diamonds['price'].std()
                    n = len(similar_diamonds)

                    # Use t-distribution for small samples
                    confidence = 0.95
                    degrees_freedom = n - 1
                    t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)

                    margin_error = t_value * (std_price / np.sqrt(n))
                    lower_bound = max(0, mean_price - margin_error)
                    upper_bound = mean_price + margin_error

                    st.metric("Price Range (95% CI)", f"${lower_bound:.2f} - ${upper_bound:.2f}")
                    st.metric("Similar Diamonds Found", f"{n}")
                else:
                    st.write("Insufficient similar diamonds found to estimate accuracy.")
                    st.metric("Model Confidence", "Based on test R² score")

            st.markdown("</div>", unsafe_allow_html=True)

            # If multiple models are trained, show comparison
            if len(st.session_state.models) > 1:
                st.write("### Price Predictions from All Models")

                comparison_data = []

                for name, info in st.session_state.models.items():
                    model_obj = info["model"]

                    # Apply scaling if needed
                    if st.session_state.scaler is not None:
                        input_scaled = st.session_state.scaler.transform(input_df)
                    else:
                        input_scaled = input_df

                    # Make prediction
                    pred = model_obj.predict(input_scaled)[0]

                    comparison_data.append({
                        "Model": name,
                        "Predicted Price": f"${pred:.2f}",
                        "Test R²": f"{info['test_r2']:.4f}",
                        "Test RMSE": f"${info['test_rmse']:.2f}"
                    })

                st.dataframe(pd.DataFrame(comparison_data).set_index("Model"))

                # Plot comparison
                fig, ax = plt.subplots(figsize=(10, 6))

                model_names = [data["Model"] for data in comparison_data]
                predictions = [float(data["Predicted Price"].replace("$", "")) for data in comparison_data]

                bars = ax.bar(model_names, predictions, color=colors['primary']['plum'], alpha=0.8)

                # Highlight selected model
                selected_index = model_names.index(selected_model)
                bars[selected_index].set_color(colors['accent']['gold'])

                plt.title('Price Predictions by Different Models', fontsize=14, color=colors['primary']['bordeaux'])
                plt.ylabel('Predicted Price ($)', fontsize=12, color=colors['primary']['charcoal'])
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)

                # Add price labels on top of bars
                for i, v in enumerate(predictions):
                    ax.text(i, v + 100, f"${v:.2f}", ha='center', fontweight='bold', color=colors['primary']['charcoal'])

                plt.tight_layout()
                st.pyplot(fig)

            # Show comparable diamonds
            if similar_diamonds is not None and not similar_diamonds.empty:
                st.write("### Similar Diamonds in Dataset")
                st.dataframe(similar_diamonds.head(5))
            else:
                st.write("No similar diamonds found in the dataset.")
