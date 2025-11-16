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

# Set Matplotlib style
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

# Streamlit config
st.set_page_config(
    page_title="Diamond Price Analysis",
    page_icon="💎",
    layout="wide"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>💎 Diamond Price Analysis & Prediction</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/diamond.png", width=100)
st.sidebar.title("Navigation")
st.sidebar.markdown(
    f"<div style='background-color: {colors['accent']['ivory']}; padding: 10px; border-radius: 5px;'>",
    unsafe_allow_html=True
)
page = st.sidebar.radio("Go to", [
    "Data Overview", "Exploratory Analysis", "Price vs Attributes",
    "Distribution Analysis", "Model Building", "Model Evaluation",
    "Feature Correlation", "Error Analysis", "Price Prediction"
])
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Session state init
for key in ['models', 'X_train', 'X_test', 'y_train', 'y_test', 'features',
            'current_model', 'scaler', 'feature_names']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'models' else {}

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("diamonds.csv")

df = load_data()

# ========================
#  DATA OVERVIEW SECTION
# ========================

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
        st.text(buffer.getvalue())

        st.write("### Missing Values")
        st.dataframe(df.isna().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

        st.write("### Summary Statistics")
        st.dataframe(df.describe())

# ================================
#  EXPLORATORY ANALYSIS SECTION
# ================================

elif page == "Exploratory Analysis":
    st.markdown("<h2 class='sub-header'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    feature = st.selectbox("Select Feature", ["cut", "color", "clarity"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x=df[feature],
        y=df['price'],
        ax=ax,
        palette=[colors['primary']['bordeaux'],
                 colors['primary']['plum'],
                 colors['primary']['merlot'],
                 colors['primary']['charcoal']]
    )
    plt.title(f'Price Distribution Across {feature.upper()}', fontsize=14, color=colors['primary']['bordeaux'])
    ax.set_xlabel(feature.capitalize())
    ax.set_ylabel("Price ($)")
    st.pyplot(fig)

    st.write(f"### Insights for {feature.upper()}")
    if feature == "cut":
        st.write("Higher cut quality generally yields higher prices due to better brilliance.")
    elif feature == "color":
        st.write("Colorless diamonds (D–F) typically command higher market prices.")
    else:
        st.write("Clarity strongly influences price, with IF/VVS being the most valuable.")
# ============================
#  PRICE vs ATTRIBUTES SECTION
# ============================

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
    plt.title(f'{attribute.upper()} vs PRICE', fontsize=14, color=colors['primary']['bordeaux'])
    ax.set_xlabel(attribute.capitalize())
    ax.set_ylabel('Price ($)')
    st.pyplot(fig)

    corr = df[attribute].corr(df['price'])
    st.write(f"### Correlation with Price: **{corr:.4f}**")

# ==============================
#  DISTRIBUTION ANALYSIS SECTION
# ==============================

elif page == "Distribution Analysis":
    st.markdown("<h2 class='sub-header'>Price Distribution Analysis</h2>", unsafe_allow_html=True)

    plot_type = st.radio("Select Plot Type", ["Distribution Plot", "Probability Density Function"])

    if plot_type == "Distribution Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(df['price'], color=colors['primary']['bordeaux'], fill=True, alpha=0.3)
        plt.axvline(df['price'].mean(), linestyle='--', color=colors['primary']['charcoal'])
        plt.title("Price Distribution")
        st.pyplot(fig)

    else:
        def prob(arr, mean, std):
            coeff = 1/(std*np.sqrt(2*np.pi))
            return coeff * np.exp(-((arr-mean)**2/(2*(std**2))))

        rho = prob(df["price"].sort_values(), df["price"].mean(), df["price"].std())
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(df["price"].sort_values(), rho, color=colors['primary']['bordeaux'], s=10)
        plt.title("Normal Distribution Curve")
        st.pyplot(fig)

# ============================
#     MODEL BUILDING SECTION
# ============================

elif page == "Model Building":
    st.markdown("<h2 class='sub-header'>Advanced Model Building</h2>", unsafe_allow_html=True)

    df_encoded = df.copy()

    # Encode categorical features
    df_encoded["cut"].replace({"Fair":1,"Good":2,"Very Good":3,"Premium":4,"Ideal":5}, inplace=True)
    df_encoded["color"].replace({"D":1,"E":2,"F":3,"G":4,"H":5,"I":6,"J":7}, inplace=True)
    df_encoded["clarity"].replace({"I1":1,"SI2":2,"SI1":3,"VS2":4,"VS1":5,"VVS2":6,"VVS1":7,"IF":8}, inplace=True)

    # Preprocessing selection
    model_type = st.radio("Feature Selection", ["Full Model (All Features)", "Reduced Model (Without x, y, z)"])
    if model_type == "Full Model (All Features)":
        features = list(df_encoded.columns)
        features.remove("price")
    else:
        df_encoded = df_encoded.drop(columns=["x","y","z"])
        features = list(df_encoded.columns)
        features.remove("price")

    scale_features = st.checkbox("Scale Features (Recommended for SVM)", value=True)
    use_cross_validation = st.checkbox("Use Cross-Validation", value=True)
    if use_cross_validation:
        cv_folds = st.slider("Number of CV Folds", 3, 10, 5)

    # Train-test split
    X = df_encoded[features]
    Y = df_encoded["price"]

    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.33)
    random_state = st.number_input("Random State", 0, 100, 42)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Scaling with feature name fix
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.session_state.scaler = scaler
        st.session_state.feature_names = list(X_train.columns)   # ⭐ FIX ADDED HERE
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        st.session_state.scaler = None
        st.session_state.feature_names = list(X_train.columns)

    # Store session data
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.features = features

    # Algorithm definitions
    algorithms = {
        "Linear Regression": {"model": LinearRegression(), "params": {}},

        "Decision Tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "max_depth": st.slider("DT Max Depth", 2, 30, 10),
                "min_samples_split": st.slider("DT Min Samples Split", 2, 20, 2),
                "min_samples_leaf": st.slider("DT Min Samples Leaf", 1, 20, 1)
            }
        },

        "Random Forest": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": st.slider("RF Estimators", 10, 200, 100),
                "max_depth": st.slider("RF Max Depth", 2, 30, 10),
                "min_samples_split": st.slider("RF Min Samples Split", 2, 20, 2)
            }
        },

        "Gradient Boosting": {
            "model": GradientBoostingRegressor(),
            "params": {
                "n_estimators": st.slider("GB Estimators", 10, 200, 100),
                "learning_rate": st.slider("GB Learning Rate", 0.01, 0.3, 0.1),
                "max_depth": st.slider("GB Max Depth", 2, 10, 3)
            }
        },

        "Support Vector Machine": {
            "model": SVR(),
            "params": {
                "C": st.slider("SVM C", 0.1, 10.0, 1.0),
                "kernel": st.selectbox("Kernel", ["linear","poly","rbf","sigmoid"]),
                "gamma": st.selectbox("Gamma", ["scale","auto"])
            }
        }
    }

    selected_algorithms = []
    st.write("### Select Models to Train")
    if st.checkbox("Linear Regression", True): selected_algorithms.append("Linear Regression")
    if st.checkbox("Decision Tree"): selected_algorithms.append("Decision Tree")
    if st.checkbox("Random Forest"): selected_algorithms.append("Random Forest")
    if st.checkbox("Gradient Boosting"): selected_algorithms.append("Gradient Boosting")
    if st.checkbox("Support Vector Machine"): selected_algorithms.append("Support Vector Machine")

    # Train model button
    if st.button("Train Selected Models"):
        st.session_state.models = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, algo in enumerate(selected_algorithms):
            status_text.text(f"Training {algo}...")

            model_obj = algorithms[algo]["model"]
            params = algorithms[algo]["params"]
            model_obj.set_params(**params)

            model_obj.fit(X_train_scaled, y_train)

            train_pred = model_obj.predict(X_train_scaled)
            test_pred = model_obj.predict(X_test_scaled)

            model_data = {
                "model": model_obj,
                "params": params,
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
                "cv_scores": cross_val_score(model_obj, X_train_scaled, y_train, cv=cv_folds)
                    if use_cross_validation else None
            }

            st.session_state.models[algo] = model_data
            progress_bar.progress((i+1) / len(selected_algorithms))

        st.success("All selected models trained successfully!")

# ============================
#     MODEL EVALUATION
# ============================

elif page == "Model Evaluation":
    st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)

    if not st.session_state.models:
        st.warning("Train at least one model first.")
    else:
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model", model_names)

        st.session_state.current_model = selected_model
        model_info = st.session_state.models[selected_model]
        model = model_info["model"]

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        # ⭐ FIX ADDED — reorder columns before scaling
        X_train = X_train[st.session_state.feature_names]
        X_test = X_test[st.session_state.feature_names]

        if st.session_state.scaler is not None:
            X_train_eval = st.session_state.scaler.transform(X_train)
            X_test_eval = st.session_state.scaler.transform(X_test)
        else:
            X_train_eval = X_train
            X_test_eval = X_test

        y_train_pred = model.predict(X_train_eval)
        y_test_pred = model.predict(X_test_eval)

        st.write("### Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Train R²", f"{r2_score(y_train, y_train_pred):.3f}")
            st.metric("Train RMSE", f"{np.sqrt(mean_squared_error(y_train, y_train_pred)):.3f}")

        with col2:
            st.metric("Test R²", f"{r2_score(y_test, y_test_pred):.3f}")
            st.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}")

# ============================
#      ERROR ANALYSIS
# ============================

elif page == "Error Analysis":
    st.markdown("<h2 class='sub-header'>Error Analysis</h2>", unsafe_allow_html=True)

    if not st.session_state.models:
        st.warning("Train a model first.")
    else:
        selected_model = st.session_state.current_model
        model_info = st.session_state.models[selected_model]
        model = model_info["model"]

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        # ⭐ FIX — reorder before scaling
        X_train = X_train[st.session_state.feature_names]
        X_test = X_test[st.session_state.feature_names]

        if st.session_state.scaler:
            X_train_scaled = st.session_state.scaler.transform(X_train)
            X_test_scaled = st.session_state.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        y_pred = model.predict(X_test_scaled)
        residuals = y_test - y_pred

        st.write("### Residual Distribution")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, bins=30, kde=True, color=colors['primary']['bordeaux'])
        plt.title(f"Residual Plot ({selected_model})")
        plt.xlabel("Residual Error")
        st.pyplot(fig)


# ============================
#      PRICE PREDICTION
# ============================

elif page == "Price Prediction":

    st.markdown("<h2 class='sub-header'>Diamond Price Prediction</h2>", unsafe_allow_html=True)

    if not st.session_state.models:
        st.warning("Please train a model first.")
    else:
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model for Prediction", model_names)

        st.write("### Enter Diamond Features")

        # User Inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            carat = st.number_input("Carat", 0.2, 5.0, 1.0)
            depth = st.number_input("Depth", 55.0, 70.0, 61.0)

        with col2:
            table = st.number_input("Table", 50.0, 75.0, 57.0)
            x = st.number_input("X (mm)", 0.0, 10.0, 5.5)

        with col3:
            y = st.number_input("Y (mm)", 0.0, 10.0, 5.4)
            z = st.number_input("Z (mm)", 0.0, 10.0, 3.5)

        cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

        if st.button("Predict Price"):
            model_info = st.session_state.models[selected_model]
            model = model_info["model"]

            # Encoding
            input_data = {
                "carat": carat,
                "depth": depth,
                "table": table,
                "x": x,
                "y": y,
                "z": z,
                "cut": {"Fair":1,"Good":2,"Very Good":3,"Premium":4,"Ideal":5}[cut],
                "color": {"D":1,"E":2,"F":3,"G":4,"H":5,"I":6,"J":7}[color],
                "clarity": {"I1":1,"SI2":2,"SI1":3,"VS2":4,"VS1":5,"VVS2":6,"VVS1":7,"IF":8}[clarity]
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # ⭐ FIX — reorder columns to match training order
            input_df = input_df[st.session_state.feature_names]

            # Scale if required
            if st.session_state.scaler is not None:
                input_scaled = st.session_state.scaler.transform(input_df)
            else:
                input_scaled = input_df

            # Predict
            predicted_price = model.predict(input_scaled)[0]

            st.success(f"### 💎 Predicted Price: **${predicted_price:,.2f}**")


# ================================
#      APPLICATION FOOTER
# ================================

st.markdown("<br><hr><center>© 2025 Diamond Price Predictor | Streamlit App</center>", unsafe_allow_html=True)

