import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set page configuration
st.set_page_config(
    page_title="Diamond Price Analysis",
    page_icon="💎",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6633ff;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>💎 Diamond Price Analysis & Prediction</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/diamond.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Price vs Attributes", 
                                  "Distribution Analysis", "Model Building", "Model Evaluation", 
                                  "Feature Correlation", "Error Analysis", "Price Prediction"])

# Initialize session state if not already done
if 'model' not in st.session_state:
    st.session_state.model = None
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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/diamonds.csv')
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
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
    sns.boxplot(x=df[feature], y=df['price'], ax=ax)
    plt.title(f'Boxplot between "{feature.upper()}" and "PRICE"')
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
    
    colors = {
        "carat": "#3875E3", "depth": "#A038E3", "table": "#38E37A",
        "x": "#C13481", "y": "#C13443", "z": "#C18534"
    }
    markers = {"carat": "*", "depth": "+", "table": "+", "x": "*", "y": "*", "z": "*"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(df[attribute], df['price'], color=colors[attribute], marker=markers[attribute])
    plt.title(f'Scatter plot between "{attribute.upper()}" and "PRICE"')
    st.pyplot(fig)
    
    # Add correlation coefficient
    corr = df[attribute].corr(df['price'])
    st.write(f"### Correlation with Price: {corr:.4f}")
    
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
        sns.distplot(df['price'], bins='sturges', hist=False, color='#34C1B7', ax=ax)
        plt.axvline(df['price'].mean(), label=f'Mean price = {df["price"].mean():.3f}', color='#000000')
        plt.title('Normal Distribution Curve for the "price"')
        plt.legend()
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
        plt.scatter(df["price"].sort_values(), rho)
        plt.axvline(x=df["price"].mean(), label="mean of price", color='#000000')
        plt.title('Normal Distribution curve using "plt.scatter()"')
        plt.legend()
        st.pyplot(fig)
    
    st.write("### Price Distribution Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"${df['price'].mean():.2f}")
    col2.metric("Median", f"${df['price'].median():.2f}")
    col3.metric("Std Dev", f"${df['price'].std():.2f}")
    col4.metric("Range", f"${df['price'].max() - df['price'].min():.2f}")

# Model Building
elif page == "Model Building":
    st.markdown("<h2 class='sub-header'>Linear Regression Model Building</h2>", unsafe_allow_html=True)
    
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
    
    # Model selection
    model_type = st.radio("Select Model Type", ["Full Model (All Features)", "Reduced Model (Without x, y, z)"])
    
    if model_type == "Full Model (All Features)":
        features = list(df_encoded.columns)
        features.remove('price')
    else:
        # Drop features highly correlated with 'carat'
        df_encoded_reduced = df_encoded.drop(columns=['x', 'y', 'z'])
        features = list(df_encoded_reduced.columns)
        features.remove('price')
        df_encoded = df_encoded_reduced
    
    # Split data
    X = df_encoded[features]
    Y = df_encoded['price']
    
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.33, 0.01)
    random_state = st.number_input("Random State", 0, 100, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    st.write(f"### Training set: {X_train.shape[0]} samples")
    st.write(f"### Test set: {X_test.shape[0]} samples")
    
    # Build model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            linear_regression = LinearRegression()
            linear_regression.fit(X_train, y_train)
            
            # Store model in session state for later use
            st.session_state.model = linear_regression
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.features = features
            
            # Display coefficients
            st.write("### Model Coefficients")
            
            coef_df = pd.DataFrame({
                'Feature': ['Constant'] + features,
                'Coefficient': [linear_regression.intercept_] + list(linear_regression.coef_)
            })
            
            st.dataframe(coef_df)
            
            st.success("Model trained successfully!")

# Model Evaluation
elif page == "Model Evaluation":
    st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Building' section.")
    else:
        model = st.session_state.model
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.write("### Training Set Metrics")
            st.metric("R-squared", f"{train_r2:.3f}")
            st.metric("Mean Squared Error", f"{train_mse:.3f}")
            st.metric("Root Mean Squared Error", f"{train_rmse:.3f}")
            st.metric("Mean Absolute Error", f"{train_mae:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.write("### Test Set Metrics")
            st.metric("R-squared", f"{test_r2:.3f}")
            st.metric("Mean Squared Error", f"{test_mse:.3f}")
            st.metric("Root Mean Squared Error", f"{test_rmse:.3f}")
            st.metric("Mean Absolute Error", f"{test_mae:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
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
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'Actual vs Predicted Prices ({dataset})')
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
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title('Feature Correlation Heatmap')
    st.pyplot(fig)
    
    # VIF calculation
    if st.session_state.model is not None:
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
        
        st.write("""
        ### VIF Interpretation
        - VIF > 10: High multicollinearity
        - 5 < VIF < 10: Moderate multicollinearity
        - VIF < 5: Low multicollinearity
        
        Features with high VIF values may be redundant and could potentially be removed to simplify the model.
        """)

# Error Analysis
elif page == "Error Analysis":
    st.markdown("<h2 class='sub-header'>Error Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Building' section.")
    else:
        model = st.session_state.model
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate errors
        train_errors = y_train - y_train_pred
        test_errors = y_test - y_test_pred
        
        # Error distribution
        dataset = st.radio("Select Dataset", ["Training Set", "Test Set"], key="error_dataset")
        
        if dataset == "Training Set":
            errors = train_errors
            y_actual = y_train
        else:
            errors = test_errors
            y_actual = y_test
        
        st.write(f"### Error Distribution ({dataset})")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.hist(errors, bins='sturges', color='#3175D9')
        plt.axvline(errors.mean(), color='#D93131', label=f'Mean Error: {errors.mean():.2f}')
        plt.title(f'Histogram for the errors obtained in the predicted values of the {dataset.lower()}')
        plt.legend()
        st.pyplot(fig)
        
        st.write(f"### Error vs Actual Price ({dataset})")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(errors, y_actual, color='#D93131', marker='*', alpha=0.5)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'Scatter plot between the errors and the dependent variable for the {dataset.lower()}')
        plt.xlabel('Errors')
        plt.ylabel('Actual Price')
        st.pyplot(fig)
        
        # Error statistics
        st.write(f"### Error Statistics ({dataset})")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Error", f"{errors.mean():.2f}")
        col2.metric("Median Error", f"{np.median(errors):.2f}")
        col3.metric("Std Dev", f"{errors.std():.2f}")
        col4.metric("Max Abs Error", f"{np.abs(errors).max():.2f}")

# Price Prediction
elif page == "Price Prediction":
    st.markdown("<h2 class='sub-header'>Diamond Price Prediction</h2>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Building' section.")
    else:
        model = st.session_state.model
        features = st.session_state.features
        
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
            # Make prediction
            predicted_price = model.predict(input_df)[0]
            
            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                <h2 style="color: #3366ff;">Predicted Diamond Price</h2>
                <h1 style="color: #3366ff; font-size: 3rem;">${predicted_price:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Show comparable diamonds
            st.write("### Similar Diamonds in Dataset")
            
            # Find similar diamonds based on carat weight
            if "carat" in df.columns:
                similar_diamonds = df[(df['carat'] >= carat*0.9) & (df['carat'] <= carat*1.1)]
                
                if not similar_diamonds.empty:
                    st.dataframe(similar_diamonds.head(5))
                else:
                    st.write("No similar diamonds found in the dataset.")
            else:
                st.write("Cannot find similar diamonds: carat column not available.")