import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Real-World ML Deployment",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .regression-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classification_model' not in st.session_state:
    st.session_state.classification_model = None
if 'regression_model' not in st.session_state:
    st.session_state.regression_model = None
if 'classification_scaler' not in st.session_state:
    st.session_state.classification_scaler = None
if 'regression_scaler' not in st.session_state:
    st.session_state.regression_scaler = None
if 'iris_data' not in st.session_state:
    st.session_state.iris_data = None
if 'housing_data' not in st.session_state:
    st.session_state.housing_data = None
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'regression_results' not in st.session_state:
    st.session_state.regression_results = None

@st.cache_data
def load_iris_data():
    """Load and prepare Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df, iris.feature_names, iris.target_names

@st.cache_data
def load_housing_data():
    """Load and prepare California Housing dataset"""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    return df, housing.feature_names

def train_classification_model(df, feature_names, model_type='RandomForest'):
    """Train classification model for Iris dataset"""
    X = df[feature_names]
    y = df['species']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:  # SVM
        model = SVC(random_state=42, probability=True)
    
    model.fit(X_train_scaled, y_train)
    
    # Store in session state
    st.session_state.classification_model = model
    st.session_state.classification_scaler = scaler
    
    # Calculate metrics
    accuracy, cm, report, y_pred = get_classification_metrics(model, X_test_scaled, y_test)
    
    # Store results
    st.session_state.classification_results = {
        'accuracy': accuracy,
        'cm': cm,
        'report': report,
        'y_pred': y_pred,
        'y_test': y_test,
        'X_test': X_test_scaled,
        'model_type': model_type
    }
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def train_regression_model(df, feature_names, model_type='RandomForest'):
    """Train regression model for Housing dataset"""
    X = df[feature_names]
    y = df['MedHouseVal']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:  # LinearRegression
        model = LinearRegression()
    
    model.fit(X_train_scaled, y_train)
    
    # Store in session state
    st.session_state.regression_model = model
    st.session_state.regression_scaler = scaler
    
    # Calculate metrics
    mse, rmse, r2, y_pred = get_regression_metrics(model, X_test_scaled, y_test)
    
    # Store results
    st.session_state.regression_results = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test,
        'X_test': X_test_scaled,
        'model_type': model_type
    }
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def get_classification_metrics(model, X_test, y_test):
    """Calculate classification model performance metrics"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return accuracy, cm, report, y_pred

def get_regression_metrics(model, X_test, y_test):
    """Calculate regression model performance metrics"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, r2, y_pred

def create_iris_scatter_plot(df):
    """Create scatter plot for Iris dataset"""
    fig = px.scatter(
        df, 
        x='sepal length (cm)', 
        y='sepal width (cm)',
        color='species_name',
        size='petal length (cm)',
        title='Iris Dataset - Sepal Length vs Sepal Width',
        hover_data=['petal width (cm)']
    )
    return fig

def create_housing_scatter_plot(df):
    """Create scatter plot for Housing dataset"""
    fig = px.scatter(
        df.sample(1000), 
        x='MedInc', 
        y='MedHouseVal',
        color='HouseAge',
        title='California Housing - Median Income vs House Value',
        labels={'MedInc': 'Median Income', 'MedHouseVal': 'Median House Value'}
    )
    return fig

def create_confusion_matrix_plot(cm, class_names):
    """Create confusion matrix visualization"""
    # Create annotations for the confusion matrix
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations,
        height=400
    )
    
    return fig

def create_feature_importance_plot(model, feature_names):
    """Create feature importance visualization"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        return fig
    return None

def create_prediction_probability_plot(proba, class_names):
    """Create prediction probability visualization"""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=proba,
            marker_color=['lightcoral', 'lightblue', 'lightgreen'][:len(proba)],
            text=[f'{p:.3f}' for p in proba],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Species',
        yaxis_title='Probability',
        height=400
    )
    return fig

def create_residual_plot(y_test, y_pred):
    """Create residual plot for regression"""
    residuals = y_test - y_pred
    fig = px.scatter(
        x=y_pred, 
        y=residuals,
        title='Residual Plot',
        labels={'x': 'Predicted Values', 'y': 'Residuals'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig

# Main app
def main():
    st.title("🌸 Real-World ML Model Deployment")
    st.markdown("Deploy machine learning models for **Iris Classification** and **House Price Prediction**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choose a task",
        ["🏠 Home", "🌸 Iris Classification", "🏘️ House Price Prediction", "📊 Model Comparison", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        st.header("Welcome to Real-World ML Deployment!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌸 Iris Flower Classification")
            st.write("""
            **Dataset:** Famous Iris flower dataset
            **Task:** Classify iris flowers into 3 species
            **Features:** Sepal length, sepal width, petal length, petal width
            **Classes:** Setosa, Versicolor, Virginica
            """)
            
            # Load and display Iris data
            iris_df, iris_features, iris_targets = load_iris_data()
            st.session_state.iris_data = (iris_df, iris_features, iris_targets)
            
            st.write("Sample data:")
            st.dataframe(iris_df.head())
            
            # Iris visualization
            fig_iris = create_iris_scatter_plot(iris_df)
            st.plotly_chart(fig_iris, use_container_width=True)
        
        with col2:
            st.subheader("🏘️ California Housing Price Prediction")
            st.write("""
            **Dataset:** California Housing dataset
            **Task:** Predict median house values
            **Features:** Income, house age, rooms, population, etc.
            **Target:** Median house value in hundreds of thousands of dollars
            """)
            
            # Load and display Housing data
            housing_df, housing_features = load_housing_data()
            st.session_state.housing_data = (housing_df, housing_features)
            
            st.write("Sample data:")
            st.dataframe(housing_df.head())
            
            # Housing visualization
            fig_housing = create_housing_scatter_plot(housing_df)
            st.plotly_chart(fig_housing, use_container_width=True)
    
    elif page == "🌸 Iris Classification":
        st.header("🌸 Iris Flower Classification")
        
        if st.session_state.iris_data is None:
            iris_df, iris_features, iris_targets = load_iris_data()
            st.session_state.iris_data = (iris_df, iris_features, iris_targets)
        else:
            iris_df, iris_features, iris_targets = st.session_state.iris_data
        
        tab1, tab2, tab3 = st.tabs(["🔧 Train Model", "🔮 Make Predictions", "📈 Model Analysis"])
        
        with tab1:
            st.subheader("Model Training")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Dataset Overview:**")
                st.dataframe(iris_df.describe())
                
                # Class distribution
                fig_dist = px.histogram(
                    iris_df, 
                    x='species_name', 
                    title='Class Distribution',
                    color='species_name'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.write("**Model Configuration:**")
                model_type = st.selectbox(
                    "Select Model",
                    ["RandomForest", "LogisticRegression", "SVM"]
                )
                
                if st.button("Train Classification Model", type="primary"):
                    with st.spinner("Training model..."):
                        model, scaler, X_train, X_test, y_train, y_test = train_classification_model(
                            iris_df, iris_features, model_type
                        )
                        
                        st.success(f"Model trained successfully!")
                        st.metric("Accuracy", f"{st.session_state.classification_results['accuracy']:.3f}")
            
            # Display training results
            if st.session_state.classification_results is not None:
                st.markdown("---")
                st.subheader("Training Results")
                
                results = st.session_state.classification_results
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{results['report']['weighted avg']['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{results['report']['weighted avg']['recall']:.3f}")
                
                # Confusion matrix
                fig_cm = create_confusion_matrix_plot(results['cm'], iris_targets)
                st.plotly_chart(fig_cm, use_container_width=True)
        
        with tab2:
            st.subheader("Make Predictions")
            
            if st.session_state.classification_model is None:
                st.warning("Please train a model first!")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Enter Flower Measurements:**")
                sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
                sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
            
            with col2:
                st.write("**&nbsp;**")  # Spacing
                petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
                petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)
            
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            if st.button("Predict Species", type="primary"):
                input_scaled = st.session_state.classification_scaler.transform(input_data)
                prediction = st.session_state.classification_model.predict(input_scaled)[0]
                prediction_proba = st.session_state.classification_model.predict_proba(input_scaled)[0]
                
                species_name = iris_targets[prediction]
                confidence = max(prediction_proba)
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Species: {species_name}</h2>
                    <p><strong>Confidence:</strong> {confidence:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability visualization
                fig_proba = create_prediction_probability_plot(prediction_proba, iris_targets)
                st.plotly_chart(fig_proba, use_container_width=True)
        
        with tab3:
            st.subheader("Model Analysis")
            
            if st.session_state.classification_model is None:
                st.warning("Please train a model first!")
                return
            
            # Feature importance
            fig_importance = create_feature_importance_plot(
                st.session_state.classification_model, 
                iris_features
            )
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Pairplot
            st.subheader("Feature Relationships")
            fig_pair = px.scatter_matrix(
                iris_df, 
                dimensions=iris_features, 
                color='species_name',
                title='Feature Pairplot'
            )
            st.plotly_chart(fig_pair, use_container_width=True)
    
    elif page == "🏘️ House Price Prediction":
        st.header("🏘️ House Price Prediction")
        
        if st.session_state.housing_data is None:
            housing_df, housing_features = load_housing_data()
            st.session_state.housing_data = (housing_df, housing_features)
        else:
            housing_df, housing_features = st.session_state.housing_data
        
        tab1, tab2, tab3 = st.tabs(["🔧 Train Model", "🔮 Make Predictions", "📈 Model Analysis"])
        
        with tab1:
            st.subheader("Model Training")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Dataset Overview:**")
                st.dataframe(housing_df.describe())
                
                # Price distribution
                fig_price = px.histogram(
                    housing_df, 
                    x='MedHouseVal', 
                    title='House Price Distribution',
                    nbins=50
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            with col2:
                st.write("**Model Configuration:**")
                model_type = st.selectbox(
                    "Select Model",
                    ["RandomForest", "LinearRegression"]
                )
                
                if st.button("Train Regression Model", type="primary"):
                    with st.spinner("Training model..."):
                        model, scaler, X_train, X_test, y_train, y_test = train_regression_model(
                            housing_df, housing_features, model_type
                        )
                        
                        st.success(f"Model trained successfully!")
                        st.metric("R² Score", f"{st.session_state.regression_results['r2']:.3f}")
            
            # Display training results
            if st.session_state.regression_results is not None:
                st.markdown("---")
                st.subheader("Training Results")
                
                results = st.session_state.regression_results
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{results['r2']:.3f}")
                with col2:
                    st.metric("RMSE", f"{results['rmse']:.3f}")
                with col3:
                    st.metric("MSE", f"{results['mse']:.3f}")
                
                # Residual plot
                fig_residual = create_residual_plot(results['y_test'], results['y_pred'])
                st.plotly_chart(fig_residual, use_container_width=True)
        
        with tab2:
            st.subheader("Make Predictions")
            
            if st.session_state.regression_model is None:
                st.warning("Please train a model first!")
                return
            
            st.write("**Enter House Characteristics:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                med_inc = st.slider("Median Income (in 10k$)", 0.5, 15.0, 5.0, 0.1)
                house_age = st.slider("House Age (years)", 1.0, 52.0, 20.0, 1.0)
                ave_rooms = st.slider("Average Rooms", 1.0, 20.0, 6.0, 0.1)
                ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)
            
            with col2:
                population = st.slider("Population", 3.0, 35000.0, 3000.0, 100.0)
                ave_occup = st.slider("Average Occupancy", 0.5, 20.0, 3.0, 0.1)
                latitude = st.slider("Latitude", 32.0, 42.0, 35.0, 0.1)
                longitude = st.slider("Longitude", -125.0, -114.0, -120.0, 0.1)
            
            input_data = np.array([[
                med_inc, house_age, ave_rooms, ave_bedrms, 
                population, ave_occup, latitude, longitude
            ]])
            
            if st.button("Predict House Price", type="primary"):
                input_scaled = st.session_state.regression_scaler.transform(input_data)
                prediction = st.session_state.regression_model.predict(input_scaled)[0]
                
                st.markdown(f"""
                <div class="regression-box">
                    <h2>Predicted House Price: ${prediction:.2f}00</h2>
                    <p><strong>Median House Value:</strong> ${prediction * 100000:.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input summary
                st.subheader("Input Summary")
                input_df = pd.DataFrame([{
                    'Median Income': f"${med_inc * 10000:.0f}",
                    'House Age': f"{house_age} years",
                    'Avg Rooms': f"{ave_rooms:.1f}",
                    'Avg Bedrooms': f"{ave_bedrms:.1f}",
                    'Population': f"{population:.0f}",
                    'Avg Occupancy': f"{ave_occup:.1f}",
                    'Latitude': f"{latitude:.1f}",
                    'Longitude': f"{longitude:.1f}"
                }])
                st.dataframe(input_df.T.rename(columns={0: 'Value'}))
        
        with tab3:
            st.subheader("Model Analysis")
            
            if st.session_state.regression_model is None:
                st.warning("Please train a model first!")
                return
            
            # Feature importance
            fig_importance = create_feature_importance_plot(
                st.session_state.regression_model, 
                housing_features
            )
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Feature Correlations")
            corr_matrix = housing_df[housing_features + ['MedHouseVal']].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title='Feature Correlation Matrix',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    elif page == "📊 Model Comparison":
        st.header("📊 Model Comparison")
        
        st.subheader("Classification vs Regression Tasks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🌸 Iris Classification**")
            if st.session_state.classification_model is not None:
                st.success("✅ Model Trained")
                st.write("- **Task:** Multi-class classification")
                st.write("- **Classes:** 3 (Setosa, Versicolor, Virginica)")
                st.write("- **Features:** 4 numerical features")
                st.write("- **Metric:** Accuracy")
                if st.session_state.classification_results:
                    st.metric("Current Accuracy", f"{st.session_state.classification_results['accuracy']:.3f}")
            else:
                st.warning("❌ Model Not Trained")
        
        with col2:
            st.write("**🏘️ House Price Prediction**")
            if st.session_state.regression_model is not None:
                st.success("✅ Model Trained")
                st.write("- **Task:** Regression")
                st.write("- **Target:** Continuous house prices")
                st.write("- **Features:** 8 numerical features")
                st.write("- **Metric:** R² Score")
                if st.session_state.regression_results:
                    st.metric("Current R² Score", f"{st.session_state.regression_results['r2']:.3f}")
            else:
                st.warning("❌ Model Not Trained")
        
        st.markdown("---")
        st.subheader("Key Differences")
        
        comparison_df = pd.DataFrame({
            'Aspect': ['Problem Type', 'Output', 'Evaluation Metric', 'Use Case'],
            'Classification (Iris)': ['Classification', 'Discrete Classes', 'Accuracy, Precision, Recall', 'Species Identification'],
            'Regression (Housing)': ['Regression', 'Continuous Values', 'R², RMSE, MSE', 'Price Prediction']
        })
        
        st.dataframe(comparison_df, use_container_width=True)
    
    elif page == "ℹ️ About":
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ## 🎯 Purpose
        This application demonstrates real-world machine learning model deployment using Streamlit. 
        It showcases both **classification** and **regression** tasks with popular datasets.
        
        ## 📊 Datasets Used
        
        ### 🌸 Iris Dataset
        - **Source:** Built into scikit-learn
        - **Samples:** 150 flowers
        - **Features:** 4 (sepal length, sepal width, petal length, petal width)
        - **Classes:** 3 species of iris flowers
        - **Task:** Multi-class classification
        
        ### 🏘️ California Housing Dataset
        - **Source:** Built into scikit-learn
        - **Samples:** 20,640 districts
        - **Features:** 8 (median income, house age, average rooms, etc.)
        - **Target:** Median house value
        - **Task:** Regression
        
        ## 🤖 Models Implemented
        
        ### Classification Models
        - **Random Forest Classifier**
        - **Logistic Regression**
        - **Support Vector Machine (SVM)**
        
        ### Regression Models
        - **Random Forest Regressor**
        - **Linear Regression**
        
        ## 🚀 Features
        - Interactive model training
        - Real-time predictions
        - Comprehensive visualizations
        - Model performance metrics
        - Feature importance analysis
        - Responsive design
        
        ## 🛠️ Technology Stack
        - **Streamlit:** Web application framework
        - **Scikit-learn:** Machine learning library
        - **Plotly:** Interactive visualizations
        - **Pandas:** Data manipulation
        - **NumPy:** Numerical computing
        
        ## 📈 Model Interpretability
        - Confusion matrices for classification
        - Feature importance plots
        - Residual plots for regression
        - Correlation matrices
        - Prediction probability distributions
        
        ---
        
        **Built for educational purposes to demonstrate ML model deplyment**
