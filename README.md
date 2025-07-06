# Iris-Classification-and-House-Prediction

# 🌸 Real-World ML Model Deployment with Streamlit

A comprehensive web application that demonstrates machine learning model deployment using Streamlit, featuring both classification and regression tasks with real-world datasets.

## Deplyment Link:  https://iris-classifier-and-house-predictor.streamlit.app

## 🎯 Overview

This application showcases two fundamental machine learning tasks:
- **🌸 Iris Flower Classification**: Multi-class classification using the famous Iris dataset
- **🏘️ House Price Prediction**: Regression analysis using the California Housing dataset

## 🚀 Features

### 🔧 Model Training
- Interactive model selection and training
- Real-time performance metrics
- Multiple algorithm options for each task
- Comprehensive model evaluation

### 🔮 Predictions
- User-friendly input interfaces
- Real-time prediction capabilities
- Confidence scores and probability distributions
- Interactive sliders for easy data input

### 📈 Visualizations
- Feature importance plots
- Confusion matrices
- Residual plots for regression
- Correlation matrices
- Interactive scatter plots
- Model performance comparisons

### 🎨 User Experience
- Modern, responsive design
- Intuitive navigation
- Real-time feedback
- Educational content about ML concepts

## 📊 Datasets

### 🌸 Iris Dataset
- **Samples**: 150 flower measurements
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 species (Setosa, Versicolor, Virginica)
- **Task**: Multi-class classification

### 🏘️ California Housing Dataset
- **Samples**: 20,640 housing districts
- **Features**: 8 (median income, house age, average rooms, etc.)
- **Target**: Median house value
- **Task**: Regression

## 🤖 Machine Learning Models

### Classification Models
- **Random Forest Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

### Regression Models
- **Random Forest Regressor**
- **Linear Regression**

## 🛠️ Technology Stack

- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Seaborn & Matplotlib**: Additional plotting capabilities

## 📦 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone <https://github.com/DuPontKevlar/Iris-Classification-and-House-Prediction>
   cd real-world-ml-deployment
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

### 🌐 Streamlit Cloud Deployment

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Choose `app.py` as the main file
   - Click "Deploy"

3. **Your app will be live** at `https://your-app-name.streamlit.app`

## 📁 Project Structure

```
real-world-ml-deployment/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore file (optional)
```

## 🎮 How to Use

### 1. **Home Page**
- Overview of both datasets
- Interactive visualizations
- Dataset statistics and sample data

### 2. **🌸 Iris Classification**
- **Train Model**: Select algorithm and train on Iris dataset
- **Make Predictions**: Input flower measurements and get species predictions
- **Model Analysis**: View feature importance and model performance

### 3. **🏘️ House Price Prediction**
- **Train Model**: Select algorithm and train on housing dataset
- **Make Predictions**: Input house characteristics and get price predictions
- **Model Analysis**: View feature correlations and residual plots

### 4. **📊 Model Comparison**
- Compare classification vs regression tasks
- View model performance side-by-side
- Understand key differences between problem types

## 📈 Model Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Confusion Matrix**: Detailed breakdown of predictions

### Regression Metrics
- **R² Score**: Coefficient of determination (explained variance)
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Squared Error
- **Residual Plots**: Visualization of prediction errors

## 🔧 Customization

### Adding New Models
To add a new machine learning model:

1. Import the model in `app.py`
2. Add the model option to the selectbox
3. Implement the training logic in the respective function
4. Update the prediction and analysis sections

### Adding New Datasets
To add a new dataset:

1. Create a data loading function
2. Add a new page in the navigation
3. Implement training, prediction, and analysis tabs
4. Update the model comparison section

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

2. **Streamlit Not Found**
   - Install Streamlit: `pip install streamlit`
   - Verify installation: `streamlit --version`

3. **Port Already in Use**
   - Use a different port: `streamlit run app.py --server.port 8502`

4. **Memory Issues**
   - Reduce dataset size for large datasets
   - Use `@st.cache_data` decorator for expensive operations

### Performance Optimization

- Use caching for data loading and model training
- Implement session state for model persistence
- Optimize visualizations for large datasets

## 📚 Learning Outcomes

This project demonstrates:
- **ML Model Deployment**: From training to web application
- **Interactive Visualizations**: Using Plotly for engaging charts
- **User Interface Design**: Creating intuitive ML applications
- **Model Interpretability**: Understanding and explaining ML predictions
- **Real-world Applications**: Working with actual datasets

## 🎓 Educational Value

Perfect for:
- **Students**: Learning ML deployment concepts
- **Developers**: Understanding Streamlit framework
- **Data Scientists**: Building interactive ML applications
- **Educators**: Teaching ML concepts with visual examples

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new machine learning models
- Improve visualizations
- Enhance user interface
- Add more datasets
- Fix bugs and improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the Streamlit documentation
3. Check scikit-learn documentation for ML-related questions

## 🌟 Acknowledgments

- **Streamlit** for the amazing framework
- **Scikit-learn** for the datasets and ML algorithms
- **Plotly** for interactive visualizations
- **UCI ML Repository** for the original Iris dataset

---

**🚀 Ready to deploy your machine learning models? Get started now!**

### Quick Start Commands
```bash
# Clone and setup
git clone <https://github.com/DuPontKevlar/Iris-Classification-and-House-Prediction>
cd real-world-ml-deployment
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Deploy to Streamlit Cloud
# Push to GitHub and deploy via share.streamlit.io
```

**Happy Machine Learning! 🤖✨**
