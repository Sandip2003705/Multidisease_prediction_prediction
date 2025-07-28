# Multidisease_prediction_prediction
This project is a machine learning-based web application that predicts the likelihood of three common diseases:

- **Diabetes**
- **Heart Disease**
- **Parkinson’s Disease**

Built using Python and ML libraries, the system takes medical input parameters and returns disease prediction results.

---

## ⭐ Features

- 🔍 Predicts 3 diseases from user input
- 📊 Uses trained ML models (Logistic Regression, Random Forest, SVM, etc.)
- 🧠 Modular architecture – separate models and logic for each disease
- 📈 Displays model performance metrics
- 💻 Simple UI built with Streamlit (optional)
- 🔧 Easy to customize and expand

---

## 📦 Requirements

Before running the project, make sure you have:

- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for cloning repo)

---

## 📚 Libraries Used

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- streamlit *(for UI, optional)*
---
## Setup & Usage
 # 1. Clone the Respository
     git clone https://github.com/yourusername/multi-disease-prediction.git
     cd multi-disease-prediction

  ---   
# 2. Install the Requirements
      pip install -r requirements.txt
   ---
# 3. Run the App (if using Streamlit)
        streamlit run main_app.py
---
# Folder Structure
        multi-disease-prediction/
<pre> ```
│
├── diabetes/
│   ├── diabetes_model.pkl
│   ├── diabetes_data.csv
│   └── diabetes_predict.py
│
├── heart/
│   ├── heart_model.pkl
│   ├── heart_data.csv
│   └── heart_predict.py
│
├── parkinsons/
│   ├── parkinsons_model.pkl
│   ├── parkinsons_data.csv
│   └── parkinsons_predict.py
│
├── main_app.py         # UI or main integration file
├── requirements.txt    # Required Python packages
└── README.md
 ``` </pre>

# How It Works
1. Data Preprocessing: Each dataset is cleaned and normalized.

2. Model Training: Separate ML models are trained for each disease.

3. Model Saving: Trained models are saved using pickle or joblib.

4. Prediction: Takes user input (via UI or command line) and returns the prediction.

5. Output: Displays whether the person is likely to have the disease.

---
# Customization
  You can easily 

 - Replace or update the datasets for better quality data

 - Tune hyperparameters in GridSearchCV for improved accuracy

 -  Add new diseases by:
 
 -  Creating a new folder

 -  Adding a new model + predict script

 -  Integrating into main_app.py

# Limitations
  - Predictions depend heavily on the dataset quality.

  - Not suitable for real medical decisions — consult a professional.

  - Parkinson’s model is based on vocal data — requires proper audio metrics.

  - UI is basic; needs improvement for real-world use.

## Acknowledgment

- UCI Machine Learning Repository

- Kaggle Datasets

- Scikit-learn and Streamlit community

- Open-source contributors and healthcare researchers
