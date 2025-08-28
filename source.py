
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.express as px
import pickle
import os
import time
from datetime import datetime

# Loading and preprocessing the dataset
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('VN_housing_dataset.csv', encoding='utf-8')
    
    if df.columns[0] == '':
        df = df.drop(columns=[''])
    
    def clean_price(price):
        if isinstance(price, str):
            price = price.replace('triệu/m²', '').replace('đ/m²', '').replace(',', '.').strip()
            try:
                return float(price)
            except ValueError:
                return np.nan
        return np.nan
    
    df['Giá/m2'] = df['Giá/m2'].apply(clean_price)
    df['Diện tích'] = df['Diện tích'].str.replace(' m²', '').astype(float)
    df['Dài'] = df['Dài'].str.replace(' m', '').replace('NaN', np.nan).astype(float)
    df['Rộng'] = df['Rộng'].str.replace(' m', '').replace('NaN', np.nan).astype(float)
    
    def clean_numeric(value):
        if isinstance(value, str):
            if 'nhiều hơn 10' in value.lower():
                return 11
            return float(value.replace(' phòng', ''))
        return value
    
    df['Số tầng'] = df['Số tầng'].replace('NaN', np.nan).astype(str).apply(clean_numeric).astype(float)
    df['Số phòng ngủ'] = df['Số phòng ngủ'].astype(str).apply(clean_numeric).astype(float)
    
    df = df.dropna(subset=['Giá/m2'])
    
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    for col in categorical_cols:
        df[col] = df[col].fillna('missing').astype(str)
    
    features = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý', 'Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    X = df[features]
    y = df['Giá/m2']
    
    if len(X) > 20000:
        X, _, y, _ = train_test_split(X, y, train_size=20000, random_state=42)
    else:
        st.write(f"Using full dataset with {len(X)} rows.")
    
    return X, y, df

# Load or initialize prediction history
def load_prediction_history():
    history_file = 'prediction_history.csv'
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    return pd.DataFrame(columns=[
        'Timestamp', 'District', 'House Type', 'Legal Documents', 'Floors', 
        'Bedrooms', 'Area', 'Length', 'Width', 'Model', 'Price (million/m²)'
    ])

# Save prediction to history
def save_prediction_to_history(input_data, model_name, price):
    history_file = 'prediction_history.csv'
    history_df = load_prediction_history()
    
    new_entry = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'District': input_data['Quận'].iloc[0],
        'House Type': input_data['Loại hình nhà ở'].iloc[0],
        'Legal Documents': input_data['Giấy tờ pháp lý'].iloc[0],
        'Floors': input_data['Số tầng'].iloc[0],
        'Bedrooms': input_data['Số phòng ngủ'].iloc[0],
        'Area': input_data['Diện tích'].iloc[0],
        'Length': input_data['Dài'].iloc[0],
        'Width': input_data['Rộng'].iloc[0],
        'Model': model_name,
        'Price (million/m²)': price
    }
    
    history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
    history_df.to_csv(history_file, index=False, encoding='utf-8')

# Training and saving the XGBoost model
@st.cache_resource
def train_xgboost_model(X, y):
    model_path = 'xgboost_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 0.0
    
    start_time = time.time()
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1))
    ])
    
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    return model, training_time

# Training and saving the Random Forest model
@st.cache_resource
def train_rf_model(X, y):
    model_path = 'rf_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 0.0
    
    start_time = time.time()
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols),
            ('num', SimpleImputer(strategy='median'), numerical_cols)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    return model, training_time

# Training and saving the Linear Regression model
@st.cache_resource
def train_linear_model(X, y):
    model_path = 'linear_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 0.0
    
    start_time = time.time()
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    return model, training_time

# Training and saving the MLP model
@st.cache_resource
def train_mlp_model(X, y):
    model_path = 'mlp_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 0.0
    
    start_time = time.time()
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(30,), max_iter=100, random_state=42, 
                                 early_stopping=True, n_iter_no_change=10))
    ])
    
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    return model, training_time

# Training and saving the kNN model
@st.cache_resource
def train_knn_model(X, y):
    model_path = 'knn_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 0.0
    
    start_time = time.time()
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(n_neighbors=5, n_jobs=-1))
    ])
    
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    return model, training_time

# Training and saving the CatBoost model
@st.cache_resource
def train_catboost_model(X, y):
    model_path = 'catboost_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 0.0
    
    start_time = time.time()
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ]), categorical_cols),
            ('num', SimpleImputer(strategy='median'), numerical_cols)
        ], remainder='passthrough')
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor(iterations=50, depth=8, random_seed=42, verbose=False,
                                       cat_features=[0, 1, 2]))
    ])
    
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    return model, training_time

# Training and saving the SVR model
@st.cache_resource
def train_svr_model(X, y):
    model_path = 'svr_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 0.0
    
    start_time = time.time()
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ])
    
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    return model, training_time

# Training and saving the Voting Regressor model
@st.cache_resource
def train_voting_model(X, y):
    model_path = 'voting_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 0.0
    
    start_time = time.time()
    categorical_cols = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
    numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng']
    
    preprocessor_xgb = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ])
    xgb_pipe = Pipeline([
        ('preprocessor', preprocessor_xgb),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=1))
    ])
    
    preprocessor_rf = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols),
            ('num', SimpleImputer(strategy='median'), numerical_cols)
        ])
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor_rf),
        ('regressor', RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=1))
    ])
    
    preprocessor_cat = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ]), categorical_cols),
            ('num', SimpleImputer(strategy='median'), numerical_cols)
        ], remainder='passthrough')
    cat_pipe = Pipeline([
        ('preprocessor', preprocessor_cat),
        ('regressor', CatBoostRegressor(iterations=50, depth=8, random_seed=42, verbose=False,
                                       cat_features=[0, 1, 2]))
    ])
    
    model = VotingRegressor(estimators=[('xgb', xgb_pipe), ('rf', rf_pipe), ('cat', cat_pipe)], n_jobs=1)
    
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    training_time = time.time() - start_time
    return model, training_time

# Evaluate models
def evaluate_models(X, y, models):
    metrics = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, info in models.items():
        model = info['model']
        y_pred = model.predict(X_test)
        metrics[name] = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        }
    return metrics

def get_market_segment(price):
    if price < 50:
        return 'Economy', "#9ed8ff"
    elif price < 100:
        return 'Mid range', "#ffeb9d"
    elif price < 200:
        return 'Premium', "#ffada3"
    else:
        return 'Luxury', "#e6a7ff"

# Main app
st.set_page_config(page_title="Hanoi Real Estate Prediction", layout="wide")

# Centered title at the top
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #333333;
            margin-top: 0px;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
    </style>
    <div class="main-title">Hanoi Real Estate Price Estimate</div>
""", unsafe_allow_html=True)

# Improved CSS with fixed elements and no auto-scaling
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow-x: hidden;
        }
        .stApp {
            background-color: #ffffff;
            padding: 20px;
            min-width: 1200px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .stSidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 250px;
            height: 100%;
            background-color: #f9f9f9;
            padding: 20px;
            border-right: 1px solid #e0e0e0;
            overflow-y: auto;
            z-index: 1000;
        }
        .stSidebar .stRadio > div {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
        }
        .stSidebar .stRadio > label {
            font-weight: 500;
            color: #333333;
            padding: 8px 12px;
            border-radius: 6px;
            transition: background-color 0.3s ease;
        }
        .stSidebar .stRadio > div > label:hover {
            background-color: #e0e0e0;
        }
        .stSidebar .stRadio > div > label[data-baseweb="radio"] > div {
            color: #333333 !important;
        }
        .main-content {
            margin-left: 270px;
            padding: 20px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: #ffffff;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .stSelectbox, .stNumberInput, .stMultiSelect {
            background-color: #f9f9f9;
            color: #333333;
            border-radius: 8px;
            padding: 8px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            min-width: 200px;
        }
        .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input, .stMultiSelect div[data-baseweb="select"] > div {
            background-color: #f9f9f9 !important;
            color: #333333 !important;
            border-color: #ddd !important;
            font-weight: normal !important;
        }
        .stSelectbox div[data-baseweb="select"] > div > div > div, .stMultiSelect div[data-baseweb="select"] > div > div > div {
            color: #333333 !important;
            font-weight: normal !important;
        }
        .stContainer {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
            min-width: 0;
            max-width: 100%;
        }
        .stExpander {
            background-color: #ffffff;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
        .stExpander div[data-testid="stExpanderToggle"] {
            background-color: #ffffff !important;
            color: #333333 !important;
            font-weight: 500;
        }
        .prediction-box {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3em;
            color: #333333;
            border: 1px solid #ddd;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin: 15px 0;
            min-width: 0;
        }
        .model-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            flex: 1;
            text-align: center;
            font-size: 1.1em;
            color: #333333;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin: 10px;
            transition: all 0.3s ease;
            min-width: 200px;
        }
        .model-card:hover {
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
            border-color: #bbb;
            transform: translateY(-2px);
        }
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: #333333 !important;
        }
        .stDataFrame {
            background-color: #ffffff !important;
            color: #333333 !important;
        }
        .stDataFrame th, .stDataFrame td {
            background-color: #ffffff !important;
            color: #333333 !important;
            border-color: #e0e0e0 !important;
        }
        .segment-label {
            font-size: 0.9em;
            padding: 8px 15px;
            border-radius: 6px;
            display: inline-block;
            margin-top: 8px;
            font-weight: 500;
            color: #333333;
        }
        .best-model {
            border: 2px solid #4CAF50 !important;
            background-color: #e8f5e9 !important;
            font-weight: bold;
        }
        @media (max-width: 768px) {
            .stSidebar {
                width: 200px;
            }
            .main-content {
                margin-left: 220px;
            }
            .model-card { width: 100%; margin: 10px 0; }
            .stContainer { padding: 15px; }
            .stApp { min-width: 100%; max-width: 100%; }
            .main-title { font-size: 2em; }
        }
        .segment-label[style*="#3498db"] { background-color: #3498db80; }
        .segment-label[style*="#f1c40f"] { background-color: #f1c40f80; }
        .segment-label[style*="#e74c3c"] { background-color: #e74c3c80; }
        .segment-label[style*="#9b59b6"] { background-color: #9b59b680; }
    </style>
""", unsafe_allow_html=True)

# Wrap main content to avoid overlap with fixed sidebar
st.markdown('<div class="main-content">', unsafe_allow_html=True)

X, y, df = load_and_preprocess_data()

# Load models
xg_model, xg_time = train_xgboost_model(X, y)  # Fixed from train_xgboost_model boutons(X, y)
rf_model, rf_time = train_rf_model(X, y)
linear_model, linear_time = train_linear_model(X, y)
mlp_model, mlp_time = train_mlp_model(X, y)
knn_model, knn_time = train_knn_model(X, y)
cat_model, cat_time = train_catboost_model(X, y)
svr_model, svr_time = train_svr_model(X, y)
voting_model, voting_time = train_voting_model(X, y)

models = {
    'XGBoost': {'model': xg_model},
    'Random Forest': {'model': rf_model},
    'Linear Regression': {'model': linear_model},
    'MLP': {'model': mlp_model},
    'KNN': {'model': knn_model},
    'CatBoost': {'model': cat_model},
    'SVR': {'model': svr_model},
    'Voting Regressor': {'model': voting_model}
}

training_times = {
    'XGBoost': xg_time,
    'Random Forest': rf_time,
    'Linear Regression': linear_time,
    'MLP': mlp_time,
    'KNN': knn_time,
    'CatBoost': cat_time,
    'SVR': svr_time,
    'Voting Regressor': voting_time
}

# Taskbar using sidebar
st.sidebar.title("Navigation")
task = st.sidebar.radio(
    "Select Task",
    ["📊 Model Evaluation", "🏡 Real Estate Prediction", "📜 Prediction History"],
    index=0
)

if task == "📊 Model Evaluation":
    with st.container():
        st.markdown("### Model Evaluation")
        
        metrics = evaluate_models(X, y, models)
        metrics_df = pd.DataFrame({
            model: {
                'R2': metrics[model]['R2'],
                'MAE': metrics[model]['MAE'],
                'MSE': metrics[model]['MSE'],
                'RMSE': metrics[model]['RMSE'],
                'Training Time (s)': training_times[model]
            } for model in models.keys()
        }).T
        metrics_df = metrics_df.round(2)
        st.markdown("#### 📋 Performance Table")
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("#### 📈 Metrics Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            mae_data = pd.DataFrame({
                'Model': list(models.keys()),
                'MAE': [metrics[model]['MAE'] for model in models.keys()]
            })
            best_mae_model = mae_data.loc[mae_data['MAE'].idxmin()]['Model']
            fig_mae = px.bar(
                mae_data,
                x='Model',
                y='MAE',
                title="MAE (Mean Absolute Error)",
                height=400,
                text='MAE'
            )
            fig_mae.update_traces(
                textposition='auto',
                marker_color=["#4a65ff" if model == best_mae_model else '#aeaeae' for model in models.keys()]
            )
            fig_mae.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color='#333333',
                xaxis_title="Model",
                yaxis_title="MAE",
                showlegend=False
            )  # Fixed from fig_maeCatalà
            st.plotly_chart(fig_mae, use_container_width=True)

        with col2:
            mse_data = pd.DataFrame({
                'Model': list(models.keys()),
                'MSE': [metrics[model]['MSE'] for model in models.keys()]
            })
            best_mse_model = mse_data.loc[mse_data['MSE'].idxmin()]['Model']
            fig_mse = px.bar(
                mse_data,
                x='Model',
                y='MSE',
                title="MSE (Mean Squared Error)",
                height=400,
                text='MSE'
            )
            fig_mse.update_traces(
                textposition='auto',
                marker_color=['#4a65ff' if model == best_mse_model else '#aeaeae' for model in models.keys()]
            )
            fig_mse.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color='#333333',
                xaxis_title="Model",
                yaxis_title="MSE",
                showlegend=False
            )
            st.plotly_chart(fig_mse, use_container_width=True)

        with col1:
            rmse_data = pd.DataFrame({
                'Model': list(models.keys()),
                'RMSE': [metrics[model]['RMSE'] for model in models.keys()]
            })
            best_rmse_model = rmse_data.loc[rmse_data['RMSE'].idxmin()]['Model']
            fig_rmse = px.bar(
                rmse_data,
                x='Model',
                y='RMSE',
                title="RMSE (Root Mean Squared Error)",
                height=400,
                text='RMSE'
            )
            fig_rmse.update_traces(
                textposition='auto',
                marker_color=['#4a65ff' if model == best_rmse_model else '#aeaeae' for model in models.keys()]
            )
            fig_rmse.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color='#333333',
                xaxis_title="Model",
                yaxis_title="RMSE",
                showlegend=False
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col2:
            r2_data = pd.DataFrame({
                'Model': list(models.keys()),
                'R2': [metrics[model]['R2'] for model in models.keys()]
            })
            best_r2_model = r2_data.loc[r2_data['R2'].idxmax()]['Model']
            fig_r2 = px.bar(
                r2_data,
                x='Model',
                y='R2',
                title="R² (Coefficient of Determination)",
                height=400,
                text='R2'
            )
            fig_r2.update_traces(
                textposition='auto',
                marker_color=['#4a65ff' if model == best_r2_model else '#aeaeae' for model in models.keys()]
            )
            fig_r2.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font_color='#333333',
                xaxis_title="Model",
                yaxis_title="R²",
                yaxis_range=[-0.1, 1.1],
                showlegend=False
            )
            st.plotly_chart(fig_r2, use_container_width=True)

elif task == "🏡 Real Estate Prediction":
    with st.container():
        st.markdown("### Enter House Information")
        with st.form(key='prediction_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                districts = df['Quận'].dropna().unique()
                district = st.selectbox("🏙️ District", options=districts)
                
                house_types = df['Loại hình nhà ở'].dropna().unique()
                house_type = st.selectbox("🏘️ House Type", options=house_types)
                
                legal_docs = df['Giấy tờ pháp lý'].dropna().unique()
                legal_doc = st.selectbox("📜 Legal Documents", options=legal_docs)
                
                compare_models = st.multiselect(
                    "🔎 Select models to compare (choose fewer for faster results)",
                    options=list(models.keys()),
                    default=['XGBoost', 'Random Forest', 'Linear Regression']
                )
            
            with col2:
                floors = st.number_input("🏛️ Floors", min_value=1, max_value=20, value=4, step=1)
                bedrooms = st.number_input("🛏️ Bedrooms", min_value=1, max_value=20, value=3, step=1)
                area = st.number_input("📏 Area (m²)", min_value=10.0, max_value=1000.0, value=50.0, step=1.0)
                length = st.number_input("📐 Length (m)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
                width = st.number_input("📐 Width (m)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            
            submit_button = st.form_submit_button("🔍 Predict Price")
        
        if submit_button:
            input_data = pd.DataFrame({
                'Quận': [district],
                'Loại hình nhà ở': [house_type],
                'Giấy tờ pháp lý': [legal_doc],
                'Số tầng': [floors],
                'Số phòng ngủ': [bedrooms],
                'Diện tích': [area],
                'Dài': [length],
                'Rộng': [width]
            })
            
            with st.spinner("Predicting..."):
                progress_bar = st.progress(0)
                status_text = st.empty()  # Fixed from status_test
                
                if compare_models:
                    st.markdown("### ☁️ Prediction Results")
                    model_preds = {}
                    total_models = len(compare_models)
                    
                    for i, model_name in enumerate(compare_models):
                        status_text.text(f"Predicting with {model_name} ({i+1}/{total_models})...")
                        model_info = models[model_name]
                        model = model_info['model']
                        pred = float(model.predict(input_data).ravel()[0])
                        model_preds[model_name] = pred
                        save_prediction_to_history(input_data, model_name, pred)
                        progress_bar.progress((i + 1) / total_models)
                    
                    selected_models_dict = {k: v for k, v in models.items() if k in compare_models}
                    metrics = evaluate_models(X, y, selected_models_dict)
                    best_model = max(metrics, key=lambda x: metrics[x]['R2']) if compare_models else None
                    
                    col1, col2, col3 = st.columns(3)
                    cols = [col1, col2, col3]
                    for idx, (model_name, pred) in enumerate(model_preds.items()):
                        current_col = cols[idx % 3]
                        is_best = " best-model" if model_name == best_model else ""
                        with current_col:
                            st.markdown(f"""
                                <div class='model-card{is_best}'>
                                    <b>{model_name}</b><br>
                                    💵 {pred:.2f} million/m²<br>
                                    <span class='segment-label' style='background-color: {get_market_segment(pred)[1]}; color: #333333;'>
                                        {get_market_segment(pred)[0]}
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    preds_df = pd.DataFrame(model_preds.items(), columns=['Model', 'Price (million/m²)'])
                    st.download_button(
                        "📥 Download Predictions",
                        preds_df.to_csv(index=False).encode('utf-8'),
                        "predictions.csv",
                        mime='text/csv'
                    )
                
                status_text.empty()
                progress_bar.empty()

elif task == "📜 Prediction History":
    with st.container():
        st.markdown("### Prediction History")
        history_df = load_prediction_history()
        
        if not history_df.empty:
            st.markdown("#### 📋 Historical Predictions")
            st.dataframe(history_df, use_container_width=True)
            
            st.download_button(
                "📥 Download History",
                history_df.to_csv(index=False).encode('utf-8'),
                "prediction_history.csv",
                mime='text/csv'
            )
            


# Close the main-content div
st.markdown('</div>', unsafe_allow_html=True)
