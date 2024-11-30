import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
# Setting page layout ( This command should be after importing libraries )
st.set_page_config(page_title='Machine Learning - Maintenance Cost Prediction',page_icon=None,
                   layout='wide',initial_sidebar_state='auto', menu_items=None)
with st.sidebar:
    st.markdown("""
    <style>
    :root {
      --header-height: 50px;
    }
    .css-z5fcl4 {
      padding-top: 2.5rem;
      padding-bottom: 5rem;
      padding-left: 2rem;
      padding-right: 2rem;
      color: blue;
    }
    .css-1544g2n {
      padding: 0rem 0.5rem 1.0rem;
    }
    [data-testid="stHeader"] {
        background-image: url(/app/static/icons8-astrolabe-64.png);
        background-repeat: no-repeat;
        background-size: contain;
        background-origin: content-box;
        color: blue;
    }

    [data-testid="stHeader"] {
        background-color: rgba(28, 131, 225, 0.1);
        padding-top: var(--header-height);
    }

    [data-testid="stSidebar"] {
        background-color: #e3f2fd; /* Soft blue */
        margin-top: var(--header-height);
        color: blue;
        position: fixed; /* Ensure sidebar is fixed */
        width: 250px; /* Fixed width */
        height: 100vh; /* Full height of the viewport */
        z-index: 999; /* Ensure it stays on top */
        overflow-y: auto; /* Enable scrolling for overflow content */
        padding-bottom: 2rem; /* Extra padding at the bottom */
    }

    [data-testid="stToolbar"]::before {
        content: "Machine Learning - Maintenance Cost Prediction";
    }

    [data-testid="collapsedControl"] {
        margin-top: var(--header-height);
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 100%; /* Sidebar takes full width on small screens */
            height: auto; /* Adjust height for small screens */
            position: relative; /* Sidebar is not fixed on small screens */
            z-index: 1000; /* Ensure it stays on top */
        }

        .css-z5fcl4 {
            padding-left: 1rem; /* Adjust padding for smaller screens */
            padding-right: 1rem;
        }

        [data-testid="stHeader"] {
            padding-top: 1rem; /* Adjust header padding */
        }

        [data-testid="stToolbar"] {
            font-size: 1.2rem; /* Adjust font size for the toolbar */
        }
    }
    </style>
    """, unsafe_allow_html=True)
st.sidebar.header("Preparing Models")
yn = st.sidebar.selectbox('Model are already prepated,do you want to prepare them again?', ['No' ,
                                                     'Yes'])
def PreparingModels():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    import pandas as pd
    import pickle
    df = pd.read_excel('maintenance_coded_outliers_replaced.xlsx')
    # Select relevant features and target variable
    features = [
        'service_duration', 'KMs IN', 'Fuel in', 'damage_code', 
        'car_code', 'Class_code', 'Make_code', 'service_probability'
    ]
    target = 'cost'
    # Drop rows with missing values in the relevant columns
    df_cleaned = df[features + [target]].dropna()
    # Split data into features (X) and target (y)
    X = df_cleaned[features]
    y = df_cleaned[target]
    # Train-test split (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardize numeric features for Ridge and Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, solver='auto'),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        if "Ridge" in name or "Lasso" in name:
            model.fit(X_train_scaled, y_train)  # Use scaled data for Ridge/Lasso
            y_pred = model.predict(X_test_scaled)
            # Save the  model
            with open(name+'-gb_model.pkl', 'wb') as f:
                pickle.dump(model, f)
                print('File Created')
        else:
            model.fit(X_train, y_train)  # Use raw data for others
            y_pred = model.predict(X_test)
            with open(name+'-gb_model.pkl', 'wb') as f:
                pickle.dump(model, f)
                print('File Created')
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "MAE": mae, "R²": r2}
    # Display results
    #st.write(results)
# If user select to prepare the models again, just call fuction PrepareModels
if yn:
    PreparingModels()
    
# Load the saved model
def loadModel(M):
    try:
        with open(f'{M}-gb_model.pkl', 'rb') as f:
             model = pickle.load(f)
             return model
            
    except FileNotFoundError:
        st.error(f"Model file not found. Please upload {M}-gb_model.pkl.")
        st.stop()
st.sidebar.header("Regression Models")
rm = st.sidebar.selectbox('Select Regression Model',['Linear Regression',
                                                     'Ridge Regression' ,
                                                     'Lasso Regression' ,
                                                     'Random Forest'    ,
                                                     'Gradient Boosting'])

models = {
    "Linear Regression": 'Linear Regression',
    "Ridge Regression" : 'Ridge Regression',
    "Lasso Regression" : 'Lasso Regression',
    "Random Forest"    : 'Random Forest',
    "Gradient Boosting": 'Gradient Boosting'
}
model = loadModel(models[rm])
#Load Dataset
try:
    with open('maintenance_coded_outliers_replaced.xlsx', 'rb') as d:
        df = pd.read_excel(d)
except FileNotFoundError:
    st.error("Dataset not found. Please upload 'maintenance_coded_outliers_replaced.xlsx'.")
    st.stop()
    
# Define features list
features = ['service_duration', 'KMs IN', 'Fuel in', 'damage_code', 'car_code', 'Class_code', 'Make_code', 'service_probability']

# Streamlit App Title
st.title(f"Maintenance {rm} Cost Prediction ")

# Section: Input Features
st.sidebar.header("Input Features")
service_duration = st.sidebar.slider("Service Duration (days)", min_value=0, max_value=14, step=1, help="Duration of the service in days.")
kms_in = st.sidebar.slider("Kilometers Driven (km)",min_value=0, max_value=100000, value=0, step=1000, help="Total kilometers driven.")
fuel_in = st.sidebar.slider("Fuel Used (Liters)", min_value=0.0, max_value=1.0, step=0.01, help="Fuel used during service (in liters).")
###
damage_type = st.sidebar.selectbox('Select Damage Type', sorted(df['damage type'].unique()))
temp_df = df.loc[df['damage type'] == damage_type,'damage_code'].values[0]
damage_code = temp_df
#damage_code = st.sidebar.selectbox("Damage Code", [0, 1, 2, 3, 4, 5, 6], help="Type of damage classified as codes.")
#st.sidebar.write("Damage Code",damage_code)

Car = st.sidebar.selectbox('Select Car', sorted(df['car'].unique()))
temp_df = df.loc[df['car'] == Car,'car_code'].values[0]
car_code = temp_df
#car_code = st.sidebar.selectbox("Car Code", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], help="Code representing car type.")
#st.sidebar.write("Car Code",car_code)
#st.sidebar.write(df.loc[df['car_code'] == car_code ,'car'].unique())
class_code = st.sidebar.selectbox('Select Class', sorted(df['Class'].unique()))
temp_df = df.loc[df['Class'] == class_code,'Class_code'].values[0]
class_code = temp_df

#class_code = st.sidebar.selectbox("Car Class", [0, 1, 2, 3, 4], help="Code representing car class.")
#st.sidebar.write(df.loc[df['Class_code'] == class_code ,'Class'].unique())

make_code = st.sidebar.selectbox('Select Make', sorted(df['Make'].unique()))
temp_df = df.loc[df['Make'] == make_code,'Make_code'].values[0]
make_code = temp_df
#make_code = st.sidebar.selectbox("Car Make", [0, 1, 2, 3, 4], help="Code representing car manufacturer.")
#st.sidebar.write(df.loc[df['Make_code'] == make_code ,'Make'].unique())
service_probability = st.sidebar.slider("Service Probability", min_value=0.0, max_value=1.0, step=0.01, help="Likelihood of service being required.")

# Prepare user input
user_input = np.array([service_duration, kms_in, fuel_in, damage_code, car_code, class_code, make_code, service_probability]).reshape(1, -1)
# Initialize prediction to None
prediction = None

# Prediction
st.header("Prediction")
if st.button("Predict Cost"):
    try:
        prediction = model.predict(user_input)[0]
        st.success(f"Predicted Maintenance Cost: ${prediction:.2f}")
        st.write("### Input Data Summary")
        st.dataframe(pd.DataFrame(user_input, columns=features))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
st.header("Model Insights")
# Visualization (Optional)

# Actual vs Predicted Plot Section
if st.checkbox('Show Actual Vs Predicted'):
    # Define relevant features and target variable
    features = [
        'service_duration', 'KMs IN', 'Fuel in', 'damage_code', 
        'car_code', 'Class_code', 'Make_code', 'service_probability'
    ]
    target = 'cost'

    # Drop rows with missing values in the relevant columns
    df_cleaned = df[features + [target]].dropna()

    # Separate features and target variable
    X = df_cleaned[features]
    y = df_cleaned[target]

    # Split dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define and train the Gradient Boosting model
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = gb_model.predict(X_test)

    # Create a DataFrame for Plotly visualization
    actual_vs_pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    # Highlight the user's prediction
    if prediction is not None:
        user_actual = y_test.mean()  # Placeholder for user input (or replace with real value if available)
        user_pred_df = pd.DataFrame({
            'Actual': [user_actual],
            'Predicted': [prediction],
            'Type': ['User Input'],
            'Size': [20]  # Larger size for the user's dot
        })
        actual_vs_pred_df['Type'] = 'Test Data'
        actual_vs_pred_df['Size'] = 2  # Default smaller size for test data
        actual_vs_pred_df = pd.concat([actual_vs_pred_df, user_pred_df], ignore_index=True)
    else:
        actual_vs_pred_df['Type'] = 'Test Data'
        actual_vs_pred_df['Size'] = 2 # Default size

    # Plot using Plotly Express
    fig = px.scatter(
        actual_vs_pred_df,
        x='Actual',
        y='Predicted',
        color='Type',
        size='Size',  # Use the size column for dynamic point sizes
        title='Prediction vs Actual',
        labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
        hover_data={'Type': True,'Size': False},  # Remove 'Size' from hover tooltips
    )

    # Add a perfect prediction line (45-degree line)
    fig.add_shape(
        type='line',
        x0=actual_vs_pred_df['Actual'].min(),
        y0=actual_vs_pred_df['Actual'].min(),
        x1=actual_vs_pred_df['Actual'].max(),
        y1=actual_vs_pred_df['Actual'].max(),
        line=dict(color='Red', dash='dash'),
        name='Perfect Prediction'
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

# Feature Importance Plot section
st.header("Feature Importance")
if st.checkbox("Show Feature Importance"):
    try:
        if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
            importances = model.feature_importances_
        elif isinstance(model, (LinearRegression, Ridge, Lasso)):
            if hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            else:
                st.error(f"Feature importance not supported for {rm}.")
                st.stop()
        else:
            st.error(f"Feature importance not available for {rm}.")
            st.stop()
        
        # Sort features by importance
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = np.array(features)[sorted_idx]
        sorted_importances = importances[sorted_idx]

        # Plot
        importance_df = pd.DataFrame({
            'Feature': sorted_features,
            'Importance': sorted_importances
        })
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title=f'Feature Importance ({rm})',
                     labels={'Feature': 'Features', 'Importance': 'Importance'},
                     color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to compute feature importance: {e}")
