import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('influxdb')
from influxdb import InfluxDBClient 
import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pickle

# Function to convert database data to dataframe
def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index
    return df

# Access database
client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
client.switch_database('orkney')
# ----------------------------------------------------------------------

# Get the last 90 days of power generation data
days = 90
sql_query = f"SELECT * FROM Generation where time > now()-{days}d"
generation = client.query(sql_query)
gen_df = get_df(generation)

# Get the last 90 days of weather forecasts with the shortest lead time
wind = client.query("SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'")
wind_df = get_df(wind)

# Align the dataframes
df = gen_df.merge(wind_df, on = 'time')
X = df[['Direction', 'Lead_hours', 'Source_time', 'Speed']]
y = df['Total']

# Train and test split
X_train = X[:int(X.shape[0]*0.85)]
X_test = X[int(X.shape[0]*0.85):]
y_train = y[:int(X.shape[0]*0.85)]
y_test = y[int(X.shape[0]*0.85):]
# ----------------------------------------------------------------------
for degree in range(1,6):
    # Start run
    with mlflow.start_run():
        # Preprocessing
        Preprocessing = ColumnTransformer(transformers=[
        ('encode', OneHotEncoder(), ['Direction']), #Altering the wind direction to be a usable feature 
        ('normalize', StandardScaler(), ['Speed']), #Scaling the data to be within a set range
        ('polynomial', PolynomialFeatures(degree), ['Speed']) # Create polynomial features for polynomial regression
        ])   

        # Training model
        complete_pipeline = Pipeline([
        ('preprocess', Preprocessing),
        ('regression', LinearRegression()),  
        ])

        # Fit the model 
        model = complete_pipeline.fit(X_train, y_train)
        
        # Load in the currently saved model (if there is one)
        try:
            current_model = pickle.load(open( "model.p", "rb" ))  
        except:
            pickle.dump(model, open('model.p', 'wb'))
            current_model = pickle.load(open( "model.p", "rb" )) 

        # Logging metrics
        accuracy = model.score(X_test, y_test)
        saved_accuracy = current_model.score(X_test, y_test)
        mlflow.log_metric('New model accuracy', accuracy)
        mlflow.log_metric('Saved model accuracy', saved_accuracy)
        
        # If new model is better, save that one instead, if not use the currently saved model to do forecasting
        if current_model.score(X_test, y_test) < model.score(X_test, y_test):
            pickle.dump(model, open( "model.p", "wb" ))
            print('The new model is better')
        else: 
            model = current_model
            print('The current model is better')

        # Logging parameters
        mlflow.log_param('degree', degree)
        mlflow.log_param('days', days)

        # Logging final model
        if degree == 5:
       		mlflow.sklearn.log_model(model, 'model')

      








