import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Define the LSTM model
def create_lstm_model(X_train_scaled, y_train, units=50, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Define the GRU model
def create_gru_model(X_train_scaled, y_train, units=50, optimizer='adam'):
    model = Sequential()
    model.add(GRU(units, activation='relu', return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(units, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Load the necessary files
def forecast_profit(df1, df2):
    treatment_profits = df1[['Treatment Menu Names', 'Profit (Price - COGS)']]
    forecast_df_merged = df2.melt(id_vars=['Date'], var_name='Treatment Menu Names', value_name='Count')

    # Merged & Calculate
    merged_df = pd.merge(forecast_df_merged, treatment_profits, on='Treatment Menu Names')
    merged_df['Total Revenue'] = merged_df['Count'] * merged_df['Profit (Price - COGS)']

    # Summarize the total revenue
    total_revenue = format(merged_df['Total Revenue'].sum(), '.3f')
    
    return float(total_revenue)

def train_and_forecast(scenario_df, service_df):
 
    scenario_df = scenario_df.drop(columns=['Hour Required'])

    # Prepare the data
    X = scenario_df.drop(columns=['Date'])
    y = scenario_df.drop(columns=['Date'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for Grid Search for Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the Random Forest model
    rf = RandomForestRegressor(random_state=42)

    # Initialize Grid Search for Random Forest
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    # Fit the Grid Search model
    grid_search_rf.fit(X_train, y_train)

    # Get the best parameters for Random Forest
    best_params_rf = grid_search_rf.best_params_
    print(f"Best parameters for Random Forest: {best_params_rf}")

    # Train the final Random Forest model with the best parameters
    best_rf = RandomForestRegressor(**best_params_rf, random_state=42)
    best_rf.fit(X_train, y_train)

    # Make predictions with Random Forest
    y_pred_rf = best_rf.predict(X_test)

    # Calculate evaluation metrics for Random Forest
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)


    # Prepare data for LSTM and GRU models
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        # Define hyperparameters
    units_list = [50, 100, 150]
    optimizers = ['adam', 'rmsprop']
    batch_sizes = [16, 32]
    epochs_list = [10, 20, 30]

    # Custom grid search for LSTM
    best_rmse_lstm = float('inf')
    best_params_lstm = None
    best_model_lstm = None

    for units in units_list:
        for optimizer in optimizers:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    model = create_lstm_model(X_train_scaled, y_train, units=units, optimizer=optimizer)
                    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2, callbacks=[EarlyStopping(patience=3, monitor='val_loss')])
                    y_pred_lstm = model.predict(X_test_scaled)
                    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
                    if rmse_lstm < best_rmse_lstm:
                        best_rmse_lstm = rmse_lstm
                        best_params_lstm = {'units': units, 'optimizer': optimizer, 'batch_size': batch_size, 'epochs': epochs}
                        best_model_lstm = model

    print(f"Best parameters for LSTM: {best_params_lstm}")

    # Custom grid search for GRU
    best_rmse_gru = float('inf')
    best_params_gru = None
    best_model_gru = None

    for units in units_list:
        for optimizer in optimizers:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    model = create_gru_model(X_train_scaled, y_train, units=units, optimizer=optimizer)
                    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2, callbacks=[EarlyStopping(patience=3, monitor='val_loss')])
                    y_pred_gru = model.predict(X_test_scaled)
                    rmse_gru = np.sqrt(mean_squared_error(y_test, y_pred_gru))
                    if rmse_gru < best_rmse_gru:
                        best_rmse_gru = rmse_gru
                        best_params_gru = {'units': units, 'optimizer': optimizer, 'batch_size': batch_size, 'epochs': epochs}
                        best_model_gru = model

    print(f"Best parameters for GRU: {best_params_gru}")

    # Select the best model
    best_model = None
    if rmse_rf < best_rmse_lstm and rmse_rf < best_rmse_gru:
        best_model = best_rf
    elif best_rmse_lstm < rmse_rf and best_rmse_lstm < best_rmse_gru:
        best_model = best_model_lstm
    else:
        best_model = best_model_gru

    # Create a date range for forecasting
    date_range = pd.date_range(start=scenario_df['Date'].max(), periods=15, freq='B')

    # Prepare the data for forecasting
    X_future_mean = np.tile(X_train.mean().values, (len(date_range), 1))
    noise = np.random.normal(0, 0.6, X_future_mean.shape)  # Increase noise
    X_future = pd.DataFrame(X_future_mean + noise, columns=X_train.columns)
    X_future['Date'] = date_range

    # Forecast the treatments using the best model
    if best_model in [best_rf]:
        X_future_scaled = X_future.drop(columns=['Date'])
        forecast = best_model.predict(X_future_scaled)
    elif best_model in [best_model_lstm, best_model_gru]:
        X_future_scaled = scaler.transform(X_future.drop(columns=['Date']))
        X_future_scaled = X_future_scaled.reshape((X_future_scaled.shape[0], 1, X_future_scaled.shape[1]))
        forecast = best_model.predict(X_future_scaled)

    # Add randomness to the predictions
    forecast_random = forecast + np.random.normal(0, 0.6, forecast.shape)

    # Apply rounding to forecasted results
    forecast_rounded = np.where(forecast_random % 1 >= 0.5, np.ceil(forecast_random), np.floor(forecast_random)).astype(int)

    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame(forecast_rounded, columns=y.columns, index=date_range)

    # Convert the index to string to remove the time part and create a 'Date' column
    forecast_df['Date'] = forecast_df.index.strftime('%Y-%m-%d')
    forecast_df = forecast_df.reset_index(drop=True)
    forecast_df.drop(index=forecast_df.index[0], inplace=True, axis=0)


    # Extract relevant columns
    service_duration = service_df[['Treatment Menu Names', 'Duration']]
    service_duration.columns = ['Treatment', 'Duration']

    # Convert Duration from minutes to hours
    service_duration['Duration'] = service_duration['Duration'] / 60

    # Calculate the Hour Required for each day in the forecast
    hour_required = []

    for index, row in forecast_df.iterrows():
        total_hours = 0
        for treatment in forecast_df.columns[:-1]:  # Exclude the 'Date' column
            treatment_count = row[treatment]
            duration_per_treatment = service_duration.loc[service_duration['Treatment'] == treatment, 'Duration'].values[0]
            total_hours += treatment_count * duration_per_treatment
        hour_required.append(total_hours)

    # Add the Hour Required column to the forecast DataFrame
    forecast_df['Hour Required'] = hour_required

    # Reorder columns to place 'Date' before 'Hour Required'
    cols = list(forecast_df.columns)
    cols.insert(-1, cols.pop(cols.index('Date')))

    forecast_df = forecast_df[cols]

    #LSTM Evaluation Score
    print(f"{'='*10}{' Random Forest '}{'='*10}")
    print(f"Random Forest RMSE: {rmse_rf}")
    print(f"Random Forest R²: {r2_rf}")

    #LSTM Evaluation Score
    print(f"{'='*10}{' LSTM '}{'='*10}")
    print(f"LSTM RMSE: {best_rmse_lstm}")
    r2_lstm = r2_score(y_test, y_pred_lstm)
    print(f"LSTM R²: {r2_lstm}")

    #GRU Evaluation Score
    print(f"{'='*10}{' GRU '}{'='*10}")
    print(f"GRU RMSE: {best_rmse_gru}")
    r2_gru = r2_score(y_test, y_pred_gru)
    print(f"GRU R²: {r2_gru}")

    evaluation = {
        'Evaluation': [
            {
                'Model': 'Random Forest',
                'RMSE' : rmse_rf,
                'R²' : r2_lstm
            },
            {
                'Model': 'LSTM',
                'RMSE': rmse_lstm,
                'R²': r2_lstm
            },
            {
                'Model': 'GRU',
                'RMSE': rmse_gru,
                'R²': r2_gru
            }
        ] 
    }

    # Save the updated forecast to a new Excel file
    # updated_forecast_excel_path = './Data/forecast_2_weeks.xlsx'
    # forecast_df.to_excel(updated_forecast_excel_path, index=False)

    return forecast_df, evaluation, forecast_profit(service_df, forecast_df)
