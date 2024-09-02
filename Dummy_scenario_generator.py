import pandas as pd
import random
import os
from datetime import timedelta


def generate_dummy_scenario(start_date, end_date, random_maximum):
    # Define treatments and the range for treatment counts per day
    min_treatment_count = 0
    max_treatment_count = random_maximum

    # Load the service list file to get treatment durations
    service_list_file = './Data/Service_list.xlsx'
    service_list_df = pd.read_excel(service_list_file, sheet_name='Main')

    # Extract relevant columns from service_list_df
    service_list_processed = service_list_df[[
        'Treatment Menu Names', 
        'Duration'
    ]]
    service_list_processed.columns = [
        'Treatment_Names', 
        'Duration_minutes'
    ]

    treatments = service_list_processed['Treatment_Names']

    # Convert to dictionary for quick lookup
    treatment_durations = dict(zip(service_list_processed['Treatment_Names'], service_list_processed['Duration_minutes']))

    # Function to generate random treatment counts for a given day
    def generate_daily_treatment_counts():
        return {treatment: random.randint(min_treatment_count, max_treatment_count) for treatment in treatments}

    # Generate dates for the given range (Monday to Friday)
    delta = timedelta(days=1)
    dates = []
    while start_date <= end_date:
        if start_date.weekday() < 5:  # Monday to Friday are 0-4
            dates.append(start_date)
        start_date += delta

    # Generate the dataset
    data = []
    for date in dates:
        daily_counts = generate_daily_treatment_counts()
        total_hours_required = sum(daily_counts[treatment] * treatment_durations[treatment] / 60 for treatment in treatments if treatment in treatment_durations)
        daily_counts['Date'] = date.strftime('%Y-%m-%d')
        daily_counts['Hour Required'] = total_hours_required
        data.append(daily_counts)

    # Convert to DataFrame
    treatment_demand_df = pd.DataFrame(data)

    #Save the dataset to a CSV file
    output_dir = './Data/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'Dummy_Scenario.csv')
    treatment_demand_df.to_csv(output_file, index=False)

    return treatment_demand_df