import pandas as pd
import calendar

#Calculate the efficiency score
def calculate_efficiency_score_row(row):
    if row['Duration'] < row['Expected Duration']:
        return float(format((row['Expected Duration'] - row['Duration']) / row['Expected Duration'], '.3f'))
    else:
        return float(format(((row['Duration'] - row['Expected Duration']) / row['Expected Duration'] * (-1)), '.3f'))
    
def calculate_performance_score(efficiency, rating, success_rate):

    performance_score = format(((efficiency * 100 * 0.4 * 2) + (rating * 20 * 0.3) + (success_rate * 100 * 0.3)), '.3f')

    return float(performance_score)

def performance_score_sheet(dummy_data, efficiency_data):

    dummy_data['Rating'] = pd.to_numeric(dummy_data['Rating'], errors='coerce')
    dummy_data['Success'] = pd.to_numeric(dummy_data['Success'], errors='coerce')

    grouped_data = dummy_data.groupby(['Dentist Name', 'Treatment']).agg({
        'Rating': 'mean',
        'Success': 'mean'
    }).reset_index()

    grouped_data = grouped_data.merge(efficiency_data.melt(id_vars='Dentist Name', var_name='Treatment', value_name='Efficiency'), on=['Dentist Name', 'Treatment'], how='left')

    grouped_data['Performance Score'] = grouped_data.apply(
        lambda row: calculate_performance_score(row['Efficiency'], row['Rating'], row['Success']), axis=1)

    performance_df = grouped_data.pivot(index='Dentist Name', columns='Treatment', values='Performance Score').reset_index()
    performance_df = performance_df.fillna(0)

    return performance_df



def get_performance_metrics(dentist_name, treatment, df, df2):
    target_dentist = df[df['Dentist Name'] == dentist_name]
    target_treatment = target_dentist[target_dentist['Treatment'] == treatment]

    target_success = target_treatment[target_treatment['Success'] == 1]

    target_expected_duration = df2[df2['Treatment Menu Names'] == treatment]
    target_expected_duration = target_expected_duration['Duration'].values[0]

    count = target_treatment.value_counts().sum()
    average = format(target_treatment['Duration'].mean(), '.3f')
    max = format(target_treatment['Duration'].max(), '.3f')
    min = format(target_treatment['Duration'].min(), '.3f')
    std = format(target_treatment['Duration'].std(), '.3f')
    rating_avg = float(format(target_treatment['Rating'].mean(), '.1f'))
    success_rate = float(target_success['Success'].value_counts().sum())/float(count)

   # Calculate the performance score
    if float(average) < float(target_expected_duration):
        efficiency = (target_expected_duration - float(average)) / target_expected_duration * 1
    else:
        efficiency = (float(average) - target_expected_duration) / target_expected_duration * -1

    efficiency = float(format(efficiency, '.3f'))

    performance_score = calculate_performance_score(efficiency, rating_avg, success_rate)


    return(count, average, max, min, std, efficiency, rating_avg, success_rate, float(performance_score))


def get_performance_metrics_chart(dentist_name, treatment, df, df2):
    target_dentist = df[df['Dentist Name'] == dentist_name]
    target_treatment = target_dentist[target_dentist['Treatment'] == treatment]

    target_expected_duration = df2[df2['Treatment Menu Names'] == treatment]
    target_expected_duration = target_expected_duration['Duration'].values[0]

    # Extract the first and last dates
    start_date = df['Date'].min()
    end_date = df['Date'].max()

    # Convert month numbers to month names
    start_month = calendar.month_name[start_date.month]
    end_month = calendar.month_name[end_date.month]

    # Extract the series for plotting
    treatment_duration_series = target_treatment[['Date', 'Duration']].set_index('Date')

    return(treatment_duration_series, target_expected_duration, start_month, end_month)


def Anlyze_dentist_past_data(df_dentist_report, df_service):

    dentists = df_dentist_report['Dentist Name'].unique()

    # Ensure 'Duration' column is numeric
    df_dentist_report['Duration'] = pd.to_numeric(df_dentist_report['Duration'])

    # Calculate the average duration for each treatment per dentist
    average_duration_per_dentist_treatment = df_dentist_report.groupby(['Dentist Name', 'Treatment']).agg({'Duration': 'mean'}).reset_index()

    # Merge with the expected durations from the treatment data
    expected_durations = df_service.rename(columns={'Treatment Menu Names': 'Treatment', 'Duration': 'Expected Duration'})
    efficiency_data = average_duration_per_dentist_treatment.merge(expected_durations, on='Treatment')

    efficiency_data['Efficiency Score'] = efficiency_data.apply(calculate_efficiency_score_row, axis=1)


    # Pivot the data to get the desired format for average duration
    average_duration_df = average_duration_per_dentist_treatment.pivot(index='Dentist Name', columns='Treatment', values='Duration').reset_index()

    # Create payload1 in JSON format
    payload1 = {
        "analyzed_sheet": "average_durations",
        "df_content": []
    }

    for _, row in average_duration_df.iterrows():
        dentist_entry = {
            "dentist_name": row['Dentist Name']
        }
        for treatment in average_duration_df.columns[1:]:  # Skipping the 'Dentist Name' column
            dentist_entry[treatment] = row[treatment]
        payload1["df_content"].append(dentist_entry)


    # Pivot the data to get the desired format for performance evaluation
    evaluation_df = efficiency_data.pivot(index='Dentist Name', columns='Treatment', values='Efficiency Score').reset_index()

    # Create payload2 in JSON format
    payload2 = {
        "analyzed_sheet": "efficiency_scores",
        "df_content": []
    }

    for _, row in evaluation_df.iterrows():
        dentist_entry = {
            "dentist_name": row['Dentist Name']
        }
        for treatment in evaluation_df.columns[1:]:  # Skipping the 'Dentist Name' column
            dentist_entry[treatment] = row[treatment]
        payload2["df_content"].append(dentist_entry)


    # Calculate treatment score for each dentist
    performance_score_df = performance_score_sheet(df_dentist_report, evaluation_df)

    payload3 = {
        "analyzed_sheet": "performance_scores",
        "df_content": []
    }

    for _, row in performance_score_df.iterrows():
        dentist_entry = {
            "dentist_name": row['Dentist Name']
        }
        for treatment in performance_score_df.columns[1:]:  # Skipping the 'Dentist Name' column
            dentist_entry[treatment] = row[treatment]
        payload3["df_content"].append(dentist_entry)

   
    # Create the Specialty DataFrame
    specialty_data = []

    # Create the Specialty DataFrame for payload4
    payload4 = {
        "analyzed_sheet": "specialties",
        "df_content": []
    }

    for dentist in dentists:
        specialty_treatments = efficiency_data[(efficiency_data['Dentist Name'] == dentist) & (efficiency_data['Efficiency Score'] > 0.02)]
        specialties = specialty_treatments['Treatment'].tolist()
        efficiency_scores = specialty_treatments['Efficiency Score'].tolist()
        specialty_data = {
            'Dentist Name': dentist,
            'Specialty': specialties,
            'Efficiency': efficiency_scores,
            'Wage/Hour': "Edit This!"
        }
        payload4["df_content"].append(specialty_data)


    return payload1, payload2, payload3, payload4