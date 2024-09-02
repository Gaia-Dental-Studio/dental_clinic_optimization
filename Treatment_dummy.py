
# Create a list of working dates
def create_working_dates(start_date, end_date, days_of_week):
    from datetime import timedelta
    
    current_date = start_date
    working_dates = []
    while current_date <= end_date:
        if current_date.weekday() in days_of_week:
            working_dates.append(current_date)
        current_date += timedelta(days=1)
    return working_dates

# Calculate the efficiency score
def calculate_efficiency_score(row):
    if row['Duration'] < row['Expected Duration']:
        return float(format((row['Expected Duration'] - row['Duration']) / row['Expected Duration'], '.3f'))
    else:
        return float(format(((row['Duration'] - row['Expected Duration']) / row['Expected Duration'] * (-1)), '.3f'))
    
def calculate_performance_score(efficiency, rating, success_rate):

    performance_score = format(((efficiency * 100 * 0.4 * 2) + (rating * 20 * 0.3) + (success_rate * 100 * 0.3)), '.3f')

    return float(performance_score)

def performance_score_sheet(dummy_data, efficiency_data):
    import pandas as pd

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
   

# Function to generate random treatments, durations, rating and success rate
def generate_treatments_for_day(dentist, dentists, treatment_data, num_treatments):
    import numpy as np

    treatments = treatment_data.sample(n=num_treatments, replace=True)

    # Ensure the 'Duration' column is of a compatible dtype
    treatments['Duration'] = treatments['Duration'].astype(float)

    #Duration Logic for level 7 dentist
    if dentist == dentists[0]:
        treatments['Duration'] = treatments['Duration'] * np.random.uniform(0.55, 1.05, size=num_treatments)
        probabilities_rating = [0, 0, 0.1, 0.3, 0.6]  # Adjusting to sum up to 1
        probabilities_success = [0.05, 0.95]
    
    #Duration Logic for level 6 dentist
    elif dentist == dentists[1]:
        treatments['Duration'] = treatments['Duration'] * np.random.uniform(0.6, 1.1, size=num_treatments)
        probabilities_rating = [0, 0, 0.1, 0.45, 0.55]  # Adjusting to sum up to 1
        probabilities_success = [0.1, 0.9]
    
    #Duration Logic for level 5 dentist
    elif dentist == dentists[2]:
        conditions = [
            (treatments['Treatment'] == 'Premium Crown', np.random.uniform(0.62, 1.1, size=(treatments['Treatment'] == 'Premium Crown').sum())),
            (treatments['Treatment'] == 'Premium Bridge', np.random.uniform(0.68, 1.2, size=(treatments['Treatment'] == 'Premium Bridge').sum())),
            (treatments['Treatment'] == 'Advanced Filling', np.random.uniform(0.65, 1.15, size=(treatments['Treatment'] == 'Advanced Filling').sum())),
            (treatments['Treatment'] == 'Advanced Gum Treatment', np.random.uniform(0.7, 1.15, size=(treatments['Treatment'] == 'Advanced Gum Treatment').sum()))
        ]
        for condition, factor in conditions:
            treatments.loc[condition, 'Duration'] *= factor

        # Apply a default condition for treatments not specified
        default_condition = ~treatments['Treatment'].isin(['Premium Crown', 'Premium Bridge', 'Advanced Filling', 'Advanced Gum Treatment'])
        treatments.loc[default_condition, 'Duration'] *= np.random.uniform(0.85, 1.35, size=default_condition.sum())

        probabilities_rating = [0, 0, 0.25, 0.3, 0.45]  # Adjusting to sum up to 1
        probabilities_success = [0.15, 0.85]

    # Duration Logic for level 4 dentist with multiple conditions
    elif dentist == dentists[3]:
        conditions = [
            (treatments['Treatment'] == 'Basic Scaling', np.random.uniform(0.62, 1.1, size=(treatments['Treatment'] == 'Basic Scaling').sum())),
            (treatments['Treatment'] == 'Basic Filling', np.random.uniform(0.62, 1.1, size=(treatments['Treatment'] == 'Basic Filling').sum())),
            (treatments['Treatment'] == 'Basic Tooth Extraction', np.random.uniform(0.62, 1.1, size=(treatments['Treatment'] == 'Basic Tooth Extraction').sum())),
            (treatments['Treatment'] == 'Dental Spa', np.random.uniform(0.62, 1.1, size=(treatments['Treatment'] == 'Dental Spa').sum())),
            (treatments['Treatment'] == 'Mouth Guard', np.random.uniform(0.62, 1.1, size=(treatments['Treatment'] == 'Mouth Guard').sum()))
        ]
        for condition, factor in conditions:
            treatments.loc[condition, 'Duration'] *= factor

        # Apply a default condition for treatments not specified
        default_condition = ~treatments['Treatment'].isin(['Basic Scaling', 'Basic Filling', 'Basic Tooth Extraction', 'Dental Spa', 'Mouth Guard'])
        treatments.loc[default_condition, 'Duration'] *= np.random.uniform(0.85, 1.35, size=default_condition.sum())

        probabilities_rating = [0, 0.05, 0.25, 0.3, 0.4]  # Adjusting to sum up to 1
        probabilities_success = [0.2, 0.8]

    #Duration Logic for level 3 dentist
    elif dentist == dentists[4]:
        treatments['Duration'] = treatments['Duration'] * np.random.uniform(0.77, 1.23, size=num_treatments)  
        probabilities_rating = [0.1, 0.15, 0.3, 0.2, 0.35]  # Adjusting to sum up to 1
        probabilities_success = [0.25, 0.75]

    #Duration Logic for level 2 dentist
    elif dentist == dentists[5]:
        treatments['Duration'] = treatments['Duration'] * np.random.uniform(0.81, 1.3, size=num_treatments)
        probabilities_rating = [0.1, 0.15, 0.3, 0.25, 0.3]  # Adjusting to sum up to 1
        probabilities_success = [0.27, 0.73]

    #Duration Logic for level 1 dentist
    else:
        treatments['Duration'] = treatments['Duration'] * np.random.uniform(0.85, 1.35, size=num_treatments) 
        probabilities_rating = [0.15, 0.15, 0.2, 0.25, 0.25]  # Adjusting to sum up to 1   
        probabilities_success = [0.3, 0.7] 

    ratings = []
    successes = []

    for _ in range(num_treatments):

        success_score = np.random.choice([0, 1], p=probabilities_success)
        successes.append(success_score)

        if success_score == 0:
            ratings.append(np.random.randint(1, 3))
        else:
            probabilities_rating[-1] = 1 - sum(probabilities_rating[:-1])
            assert np.isclose(sum(probabilities_rating), 1)
            ratings.append(np.random.choice([1, 2, 3, 4, 5], p=probabilities_rating))

    treatments['Rating'] = ratings
    treatments['Success'] = successes

    return treatments

def create_dummy_treatments():
    import pandas as pd
    import numpy as np
    from datetime import date

    # Load the excel file
    file_path = './Data/clinic_labor.xlsx'
    treatment_path = './Data/Service_list.xlsx'
    clinic_data = pd.read_excel(treatment_path, sheet_name='Main')
    dentist_data = pd.read_excel(file_path, sheet_name='Dentists')

    # Extract the relevant columns
    treatment_data = clinic_data[['Treatment Menu Names', 'Duration']].copy()
    treatment_data.columns = ['Treatment', 'Duration']

    # Define the parameters
    dentists = dentist_data['Dentist Name']
    start_date = date(2023, 1, 1)
    end_date = date(2023, 6, 30)
    # working_days_per_week = 5
    days_of_week = [0, 1, 2, 3, 4]  # Monday to Friday

    working_dates = create_working_dates(start_date, end_date, days_of_week)

    # Generate dummy data
    data = []

    for date in working_dates:
        for dentist in dentists:
            num_treatments = np.random.randint(14, 18)  # average 16 treatments with +-2 range
            treatments_for_day = generate_treatments_for_day(dentist, dentists, treatment_data, num_treatments)
            for _, row in treatments_for_day.iterrows():
                data.append({
                    'Date': date,
                    'Dentist Name': dentist,
                    'Treatment': row['Treatment'],
                    'Duration': row['Duration'],
                    'Rating': row['Rating'],
                    'Success': row['Success']
                })

   # Create a DataFrame
    dummy_data = pd.DataFrame(data)

    # Ensure 'Duration' column is numeric
    dummy_data['Duration'] = pd.to_numeric(dummy_data['Duration'])

    # Calculate the average duration for each treatment per dentist
    average_duration_per_dentist_treatment = dummy_data.groupby(['Dentist Name', 'Treatment']).agg({'Duration': 'mean'}).reset_index()

    # Merge with the expected durations from the treatment data
    expected_durations = treatment_data.rename(columns={'Treatment': 'Treatment', 'Duration': 'Expected Duration'})
    efficiency_data = average_duration_per_dentist_treatment.merge(expected_durations, on='Treatment')

    efficiency_data['Efficiency Score'] = efficiency_data.apply(calculate_efficiency_score, axis=1)

    # Pivot the data to get the desired format for performance evaluation
    evaluation_df = efficiency_data.pivot(index='Dentist Name', columns='Treatment', values='Efficiency Score').reset_index()

    # Pivot the data to get the desired format for average duration
    average_duration_df = average_duration_per_dentist_treatment.pivot(index='Dentist Name', columns='Treatment', values='Duration').reset_index()

    # Calculate treatment score for each dentist
    performance_score_df = performance_score_sheet(dummy_data, evaluation_df)

    # Create the Specialty DataFrame
    specialty_data = []

    for dentist in dentists:
        specialty_treatments = efficiency_data[(efficiency_data['Dentist Name'] == dentist) & (efficiency_data['Efficiency Score'] > 0.02)]
        specialties = specialty_treatments['Treatment'].tolist()
        efficiency_scores = specialty_treatments['Efficiency Score'].tolist()
        specialty_data.append({
            'Dentist Name': dentist,
            'Specialty': specialties,
            'Efficiency': efficiency_scores,
            'Wage/Hour': "Edit This!"
        })

    specialty_df = pd.DataFrame(specialty_data)

    # Save the dummy data, performance evaluation, average duration, and specialty data into the same Excel file with different sheets
    output_file_path = './Data/Dummy_Treatment.xlsx'

    with pd.ExcelWriter(output_file_path) as writer:
        dummy_data.to_excel(writer, sheet_name='Dummy Treatments', index=False)
        average_duration_df.to_excel(writer, sheet_name='Average', index=False)
        evaluation_df.to_excel(writer, sheet_name='Efficiency', index=False)
        performance_score_df.to_excel(writer, sheet_name='Performance', index=False)
        specialty_df.to_excel(writer, sheet_name='Specialty', index=False)

    print(f"Data saved successfully to {output_file_path}")

    return(dummy_data, average_duration_df, evaluation_df)

#dummy_dentist_json for dentist performance assessment
dummy_dentist_json = {
    'dentist_name': 'Arnold',
    'date': '2023-07-01',
    'treatment_name': 'Basic Scaling',
    'Duration': 30.25, #Minutes
    'Rating': 3,
    'Success': 1
}

#dummy_treatment_demand_json for forecasting treatment demand dataset
dummy_treatment_demand_json = {
    'treatment_name': 'Basic Scaling',
    'treatment_count': 7,
    'date': '2023-07-01'
}

#dentist_json for explaining dentist profile (specialty & wage/hour)
dentist_json = {
    'dentist_name': 'Arnold',
    'Specialty': [
         {
            'treatment_name': 'Basic Scaling',
            'efficiency': 0.18,
            'incentive': 0.1,
        },
        {
            'treatment_name': 'Premium Crown',
            'efficiency': 0.25,
            'incentive': 0.2
        },
        {
            'treatment_name': 'Advanced Filling',
            'efficiency': 0.22,
            'incentive': 0.2
        }
    ],
    'wage_hour': 67000,
    'dentist_level': 3,
}

#constraint_json -> constraints for scheduling optimization
constraint_json = {
    'working_hour_weekly': 40,
    'working_hour_daily': 8
}

#treatment_json represent treatment information, for calculating profit for each treatment
treatment_json = {
    'treatment_name': 'Basic Scaling',
    'expected_duration': 45, #Minutes
    'price': 500000,
    'cogs' : [
        {
            'material_name':'N05 Mask',
            'total_quantity': 2,
            'uom': 'pcs',
            'changes': 0,
            'equivalent_existance': 1,
            'price_uom': 7250,
            'total_cost': 14500
        },
        {
            'material_name':'Prescription',
            'total_quantity': 1,
            'uom': 'pcs',
            'changes': 0,
            'equivalent_existance': 1,
            'price_uom': 15000,
            'total_cost': 15000
        },
        {
            'material_name':'Povidone iodine',
            'total_quantity': 2,
            'uom': 'ml',
            'changes': 0,
            'equivalent_existance': 1,
            'price_uom': 77,
            'total_cost': 154
        },
    ],
    'lab_cost': 0
}

#The output of the schedule optimization should be with this format, roster schedule is the table content
optimized_schedule_json = {
    'roster_name': 'Top 1 Roster',
    'roster_schedule': [
        {
            'date': '2023-07-03',
            'dentist_names': ['Meyden', 'Ashley'],
            'working_hours': [10, 8],
            'total_cost': 950000
        },
        {
            'date': '2023-07-04',
            'dentist_names': ['Arnold', 'Miranda', 'Arnold'],
            'working_hours': [8, 6.5, 7],
            'total_cost': 1250000
        }
    ],
    'schedule_total_cost' : 2200000 #sum of total cost for each date
}