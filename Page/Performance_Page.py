import pandas as pd
import logging
import streamlit as st
import matplotlib.pyplot as plt


import locale
locale.setlocale(locale.LC_ALL, '')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%message')

# Load the necessary files
clinic_labor_file = './Data/Dummy_Treatment.xlsx'
service_list_file = './Data/Service_list.xlsx'

#Extract necessary information
clinic_labor = pd.read_excel(clinic_labor_file, sheet_name='Average')
clinic_labor_dummy = pd.read_excel(clinic_labor_file, sheet_name='Dummy Treatments')
service_list = pd.read_excel(service_list_file, sheet_name='Main') 
dentist = clinic_labor['Dentist Name']
treatment = service_list['Treatment Menu Names']


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
    success_rate = float((target_success['Success'].value_counts().sum()) * 100)/float(count)
    

   # Calculate the performance score
    if float(average) < float(target_expected_duration):
        efficiency_score = (target_expected_duration - float(average)) / target_expected_duration * 100
    else:
        efficiency_score = (float(average) - target_expected_duration) / target_expected_duration * -100
    
    # print(f"efficiency : {(efficiency_score * 0.4)}  rating: {(rating_avg * 20 * 0.3)}  success: {(success_rate * 0.3)} ")
    performance_score = format(((efficiency_score * 0.4) + (rating_avg * 20 * 0.3) + (success_rate * 0.3)), '.3f')

    efficiency_score = format(efficiency_score, '.2f')

    # Extract the series for plotting
    treatment_duration_series = target_treatment[['Date', 'Duration']].set_index('Date')

    return(count, average, max, min, std, efficiency_score, treatment_duration_series, target_expected_duration, rating_avg, success_rate, performance_score)




# Streamlit app
def display():
    st.title("Dentist Performance Analysis")

    global clinic_labor, dentist, treatment

    with st.form("Form"):
        dentist_name = st.selectbox("Dentist Name:", dentist, key='dentist_name')
        treatment_name = st.selectbox("Treatment Name:", treatment, key='treatment_name')

        if st.form_submit_button("Analyze"):
            treatment_count, average_result, max_result, min_result, std_result, efficiency_result, treatment_series, target_duration, rating, success_rate, performance_score = get_performance_metrics(dentist_name, treatment_name, clinic_labor_dummy, service_list)

            st.subheader(f"{dentist_name} - {treatment_name}, History:")
            st.write(f"Treatment Counts: {treatment_count} times")
            st.write(f"Average Duration: {average_result} minutes")
            st.write(f"Maximum Duration: {max_result} minutes")
            st.write(f"Minimum Duration: {min_result} minutes" )
            st.write(f"Standard Deviation: {std_result} minutes")
            st.write(f"Efficiency: {efficiency_result} %")
            st.write(f"Treatment Rating (1 ~ 5): {rating}")
            st.write(f"Success Rate: {format((success_rate), '.3f')} %")
            st.write(f"Performance Score (0 ~ 100): {performance_score}")

            plt.figure(figsize=(10, 6))
            plt.plot(treatment_series, marker='o', linestyle='-', label='Treatment Duration')
            plt.axhline(y=target_duration, color='r', linestyle='--', label=f'Expected Duration ({target_duration})')
            plt.title(f"{dentist_name} - {treatment_name}, History Trend")
            plt.xlabel('Date')
            plt.ylabel('Duration (minutes)')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)


    