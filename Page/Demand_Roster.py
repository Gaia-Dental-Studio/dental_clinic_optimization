import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD
import streamlit as st
import os
from Dummy_scenario_generator import generate_dummy_scenario
from datetime import datetime


def display():
    if 'dummy_df' not in st.session_state:
        st.session_state.dummy_df = pd.DataFrame()
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    st.title("Demand Roster Optimization")

    # Generate Dummy Scenario
    st.header("Generate Dummy Scenario")
    start_date = st.date_input("Start Date", value=datetime(2023, 6, 1))
    end_date = st.date_input("End Date", value=datetime(2023, 6, 30))

    if st.button("Generate Dummy Scenario"):
        if start_date > end_date:
            st.error("Start Date should be before End Date")
        else:
            file_path = generate_dummy_scenario(start_date, end_date, random_maximum=10)
            st.session_state.dummy_df = file_path
            st.success(f"Dummy Scenario generated!")

    # Display the dummy dataframe if it exists
    if not st.session_state.dummy_df.empty:
        st.write(st.session_state.dummy_df)

    # Load data
    treatment_demand_file = './Data/Dummy_Scenario.csv'
    clinic_labor_file = './Data/clinic_labor.xlsx'
    service_list_file = './Data/Service_list.xlsx'
    
    treatment_demand_df = pd.read_csv(treatment_demand_file)
    clinic_labor_df = pd.read_excel(clinic_labor_file, sheet_name='Labor')
    service_list_df = pd.read_excel(service_list_file, sheet_name='Main')

    # Extract relevant columns from clinic_labor_df
    clinic_labor_processed = clinic_labor_df[[
        'List Name', 
        '% INCREASING SALARY', 
        'Timeline', 
        '% of expected efficiency increase in performance', 
        'Base wage hourly (IDR)'
    ]]
    
    # Renaming columns for easier access
    clinic_labor_processed.columns = [
        'List_Name', 
        'Increasing_Salary', 
        'Timeline', 
        'Efficiency_increase_percent', 
        'Base_wage_hourly_IDR'
    ]
    
    # Extract relevant columns from service_list_df
    service_list_processed = service_list_df[[
        'Treatment Menu Names', 
        'Duration', 
        'Profit (Price - COGS)'
    ]]
    
    # Renaming columns for easier access
    service_list_processed.columns = [
        'Treatment_Names', 
        'Duration_minutes',
        'Profit'
    ]

    # Filter the clinic labor data to only include Dent Lv1 ~ Lv7
    filtered_clinic_labor = clinic_labor_processed[clinic_labor_processed['List_Name'].str.contains('Dent Lv[1-7]')]
    
    # Extract necessary data from the filtered data
    dentists_filtered = filtered_clinic_labor['List_Name'].tolist()
    
    efficiency_increase_filtered = filtered_clinic_labor['Efficiency_increase_percent'].fillna(0).tolist()
    salary_increase_filtered = filtered_clinic_labor['Increasing_Salary'].tolist()
    base_wages_filtered = filtered_clinic_labor['Base_wage_hourly_IDR'].tolist()
    
    services = service_list_processed['Treatment_Names'].tolist()
    durations = service_list_processed['Duration_minutes'].tolist()
    profits = service_list_processed['Profit'].tolist()
    
    # Streamlit input for maximum working hours per dentist per day
    max_working_hours_per_day = st.number_input("Maximum working hours per dentist per day", value=8, min_value=1, max_value=24)

    if st.button("Run Optimization"):
        # Function to get the total hours required for a given day's treatments
        def get_total_hours_required(day_treatments):
            total_hours_required = 0
            for service in day_treatments.columns:
                if service in services:
                    service_count = day_treatments[service].values[0]
                    duration = durations[services.index(service)]
                    total_hours_required += (service_count * duration) / 60
            return total_hours_required
        
        results = []
        
        # Iterate through each day in the treatment demand dataset
        for date in treatment_demand_df['Date'].unique():
            day_treatments = treatment_demand_df[treatment_demand_df['Date'] == date].drop(columns=['Date', 'Hour Required'])
            total_hours_required = get_total_hours_required(day_treatments)
            
            # Create a linear programming problem
            prob = LpProblem(f"Optimal_Roster_{date}", LpMinimize)
            
            # Decision variables
            num_dentists = LpVariable.dicts("num_dentists", dentists_filtered, 0, cat='Integer')
            total_hours_worked = LpVariable.dicts("total_hours_worked", dentists_filtered, 0)
            
            # Objective function: Minimize the total cost considering efficiency
            total_cost = lpSum([
                num_dentists[dentist] * base_wages_filtered[dentists_filtered.index(dentist)] * max_working_hours_per_day
                for dentist in dentists_filtered
            ]) + lpSum([
                total_hours_worked[dentist] * base_wages_filtered[dentists_filtered.index(dentist)] * (1 - efficiency_increase_filtered[dentists_filtered.index(dentist)])
                for dentist in dentists_filtered
            ])
            prob += total_cost
            
            # Constraints
            prob += lpSum([
                total_hours_worked[dentist]
                for dentist in dentists_filtered
            ]) >= total_hours_required
            
            for dentist in dentists_filtered:
                prob += total_hours_worked[dentist] <= num_dentists[dentist] * max_working_hours_per_day * (1 + efficiency_increase_filtered[dentists_filtered.index(dentist)])
            
            # Solve the problem
            prob.solve(PULP_CBC_CMD(timeLimit=30))
            
            # Extracting the results
            optimal_num_dentists = {dentist: num_dentists[dentist].varValue for dentist in dentists_filtered}
            total_cost_result = value(prob.objective)
            
            total_working_hours = sum(total_hours_worked[dentist].varValue for dentist in dentists_filtered)
            total_dentists = sum(optimal_num_dentists[dentist] for dentist in dentists_filtered)
            average_working_hours = total_working_hours / total_dentists if total_dentists > 0 else 0
            
            daily_profit = sum(day_treatments[service].values[0] * profits[services.index(service)] for service in day_treatments.columns if service in services)
            
            results.append({
                'Date': date,
                'Optimal Roster': optimal_num_dentists,
                'Total Cost(Wage)': total_cost_result,
                'Profit': daily_profit,
                'Average Working Hours': average_working_hours
            })
        
        results_df = pd.DataFrame(results)
        st.session_state.results_df = results_df
        
        # Display the results
        st.write("Optimal Roster and Costs for Each Day:")
        st.write(st.session_state.results_df)
        
        # Save results to an Excel file
        output_dir = './Results/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, f'optimal_roster({start_date} ~ {end_date}).xlsx')
        st.session_state.results_df.to_excel(output_file, index=False)
        
        # st.write(f"Results saved to: {output_file}")

        # Option to download the results
        st.download_button(
            label="Download Results as Excel",
            data=st.session_state.results_df.to_csv(index=False).encode('utf-8'),
            file_name='optimal_roster.csv',
            mime='text/csv'
        )

