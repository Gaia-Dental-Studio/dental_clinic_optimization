import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD
import random
import streamlit as st
import io
import plotly.express as px


def display():

    # Load the Excel files
    clinic_labor_df = pd.read_excel('./Data/clinic_labor.xlsx', sheet_name='Labor')
    service_list_df = pd.read_excel('./Data/Service_list.xlsx', sheet_name='Main')
    
    # Load and display the editable scenario DataFrame
    scenarios_df = pd.read_excel('./Data/clinic_labor.xlsx', sheet_name='Experiment')
    st.write("Edit the scenarios below:")
    scenarios_df = st.data_editor(scenarios_df, key="editable_scenarios")

    # Inputs for total working and overtime hours per dentist
    total_working_hours_per_dentist = st.number_input("Total working hours per dentist per month", value=8*22, min_value=0)
    total_overtime_hours_per_dentist = st.number_input("Total overtime hours per dentist per month", value=1*3*4, min_value=0)

    # Initialize session state for results
    if "results" not in st.session_state:
        st.session_state["results"] = []

    if st.button("Run Optimization"):

        st.header("Monthly Scenario Result")

        # Extracting relevant columns from clinic_labor_df
        clinic_labor_processed = clinic_labor_df[[
            'List Name', 
            'Base wage hourly (IDR)', 
            'Timeline', 
            '% of expected efficiency increase in performance', 
            'Salary per month(IDR)',
            '% INCREASING SALARY',
            'Bonus Type/year'
        ]]

        # Renaming columns for easier access
        clinic_labor_processed.columns = [
            'List_Name', 
            'Base_wage_hourly_IDR', 
            'Timeline', 
            'Efficiency_increase_percent', 
            'Salary_per_month_IDR',
            'Salary_increase_percent',
            'Bonus_percent'
        ]

        # Extracting relevant columns from service_list_df
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
        base_wages_filtered = filtered_clinic_labor['Base_wage_hourly_IDR'].tolist()
        efficiency_increase_filtered = filtered_clinic_labor['Efficiency_increase_percent'].fillna(0).tolist()
        monthly_salaries_filtered = filtered_clinic_labor['Salary_per_month_IDR'].tolist()
        salary_increase_percent = filtered_clinic_labor['Salary_increase_percent'].tolist()
        bonus_percent = filtered_clinic_labor['Bonus_percent'].tolist()

        services = service_list_processed['Treatment_Names'].tolist()
        durations = service_list_processed['Duration_minutes'].tolist()
        profits = service_list_processed['Profit'].tolist()

        results = []
        all_services_performed_results = {}

        # Corrected function to randomize treatment counts within allowed hours per dentist
        def randomize_treatment_counts(num_dentists, services, durations, total_hours_per_dentist):
            treatment_counts = {dentist: {service: 0 for service in services} for dentist in range(num_dentists)}
            for dentist in range(num_dentists):
                remaining_hours = total_hours_per_dentist[dentist]
                while remaining_hours > 0:
                    service = random.choice(services)
                    duration = durations[services.index(service)] / 60
                    if remaining_hours - duration >= 0:
                        treatment_counts[dentist][service] += 1
                        remaining_hours -= duration
                    else:
                        break
            return treatment_counts

        # Iterate through each scenario
        for idx, row in scenarios_df.iterrows():
            # Create a linear programming problem
            prob_filtered = LpProblem(f"Optimal_Labor_Roster_Filtered_Scenario_{idx + 1}", LpMinimize)

            # Decision variables
            working_hours_filtered = LpVariable.dicts("working_hours", dentists_filtered, 0)
            overtime_filtered = LpVariable.dicts("overtime", dentists_filtered, 0)
            services_performed = LpVariable.dicts("services_performed", (dentists_filtered, services), 0, cat='Integer')

            # Calculate total available hours for each dentist level with efficiency considered
            total_working_hours = sum([
                total_working_hours_per_dentist * (1 + efficiency_increase_filtered[i]) * row[f'Dentist-Lv{i + 1}_Count'] 
                for i in range(len(dentists_filtered))
            ])
            total_overtime_hours = sum([
                total_overtime_hours_per_dentist * (1 + efficiency_increase_filtered[i]) * row[f'Dentist-Lv{i + 1}_Count'] 
                for i in range(len(dentists_filtered))
            ])

            scenario_output = f"Scenario {idx + 1} - Total Working Hours: {total_working_hours:.2f}, Total Available Overtime Hours: {total_overtime_hours:.2f}"
            # st.write(scenario_output)

           # Objective function: Minimize the total cost
            total_cost_filtered = lpSum([
                working_hours_filtered[dentist] * base_wages_filtered[dentists_filtered.index(dentist)] +
                overtime_filtered[dentist] * base_wages_filtered[dentists_filtered.index(dentist)] * 2
                for dentist in dentists_filtered
            ])
            prob_filtered += total_cost_filtered


            # Constraints
            for i, dentist in enumerate(dentists_filtered):
                prob_filtered += working_hours_filtered[dentist] <= total_working_hours_per_dentist * row[f'Dentist-Lv{i + 1}_Count'] * (1 + efficiency_increase_filtered[i])
                prob_filtered += overtime_filtered[dentist] <= total_overtime_hours_per_dentist * row[f'Dentist-Lv{i + 1}_Count'] * (1 + efficiency_increase_filtered[i])

            # Ensure the working hours consider efficiency increase
            prob_filtered += lpSum([
                working_hours_filtered[dentist] * (1 + efficiency_increase_filtered[i]) 
                for i, dentist in enumerate(dentists_filtered)
            ]) <= total_working_hours + total_overtime_hours

            for i, dentist in enumerate(dentists_filtered):
                prob_filtered += lpSum([
                    services_performed[dentist][service] * durations[j] / 60 
                    for j, service in enumerate(services)
                ]) <= working_hours_filtered[dentist] * (1 + efficiency_increase_filtered[i])

            # Ensure total service time is at least equal to the total working hours and does not exceed the total available hours
            prob_filtered += lpSum([
                services_performed[dentist][service] * durations[j] / 60
                for j, service in enumerate(services)
                for dentist in dentists_filtered
            ]) >= total_working_hours

            prob_filtered += lpSum([
                services_performed[dentist][service] * durations[j] / 60
                for j, service in enumerate(services)
                for dentist in dentists_filtered
            ]) <= total_working_hours + total_overtime_hours

            # Solve the problem
            prob_filtered.solve(PULP_CBC_CMD(timeLimit=10))

            # Extracting the results
            working_hours_result_filtered = {dentist: working_hours_filtered[dentist].varValue for dentist in dentists_filtered}
            overtime_result_filtered = {dentist: overtime_filtered[dentist].varValue for dentist in dentists_filtered}
            total_cost_result_filtered = value(prob_filtered.objective)

            # Randomize treatment counts for each dentist
            total_hours_per_dentist = [
                (working_hours_result_filtered[dentist] + (overtime_result_filtered[dentist] if overtime_result_filtered[dentist] else 0)) * (1 + efficiency_increase_filtered[dentists_filtered.index(dentist)])
                for dentist in dentists_filtered
            ]
            randomized_counts = randomize_treatment_counts(len(dentists_filtered), services, durations, total_hours_per_dentist)
            services_performed_result = {dentists_filtered[d]: {service: randomized_counts[d][service] for service in services} for d in range(len(dentists_filtered))}

            total_profit = sum([services_performed_result[dentist][service] * profits[j] 
                                for dentist in dentists_filtered 
                                for j, service in enumerate(services)])

            total_hours_per_service = {service: sum(services_performed_result[dentist][service] * durations[j] / 60 for dentist in dentists_filtered) for j, service in enumerate(services)}

            results.append({
                'Scenario': idx + 1,
                'Total Profit': total_profit,
                'Total Cost (Salary)': total_cost_result_filtered,
                'Profit to Expense Ratio': total_profit / total_cost_result_filtered if total_cost_result_filtered != 0 else float('inf')
            })

            # Store the performed services for each scenario
            services_performed_df = pd.DataFrame(services_performed_result)
            services_performed_df['Total Hours'] = pd.Series(total_hours_per_service)
            dentist_counts = {dentist: row[f'Dentist-Lv{dentists_filtered.index(dentist) + 1}_Count'] for dentist in dentists_filtered}
            dentist_counts_df = pd.DataFrame(dentist_counts, index=['Dentist Count'])
            services_performed_df = pd.concat([services_performed_df, dentist_counts_df])
            all_services_performed_results[f'Scenario_{idx + 1}'] = services_performed_df

            # Calculate 5-year projections
            yearly_results = {'Total Profit': [], 'Total Cost (Salary)': [], 'Profit to Expense Ratio': []}
            cumulative_cost = 0  # Initialize cumulative cost
            for year in range(1, 6):
                yearly_profit = total_profit * 12 * year
                yearly_cost = 0
                for level in range(len(dentists_filtered)):
                    yearly_cost += (
                        row[f'Dentist-Lv{level + 1}_Count'] * base_wages_filtered[level] * 
                        (total_working_hours_per_dentist * 12 + total_overtime_hours_per_dentist * 12 * 2) *
                        (1 + salary_increase_percent[level] + bonus_percent[level]) ** year
                    )

                cumulative_cost += yearly_cost  # Accumulate the yearly cost
                
                yearly_results['Total Profit'].append(yearly_profit)
                yearly_results['Total Cost (Salary)'].append(cumulative_cost * -1)  # Making salary cost negative
                yearly_results['Profit to Expense Ratio'].append(yearly_profit / abs(yearly_cost) if yearly_cost != 0 else float('inf'))

            projection_df = pd.DataFrame(yearly_results, index=['Year1', 'Year2', 'Year3', 'Year4', 'Year5'])
            projection_df.drop(columns='Profit to Expense Ratio', axis=1, inplace=True)
            all_services_performed_results[f'Scenario_{idx + 1}_Projection'] = projection_df

            # Create a DataFrame suitable for Plotly
            df_plot = projection_df.reset_index().melt(id_vars='index', var_name='Category', value_name='Value')
            df_plot.columns = ['Year', 'Category', 'Value']

            # Create the Plotly figure with custom colors
            color_discrete_map = {'Total Profit': 'blue', 'Total Cost (Salary)': 'red'} 

            # Create the Plotly figure
            fig = px.bar(df_plot, x='Year', y='Value', color='Category', barmode='group', title=f'Scenario {idx + 1} - 5 Year Projection',
                        color_discrete_map=color_discrete_map)
            # st.plotly_chart(fig)

            # Store results in session state
            st.session_state["results"] = results
            st.session_state["all_services_performed_results"] = all_services_performed_results
            st.session_state[f"scenario_output_{idx + 1}"] = scenario_output
            st.session_state[f"projection_plot_{idx + 1}"] = fig
                    

        # Display the results from session state
    if st.session_state["results"]:
        results_df = pd.DataFrame(st.session_state["results"])
        st.write("Scenario Comparison: ")
        st.write(results_df)

        # Prepare the data for download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, sheet_name='Monthly', index=False)
            for scenario, df in st.session_state["all_services_performed_results"].items():
                df.to_excel(writer, sheet_name=scenario, index=True)

        output.seek(0)

        st.download_button(
            label="Download Excel file",
            data=output,
            file_name="scenario_comparison_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

         # Display stored scenario outputs
        for idx in range(len(st.session_state["results"])):
            st.write(st.session_state[f"scenario_output_{idx + 1}"])
            st.plotly_chart(st.session_state[f"projection_plot_{idx + 1}"])