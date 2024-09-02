import pandas as pd
import random
import streamlit as st
import locale
# import openpyxl
# import os

locale.setlocale(locale.LC_ALL, '')

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

def get_random_schedule(dentists, hour_required, max_hours_per_day, max_hours_per_week, selected_df):
    schedule = []
    weekly_hours = {dentist['Name']: 0 for dentist in dentists}
    day_counter = 0

    for required_hours in hour_required:
        day_counter += 1
        remaining_hours = required_hours
        daily_schedule = []
        random.shuffle(dentists)

        # Include selected dentists first if selected_df is not empty
        if not selected_df.empty:
            selected_dentists = selected_df[selected_df['Selected']].to_dict(orient='records')
            for selected_dentist in selected_dentists:
                dentist_name = selected_dentist['Name']
                max_selected_hours = selected_dentist['WorkingHour']
                available_hours = min(max_selected_hours - weekly_hours[dentist_name], max_selected_hours)
                hours_to_assign = min(max_hours_per_day, remaining_hours, available_hours)
                if hours_to_assign > 0:
                    daily_schedule.append((dentist_name, hours_to_assign))
                    remaining_hours -= hours_to_assign
                    weekly_hours[dentist_name] += hours_to_assign

        for dentist in dentists:
            if remaining_hours <= 0:
                break
            if not selected_df.empty and dentist['Name'] in [d['Name'] for d in selected_dentists]:
                continue  # Skip already selected dentists
            available_hours = max_hours_per_week - weekly_hours[dentist['Name']]
            hours_to_assign = min(max_hours_per_day, remaining_hours, available_hours)
            if hours_to_assign > 0:
                daily_schedule.append((dentist['Name'], hours_to_assign))
                remaining_hours -= hours_to_assign
                weekly_hours[dentist['Name']] += hours_to_assign

        # Reset weekly hours if the week is completed
        if day_counter % 7 == 0:
            weekly_hours = {dentist['Name']: 0 for dentist in dentists}

        # Remove dentists who work less than an hour
        daily_schedule = [(dentist, hours) for dentist, hours in daily_schedule if hours >= 1]
        schedule.append(daily_schedule)

    return schedule

def get_cheapest_schedule(dentists, required_hours, max_hours_per_day, max_hours_per_week, weekly_hours, day_counter, selected_df):
    sorted_dentists = sorted(dentists, key=lambda x: x['Wage_per_hour'])
    schedule = []
    remaining_hours = required_hours

    # Include selected dentists first if selected_df is not empty
    if not selected_df.empty:
        selected_dentists = selected_df[selected_df['Selected']].to_dict(orient='records')
        for selected_dentist in selected_dentists:
            dentist_name = selected_dentist['Name']
            max_selected_hours = selected_dentist['WorkingHour']
            available_hours = min(max_selected_hours - weekly_hours[dentist_name], max_selected_hours)
            hours_to_assign = min(max_hours_per_day, remaining_hours, available_hours)
            if hours_to_assign > 0:
                schedule.append((dentist_name, hours_to_assign))
                remaining_hours -= hours_to_assign
                weekly_hours[dentist_name] += hours_to_assign

    for dentist in sorted_dentists:
        if remaining_hours <= 0:
            break
        if not selected_df.empty and dentist['Name'] in [d['Name'] for d in selected_dentists]:
            continue  # Skip already selected dentists
        available_hours = max_hours_per_week - weekly_hours[dentist['Name']]
        hours_to_assign = min(max_hours_per_day, remaining_hours, available_hours)
        if hours_to_assign > 0:
            schedule.append((dentist['Name'], hours_to_assign))
            remaining_hours -= hours_to_assign
            weekly_hours[dentist['Name']] += hours_to_assign

    # Reset weekly hours after every 7 days
    if day_counter % 7 == 0:
        weekly_hours = {dentist['Name']: 0 for dentist in dentists}

    # Remove dentists who work less than an hour
    schedule = [(dentist, hours) for dentist, hours in schedule if hours >= 1]

    return schedule, weekly_hours

# Helper function to evaluate a schedule
def evaluate_schedule(schedule, max_hours_per_week):
    total_cost = 0
    weekly_hours = {dentist['Name']: 0 for dentist in dentists}

    num_weeks = len(schedule) // 7
    extra_days = len(schedule) % 7
    adjusted_max_hours_per_week = max_hours_per_week * (num_weeks + (1 if extra_days > 0 else 0))

    for day in schedule:
        for dentist_name, hours in day:
            weekly_hours[dentist_name] += hours
            total_cost += hours * next(d['Wage_per_hour'] for d in dentists if d['Name'] == dentist_name)

    # Check if any dentist exceeds the adjusted weekly hours constraint
    if any(hours > adjusted_max_hours_per_week for hours in weekly_hours.values()):
        return float('inf')  # Invalid schedule if any dentist exceeds weekly hours

    return total_cost

#Main function to find the top schedules
def find_top_schedules(hour_required, max_hours_per_day, max_hours_per_week, selected_df, num_combinations=50000):
    all_schedules = set()  # Use a set to ensure unique schedules

    print(selected_df)

    # First combination using the cheapest dentists
    first_schedule = []
    weekly_hours = {dentist['Name']: 0 for dentist in dentists}  # Initialize weekly hours
    day_counter = 0  # Initialize day counter
    for required_hours in hour_required:
        day_counter += 1
        day_schedule, weekly_hours = get_cheapest_schedule(dentists, required_hours, max_hours_per_day, max_hours_per_week, weekly_hours, day_counter, selected_df)
        first_schedule.append(tuple(day_schedule))  # Convert to tuple to make it hashable
    all_schedules.add(tuple(first_schedule))  # Add the schedule to the set

    # Print the total cost of the cheapest schedule
    # cheapest_schedule_cost = evaluate_schedule(list(first_schedule), max_hours_per_week)
    # print(f"Total cost of the cheapest schedule: {cheapest_schedule_cost}")

    # Generate remaining combinations randomly
    while len(all_schedules) < num_combinations:
        schedule = get_random_schedule(dentists, hour_required, max_hours_per_day, max_hours_per_week, selected_df)
        all_schedules.add(tuple(tuple(day) for day in schedule))  # Add the schedule to the set

    schedule_costs = [(list(schedule), evaluate_schedule(list(schedule), max_hours_per_week)) for schedule in all_schedules]
    schedule_costs = sorted(schedule_costs, key=lambda x: x[1])

    return [schedule for schedule, cost in schedule_costs[:5]], len(all_schedules)

# Streamlit app
def display():
    processed_clinic_labor_file = './Data/Dummy_Treatment.xlsx'
    service_list_file = './Data/Service_list.xlsx'
    treatment_forecast_file = './Data/forecast_2_weeks.xlsx'

    service_list_df = pd.read_excel(service_list_file, sheet_name='Main')

    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()

    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    if 'processed_clinic_labor_df' not in st.session_state:
        processed_clinic_labor_df = pd.read_excel(processed_clinic_labor_file, sheet_name='Specialty')
        st.session_state.processed_clinic_labor_df = processed_clinic_labor_df.copy()
    else:
        processed_clinic_labor_df = st.session_state.processed_clinic_labor_df

    

    st.title("Scheduling Optimization")

    max_hours_per_week = st.number_input("Maximum Working Hours per Week", value=40, min_value=1)
    max_hours_per_day = st.number_input("Maximum Working Hours per Day", value=10, min_value=1)

    st.header("Edit Dentist Roster")
    edited_df = st.data_editor(st.session_state.processed_clinic_labor_df, num_rows="dynamic")

    if st.button("Save Changes"):
        st.session_state.processed_clinic_labor_df = edited_df

        try:
            with pd.ExcelWriter(processed_clinic_labor_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                edited_df.to_excel(writer, sheet_name='Specialty', index=False)
                
            st.success("Changes saved!")

            # Extract relevant columns from the updated clinic_labor_df
            clinic_labor_data_processed = edited_df[['Dentist Name', 'Specialty', 'Efficiency', 'Wage/Hour']]
            clinic_labor_data_processed.columns = ['Name', 'Specialty', 'Efficiency', 'Wage_per_hour']
        except Exception as e:
            st.error(f"An error occurred while saving changes: {e}")


    clinic_labor_data_processed = edited_df[['Dentist Name', 'Specialty', 'Efficiency', 'Wage/Hour']]
    clinic_labor_data_processed.columns = ['Name', 'Specialty', 'Efficiency', 'Wage_per_hour']

    # Initialize session state
    if 'selected_df2' not in st.session_state:
        st.session_state.selected_df2 = pd.DataFrame(columns=['Name', 'WorkingHour', 'Selected'])

    st.header("Select Dentist to be included to the Schedule")

    with st.form("selected_dentist"):
        selected_df = pd.DataFrame(clinic_labor_data_processed['Name'], columns=['Name'])
        selected_df['WorkingHour'] = 0
        selected_df['Selected'] = False 

        edited_df2 = st.data_editor(selected_df, num_rows="dynamic")

        if st.form_submit_button("Apply"):
            st.session_state.selected_df2 = edited_df2[edited_df2['Selected'] == True]
            st.success("Applied!")

            st.write("Selected Dentist to forced include in schedule")
            # Display the selected dentists
            if not st.session_state.selected_df2.empty:
                st.dataframe(st.session_state.selected_df2)
            else:
                st.write("No dentists selected yet.")

    
    # Prepare data for optimization
    global dentists, services, durations
    dentists = clinic_labor_data_processed.to_dict(orient='records')
    services = service_list_df['Treatment Menu Names'].tolist()
    durations = service_list_df['Duration'].tolist()

    st.header("Forecasted Scenario")

    forecast_df = pd.read_excel(treatment_forecast_file)
    st.session_state.forecast_df = forecast_df

    # Display the forecast dataframe if it exists
    if not st.session_state.forecast_df.empty:
        st.write(st.session_state.forecast_df)

    forecast_profit_result = forecast_profit(service_list_df, forecast_df)     
    st.write(f"Forecasted Profit: {locale.format_string('%d',  forecast_profit_result, grouping=True)} IDR")

    if st.button("Optimize Schedule"):
        forecast_df = st.session_state.forecast_df
        daily_demands = forecast_df.set_index('Date').to_dict(orient='index')
        hour_required = forecast_df['Hour Required'].tolist()

        top_schedules, total_combinations = find_top_schedules(hour_required, max_hours_per_day, max_hours_per_week, st.session_state.selected_df2)

        st.write(f"Total combinations evaluated: {total_combinations}")

        # Prepare the results
        results = []
        for idx, best_schedules in enumerate(top_schedules, start=1):
            results_df = pd.DataFrame([{
                'Date': day_idx, 
                'Dentist Name': [dentist for dentist, _ in day], 
                'Working hours': [round(hours, 2) for _, hours in day], 
                'Total Cost': evaluate_schedule([day], max_hours_per_week)
            } for day_idx, day in zip(daily_demands.keys(), best_schedules)])

            total_cost_sum = results_df['Total Cost'].sum()
            results.append((results_df, total_cost_sum))

        # Sort the results by total cost
        results = sorted(results, key=lambda x: x[1])

        for idx, (results_df, total_cost_sum) in enumerate(results, start=1):
            st.write(f"Roster {idx} Schedules:")
            st.write(results_df)
            st.write(f"Total Cost Sum for Roster {idx}: {locale.format_string('%d', total_cost_sum, grouping=True)} IDR")
            st.write(f"Total Revenue: {locale.format_string('%d', forecast_profit_result - total_cost_sum, grouping=True)} IDR")

            # Save the results
            # output_dir = './Results/'
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)

            # output_file = os.path.join(output_dir, f'optimal_schedule_top_{idx}.xlsx')
            # results_df.to_excel(output_file, index=False)

            st.download_button(
                label=f"Download Roster {idx} Optimal Schedules as Excel",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name=f'optimal_schedules_top_{idx}.csv',
                mime='text/csv'
            )

