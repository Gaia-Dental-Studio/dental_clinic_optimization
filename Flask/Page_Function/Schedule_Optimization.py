import pandas as pd
import random


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
def evaluate_schedule(dentists, schedule, max_hours_per_week):
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

    format(total_cost, '.3f')

    return float(total_cost)

#Main function to find the top schedules
def find_top_schedules(dentists, hour_required, max_hours_per_day, max_hours_per_week, selected_df, num_combinations=50000):
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

    schedule_costs = [(list(schedule), evaluate_schedule(dentists, list(schedule), max_hours_per_week)) for schedule in all_schedules]
    schedule_costs = sorted(schedule_costs, key=lambda x: x[1])

    return [schedule for schedule, cost in schedule_costs[:5]]



def run_scheduling_optimization(max_hours_per_week, max_hours_per_day, dentists_df, forecasted_df, selected_df):

    if selected_df.empty:
        selected_df = pd.DataFrame(columns=['Name', 'WorkingHour', 'Selected'])

    dentists_df_processed = dentists_df[['Dentist Name', 'Specialty', 'Efficiency', 'Wage/Hour']]
    dentists_df_processed.columns = ['Name', 'Specialty', 'Efficiency', 'Wage_per_hour']
    dentists = dentists_df_processed.to_dict(orient='records')
    
    daily_demands = forecasted_df.set_index('Date').to_dict(orient='index')
    hour_required = forecasted_df['Hour Required'].tolist()

    top_schedules = find_top_schedules(dentists, hour_required, max_hours_per_day, max_hours_per_week, selected_df)

    # Prepare the results
    results = []
    for best_schedules in top_schedules:
        results_df = pd.DataFrame([{
            'Date': day_idx, 
            'Dentist Name': [dentist for dentist, _ in day], 
            'Working hours': [round(hours, 2) for _, hours in day], 
            'Total Cost': evaluate_schedule(dentists, [day], max_hours_per_week)
        } for day_idx, day in zip(daily_demands.keys(), best_schedules)])

        total_cost_sum = results_df['Total Cost'].sum()
        results.append((results_df, total_cost_sum))

    # Sort the results by total cost
    results = sorted(results, key=lambda x: x[1])

    # Construct the output payload with multiple rosters
    rosters = []
    for idx, (results_df, total_cost_sum) in enumerate(results, start=1):
        roster_schedule = []
        for _, row in results_df.iterrows():
            roster_schedule.append({
                'date': row['Date'],
                'dentist_names': row['Dentist Name'],
                'working_hours': row['Working hours'],
                'total_cost': row['Total Cost']
            })
        
        rosters.append({
            'roster_rank': idx,
            'roster_schedule': roster_schedule,
            'schedule_total_cost': total_cost_sum
        })

    output_payload = {
        'rosters': rosters
    }

    return output_payload