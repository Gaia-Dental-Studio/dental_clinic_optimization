def run_sheet_optimization(Constraints_df, Worker_df, Forecasted_df, item_numbers_json):
    import json
    import pandas as pd
    import numpy as np
    from datetime import datetime

    #Load data
    scenario_df_v2 = Forecasted_df
    constraints_df = Constraints_df
    worker_df = Worker_df
   
    item_numbers_data = item_numbers_json

    # Define the clinic conditions based on constraints
    num_rooms = int(constraints_df.loc[constraints_df['Constraints'] == 'Room_num', 'Value'].values[0])
    treatment_interval = int(constraints_df.loc[constraints_df['Constraints'] == 'Treatment_Interval', 'Value'].values[0])
    clinic_open_time = constraints_df.loc[constraints_df['Constraints'] == 'Clinic_Open', 'Value'].values[0]
    clinic_close_time = constraints_df.loc[constraints_df['Constraints'] == 'Clinic_Close', 'Value'].values[0]
    max_working_hour_day = int(constraints_df.loc[constraints_df['Constraints'] == 'Max_Working_Hour_Day', 'Value'].values[0]) * 60  # Convert hours to minutes
    max_working_hour_week = int(constraints_df.loc[constraints_df['Constraints'] == 'Max_Working_Hour_Week', 'Value'].values[0]) * 60  # Convert hours to minutes

    #Preprocess
    scenario_df_v2 = scenario_df_v2.iloc[:, 1:]


    # Room Configurations
    chairs_in_rooms = {
        'Room1': ['Chair1', 'Chair2'],
        'Room2': ['Chair1', 'Chair2'],
        'Room3': ['Chair3']
    }



    worker_treatments = {
        "MD" : get_worker_treatments("MD", item_numbers_data),
        "OS" : get_worker_treatments("OS", item_numbers_data),
        "ORT" : get_worker_treatments("ORT", item_numbers_data),
        "GP" : get_worker_treatments("GP", item_numbers_data),
        "SPC" : get_worker_treatments("SPC", item_numbers_data),
        "PAT" : get_worker_treatments("PAT", item_numbers_data),
        "PED DNT" : get_worker_treatments("PED DNT", item_numbers_data),
        "ANT" : get_worker_treatments("ANT", item_numbers_data),
        "All SPC" : get_worker_treatments("All SPC", item_numbers_data),
        "LAB" : get_worker_treatments("LAB", item_numbers_data),
        "PER" : get_worker_treatments("PER", item_numbers_data),
        "END" : get_worker_treatments("END", item_numbers_data),
        "PRS" : get_worker_treatments("PRS", item_numbers_data),
        "RAD" : get_worker_treatments("RAD", item_numbers_data)   
    }


    # Reshape the forecast_2_weeks DataFrame to have a 'Treatment' and 'Count' for each date
    forecast_reshaped = scenario_df_v2.melt(id_vars=['date'], 
                                        var_name='Treatment', 
                                        value_name='Count')


    # Filter out treatments not in item_numbers
    item_numbers = get_item_numbers(item_numbers_data)
    forecast_reshaped = forecast_reshaped[forecast_reshaped['Treatment'].isin(item_numbers)]

    # Map worker types to their hourly wages and IDs
    worker_wages = worker_df.set_index('Worker_Id')['Wage_Hour'].to_dict()
    job_to_worker_ids = worker_df.groupby('Job')['Worker_Id'].apply(list).to_dict() #Change the "JOB" Column format later 

    # Function to get eligible treatments for each worker based on their Job
    def get_eligible_treatments(worker_job, worker_treatments):
        return worker_treatments.get(worker_job, [])

    # Create a new column in the DataFrame for eligible treatments
    worker_df['Eligible_Treatments'] = worker_df['Job'].apply(lambda job: get_eligible_treatments(job, worker_treatments))

    # Prepare to track worker hours
    worker_hours_daily = {worker_id: 0 for worker_id in worker_df['Worker_Id']}
    worker_hours_weekly = {worker_id: 0 for worker_id in worker_df['Worker_Id']}
    worker_last_end_time = {worker_id: pd.to_datetime(forecast_reshaped['date'].min()) for worker_id in worker_df['Worker_Id']}

    # Prepare to store the scheduled output
    output_schedule = {room: [] for room in chairs_in_rooms.keys()}

    # Initialize time tracking for each room
    current_time = {room: pd.to_datetime(f"{forecast_reshaped['date'].min()} {clinic_open_time}") for room in chairs_in_rooms.keys()}

    # Keep track of which room to assign the next non-advanced treatment
    non_advanced_room_toggle = True

    # Track the current week to reset weekly hours
    current_week = pd.to_datetime(forecast_reshaped['date'].min()).isocalendar()[1]

    # Ensure the date column is parsed as datetime objects
    forecast_reshaped['date'] = pd.to_datetime(forecast_reshaped['date'], format='%m/%d/%Y')

    # Sort the dates to ensure they are processed in chronological order
    sorted_dates = sorted(forecast_reshaped['date'].unique())

    # Iterate over each unique date in sorted order
    for date in sorted_dates:
        print(f"Processing date: {date}")
        
        # Reset weekly hours if it's a new week
        week_of_year = pd.to_datetime(date).isocalendar()[1]
        if week_of_year != current_week:
            worker_hours_weekly = {worker_id: 0 for worker_id in worker_df['Worker_Id']}
            current_week = week_of_year
        
        # Reset daily hours for workers
        worker_hours_daily = {worker_id: 0 for worker_id in worker_df['Worker_Id']}
        
        # Reset or update last end time appropriately for new day
        for worker_id in worker_last_end_time:
            worker_last_end_time[worker_id] = pd.to_datetime(date).replace(hour=0, minute=0, second=0)
        
        # Reset current time for each room to the clinic's opening time
        current_time = {room: pd.to_datetime(f"{date} {clinic_open_time}") for room in chairs_in_rooms.keys()}
        current_time['Remote'] = pd.to_datetime(f"{date} {clinic_open_time}")  # Add Remote room
        
        # Diagnostic output to ensure reset is correct
        print(f"Worker hours reset for {date}:")
        print(f"Daily hours: {worker_hours_daily}")
        print(f"Weekly hours: {worker_hours_weekly}")
        print(f"Worker last end time: {worker_last_end_time}")
        print(f"Current time for rooms: {current_time}")
        
        # Continue with the scheduling logic
        daily_treatments = forecast_reshaped[forecast_reshaped['date'] == date]

        for index, row in daily_treatments.iterrows():
            item_number = row['Treatment']
            count = int(row['Count'])  # Number of sessions for this treatment

            # Initialize variables
            treatment_info = None
            treatment_duration = None

            # Variation factor for treatment duration
            variation_factor = np.random.uniform(0.8, 1.2)  # ±20% variability

            # Find the corresponding treatment and variation in the JSON data
            for treatment in item_numbers_data['treatment']:
                for detail in treatment['treatment_details']:
                    if str(detail['item_number']) == item_number:
                        treatment_info = detail
                        treatment_duration = float(detail['treatment_duration_ind'] * variation_factor)
                        break
                if treatment_info is not None:
                    break

            # Chair/room assignment
            if treatment_info['chair'] == 3:
                room = 'Room3'  # Assign to Room3 for chair 3
            else:
                room = 'Room1' if non_advanced_room_toggle else 'Room2'  # Toggle between Room1 and Room2 for chair 1 or 2
                non_advanced_room_toggle = not non_advanced_room_toggle  # Alternate between Room1 and Room2

            # Select the worker for this treatment from all eligible workers
            eligible_workers = []
            for worker_id in worker_df['Worker_Id']:
                # Check if the worker is eligible for this treatment
                if item_number in worker_df.loc[worker_df['Worker_Id'] == worker_id, 'Eligible_Treatments'].values[0]:
                    eligible_workers.append(worker_id)

            selected_worker = None

            # Iterate through eligible workers and find the first available one
            for worker_id in eligible_workers:
                daily_available = max_working_hour_day - worker_hours_daily[worker_id]
                weekly_available = max_working_hour_week - worker_hours_weekly[worker_id]
                start_time = current_time[room]
                end_time = start_time + pd.to_timedelta(treatment_duration, unit='m')

                # Debugging output to validate worker selection logic
                # print(f"Worker: {worker_id}, daily available: {daily_available}, weekly available: {weekly_available}, start: {start_time}, end: {end_time}")

                # Check if the worker has enough daily and weekly available time and can start the treatment on time
                if daily_available >= treatment_duration and weekly_available >= treatment_duration:
                    if worker_last_end_time[worker_id] <= start_time:
                        selected_worker = worker_id
                        worker_last_end_time[worker_id] = end_time
                        break  # Stop once a worker is selected

            # If no worker is found, print debug info and raise an error
            if selected_worker is None:
                print(f"Eligible workers: {eligible_workers}")
                print(f"Daily worker hours: {worker_hours_daily}")
                print(f"Weekly worker hours: {worker_hours_weekly}")
                raise ValueError(f"No available worker for treatment {item_number} on {date}")

            # Labor Cost calculation for each procedure
            labor_cost = treatment_duration * (worker_wages[selected_worker] / 60)

            # Schedule the treatment
            output_schedule[room].append({
                'Date': pd.to_datetime(date).date(),
                'Start_Time': start_time.strftime('%I:%M %p'),
                'Finish_Time': end_time.strftime('%I:%M %p'),
                'Item_Number': item_number,
                'Treatment_Names': treatment_info['treatment_type'],
                'Duration': treatment_duration,
                'Worker_Id': selected_worker,
                'Chair': treatment_info['chair'],
                'LaborCost': labor_cost
            })

            # Update the current time for the room
            current_time[room] = end_time + pd.to_timedelta(treatment_interval, unit='m')

            # Update worker hours
            worker_hours_daily[selected_worker] += treatment_duration
            worker_hours_weekly[selected_worker] += treatment_duration

            # Check if we exceed the clinic's operating hours for this room
            if current_time[room].time() > datetime.strptime(clinic_close_time, '%H:%M:%S').time():
                break

                
    # Convert the schedule dictionary to DataFrames for each room
    room_schedules = {room: pd.DataFrame(schedule) for room, schedule in output_schedule.items()}

    # Add the Total_Cost column and calculate the total labor cost for each room
    for room, schedule in room_schedules.items():
        if 'LaborCost' in schedule.columns:
            total_cost = schedule['LaborCost'].sum()
            total_cost_series = pd.Series([total_cost] + [None] * (len(schedule) - 1))
            schedule['Total_Cost'] = total_cost_series
        else:
            schedule['Total_Cost'] = pd.Series([None] * len(schedule))  # Add an empty column

    # Initialize the JSON structure
    json_output = {
        "payload": []
    }

    # Process each room's schedule and append to the JSON output
    for room, schedule in room_schedules.items():
        room_schedule = []
        total_labor_cost = schedule['LaborCost'].sum() if 'LaborCost' in schedule.columns else 0
        
        # Iterate over each row in the room's schedule and convert it to the desired format
        for _, row in schedule.iterrows():
            room_schedule.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "appointment_start": row['Start_Time'],
                "item_number": row['Item_Number'],
                "treatment_type": row['Treatment_Names'],
                "treatment_duration": row['Duration'],
                "worker_id": row['Worker_Id'],
                "chair": row['Chair'],
                "labor_cost": row['LaborCost']
            })
        
        # Append the room information and its schedule to the payload
        json_output['payload'].append({
            "room": int(room[-1]),  # Extract the room number from the room key (e.g., 'Room1' -> 1)
            "total_labor_cost": total_labor_cost,
            "schedule": room_schedule
        })

    # Output the final JSON format
    output_json = json.dumps(json_output, indent=4)

   
    return output_json 


# Determine which worker do which treatment
def get_worker_treatments(worker_code, json_data):
    worker_treatments_list = []

    for treatment in json_data['treatment']:
        for detail in treatment['treatment_details']:
            if worker_code in detail.get('worker_responsibility_ind', []):
                worker_treatments_list.append(str(detail['item_number']))

    return worker_treatments_list

def get_item_numbers(json_data):
    item_numbers = []

    for treatment in json_data['treatment']:
        for detail in treatment['treatment_details']:
            item_numbers.append(str(detail['item_number']))

    return item_numbers