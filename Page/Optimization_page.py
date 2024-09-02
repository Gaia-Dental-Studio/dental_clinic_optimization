import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import locale
locale.setlocale(locale.LC_ALL, '')

import streamlit as st
import sys


sys.path.insert(0, '../')
from Financial_model import *
from Optimization_Functions import *

# Load the service history dataset
service_history_file = "./Data/Service_History.xlsx"
service_history_df = pd.read_excel(service_history_file)

# Calculate the number of rows for each Treatment Name
service_counts = service_history_df['Treatment Name'].value_counts().to_dict()

def display():
    st.title("Maximize Profit Optimization")
    
   # Input forms for clinic operations
    st.header("Input Clinic Operation Details")

    if 'clinic_operation' not in st.session_state:
        st.session_state.clinic_operation = None

    if st.session_state.clinic_operation:
        clinic_operation = st.session_state.clinic_operation

        with st.expander("Clinic Operation Details"):
            # Display clinic operation details
            st.write(f"Number of Clinics: {clinic_operation.get_clinic_number()}")
            st.write(f"Clinic Open Hours per Month: {clinic_operation.get_clinicOpen_hour()}")
            st.write(f"Number of Workers: {clinic_operation.get_worker_number()}")
            st.write(f"Number of Doctors: {clinic_operation.get_doctor_number()}")
            st.write(f"Worker Salary per Month: {clinic_operation.get_worker_salary()}")
            st.write(f"Doctor Salary per Month: {clinic_operation.get_doctor_salary()}")
            st.write(f"Clinic Productive Hours per Month: {clinic_operation.get_clinic_productive_hours()}")
            st.write(f"Total Clinic Labor Expense per Month: {clinic_operation.get_clinic_expense()}")

        with st.form("edit_operation_form"):
            clinic_number = st.number_input("Number of Clinics", min_value=1, value=clinic_operation.get_clinic_number())
            clinicOpen_hour = st.number_input("Clinic Open Hours per Month", min_value=0, value=clinic_operation.get_clinicOpen_hour())
            worker_number = st.number_input("Number of Workers", min_value=0, value=clinic_operation.get_worker_number())
            doctor_number = st.number_input("Number of Doctors", min_value=0, value=clinic_operation.get_doctor_number())
            worker_salary = st.number_input("Worker Salary per Month", min_value=0.0, value=clinic_operation.get_worker_salary(), step=0.01)
            doctor_salary = st.number_input("Doctor Salary per Month", min_value=0.0, value=clinic_operation.get_doctor_salary(), step=0.01)
            
            if st.form_submit_button("Edit Clinic Operation"):
                st.session_state.clinic_operation.set_clinic_number(clinic_number)
                st.session_state.clinic_operation.set_clinicOpen_hour(clinicOpen_hour)
                st.session_state.clinic_operation.set_worker_number(worker_number)
                st.session_state.clinic_operation.set_doctor_number(doctor_number)
                st.session_state.clinic_operation.set_worker_salary(worker_salary)
                st.session_state.clinic_operation.set_doctor_salary(doctor_salary)
                st.success("Clinic operation details updated successfully!")
    else:
        with st.form("operation_form"):
            clinic_number = st.number_input("Number of Clinics", min_value=1)
            clinicOpen_hour = st.number_input("Clinic Open Hours per Month", min_value=0)
            worker_number = st.number_input("Number of Workers", min_value=0)
            doctor_number = st.number_input("Number of Doctors", min_value=0)
            worker_salary = st.number_input("Worker Salary per Month", min_value=0.0, step=0.01)
            doctor_salary = st.number_input("Doctor Salary per Month", min_value=0.0, step=0.01)
            
            if st.form_submit_button("Add Clinic Operation"):
                clinic_operation = ClinicOperation(clinic_number, clinicOpen_hour, worker_number, doctor_number, worker_salary, doctor_salary)
                st.session_state.clinic_operation = clinic_operation
                st.success("Clinic operation details added successfully!")


    # Option to upload service list via Excel or CSV
    st.subheader("Upload Service List")
    duration_type = st.selectbox("Select Duration Type", ["Hours", "Minutes"], index=0)
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        if 'uploaded_data' not in st.session_state:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.session_state.uploaded_data = df
                st.session_state.duration_type = duration_type
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error processing the file: {e}")

    if 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        duration_type = st.session_state.duration_type

        # Display the dataframe
        st.write("Preview of uploaded file:")
        st.write(df)
        
        # Extract necessary columns
        necessary_columns = ['Treatment Menu Names', 'Price', 'Duration', 'COGS', 'Discount Rate', 'Debit Charge']
        if set(necessary_columns).issubset(df.columns):
            df = df[necessary_columns]  # Keep only the necessary columns
            
            # Convert duration to hours if it's in minutes
            if duration_type == "Minutes":
                df['Duration'] = df['Duration'] / 60
                df['Duration'] = df['Duration'].apply(lambda x: round(x, 3))  # Round to 3 decimal places
            
            if 'services_processed' not in st.session_state:
                st.session_state.services_processed = []

            if 'services' not in st.session_state:
                st.session_state.services = []

            for index, row in df.iterrows():
                if index not in st.session_state.services_processed:
                    service = {
                        "service_id": index,
                        "service_name": row['Treatment Menu Names'],
                        "service_fee": row['Price'],
                        "service_COGS": row['COGS'],
                        "service_times": row['Duration'],
                        "discount_rate": row['Discount Rate'],
                        "debit_charge": row['Debit Charge']   
                    }
                    st.session_state.services.append(service)
                    st.session_state.services_processed.append(index)
            st.success("Services added from the uploaded file successfully!")
        else:
            st.error(f"The file must contain columns: {necessary_columns}")

        if st.button("Remove Uploaded File"):
            del st.session_state.uploaded_data
            del st.session_state.services_processed
            st.session_state.services = [service for service in st.session_state.services if 'service_id' not in service or not isinstance(service['service_id'], int)]
            st.success("Uploaded file removed successfully!")

    st.subheader("Upload Service List Manually")
    services = []
    if 'services' not in st.session_state:
        st.session_state.services = []

    with st.form("service_form"):
        service_id = st.text_input("Service ID")
        service_name = st.text_input("Service Name")
        service_fee = st.number_input("Service Fee", min_value=0.0, step=0.01)
        service_COGS = st.number_input("Service COGS", min_value=0.0, step=0.01)
        service_times = st.number_input("Service Duration (hours)", min_value=0.0, step=0.1)
        discount_rate = st.number_input("Discount Rate", min_value=0.0, step=0.01)
        debit_charge = st.number_input("Debit Charge", min_value=0.0, step=0.01)
        
        if st.form_submit_button("Add Service"):
            service = {
                "service_id": service_id,
                "service_name": service_name,
                "service_fee": service_fee,
                "service_COGS": service_COGS,
                "service_times": service_times,
                "discount_rate": discount_rate,
                "debit_charge": debit_charge
            }
            st.session_state.services.append(service)
            st.success("Service added successfully!")

    # Display added services
    with st.expander("Added Services"):
        for service in st.session_state.services:
            st.write(service)

    #Move to Costs Form
    # Function to load cost details into session state for editing
    def load_cost_details(cost):
        st.session_state.cost_id = cost["cost_id"]
        st.session_state.cost_name = cost["cost_name"]
        st.session_state.cost_fee = cost["cost_fee"]
        st.session_state.cost_type = cost["cost_type"]

    # Input forms for costs
    st.header("Input Costs/Expenses")
    costs = []
    if 'costs' not in st.session_state:
        st.session_state.costs = []

    # Form to add or edit a cost
    with st.form("cost_form"):
        st.write("Add or Edit Cost")
        cost_id = st.text_input("Cost ID", key="cost_id")
        cost_name = st.text_input("Cost Name", key="cost_name")
        cost_fee = st.number_input("Cost Fee", min_value=0.0, step=0.01, key="cost_fee")
        cost_type = st.selectbox("Cost Type", ["Operational", "General"], key="cost_type")

        if st.form_submit_button("Add/Update Cost"):
            # Check if cost_id exists
            existing_cost = next((cost for cost in st.session_state.costs if cost["cost_id"] == cost_id), None)
            if existing_cost:
                # Update existing cost
                for cost in st.session_state.costs:
                    if cost["cost_id"] == cost_id:
                        cost["cost_name"] = cost_name
                        cost["cost_fee"] = cost_fee
                        cost["cost_type"] = cost_type
                st.success("Cost updated successfully!")
            else:
                # Add new cost
                cost = {
                    "cost_id": cost_id,
                    "cost_name": cost_name,
                    "cost_fee": cost_fee,
                    "cost_type": cost_type
                }
                st.session_state.costs.append(cost)
                st.success("Cost added successfully!")

    # Display added costs
    with st.expander("Added Costs"):
        for cost in st.session_state.costs:
            st.write(cost)



    # Choose optimization model type
    model_type = st.selectbox("Select Model Type", ["General Model", "Model with Bias/Weight from Service History"])

    # Choose optimization constraint type
    constraint_type = st.selectbox("Select Optimization Constraint Type", [
        "Optimization with Hour Limit", 
        "Optimization with Patient Limit",
        "Optimization with COGS Limit"
        ])


    patient_limit = None
    patient_limit_trigger = False
    if constraint_type == "Optimization with Patient Limit":
        patient_limit = st.number_input("Enter the maximum number of patients a doctor can handle per month", min_value=1, step=1)
        patient_limit = patient_limit * clinic_operation.get_doctor_number()
        patient_limit_trigger = True


    COGS_limit = None
    COGS_limit_trigger = False
    if constraint_type == "Optimization with COGS Limit":
        COGS_limit = st.number_input("Enter monthly COGS limit", min_value=1, step=1)
        COGS_limit_trigger = True


    if st.button("Optimize for Maximum Profit"):
        clinic_service_list = ClinicServiceList({"services": st.session_state.services, "discount_rate": 0.0, "debit_charge": 0.0})
        operational_cost_list = CostList({"cost_list": [cost for cost in st.session_state.costs if cost["cost_type"] == "Operational"]}, "Operational")
        general_cost_list = CostList({"cost_list": [cost for cost in st.session_state.costs if cost["cost_type"] == "General"]}, "General")

        total_cost_per_clinic = operational_cost_list.get_total_cost() + general_cost_list.get_total_cost() + st.session_state.clinic_operation.get_clinic_expense()
        total_working_hours_per_month = st.session_state.clinic_operation.get_clinic_productive_hours()
        prices = [service.get_service_fee() for service in clinic_service_list.get_services()]
        durations = [service.get_service_times() for service in clinic_service_list.get_services()]
        COGS = [service.get_service_COGS() for service in clinic_service_list.get_services()]

        services = clinic_service_list.get_services()

        # Net profits calculation
        profits = [service.calculate_service_netProfit() for service in services]

        #Some Declared Variables
        use_bias = False
        service_counts = None

        iteration = 0
        max_iterations = 10 
        efficiency = 0
        best_efficiency = 0
        best_individual = None
        best_logbook = None
        # total_profit = 0


        if model_type == "Model with Bias/Weight from Service History":
            # Calculate service counts based on historical data
            service_counts = {service.get_service_name(): service_history_df[service_history_df['Treatment Name'] == service.get_service_name()].shape[0] for service in clinic_service_list.get_services()}
            # service_counts = service_counts * 1000
            use_bias = True
            st.write(f"Bias: {service_counts}")

        

        while iteration < max_iterations:
            ga_optimizer = GAOptimization(prices, durations, total_cost_per_clinic, total_working_hours_per_month, profits, COGS,  service_counts, use_bias, COGS_limit, patient_limit, bias_weight=0.9)
            current_individual, logbook = ga_optimizer.optimize()

            total_hours_used = np.dot(current_individual, durations)
            efficiency = (total_hours_used / total_working_hours_per_month) * 100

            if constraint_type == "Optimization with Hour Limit" and efficiency >= 50:
                best_individual = current_individual
                best_logbook = logbook
                break

            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_individual = current_individual
                best_logbook = logbook

            iteration += 1

        if best_individual is not None:
            st.subheader("Optimization Results (Monthly)")
            st.write(f"Best individual: {best_individual}")
            st.write(f"Fitness Value: {best_individual.fitness.values[0]}")#Fitness maximum Profit from Algorithm

            max_profit = sum(np.array(best_individual) * np.array(profits))
            max_profit_clinics = max_profit * st.session_state.clinic_operation.get_clinic_number()

            st.write(f"Maximum Profit: {locale.format_string('%d', max_profit , grouping=True)} IDR")#Manually maximum profit calculation
            st.write(f"Maximum Profit for All clinics: {locale.format_string('%d', max_profit_clinics , grouping=True)} IDR")

            if patient_limit_trigger:
                st.write(f"Number of Service/Patient per doctor: {int(sum(best_individual)/clinic_operation.get_doctor_number())}")
                st.write(f"Total hours used per doctor: {(total_hours_used/clinic_operation.get_doctor_number()):.3f}")
            else:
                st.write(f"Total hours used: {total_hours_used:.3f}")

            if COGS_limit_trigger:
                st.write(f"Total Number of Services: {int(sum(best_individual))}")
                st.write(f"Total COGS used: {sum(np.array(best_individual) * np.array(COGS))}")
                
            st.write(f"Model Efficiency: {efficiency:.2f}%")

             # Plot the convergence curve
            fig_conv, ax_conv = plt.subplots()
            gen = best_logbook.select("gen")
            fit_max = best_logbook.select("max")
            fit_avg = best_logbook.select("avg")

            ax_conv.plot(gen, fit_max, label="Maximum Fitness(Bigger Better)")
            ax_conv.plot(gen, fit_avg, label=f"Average Fitness: {np.mean(fit_avg):.2f}")
            ax_conv.set_xlabel("Iteration")
            ax_conv.set_ylabel("Fitness")
            ax_conv.set_title("Genetic Algorithm Convergence Curve")
            ax_conv.legend(loc="best")
            ax_conv.grid(True)
            
            st.pyplot(fig_conv)
   
            # Display the service counts for each service
            service_names = [service.get_service_name() for service in clinic_service_list.get_services()]
            st.write("Service counts for optimal solution:")

            #Put output data into dataframe
            data_output = {
                "Service Name": service_names,
                "Optimize Count": best_individual,
                "Total Hours": [count * duration for count, duration in zip(best_individual, durations)],
                "Profit Contribution": [ count * profit for count, profit in zip(best_individual, profits)]
            }

            df_output = pd.DataFrame(data_output)
            st.dataframe(df_output)

            # Filter out services with zero count
            df_filtered = df_output[df_output['Optimize Count'] > 0]

            # Pie Chart
            # Calculate the top services
            df_filtered['Service Count Rank'] = df_filtered['Optimize Count'].rank(method='min', ascending=False)
            top_services_df = df_filtered[df_filtered['Service Count Rank'] <= 5]
            remaining_df = df_filtered[df_filtered['Service Count Rank'] > 5]

            # Calculate the remaining services total
            remaining_total = remaining_df['Optimize Count'].sum()

            # Add the remaining total as a new row in the top_services_df if there are more than 5 services
            if len(top_services_df) < len(df_filtered):
                remaining_row = pd.DataFrame({'Service Name': ['Others'], 'Optimize Count': [remaining_total], 'Total Hours': [0], 'Profit Contribution': [0]})
                top_services_df = pd.concat([top_services_df, remaining_row], ignore_index=True)

            # Create the pie chart
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(top_services_df['Optimize Count'], labels=top_services_df['Service Name'], autopct='%1.1f%%', startangle=90)
            ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # Display the pie chart
            st.pyplot(fig_pie)

            max_profit_yearly = max_profit * 12

            st.write("Profit & Loss Table:")
            #CashFlow Output
            cf_data = [
            {
                "Case": "Gross Profit(+)",
                "Year 0": 0,
                "Year 1": max_profit_yearly,
                "Year 2": max_profit_yearly * 2,
                "Year 3": max_profit_yearly * 3,
                "Year 4": max_profit_yearly * 4,
                "Year 5": max_profit_yearly * 5
            },
            {
                "Case": "Operational Costs(-)",
                "Year 0": 0,
                "Year 1": operational_cost_list.get_total_cost() * 12,
                "Year 2": operational_cost_list.get_total_cost() * 12 * 2,
                "Year 3": operational_cost_list.get_total_cost() * 12 * 3,
                "Year 4": operational_cost_list.get_total_cost() * 12 * 4,
                "Year 5": operational_cost_list.get_total_cost() * 12 * 5
            },
            {
                "Case": "General Costs(-)",
                "Year 0": 0,
                "Year 1": general_cost_list.get_total_cost() * 12,
                "Year 2": general_cost_list.get_total_cost() * 12 * 1.5,
                "Year 3": general_cost_list.get_total_cost() * 12 * 2.5,
                "Year 4": general_cost_list.get_total_cost() * 12 * 3.5,
                "Year 5": general_cost_list.get_total_cost() * 12 * 4.5
            },
            {
                "Case": "Salary Expenses(-)",
                "Year 0": 0,
                "Year 1": clinic_operation.get_clinic_expense() * 12,
                "Year 2": clinic_operation.get_clinic_expense() * 12 * 2,
                "Year 3": clinic_operation.get_clinic_expense() * 12 * 3,
                "Year 4": clinic_operation.get_clinic_expense() * 12 * 4,
                "Year 5": clinic_operation.get_clinic_expense() * 12 * 5
            },
            {
                "Case": "EBIT",
                "Year 0": 0,
                "Year 1": (max_profit - operational_cost_list.get_total_cost() - general_cost_list.get_total_cost() - clinic_operation.get_clinic_expense()) * 12,
                "Year 2": ((max_profit * 2) - (operational_cost_list.get_total_cost() * 2) - (general_cost_list.get_total_cost() * 2) - (clinic_operation.get_clinic_expense() * 1.5)) * 12,
                "Year 3": ((max_profit * 3) - (operational_cost_list.get_total_cost() * 3) - (general_cost_list.get_total_cost() * 3) - (clinic_operation.get_clinic_expense() * 2.5)) * 12,
                "Year 4": ((max_profit * 4) - (operational_cost_list.get_total_cost() * 4) - (general_cost_list.get_total_cost() * 4) - (clinic_operation.get_clinic_expense() * 3.5)) * 12,
                "Year 5": ((max_profit * 5) - (operational_cost_list.get_total_cost() * 5) - (general_cost_list.get_total_cost() * 5) - (clinic_operation.get_clinic_expense() * 4.5)) * 12
            }
        ]

        # Cashflow Output
        df_cash_flow = pd.DataFrame(cf_data)
        st.dataframe(df_cash_flow)

        # Data for waterfall chart
        waterfall_data = [
            go.Waterfall(
                x=[["Year 1", "Year 1", "Year 1", "Year 2", "Year 2", "Year 2", "Year 3", "Year 3", "Year 3", "Year 4", "Year 4", "Year 4","Year 5", "Year 5", "Year 5"], 
                ["Profit", "Expense", "Total EBIT", "Profit", "Expense", "Total EBIT", "Profit", "Expense", "Total EBIT", "Profit", "Expense", "Total EBIT", "Profit", "Expense", "Total EBIT"]],
                measure=["absolute", "relative", "total", "relative", "relative", "total", "relative", "relative", "total", "relative", "relative", "total", "relative", "relative", "total"],
                y=[       
                    max_profit_yearly, -((operational_cost_list.get_total_cost() * 12) + (general_cost_list.get_total_cost() * 12) + (clinic_operation.get_clinic_expense() * 12)), None,
                    (max_profit_yearly * 2), -((operational_cost_list.get_total_cost() * 12 * 2) + (general_cost_list.get_total_cost() * 12 * 1.5) + (clinic_operation.get_clinic_expense() * 12 * 2)), None,
                    (max_profit_yearly * 3), -((operational_cost_list.get_total_cost() * 12 * 3) + (general_cost_list.get_total_cost() * 12 * 2.5) + (clinic_operation.get_clinic_expense() * 12 * 3)), None,
                    (max_profit_yearly * 4), -((operational_cost_list.get_total_cost() * 12 * 4) + (general_cost_list.get_total_cost() * 12 * 3.5) + (clinic_operation.get_clinic_expense() * 12 * 4)), None,
                    (max_profit_yearly * 5), -((operational_cost_list.get_total_cost() * 12 * 5) + (general_cost_list.get_total_cost() * 12 * 4.5) + (clinic_operation.get_clinic_expense() * 12 * 5)), None
                ],
                base=0,
                decreasing={"marker": {"color": "Maroon", "line": {"color": "red", "width": 2}}},
                increasing={"marker": {"color": "Teal"}},
                totals={"marker": {"color": "deep sky blue", "line": {"color": "blue", "width": 3}}},
                textposition="outside",
            )
        ]

       # Layout for the chart
        layout = go.Layout(
            title="Profit & Loss Projection",
            waterfallgap=0.3,
            height=400,  # Adjust the height as needed
            xaxis=dict(
                tickfont=dict(
                    size=10,  # Adjust the font size
                ),
            ),
        )


        # Create the figure
        fig = go.Figure(data=waterfall_data, layout=layout)

        # Show plot in Streamlit
        st.plotly_chart(fig)

    else:
        st.error("Unable to achieve a feasible solution after maximum iterations.")


          