import streamlit as st
import sys
import matplotlib.pyplot as plt


sys.path.insert(0, '../')
from Financial_model import *


def display():

    st.title("Total Revenue Calculation")
    
    # Input forms for clinic operations
    st.header("Input Clinic Operation Details")
    if 'clinic_operation' not in st.session_state:
        st.session_state.clinic_operation = None

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

    # Display clinic operation details
    st.subheader("Clinic Operation Details")
    if st.session_state.clinic_operation:
        clinic_operation = st.session_state.clinic_operation
        st.write(f"Clinic Open Hours per Month: {clinic_operation.get_clinicOpen_hour()}")
        st.write(f"Number of Workers: {clinic_operation.get_worker_number()}")
        st.write(f"Number of Doctors: {clinic_operation.get_doctor_number()}")
        st.write(f"Worker Salary per Month: {clinic_operation.get_worker_salary()}")
        st.write(f"Doctor Salary per Month: {clinic_operation.get_doctor_salary()}")
        st.write(f"Clinic Productive Hours per Month: {clinic_operation.get_clinic_productive_hours()}")
        st.write(f"Total Clinic Expense per Month: {clinic_operation.get_clinic_expense()}")

    # Input forms for services
    st.header("Input Clinic Services")
    services = []
    if 'services' not in st.session_state:
        st.session_state.services = []

    with st.form("service_form"):
        service_id = st.text_input("Service ID")
        service_name = st.text_input("Service Name")
        service_fee = st.number_input("Service Fee", min_value=0.0, step=0.01)
        service_COGS = st.number_input("Service COGS", min_value=0.0, step=0.01)
        service_counts = st.number_input("Service Counts", min_value=0)
        service_times = st.number_input("Service Duration (hours)", min_value=0.0, step=0.1)
        discount_rate = st.number_input("Discount Rate", min_value=0.0, step=0.01)
        debit_charge = st.number_input("Debit Charge", min_value=0.0, step=0.01)
        
        if st.form_submit_button("Add Service"):
            service = {
                "service_id": service_id,
                "service_name": service_name,
                "service_fee": service_fee,
                "service_COGS": service_COGS,
                "service_counts": service_counts,
                "service_times": service_times,
                "discount_rate": discount_rate,
                "debit_charge": debit_charge
            }
            st.session_state.services.append(service)
            st.success("Service added successfully!")

    # Display added services
    st.subheader("Added Services")
    for service in st.session_state.services:
        st.write(service)

    # Input forms for costs
    st.header("Input Costs/Expenses")
    costs = []
    if 'costs' not in st.session_state:
        st.session_state.costs = []

    with st.form("cost_form"):
        cost_id = st.text_input("Cost ID")
        cost_name = st.text_input("Cost Name")
        cost_fee = st.number_input("Cost Fee", min_value=0.0, step=0.01)
        cost_type = st.selectbox("Cost Type", ["Operational", "General"])
        
        if st.form_submit_button("Add Cost"):
            cost = {
                "cost_id": cost_id,
                "cost_name": cost_name,
                "cost_fee": cost_fee,
                "cost_type": cost_type
            }
            st.session_state.costs.append(cost)
            st.success("Cost added successfully!")

    # Display added costs
    st.subheader("Added Costs")
    for cost in st.session_state.costs:
        st.write(cost)

    # Time variable input
    st.header("Time Variable for Financial Projection")
    time_period = st.selectbox("Select Time Period", ["Month", "Year"])
    time_multiplier = st.number_input(f"Enter number of {time_period}s", min_value=1, step=1)

    # Calculate total revenue
    if st.button("Calculate Total Revenue"):
        clinic_service_list = ClinicServiceList({"services": st.session_state.services, "discount_rate": 0.0, "debit_charge": 0.0})
        operational_cost_list = CostList({"cost_list": [cost for cost in st.session_state.costs if cost["cost_type"] == "Operational"]}, "Operational")
        general_cost_list = CostList({"cost_list": [cost for cost in st.session_state.costs if cost["cost_type"] == "General"]}, "General")
        
        total_revenue = clinic_service_list.get_gross() - (operational_cost_list.get_total_cost() + general_cost_list.get_total_cost())
        if st.session_state.clinic_operation:
            total_revenue -= st.session_state.clinic_operation.get_clinic_expense()
        
        total_revenue_projection = total_revenue * time_multiplier
        st.subheader(f"Total Revenue for {time_multiplier} {time_period}(s): {total_revenue_projection:.2f}")
        
        # Calculate Productivity Efficiency
        if st.session_state.clinic_operation:
            total_service_hours = clinic_service_list.get_total_service_hours()
            clinic_productive_hours = clinic_operation.get_clinic_productive_hours()
            st.write(f"Total service hours: {total_service_hours}")
            st.write(f"Total clinic operation hours: {clinic_productive_hours}")
            if total_service_hours <= clinic_productive_hours:
                productivity_efficiency = total_service_hours / clinic_productive_hours
                st.write(f"Productivity Efficiency for Each Clinic: {(productivity_efficiency * 100):.2f}%")
            else:
                st.subheader("Service Total Hours is bigger than clinic Total Hours available, so the proposition doesn't make sense")
                
        # Plotting the financial projection
        months_or_years = range(1, time_multiplier + 1)
        revenue_projection = [total_revenue * i for i in months_or_years]   

        plt.figure(figsize=(10, 6))
        plt.plot(months_or_years, revenue_projection, marker='o', label='Revenue Projecation')
        plt.title('Financial Projection')
        plt.xlabel(f'Time ({time_period}s)')
        plt.ylabel('Total Revenue')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        st.pyplot(plt)
