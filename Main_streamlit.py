import streamlit as st


#To reset tables if we move to another page
def reset_session_state():
    keys_to_reset = ['dummy_df', 'results_df']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# Define the sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Go to", ["Total Revenue Calculation", "Maximize Profit Optimization", "Scenario Simulation", "Demand Roster Optimization", "Generate Data", "Scheduling Optimization 2.0", "Performance Analysis"])

if st.session_state.get('current_page') != selection:
        reset_session_state()
        st.session_state['current_page'] = selection

if selection == "Total Revenue Calculation":
    from Page import Revenue_page
    Revenue_page.display()
elif selection == "Maximize Profit Optimization":
    from Page import Optimization_page
    Optimization_page.display()
elif selection == "Scenario Simulation":
    from Page import Scenario_Simulation
    Scenario_Simulation.display()
elif selection == 'Demand Roster Optimization':
    from Page import Demand_Roster
    Demand_Roster.display()
elif selection == 'Generate Data':
    from Page import GenerateData_Page
    GenerateData_Page.display()
elif selection == 'Scheduling Optimization 2.0':
    from Page import Scheduling_Page2
    Scheduling_Page2.display()
elif selection == 'Performance Analysis':
    from Page import Performance_Page
    Performance_Page.display()
