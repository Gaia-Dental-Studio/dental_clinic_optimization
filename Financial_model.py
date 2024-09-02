
class ClinicService:
    def __init__(self, service_id, service_name, service_fee, service_COGS, service_times, discount_rate=0, debit_charge=0, service_counts=0):
        self.service_id = service_id  # Service id or code
        self.service_name = service_name  # Service name
        self.service_fee = service_fee  # Service fee/price
        self.service_COGS = service_COGS  # Service Cost of Goods Selling
        self.service_counts = service_counts  # Total Service Counts, how many times service performed (optional)
        self.service_times = service_times  # Service duration per client in hour
        self.discount_rate = discount_rate
        self.debit_charge = debit_charge
        self.service_revenue = self.calculate_service_sales()

    def get_id(self):
        return self.service_id
    
    def get_service_name(self):
        return self.service_name
    
    def get_service_fee(self):
        return self.service_fee
    
    def get_service_COGS(self):
        return self.service_COGS
    
    def get_service_counts(self):
        return self.service_counts
    
    def get_service_times(self):
        return self.service_times
    
    def get_discount_rate(self):
        return self.discount_rate
    
    def get_debit_charge(self):
        return self.debit_charge

    def calculate_service_sales(self):
        discounted_fee = self.service_fee * (1 - self.discount_rate)
        total_fee_after_charge = discounted_fee * (1 - self.debit_charge)
        revenue = total_fee_after_charge - self.service_COGS
        return revenue * self.service_counts
    
    def calculate_service_netProfit(self):
        discounted_fee = self.service_fee * (1 - self.discount_rate)
        total_fee_after_charge = discounted_fee * (1 - self.debit_charge)
        revenue = total_fee_after_charge - self.service_COGS
        # print(f"Service: {self.service_name}, Original Fee: {self.service_fee}, Discounted Fee: {discounted_fee}, Total Fee After Charge: {total_fee_after_charge}, COGS: {self.service_COGS}, Net Profit: {revenue}")
        return revenue

class ClinicServiceList:
    def __init__(self, service_list):
        self.services = [
            ClinicService(
                service_id=service["service_id"],
                service_name=service["service_name"],
                service_fee=service["service_fee"],
                service_COGS=service["service_COGS"],
                service_times=service["service_times"],
                discount_rate=service['discount_rate'],
                debit_charge=service['debit_charge'],
                service_counts=service.get("service_counts", 0)
            ) for service in service_list["services"]
        ]

    def get_services(self):
        return self.services
    
    def get_gross(self):
        return sum(service.service_revenue for service in self.services)
    
    def get_total_service_hours(self):
        return sum(service.get_service_counts() * service.get_service_times() for service in self.services)
    
class Cost:
    def __init__(self, cost_id, cost_name, cost_fee, cost_type):
        self.cost_id = cost_id
        self.cost_name = cost_name
        self.cost_fee = cost_fee
        self.cost_type = cost_type

    def get_cost_id(self):
        return self.cost_id
    
    def get_cost_name(self):
        return self.cost_name
    
    def get_cost_fee(self):
        return self.cost_fee

    def get_cost_type(self):
        return self.cost_type

    def set_cost_name(self, cost_name):
        self.cost_name = cost_name
    
    def set_cost_fee(self, cost_fee):
        self.cost_fee = cost_fee
    
    def set_cost_type(self, cost_type):
        self.cost_type = cost_type

    def __repr__(self):
        return (f"Cost(cost_id={self.cost_id}, cost_name='{self.cost_name}', "
                f"cost_fee={self.cost_fee}, cost_type='{self.cost_type}')")


class CostList:
    def __init__(self, cost_list, cost_type):   
        self.costs = [
            Cost(
                cost_id=cost["cost_id"],
                cost_name=cost["cost_name"],
                cost_fee=cost["cost_fee"],
                cost_type=cost_type
            ) for cost in cost_list["cost_list"]
        ]
        self.cost_type = cost_type
        
    def get_costs(self):
        return self.costs
    
    def get_total_cost(self):
        return sum(cost.get_cost_fee() for cost in self.costs)
    
    def get_cost_names(self):
        return [cost.get_cost_name() for cost in self.costs]

    def update_cost(self, cost_id, cost_name, cost_fee, cost_type):
        for cost in self.costs:
            if cost.get_cost_id() == cost_id:
                cost.set_cost_name(cost_name)
                cost.set_cost_fee(cost_fee)
                cost.set_cost_type(cost_type)
                break

    def __repr__(self):
        return f"CostList(cost_type='{self.cost_type}', costs={self.costs})"
    

class ClinicOperation:
    def __init__(self, clinic_number, clinicOpen_hour, worker_number, doctor_number, worker_salary, doctor_salary):
        self.clinic_number = clinic_number
        self.clinicOpen_hour = clinicOpen_hour #Custom clinic open hour per month
        self.worker_number = worker_number
        self.doctor_number = doctor_number
        self.worker_salary = worker_salary #Salary per month
        self.doctor_salary = doctor_salary 

    def get_clinic_number(self):
        return self.clinic_number
    
    def set_clinic_number(self, clinic_number):
        self.clinic_number = clinic_number

    def get_clinicOpen_hour(self):
        return self.clinicOpen_hour
    
    def set_clinicOpen_hour(self, clinic_Open_hour):
        self.clinicOpen_hour = clinic_Open_hour
    
    def get_worker_number(self):
        return self.worker_number
    
    def set_worker_number(self, worker_number):
        self.worker_number = worker_number
    
    def get_doctor_number(self):
        return self.doctor_number
    
    def set_doctor_number(self, doctor_number):
        self.doctor_number = doctor_number
    
    def get_worker_salary(self):
        return self.worker_salary
    
    def set_worker_salary(self, worker_salary):
        self.worker_salary = worker_salary
    
    def get_doctor_salary(self):
        return self.doctor_salary
    
    def set_doctor_salary(self, doctor_salary):
        self.doctor_salary = doctor_salary
    
    # Get clinic total operation hour per month
    def get_clinic_productive_hours(self):
        return self.clinicOpen_hour * self.doctor_number
    
    # Get clinic expense cost per month
    def get_clinic_expense(self):
        return (self.worker_number * self.worker_salary) + (self.doctor_number * self.doctor_salary)
    

    
# # Example usage for OperationCost
# operation_cost_list_data = {
#     "cost_list": [
#         {
#             "cost_id": 1,
#             "cost_name": "Electricity",
#             "cost_fee": 1500,
#         },
#         {
#             "cost_id": 2,
#             "cost_name": "Water",
#             "cost_fee": 800,
#         },
#         {
#             "cost_id": 3,
#             "cost_name": "Rent",
#             "cost_fee": 3000,
#         },
#     ]
# }

# # Example usage for GeneralCost
# general_cost_list_data = {
#     "cost_list": [
#         {
#             "cost_id": 1,
#             "cost_name": "Office Supplies",
#             "cost_fee": 500,
#         },
#         {
#             "cost_id": 2,
#             "cost_name": "Marketing",
#             "cost_fee": 1200,
#         },
#     ]
# }

    
# operation_cost_list = CostList(operation_cost_list_data, "Operational")
# general_cost_list = CostList(general_cost_list_data, "General")


# service_list = {
#     "services":[
#         {
#             "service_id": 1,
#             "service_name": "Basic Scaling",
#             "service_fee": 4600,
#             "service_COGS": 2000,
#             "service_counts": 500,
#             "service_times": 0.5
#         },
#          {
#             "service_id": 2,
#             "service_name": "Consultation",
#             "service_fee": 300,
#             "service_COGS": 40,
#             "service_counts": 60,
#             "service_times": 1
#         },
#     ],
#     "discount_rate":0.1,
#     "debit_charge":0.05
# }



# clinic_service_list = ClinicServiceList(service_list)

# # Accessing attributes of each service
# for service in clinic_service_list.get_services():
#     print(f"Service ID: {service.get_id()}")
#     print(f"Service Name: {service.get_service_name()}")
#     print(f"Service Fee: {service.get_service_fee()}")
#     print(f"Service COGS: {service.get_service_COGS()}")
#     print(f"Service Counts: {service.get_service_counts()}")
#     print(f"Service Revenue: {service.service_revenue:.2f}\n")

# print(f"Total gross Revenue: {clinic_service_list.get_gross()} $")


# total_revenue = clinic_service_list.get_gross() - (operation_cost_list.get_total_cost() + general_cost_list.get_total_cost())
# print(f"Total Revenue: {total_revenue:.2f}")