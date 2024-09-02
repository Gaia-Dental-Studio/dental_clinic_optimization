class ClinicTreatment:
    def __init__(self, treatment_id, treatment_name, treatment_fee, treatment_COGS, treatment_duration, treatment_discount_rate, treatment_debit_charge, treatment_counts=0):
        self.treatment_id = treatment_id  # treatment id or code
        self.treatment_name = treatment_name  # treatment name
        self.treatment_fee = treatment_fee  # treatment fee/price
        self.treatment_COGS = treatment_COGS  # treatment Cost of Goods Selling
        self.treatment_counts = treatment_counts  # Total treatment Counts, how many times treatment performed (optional)
        self.treatment_duration = treatment_duration  # treatment duration per client in hour
        self.treatment_discount_rate = treatment_discount_rate
        self.treatment_debit_charge = treatment_debit_charge
        self.treatment_revenue = self.calculate_treatment_sales()

    def get_id(self):
        return self.treatment_id
    
    def get_treatment_name(self):
        return self.treatment_name
    
    def get_treatment_fee(self):
        return self.treatment_fee
    
    def get_treatment_COGS(self):
        return self.treatment_COGS
    
    def get_treatment_counts(self):
        return self.treatment_counts
    
    def get_treatment_duration(self):
        return self.treatment_duration
    
    def get_discount_rate(self):
        return self.discount_rate
    
    def get_debit_charge(self):
        return self.debit_charge

    def calculate_treatment_sales(self):
        discounted_fee = self.treatment_fee * (1 - self.treatment_discount_rate)
        total_fee_after_charge = discounted_fee * (1 - self.treatment_debit_charge)
        revenue = total_fee_after_charge - self.treatment_COGS
        return revenue * self.treatment_counts
    
    def calculate_treatment_netProfit(self):
        discounted_fee = self.treatment_fee * (1 - self.treatment_discount_rate)
        total_fee_after_charge = discounted_fee * (1 - self.treatment_debit_charge)
        revenue = total_fee_after_charge - self.treatment_COGS
        # print(f"treatment: {self.treatment_name}, Original Fee: {self.treatment_fee}, Discounted Fee: {discounted_fee}, Total Fee After Charge: {total_fee_after_charge}, COGS: {self.treatment_COGS}, Net Profit: {revenue}")
        return revenue
    
    def __repr__(self):
        return (f"Treatment(treatment_id={self.treatment_id}, treatment_name='{self.treatment_name}', "
                f"treatment_fee={self.treatment_fee}, treatment_cogs={self.treatment_COGS},"
                f"treatment_duration={self.treatment_duration}, treatment_discount_rate={self.treatment_discount_rate})"
                f"treatment_debit_charge={self.treatment_debit_charge}")

class ClinicTreatmentList:
    def __init__(self, treatment_list):
        self.treatments = [
            ClinicTreatment(
                treatment_id=treatment["treatment_id"],
                treatment_name=treatment["treatment_name"],
                treatment_fee=treatment["treatment_fee"],
                treatment_COGS=treatment["treatment_cogs"],
                treatment_duration=treatment["treatment_duration"],
                treatment_discount_rate=treatment['treatment_discount_rate'],
                treatment_debit_charge=treatment['treatment_debit_charge'],
                treatment_counts=treatment.get("treatment_counts", 0)
            ) for treatment in treatment_list["treatments"]
        ]

    def get_treatments(self):
        return self.treatments
    
    def get_gross(self):
        return sum(treatment.treatment_revenue for treatment in self.treatments)
    
    def get_total_treatment_hours(self):
        return sum(treatment.get_treatment_counts() * treatment.get_treatment_duration() for treatment in self.treatments)
    
    def __repr__(self):
        return f"CostList(costs={self.treatments})"
    
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
    def __init__(self, cost_list):   
        self.costs = [
            Cost(
                cost_id=cost["cost_id"],
                cost_name=cost["cost_name"],
                cost_fee=cost["cost_fee"],
                cost_type=cost["cost_type"]
            ) for cost in cost_list["cost_list"]
        ]
        
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
        return f"CostList(costs={self.costs})"
    