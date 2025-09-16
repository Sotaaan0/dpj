import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# Parameters
# -----------------------------
years = range(2020, 2031)  # simulate 10 years
initial_fleet_size = 10000

fuel_types = ["Petrol", "Diesel", "Hybrid", "EV", "Ethanol"]

# Base consumption (liters/100km; 0 for EVs)
base_consumption = {"Petrol": 7.5, "Diesel": 6.0, "Hybrid": 4.0, "EV": 0.0, "Ethanol": 6.5}

# Example prices (€) and running costs (€ per year)
base_price = {"Petrol": 25000, "Diesel": 27000, "Hybrid": 30000, "EV": 35000, "Ethanol": 26000}
base_run = {"Petrol": 1500, "Diesel": 1300, "Hybrid": 1000, "EV": 400, "Ethanol": 1200}

# Coefficients (toy example; normally estimated from data!)
beta_price = -0.0001
beta_run = -0.002
beta_hybrid = 0.2
beta_ev = 0.5

# -----------------------------
# Initial fleet
# -----------------------------
fleet = pd.DataFrame({
    "year_added": np.random.randint(2000, 2020, initial_fleet_size),
    "fuel_type": np.random.choice(fuel_types, initial_fleet_size, p=[0.5, 0.3, 0.1, 0.05, 0.05])
})
fleet["age"] = 2020 - fleet["year_added"]
fleet["fuel_consumption"] = fleet["fuel_type"].map(base_consumption) + np.random.normal(0, 0.5, initial_fleet_size)

# -----------------------------
# Discrete choice model
# -----------------------------
def compute_utilities():
    """Return a DataFrame of car alternatives with utilities and logit probabilities."""
    cars = pd.DataFrame({
        "fuel_type": fuel_types,
        "price": [base_price[f] for f in fuel_types],
        "running_cost": [base_run[f] for f in fuel_types]
    })
    
    def utility(row):
        u = beta_price * row["price"] + beta_run * row["running_cost"]
        if row["fuel_type"] == "EV":
            u += beta_ev
        elif row["fuel_type"] == "Hybrid":
            u += beta_hybrid
        return u
    
    cars["utility"] = cars.apply(utility, axis=1)
    cars["expU"] = np.exp(cars["utility"])
    cars["prob"] = cars["expU"] / cars["expU"].sum()
    return cars

def sample_new_cars(n_new):
    """Sample new cars for buyers based on logit probabilities."""
    cars = compute_utilities()
    choices = np.random.choice(cars["fuel_type"], size=n_new, p=cars["prob"])
    new_cars = pd.DataFrame({
        "year_added": [year]*n_new,
        "fuel_type": choices,
    })
    new_cars["age"] = 0
    new_cars["fuel_consumption"] = new_cars["fuel_type"].map(base_consumption) + np.random.normal(0, 0.3, n_new)
    return new_cars

# -----------------------------
# Scrapping function
# -----------------------------
def scrapping_probability(age):
    """Simple scrapping probability by age."""
    return min(0.02 * age, 0.9)  # grows with age, capped at 90%

# -----------------------------
# Simulation
# -----------------------------
results = []
fleet_year = fleet.copy()

for year in years:
    # Age cars
    fleet_year["age"] += 1
    
    # Scrapping step
    scrap_flags = np.random.rand(len(fleet_year)) < fleet_year["age"].apply(scrapping_probability)
    fleet_year = fleet_year.loc[~scrap_flags]
    
    # Add new cars (fleet grows slightly each year)
    n_new = int(len(fleet_year) * 0.01) + 1000
    new_cars = sample_new_cars(n_new)
    fleet_year = pd.concat([fleet_year, new_cars], ignore_index=True)
    
    # Metrics
    avg_fuel = fleet_year["fuel_consumption"].mean()
    share_electric = (fleet_year["fuel_type"] == "EV").mean()
    share_hybrid = (fleet_year["fuel_type"] == "Hybrid").mean()
    results.append({
        "year": year,
        "fleet_size": len(fleet_year),
        "avg_fuel_consumption": avg_fuel,
        "share_electric": share_electric,
        "share_hybrid": share_hybrid
    })

results_df = pd.DataFrame(results)

# -----------------------------
# Results visualization
# -----------------------------
print(results_df)

plt.figure(figsize=(10,6))
plt.plot(results_df["year"], results_df["avg_fuel_consumption"], label="Avg Fuel Consumption (l/100km)")
plt.plot(results_df["year"], results_df["share_electric"]*100, label="EV Share (%)")
plt.plot(results_df["year"], results_df["share_hybrid"]*100, label="Hybrid Share (%)")
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend()
plt.title("Toy Swedish Car Fleet Model with Discrete Choice")
plt.show()
