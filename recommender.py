import pulp
import re

def generate_recommendations(duration_mins, event_type, event_cause, requires_road_closure, crowd_size=0, desc_text=""):
    """
    V2 Recommendation Engine: Operations Research (Linear Programming)
    Uses the PuLP library to calculate the mathematically optimal distribution 
    of resources (Cops, Barricades, Tow Trucks) under budget constraints.
    """
    
    # 1. Parse description for vehicle counts to dynamically allocate tow trucks
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'a': 1, 'an': 1
    }
    
    num_vehicles = 0
    desc_lower = desc_text.lower()
    
    vehicle_words = r'(car|truck|vehicle|bus|lorry|auto|van|jeep|bike|scooter)s?'
    number_words = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)'
    
    matches = re.findall(rf'{number_words}\s+{vehicle_words}', desc_lower)
    for match in matches:
        val_str = match[0]
        if val_str.isdigit():
            num_vehicles += int(val_str)
        else:
            num_vehicles += word_to_num.get(val_str, 0)
            
    num_vehicles = max(0, num_vehicles)
    
    # 2. Define the Problem
    prob = pulp.LpProblem("Optimal_Resource_Allocation", pulp.LpMaximize)
    
    # Dynamic scaling
    base_cops_limit = 10
    base_budget = 200
    crowd_cops = 0
    
    if crowd_size > 0:
        crowd_cops = max(1, int(crowd_size / 500))
        crowd_barricades = max(5, int(crowd_size / 100)) # 1 barricade per 100 people
        base_cops_limit += crowd_cops
        base_budget += (crowd_cops * 20) + (crowd_barricades * 5)
        
    # Scale budget for tow trucks if there are many vehicles (each tow truck costs 40)
    base_budget += (num_vehicles * 40)
    
    # Increase the maximum available budget and cops if the incident is extremely severe 
    # to guarantee the LP solver doesn't fail to find a valid solution
    if duration_mins > 60:
        extra_severity_budget = int(duration_mins - 60) * 2
        base_budget += extra_severity_budget
        base_cops_limit += int(extra_severity_budget / 20)
    
    # 3. Define Variables (Integer)
    cops = pulp.LpVariable("Police_Officers", lowBound=0, upBound=base_cops_limit, cat='Integer')
    barricades = pulp.LpVariable("Barricades", lowBound=0, upBound=max(50, int(duration_mins/15)*2 + 10), cat='Integer')
    max_tow_limit = max(2, num_vehicles + 1)
    tow_trucks = pulp.LpVariable("Tow_Trucks", lowBound=0, upBound=max_tow_limit, cat='Integer')
    
    # 4. Objective Function (Tactical Effectiveness)
    prob += 10 * cops + 15 * tow_trucks + 2 * barricades, "Total_Effectiveness"
    
    # 5. Constraints
    prob += 20 * cops + 40 * tow_trucks + 5 * barricades <= base_budget, "Zone_Budget_Constraint"
    
    if crowd_size > 0:
        prob += cops >= crowd_cops, "Min_Cops_Crowd_Control"
        prob += barricades >= crowd_barricades, "Min_Barricades_Crowd_Control"
    
    # Dynamic proportional requirements based strictly on clearance time
    # 1 cop per 30 minutes of predicted clearance time
    min_cops_duration = max(1, int(duration_mins / 30))
    prob += cops >= min_cops_duration, "Min_Cops_Duration"
    
    # 2 barricades for every 15 minutes of predicted blockage
    min_barricades_duration = int(duration_mins / 15) * 2
    prob += barricades >= min_barricades_duration, "Min_Barricades_Duration"
    
    # Minimum tow truck requirements
    if num_vehicles > 0:
        prob += tow_trucks >= num_vehicles, "Min_Tow_NLP"
    elif event_cause == "accident":
        prob += tow_trucks >= 1, "Min_Tow_Accident"
        
    # Additional baseline overrides for severe event types
    if event_cause in ["tree_fall", "water_logging"] or requires_road_closure in ["TRUE", "yes", True]:
        prob += barricades >= max(10, min_barricades_duration), "Min_Barricades_Closure"
        prob += cops >= max(2, min_cops_duration), "Min_Cops_Closure"
        
    # 5. Solve the Optimization Problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 6. Extract Results
    rec = {
        "police_required": int(cops.varValue) if cops.varValue is not None else 1,
        "barricades": int(barricades.varValue) if barricades.varValue is not None else 0,
        "tow_trucks": int(tow_trucks.varValue) if tow_trucks.varValue is not None else 0,
        "priority_alert": "High" if duration_mins > 60 else "Medium" if duration_mins > 30 else "Low",
        "diversion_advised": True if duration_mins > 45 or requires_road_closure in ["TRUE", "yes", True] else False,
        "action_plan": []
    }
    
    # Generate action plan text
    if crowd_size > 0:
        rec["action_plan"].append(f"MASS GATHERING ALERT: Deployed {rec['police_required']} personnel (1 per 500 ratio) to manage estimated crowd of {int(crowd_size):,}.")
    else:
        rec["action_plan"].append(f"Deployed {rec['police_required']} personnel using MILP optimization solver.")
        
    if rec['tow_trucks'] > 0:
        rec["action_plan"].append("Heavy tow/crane dispatch authorized.")
    if rec['diversion_advised']:
        rec["action_plan"].append("Initiating Automated OpenStreetMap (OSM) Graph Diversion Routing.")
        
    return rec
