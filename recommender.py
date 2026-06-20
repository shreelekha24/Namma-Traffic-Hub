import pulp

def generate_recommendations(duration_category, event_type, event_cause, requires_road_closure, num_vehicles=0):
    """
    V2 Recommendation Engine: Operations Research (Linear Programming)
    Uses the PuLP library to calculate the mathematically optimal distribution 
    of resources (Cops, Barricades, Tow Trucks) under budget constraints.
    Updated for Time Buckets and VIP Event Scaling.
    """
    
    # 1. Define the Problem
    prob = pulp.LpProblem("Optimal_Resource_Allocation", pulp.LpMaximize)
    
    # VIP / High-Profile Event Check
    is_vip_event = (event_type == 'planned') and (event_cause in ['vip_movement', 'sports_event', 'procession', 'protest', 'public_event'])
    
    # Tow Truck Target based on vehicles
    tow_truck_target = num_vehicles if event_cause in ['accident', 'vehicle_breakdown', 'water_logging'] else 1 if event_cause == 'tree_fall' else 0
    
    # Dynamic constraints based on event type
    budget = 10000 # High budget to avoid infeasible solutions with higher requirements
    
    # 2. Define Variables (Integer)
    cops = pulp.LpVariable("Police_Officers", lowBound=0, upBound=100, cat='Integer')
    barricades = pulp.LpVariable("Barricades", lowBound=0, upBound=200, cat='Integer')
    tow_trucks = pulp.LpVariable("Tow_Trucks", lowBound=tow_truck_target, upBound=tow_truck_target, cat='Integer')
    
    # 3. Define the Objective Function (Tactical Effectiveness Score)
    prob += 10 * cops + 15 * tow_trucks + 2 * barricades, "Total_Effectiveness"
    
    # 4. Define Budget Constraint
    prob += 20 * cops + 40 * tow_trucks + 5 * barricades <= budget, "Zone_Budget_Constraint"
    
    # 5. Define Minimum Requirements
    if is_vip_event:
        prob += cops >= 50
        prob += cops <= 60
        prob += barricades >= 100
        prob += barricades <= 120
    else:
        # Minimum requirements based on duration category as requested
        if duration_category == "< 30 mins":
            prob += cops >= 7
            prob += cops <= 8
            prob += barricades >= 5
            prob += barricades <= 6
        elif duration_category == "30-60 mins":
            prob += cops >= 10
            prob += cops <= 12
            prob += barricades >= 10
            prob += barricades <= 15
        elif duration_category == "1-2 hours":
            prob += cops >= 15
            prob += cops <= 20
            prob += barricades >= 20
            prob += barricades <= 30
        elif duration_category == "> 2 hours":
            prob += cops >= 25
            prob += cops <= 30
            prob += barricades >= 40
            prob += barricades <= 50
        
    # Additional requirements if road is closed
    if event_cause in ["tree_fall", "water_logging"] or requires_road_closure in ["TRUE", "yes", True]:
        prob += barricades >= 15
        
    # 6. Solve the Optimization Problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 7. Extract Results
    rec = {
        "police_required": int(cops.varValue) if cops.varValue is not None else 1,
        "barricades": int(barricades.varValue) if barricades.varValue is not None else 0,
        "tow_trucks": int(tow_trucks.varValue) if tow_trucks.varValue is not None else 0,
        "priority_alert": "Critical" if is_vip_event else "High" if duration_category in ["> 2 hours", "1-2 hours"] else "Medium" if duration_category == "30-60 mins" else "Low",
        "diversion_advised": True if is_vip_event or event_cause in ["accident", "tree_fall", "water_logging"] or duration_category in ["> 2 hours", "1-2 hours"] or requires_road_closure in ["TRUE", "yes", True] else False,
        "action_plan": []
    }
    
    # Generate action plan text
    if is_vip_event:
        rec["action_plan"].append("VIP/Public Event Protocol Initiated: Authorized mass deployment.")
    
    rec["action_plan"].append(f"Deployed {rec['police_required']} personnel using MILP optimization solver.")
    
    if rec['tow_trucks'] > 0:
        rec["action_plan"].append(f"Heavy tow/crane dispatch authorized ({rec['tow_trucks']} units).")
    if rec['diversion_advised']:
        rec["action_plan"].append("Initiating Automated OpenStreetMap (OSM) Graph Diversion Routing.")
        
    return rec
