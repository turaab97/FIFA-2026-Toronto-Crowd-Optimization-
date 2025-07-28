# Final Project - FIFA 2026 BMO Field Evacuation Optimization: Technical Report
#
# Objective: Clearing the Crowd: Optimizing Toronto's Post-Match Transit and Traffic Flow
#
# MMAI 861
# Professor Yuri Levin
# Team Broadview
# MMAI
# Smith School of Business, Queen's University
# July 2025
#
# Dumebi Onyeagwu, Ethan He, Fergie Feng, Jeremy Burbano, Syed Turab, Umair Mumtaz
#
# Problem Statement:
# The objective is to optimize the evacuation of 100,000 fans (80,000 public transit users,
# 10,000 private vehicle passengers, and 10,000 pedestrians) from BMO Field after a FIFA
# 2026 match in Toronto, within a $200,000 CAD budget. The optimization targets using
# approximately 10 trains, supplemented by buses and streetcars, to transport 70,000
# outbound and 10,000 inbound public transit passengers. The model accounts for vehicle
# capacities, traffic light cycles, road space constraints, and evacuation times, ensuring
# all passengers are evacuated efficiently while respecting budget and fleet limits.
#
# What the Code Solves:
# This Python script simulates and optimizes the evacuation process by determining the
# optimal number of buses, streetcars, and trains, as well as the traffic light green times
# for outbound and inbound directions. It uses a simulation-based approach with
# multiprocessing to evaluate multiple configurations, prioritizing approximately 10 trains
# and balancing the remaining capacity with buses and streetcars. The code outputs the
# optimal configuration, total evacuation time, cost, and a breakdown of passengers by
# mode, ensuring all constraints (budget, capacity, and traffic flow) are met.

import math
from itertools import product
import random
from multiprocessing import Pool, cpu_count

# Configuration Settings
# These constants define the problem constraints and parameters.

# Total fans to evacuate: 100,000 (80,000 public transit, 10,000 private vehicles, 10,000 pedestrians)
PEDS_OUT, PEDS_IN = 8000, 2000  # Pedestrians: 8000 outbound, 2000 inbound
CAR_PAX_OUT, CAR_PAX_IN = 7000, 3000  # Private car passengers: 7000 outbound, 3000 inbound
CAR_VEHICLES_OUT, CAR_VEHICLES_IN = 3000, 1500  # Private vehicles: 3000 outbound, 1500 inbound
PT_TOTAL_OUT, PT_TOTAL_IN = 70000, 10000  # Public transit passengers: 70,000 outbound, 10,000 inbound

# Vehicle fleet limits (maximum available vehicles)
TOTAL_BUS_LIMIT = 400  # Maximum number of buses
TOTAL_STREETCAR_LIMIT = 350  # Maximum number of streetcars
TOTAL_TRN_LIMIT = 15  # Maximum number of trains (target ~10 trains)

# Vehicle capacities (passengers per vehicle)
CAP_BUS, CAP_STREETCAR, CAP_TRN = 70, 200, 2000  # Bus: 70, Streetcar: 200, Train: 2000

# Road space units (traffic flow impact)
UNIT_BUS, UNIT_STREETCAR, UNIT_CAR = 2, 5, 1  # Bus: 2 units, Streetcar: 5 units, Car: 1 unit (Trains use 0 units)

# Boarding and round trip times (in minutes)
BUS_BOARD, STREETCAR_BOARD, TRN_BOARD = 3, 3, 3  # Boarding time: 3 minutes for all vehicles
BUS_RT = 20  # Bus round trip: 20 minutes
STREETCAR_RT = 20  # Streetcar round trip: 20 minutes
TRN_RT = 30  # Train round trip: 30 minutes

# Traffic light capacity (vehicles or pedestrians per minute, decreasing by 10 per minute until red light resets)
RATE_OUT_VEHICLE, RATE_OUT_PED = 150, 200  # Outbound: 150 vehicle units/min, 200 pedestrians/min
RATE_IN_VEHICLE, RATE_IN_PED = 100, 150  # Inbound: 100 vehicle units/min, 150 pedestrians/min

# Budget and costs (in CAD)
BUDGET = 200000  # Total budget: $200,000
COST_BUS = 500  # Cost per bus: $500
COST_STREETCAR = 500  # Cost per streetcar: $500
COST_TRN = 2000  # Cost per train: $2000

def units_available(base_rate, minute_idx, dt):
    """
    Calculate available traffic light capacity for vehicles or pedestrians.
    Capacity decreases by 10 units per minute until the red light resets.

    Args:
        base_rate (int): Base capacity per minute (vehicles or pedestrians)
        minute_idx (int): Current minute in the traffic light cycle
        dt (float): Time step (in minutes, typically 0.5)

    Returns:
        float: Available capacity for the time step
    """
    return max(base_rate - 10 * minute_idx, 0) * dt

def simulate(Lo, Li, bus_out, bus_in, streetcar_out, streetcar_in, trn_used, dt=0.5, max_time=300):
    """
    Simulate the evacuation process for a given configuration.

    Args:
        Lo (float): Outbound green light duration (minutes)
        Li (float): Inbound green light duration (minutes)
        bus_out (int): Number of outbound buses
        bus_in (int): Number of inbound buses
        streetcar_out (int): Number of outbound streetcars
        streetcar_in (int): Number of inbound streetcars
        trn_used (int): Number of trains used
        dt (float): Time step (default 0.5 minutes)
        max_time (float): Maximum simulation time (default 300 minutes)

    Returns:
        tuple: (evacuation_time, counts, done_time, total_cost)
            - evacuation_time (float): Total time to evacuate all passengers
            - counts (dict): Number of passengers evacuated by mode and direction
            - done_time (dict): Completion time for each mode
            - total_cost (float): Total cost of the configuration
    """
    # Check if the configuration exceeds the budget
    total_cost = ((bus_out + bus_in) * COST_BUS +
                  (streetcar_out + streetcar_in) * COST_STREETCAR +
                  trn_used * COST_TRN)
    if total_cost > BUDGET:
        return math.inf, {}, {}, total_cost

    # Calculate total capacity for each vehicle type
    total_bus_capacity_out = bus_out * CAP_BUS
    total_bus_capacity_in = bus_in * CAP_BUS
    total_streetcar_capacity_out = streetcar_out * CAP_STREETCAR
    total_streetcar_capacity_in = streetcar_in * CAP_STREETCAR
    total_train_capacity = trn_used * CAP_TRN

    # Allocate public transit passengers: prioritize trains (high capacity, no road usage)
    train_pax_out = min(PT_TOTAL_OUT, total_train_capacity)
    remaining_train_capacity = total_train_capacity - train_pax_out
    train_pax_in = min(PT_TOTAL_IN, remaining_train_capacity)

    # Allocate remaining passengers to streetcars (higher capacity per dollar)
    remaining_out = PT_TOTAL_OUT - train_pax_out
    remaining_in = PT_TOTAL_IN - train_pax_in
    streetcar_pax_out = min(remaining_out, total_streetcar_capacity_out)
    streetcar_pax_in = min(remaining_in, total_streetcar_capacity_in)

    # Allocate remaining passengers to buses
    remaining_out -= streetcar_pax_out
    remaining_in -= streetcar_pax_in
    bus_pax_out = min(remaining_out, total_bus_capacity_out)
    bus_pax_in = min(remaining_in, total_bus_capacity_in)

    # Verify if all public transit passengers can be served
    total_pt_served = (train_pax_out + train_pax_in +
                       streetcar_pax_out + streetcar_pax_in +
                       bus_pax_out + bus_pax_in)
    total_pt_demand = PT_TOTAL_OUT + PT_TOTAL_IN
    if total_pt_served < total_pt_demand:
        return math.inf, {}, {}, total_cost  # Insufficient capacity

    # Initialize remaining passengers and vehicles
    ped_out, ped_in = PEDS_OUT, PEDS_IN
    car_pax_out, car_pax_in = CAR_PAX_OUT, CAR_PAX_IN
    car_vehicles_out, car_vehicles_in = CAR_VEHICLES_OUT, CAR_VEHICLES_IN
    remaining_bus_pax_out, remaining_bus_pax_in = bus_pax_out, bus_pax_in
    remaining_streetcar_pax_out, remaining_streetcar_pax_in = streetcar_pax_out, streetcar_pax_in
    remaining_train_pax = train_pax_out + train_pax_in

    # Track passengers evacuated by mode and direction
    counts = {
        'ped_out': 0, 'ped_in': 0,
        'car_out': 0, 'car_in': 0,
        'bus_out': 0, 'bus_in': 0,
        'streetcar_out': 0, 'streetcar_in': 0,
        'train_out': 0, 'train_in': 0
    }

    # Track completion times for each mode
    done_time = {'ped': None, 'bus': None, 'streetcar': None, 'train': None}

    # Initialize timers for vehicle availability (time until vehicle is ready again)
    bus_out_timers = [0] * bus_out
    bus_in_timers = [0] * bus_in
    streetcar_out_timers = [0] * streetcar_out
    streetcar_in_timers = [0] * streetcar_in
    train_timers = [0] * trn_used

    def tick_timers(timer_list):
        """Update vehicle timers by subtracting the time step."""
        for i in range(len(timer_list)):
            if timer_list[i] > 0:
                timer_list[i] = max(0, timer_list[i] - dt)

    t = 0.0  # Current simulation time
    cycle_pos = 0.0  # Position in traffic light cycle
    cycle_length = Lo + Li  # Total cycle length

    while t < max_time:
        # Determine if current cycle is outbound or inbound
        is_outbound = cycle_pos < Lo
        minute_idx = int(cycle_pos if is_outbound else cycle_pos - Lo)
        base_vehicle_rate = RATE_OUT_VEHICLE if is_outbound else RATE_IN_VEHICLE
        base_ped_rate = RATE_OUT_PED if is_outbound else RATE_IN_PED

        # Calculate available vehicle capacity for this time step
        vehicle_capacity = units_available(base_vehicle_rate, minute_idx, dt)
        remaining_capacity = vehicle_capacity

        # Process trains (no road space usage)
        if remaining_train_pax > 0:
            for i in range(trn_used):
                if remaining_train_pax <= 0:
                    break
                if train_timers[i] <= 0:  # Train is available
                    passengers_to_load = min(remaining_train_pax, CAP_TRN)
                    remaining_train_pax -= passengers_to_load
                    if is_outbound and train_pax_out > counts['train_out']:
                        actual_load = min(passengers_to_load, train_pax_out - counts['train_out'])
                        counts['train_out'] += actual_load
                    elif not is_outbound and train_pax_in > counts['train_in']:
                        actual_load = min(passengers_to_load, train_pax_in - counts['train_in'])
                        counts['train_in'] += actual_load
                    else:
                        # Load passengers from the other direction if current direction is full
                        if train_pax_out > counts['train_out']:
                            actual_load = min(passengers_to_load, train_pax_out - counts['train_out'])
                            counts['train_out'] += actual_load
                        else:
                            actual_load = min(passengers_to_load, train_pax_in - counts['train_in'])
                            counts['train_in'] += actual_load
                    train_timers[i] = TRN_BOARD + TRN_RT
                    if done_time['train'] is None or t > done_time['train']:
                        done_time['train'] = t

        # Process streetcars (outbound or inbound based on cycle)
        if is_outbound and remaining_streetcar_pax_out > 0:
            for i in range(streetcar_out):
                if remaining_streetcar_pax_out <= 0 or remaining_capacity < UNIT_STREETCAR:
                    break
                if streetcar_out_timers[i] <= 0:  # Streetcar is available
                    passengers_to_load = min(remaining_streetcar_pax_out, CAP_STREETCAR)
                    remaining_streetcar_pax_out -= passengers_to_load
                    remaining_capacity -= UNIT_STREETCAR
                    counts['streetcar_out'] += passengers_to_load
                    streetcar_out_timers[i] = STREETCAR_BOARD + STREETCAR_RT
                    if done_time['streetcar'] is None or t > done_time['streetcar']:
                        done_time['streetcar'] = t
        elif not is_outbound and remaining_streetcar_pax_in > 0:
            for i in range(streetcar_in):
                if remaining_streetcar_pax_in <= 0 or remaining_capacity < UNIT_STREETCAR:
                    break
                if streetcar_in_timers[i] <= 0:  # Streetcar is available
                    passengers_to_load = min(remaining_streetcar_pax_in, CAP_STREETCAR)
                    remaining_streetcar_pax_in -= passengers_to_load
                    remaining_capacity -= UNIT_STREETCAR
                    counts['streetcar_in'] += passengers_to_load
                    streetcar_in_timers[i] = STREETCAR_BOARD + STREETCAR_RT
                    if done_time['streetcar'] is None or t > done_time['streetcar']:
                        done_time['streetcar'] = t

        # Process buses (outbound or inbound based on cycle)
        if is_outbound and remaining_bus_pax_out > 0:
            for i in range(bus_out):
                if remaining_bus_pax_out <= 0 or remaining_capacity < UNIT_BUS:
                    break
                if bus_out_timers[i] <= 0:  # Bus is available
                    passengers_to_load = min(remaining_bus_pax_out, CAP_BUS)
                    remaining_bus_pax_out -= passengers_to_load
                    remaining_capacity -= UNIT_BUS
                    counts['bus_out'] += passengers_to_load
                    bus_out_timers[i] = BUS_BOARD + BUS_RT
                    if done_time['bus'] is None or t > done_time['bus']:
                        done_time['bus'] = t
        elif not is_outbound and remaining_bus_pax_in > 0:
            for i in range(bus_in):
                if remaining_bus_pax_in <= 0 or remaining_capacity < UNIT_BUS:
                    break
                if bus_in_timers[i] <= 0:  # Bus is available
                    passengers_to_load = min(remaining_bus_pax_in, CAP_BUS)
                    remaining_bus_pax_in -= passengers_to_load
                    remaining_capacity -= UNIT_BUS
                    counts['bus_in'] += passengers_to_load
                    bus_in_timers[i] = BUS_BOARD + BUS_RT
                    if done_time['bus'] is None or t > done_time['bus']:
                        done_time['bus'] = t

        # Process private vehicles
        if is_outbound and car_vehicles_out > 0:
            max_cars = min(remaining_capacity // UNIT_CAR, car_vehicles_out)
            if max_cars > 0:
                avg_passengers_per_car = CAR_PAX_OUT / CAR_VEHICLES_OUT
                passengers_evacuated = min(car_pax_out, max_cars * avg_passengers_per_car)
                cars_used = min(max_cars, math.ceil(passengers_evacuated / avg_passengers_per_car))
                car_pax_out -= passengers_evacuated
                car_vehicles_out -= cars_used
                remaining_capacity -= cars_used * UNIT_CAR
                counts['car_out'] += passengers_evacuated
        elif not is_outbound and car_vehicles_in > 0:
            max_cars = min(remaining_capacity // UNIT_CAR, car_vehicles_in)
            if max_cars > 0:
                avg_passengers_per_car = CAR_PAX_IN / CAR_VEHICLES_IN
                passengers_evacuated = min(car_pax_in, max_cars * avg_passengers_per_car)
                cars_used = min(max_cars, math.ceil(passengers_evacuated / avg_passengers_per_car))
                car_pax_in -= passengers_evacuated
                car_vehicles_in -= cars_used
                remaining_capacity -= cars_used * UNIT_CAR
                counts['car_in'] += passengers_evacuated

        # Process pedestrians
        ped_capacity = units_available(base_ped_rate, minute_idx, dt)
        if is_outbound:
            ped_evacuated = min(ped_out, ped_capacity)
            ped_out -= ped_evacuated
            counts['ped_out'] += ped_evacuated
        else:
            ped_evacuated = min(ped_in, ped_capacity)
            ped_in -= ped_evacuated
            counts['ped_in'] += ped_evacuated

        # Update pedestrian completion time
        if ped_out <= 0 and ped_in <= 0 and done_time['ped'] is None:
            done_time['ped'] = t

        # Update all vehicle timers
        tick_timers(bus_out_timers)
        tick_timers(bus_in_timers)
        tick_timers(streetcar_out_timers)
        tick_timers(streetcar_in_timers)
        tick_timers(train_timers)

        # Advance time and cycle position
        t += dt
        cycle_pos += dt
        if cycle_pos >= cycle_length:
            cycle_pos -= cycle_length

        # Check if evacuation is complete
        if all(x <= 0 for x in [ped_out, ped_in, car_pax_out, car_pax_in,
                                remaining_bus_pax_out, remaining_bus_pax_in,
                                remaining_streetcar_pax_out, remaining_streetcar_pax_in,
                                remaining_train_pax]):
            for k in done_time:
                if done_time[k] is None:
                    done_time[k] = t
            evacuation_time = max([tm for tm in done_time.values() if tm is not None])
            return evacuation_time, counts, done_time, total_cost

    return math.inf, counts, done_time, total_cost

def find_feasible_config():
    """
    Find a feasible vehicle configuration with approximately 10 trains.

    Returns:
        tuple: (trains, streetcars, buses) or None if no feasible configuration is found
    """
    trains = 10  # Target number of trains
    train_capacity = trains * CAP_TRN  # Total train capacity
    train_cost = trains * COST_TRN  # Total train cost
    remaining_capacity_needed = (PT_TOTAL_OUT + PT_TOTAL_IN) - train_capacity  # Remaining passengers
    remaining_budget = BUDGET - train_cost  # Remaining budget

    best_config = None
    best_cost = math.inf

    # Prioritize streetcars for remaining capacity (better capacity per dollar)
    for streetcars in range(200, min(TOTAL_STREETCAR_LIMIT, 320), 10):
        streetcar_capacity = streetcars * CAP_STREETCAR
        streetcar_cost = streetcars * COST_STREETCAR
        if streetcar_cost > remaining_budget:
            continue
        bus_capacity_needed = remaining_capacity_needed - streetcar_capacity
        if bus_capacity_needed <= 0:
            # Enough capacity with trains and streetcars
            if streetcar_cost < best_cost:
                best_cost = streetcar_cost
                best_config = (trains, streetcars, 0)
            continue
        # Calculate required buses
        buses_needed = math.ceil(bus_capacity_needed / CAP_BUS)
        bus_cost = buses_needed * COST_BUS
        total_cost = train_cost + streetcar_cost + bus_cost
        if total_cost <= BUDGET and total_cost < best_cost:
            best_cost = total_cost
            best_config = (trains, streetcars, buses_needed)

    return best_config

def evaluate_config(args):
    """
    Evaluate a single configuration by running the simulation.

    Args:
        args (tuple): (Lo, Li, bus_total, streetcar_total, bus_out_ratio, streetcar_out_ratio, trn_used)

    Returns:
        tuple: (score, config)
            - score (float): Evacuation time
            - config (tuple): Configuration parameters
    """
    Lo, Li, bus_total, streetcar_total, bus_out_ratio, streetcar_out_ratio, trn_used = args
    bus_out = int(bus_total * bus_out_ratio)
    bus_in = bus_total - bus_out
    streetcar_out = int(streetcar_total * streetcar_out_ratio)
    streetcar_in = streetcar_total - streetcar_out
    score, counts, times, cost = simulate(Lo, Li, bus_out, bus_in, streetcar_out, streetcar_in, trn_used)
    return score, (Lo, Li, bus_total, bus_out, bus_in, streetcar_total, streetcar_out, streetcar_in, trn_used)

def optimize():
    """
    Optimize the evacuation configuration using a two-phase approach:
    1. Find a feasible configuration with ~10 trains.
    2. Evaluate and fine-tune configurations using multiprocessing.

    Returns:
        tuple: (best_config, best_score)
            - best_config (tuple): Optimal configuration parameters
            - best_score (float): Optimal evacuation time
    """
    print("Starting evacuation optimization for 100,000 fans...")
    print("Public transit passengers: 80,000 (70,000 outbound + 10,000 inbound)")
    print("Target: Use approximately 10 trains")

    # Phase 1: Find a feasible configuration
    print("\nFinding feasible configuration...")
    feasible = find_feasible_config()
    if feasible:
        trains, streetcars, buses = feasible
        print(f"Feasible config found: {trains} trains, {streetcars} streetcars, {buses} buses")
        total_cost = trains * COST_TRN + streetcars * COST_STREETCAR + buses * COST_BUS
        total_capacity = trains * CAP_TRN + streetcars * CAP_STREETCAR + buses * CAP_BUS
        print(f"Cost: ${total_cost:,}, Capacity: {total_capacity:,}")
    else:
        print("No feasible configuration found!")
        return None, math.inf

    # Use multiprocessing for parallel evaluation
    num_processes = min(cpu_count(), 8)
    Lo_vals = [2.0, 2.5, 3.0, 3.5, 4.0]  # Outbound green light durations
    Li_vals = [0.5, 1.0, 1.5]  # Inbound green light durations
    configs = []

    # Generate configurations around the feasible solution
    for Lo, Li in product(Lo_vals, Li_vals):
        for train_delta in [-1, 0, 1]:
            trn_used = max(9, min(12, trains + train_delta))
            for streetcar_delta in range(-40, 41, 10):
                streetcar_total = max(150, min(TOTAL_STREETCAR_LIMIT, streetcars + streetcar_delta))
                for bus_delta in range(-30, 31, 10):
                    bus_total = max(0, min(TOTAL_BUS_LIMIT, buses + bus_delta))
                    # Check if configuration has sufficient capacity
                    total_capacity = (trn_used * CAP_TRN +
                                     streetcar_total * CAP_STREETCAR +
                                     bus_total * CAP_BUS)
                    if total_capacity < 80000:
                        continue
                    # Check budget constraint
                    total_cost = (trn_used * COST_TRN +
                                  streetcar_total * COST_STREETCAR +
                                  bus_total * COST_BUS)
                    if total_cost > BUDGET:
                        continue
                    # Try different vehicle allocation ratios (favoring outbound)
                    for bus_out_ratio in [0.7, 0.75, 0.8, 0.85, 0.9]:
                        for streetcar_out_ratio in [0.7, 0.75, 0.8, 0.85, 0.9]:
                            configs.append((Lo, Li, bus_total, streetcar_total,
                                            bus_out_ratio, streetcar_out_ratio, trn_used))

    print(f"\nPhase 1: Evaluating {len(configs)} configurations using {num_processes} processes...")
    if len(configs) == 0:
        print("No valid configurations generated!")
        return (3.0, 1.0, buses, buses * 4 // 5, buses // 5,
                streetcars, streetcars * 4 // 5, streetcars // 5, trains), math.inf

    # Parallel evaluation of configurations
    with Pool(num_processes) as pool:
        results = pool.map(evaluate_config, configs)

    # Find the best configuration
    best_score = math.inf
    best_config = None
    for score, config in results:
        if score < best_score:
            best_score = score
            best_config = config

    # Phase 2: Fine-tune the best configuration
    if best_config and best_score < math.inf:
        print(f"\nPhase 2: Fine-tuning best configuration (time: {best_score:.1f} min)...")
        Lo_base, Li_base, bus_tot_base, bus_out_base, bus_in_base, \
            streetcar_tot_base, streetcar_out_base, streetcar_in_base, trn_base = best_config
        fine_configs = []
        for Lo in [Lo_base - 0.25, Lo_base, Lo_base + 0.25]:
            for Li in [Li_base - 0.25, Li_base, Li_base + 0.25]:
                if Lo <= 0 or Li <= 0:
                    continue
                for adjustment in range(-10, 11, 5):
                    bus_out_new = max(0, min(bus_tot_base, bus_out_base + adjustment))
                    bus_in_new = bus_tot_base - bus_out_new
                    streetcar_out_new = max(0, min(streetcar_tot_base, streetcar_out_base + adjustment))
                    streetcar_in_new = streetcar_tot_base - streetcar_out_new
                    fine_configs.append((Lo, Li, bus_tot_base, streetcar_tot_base,
                                        bus_out_new / bus_tot_base if bus_tot_base > 0 else 0.8,
                                        streetcar_out_new / streetcar_tot_base if streetcar_tot_base > 0 else 0.8,
                                        trn_base))

        # Evaluate fine-tuning configurations
        with Pool(num_processes) as pool:
            fine_results = pool.map(evaluate_config, fine_configs)
        for score, config in fine_results:
            if score < best_score:
                best_score = score
                best_config = config

    return best_config, best_score

if __name__ == '__main__':
    """
    Main execution block: Run the optimization and display results.
    """
    best_cfg, best_time = optimize()
    if best_cfg is None:
        print("No feasible solution found within budget!")
    else:
        # Unpack the optimal configuration
        Lo_opt, Li_opt, bus_total, bus_out, bus_in, \
            streetcar_total, streetcar_out, streetcar_in, trn_used = best_cfg
        # Run simulation with the optimal configuration to get detailed results
        final_time, best_counts, best_times, total_cost = simulate(
            Lo_opt, Li_opt, bus_out, bus_in, streetcar_out, streetcar_in, trn_used
        )
        # Display results
        print("\n=== Optimal Configuration ===")
        print(f"Outbound green: {Lo_opt:.2f} min, Inbound green: {Li_opt:.2f} min")
        print(f"Total Bus: {bus_total}, Out/In: {bus_out}/{bus_in}")
        print(f"Total Street Cars: {streetcar_total}, Out/In: {streetcar_out}/{streetcar_in}")
        print(f"Trains used: {trn_used}")
        print(f"Total cost: ${total_cost:,} CAD (Budget: ${BUDGET:,} CAD)")
        print(f"Evacuation time: {best_time:.1f} min (~{best_time / 60:.2f} h)\n")

        print("=== Breakdown ===")
        total_evacuated = sum(best_counts.values())
        for mode, num in best_counts.items():
            percentage = (num / total_evacuated) * 100 if total_evacuated > 0 else 0
            print(f"{mode:15s}: {num:6.0f} ({percentage:5.1f}%)")
        print(f"{'Total':15s}: {total_evacuated:6.0f}")

        print("\n=== Completion Times (min) ===")
        for mode, tm in best_times.items():
            if tm is not None:
                print(f"{mode:15s}: {tm:6.1f}")
            else:
                print(f"{mode:15s}: Not completed")

        # Summarize public transit allocation
        total_pt_by_bus = best_counts['bus_out'] + best_counts['bus_in']
        total_pt_by_streetcar = best_counts['streetcar_out'] + best_counts['streetcar_in']
        total_pt_by_train = best_counts['train_out'] + best_counts['train_in']
        print(f"\n=== Public Transit Allocation ===")
        print(f"Bus passengers: {total_pt_by_bus:,.0f}")
        print(f"Street Car passengers: {total_pt_by_streetcar:,.0f}")
        print(f"Train passengers: {total_pt_by_train:,.0f}")
        print(f"Total PT passengers: {total_pt_by_bus + total_pt_by_streetcar + total_pt_by_train:,.0f} / 80,000")

        # Verify total evacuation
        expected_total = PEDS_OUT + PEDS_IN + CAR_PAX_OUT + CAR_PAX_IN + PT_TOTAL_OUT + PT_TOTAL_IN
        print(f"\n=== Verification ===")
        print(f"Expected total: {expected_total:,} people")
        print(f"Actually evacuated: {total_evacuated:,.0f} people")
        print(f"Match: {'✓' if abs(expected_total - total_evacuated) < 1 else '✗'}")
