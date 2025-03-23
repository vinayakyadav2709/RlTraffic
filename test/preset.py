# preset_simulation.py
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# SUMO configuration
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

# Fixed phase configuration
PHASE_DEFINITIONS = [
    ("yyyrrrrrrrrr", "GGGrrrrrrrrr", 30),  # Phase 0: 30 seconds duration
    ("rrryyyrrrrrr", "rrrGGGrrrrrr", 30),  # Phase 1
    ("rrrrrryyyrrr", "rrrrrrGGGrrr", 30),  # Phase 2
    ("rrrrrrrrryyy", "rrrrrrrrrGGG", 30),  # Phase 3
]

def get_vehicle_numbers(lanes):
    """Count vehicles in each lane past 10m from intersection"""
    return {
        lane: sum(
            1 for vid in traci.lane.getLastStepVehicleIDs(lane)
            if traci.vehicle.getLanePosition(vid) > 10
        )
        for lane in lanes
    }

def get_waiting_time(lanes):
    """Calculate total waiting time for vehicles in lanes"""
    return sum(traci.lane.getWaitingTime(lane) for lane in lanes)

def set_phase(junction, phase_index):
    """Set traffic light to specified phase with yellow transition"""
    yellow_state, green_state, duration = PHASE_DEFINITIONS[phase_index]
    
    # Yellow transition
    traci.trafficlight.setRedYellowGreenState(junction, yellow_state)
    traci.trafficlight.setPhaseDuration(junction, 5)  # 5 second yellow
    
    # Main green phase
    traci.trafficlight.setRedYellowGreenState(junction, green_state)
    traci.trafficlight.setPhaseDuration(junction, duration - 5)

def run_preset_simulation(config_path, steps=1000, gui=False):
    """Run simulation with fixed-time traffic lights"""
    total_waiting = 0
    total_vehicles = 0
    metrics = []
    
    sumo_binary = checkBinary("sumo-gui" if gui else "sumo")
    
    try:
        traci.start([sumo_binary, "-c", config_path], port=8873)
        junctions = traci.trafficlight.getIDList()
        
        # Initialize phase timers
        phase_timers = {junction: 0 for junction in junctions}
        current_phases = {junction: 0 for junction in junctions}
        
        print(f"Starting preset timing simulation for {steps} steps...")
        start_time = time.time()
        
        for step in range(steps):
            traci.simulationStep()
            
            for junction in junctions:
                phase_timers[junction] -= 1
                
                if phase_timers[junction] <= 0:
                    # Rotate to next phase
                    current_phases[junction] = (current_phases[junction] + 1) % 4
                    set_phase(junction, current_phases[junction])
                    phase_timers[junction] = PHASE_DEFINITIONS[current_phases[junction]][2]
                
                # Collect metrics
                lanes = traci.trafficlight.getControlledLanes(junction)
                total_waiting += get_waiting_time(lanes)
                total_vehicles += sum(get_vehicle_numbers(lanes).values())
                
            # Record metrics every 10 steps
            if step % 10 == 0:
                metrics.append((
                    step,
                    total_waiting,
                    total_vehicles,
                    traci.simulation.getDepartedNumber(),
                    traci.simulation.getArrivedNumber()
                ))
                
        duration = time.time() - start_time
        print(f"Simulation completed in {duration:.2f} seconds")
        
        return {
            'total_waiting': total_waiting,
            'total_vehicles': total_vehicles,
            'avg_waiting_per_vehicle': total_waiting / total_vehicles if total_vehicles else 0,
            'metrics': metrics
        }
        
    finally:
        traci.close()

def save_results(results, filename=None):
    """Save simulation results to file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"preset_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("Preset Timing Simulation Results\n")
        f.write("="*40 + "\n")
        f.write(f"Total Waiting Time: {results['total_waiting']:.2f}s\n")
        f.write(f"Total Vehicles Processed: {results['total_vehicles']}\n")
        f.write(f"Average Waiting per Vehicle: {results['avg_waiting_per_vehicle']:.2f}s\n")
        
        f.write("\nTime Step Metrics:\n")
        f.write("Step | Waiting (s) | Vehicles | Departed | Arrived\n")
        for entry in results['metrics']:
            f.write(f"{entry[0]:4} | {entry[1]:9.2f} | {entry[2]:8} | {entry[3]:8} | {entry[4]:7}\n")
    
    print(f"Results saved to {filename}")

def plot_results(results):
    """Generate waiting time visualization"""
    steps = [m[0] for m in results['metrics']]
    waiting = [m[1] for m in results['metrics']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, waiting)
    plt.title("Cumulative Waiting Time Over Simulation")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Waiting Time (seconds)")
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"preset_waiting_{timestamp}.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-time Traffic Light Simulation")
    parser.add_argument("-c", "--config", required=True, help="SUMO config file path")
    parser.add_argument("-s", "--steps", type=int, default=1000, 
                       help="Number of simulation steps")
    parser.add_argument("-g", "--gui", action="store_true", 
                       help="Run with SUMO GUI")
    parser.add_argument("-o", "--output", help="Output filename for results")
    parser.add_argument("-p", "--plot", action="store_true", 
                       help="Generate waiting time plot")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        sys.exit(f"Config file not found: {args.config}")
    
    results = run_preset_simulation(
        config_path=os.path.abspath(args.config),
        steps=args.steps,
        gui=args.gui
    )
    
    save_results(results, args.output)
    
    if args.plot:
        plot_results(results)