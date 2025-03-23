import os
import sys
import time
import optparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Ensure SUMO_HOME is declared
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

##############################
# Simple PyTorch Model (for RL)
##############################
class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, fc2_dims)
        self.linear3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions

##############################
# Utility functions
##############################
def get_waiting_time(lanes):
    """Returns total waiting time of vehicles in these lanes."""
    return sum(traci.lane.getWaitingTime(lane) for lane in lanes)

def get_vehicle_numbers(lanes):
    """Returns a dict of lane -> # of vehicles that have LanePosition > 10."""
    return {
        lane: sum(
            1 for vid in traci.lane.getLastStepVehicleIDs(lane)
            if traci.vehicle.getLanePosition(vid) > 10
        )
        for lane in lanes
    }

##############################
# Preset (Fixed-Time) Simulation
##############################
# We define a simple 4-phase cycle, each 30 seconds
PHASE_DEFINITIONS = [
    ("yyyrrrrrrrrr", "GGGrrrrrrrrr", 30),  # Phase 0
    ("rrryyyrrrrrr", "rrrGGGrrrrrr", 30),  # Phase 1
    ("rrrrrryyyrrr", "rrrrrrGGGrrr", 30),  # Phase 2
    ("rrrrrrrrryyy", "rrrrrrrrrGGG", 30),  # Phase 3
]

def set_phase(junction, phase_index):
    """Set traffic light to specified phase with a short yellow transition."""
    yellow_state, green_state, duration = PHASE_DEFINITIONS[phase_index]
    # For simplicity, we apply the 'yellow_state' then the 'green_state'
    traci.trafficlight.setRedYellowGreenState(junction, yellow_state)
    # (In reality, you might simulate the actual 5s yellow, but for a direct approach, we do it instantly.)
    traci.trafficlight.setRedYellowGreenState(junction, green_state)
    # You can also set phase durations if needed, but here we do minimal logic.

def run_preset_simulation(config_path, steps=500):
    """
    Runs a fixed-time simulation for 'steps' steps,
    returns (final_cumulative_wait, list_of_cumulative_wait_each_step).
    """
    sumo_binary = checkBinary("sumo")
    cumulative_wait = 0
    per_step_cumulative = []

    try:
        traci.start([sumo_binary, "-c", config_path])
        junctions = traci.trafficlight.getIDList()

        # Initialize each junction's phase
        current_phase = {j: 0 for j in junctions}
        phase_timer = {j: PHASE_DEFINITIONS[0][2] for j in junctions}

        for _ in range(steps):
            traci.simulationStep()
            step_wait = 0

            for j in junctions:
                phase_timer[j] -= 1
                if phase_timer[j] <= 0:
                    # Go to next phase
                    current_phase[j] = (current_phase[j] + 1) % len(PHASE_DEFINITIONS)
                    set_phase(j, current_phase[j])
                    phase_timer[j] = PHASE_DEFINITIONS[current_phase[j]][2]

                # Sum waiting time on all lanes for this junction
                lanes = traci.trafficlight.getControlledLanes(j)
                step_wait += get_waiting_time(lanes)

            # Add step's waiting to cumulative
            cumulative_wait += step_wait
            per_step_cumulative.append(cumulative_wait)

        return cumulative_wait, per_step_cumulative

    finally:
        traci.close()

##############################
# RL Simulation (Test Mode)
##############################
def run_RL_simulation(config_path, model_bin, steps=500):
    """
    Loads a pre-trained RL model, runs the simulation for 'steps',
    returns (final_cumulative_wait, list_of_cumulative_wait_each_step).
    """
    sumo_binary = checkBinary("sumo")
    cumulative_wait = 0
    per_step_cumulative = []

    # Example 4-phase states
    phase_states = [
        ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
        ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
        ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
        ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
    ]

    # Build RL model
    input_dims = 4  # e.g., vehicles in up to 4 lanes
    n_actions = 4
    lr = 0.1
    fc1_dims = 256
    fc2_dims = 256

    rl_model = Model(lr, input_dims, fc1_dims, fc2_dims, n_actions)
    rl_model.load_state_dict(torch.load(model_bin, map_location=rl_model.device))
    rl_model.eval()

    def choose_action(state):
        """Given a state vector (size 4), pick the best action from RL model."""
        st_tensor = torch.tensor([state], dtype=torch.float).to(rl_model.device)
        with torch.no_grad():
            actions = rl_model(st_tensor)
        return torch.argmax(actions).item()

    try:
        traci.start([sumo_binary, "-c", config_path])
        junctions = traci.trafficlight.getIDList()

        # For each junction, store how many steps remain in the current phase
        phase_timer = {j: 0 for j in junctions}

        for _ in range(steps):
            traci.simulationStep()
            step_wait = 0

            for j in junctions:
                # Sum waiting time
                lanes = traci.trafficlight.getControlledLanes(j)
                step_wait += get_waiting_time(lanes)

                # If phase is done, pick a new action
                if phase_timer[j] <= 0:
                    # Build the state vector
                    veh_counts = list(get_vehicle_numbers(lanes).values())
                    # Ensure exactly 4 elements
                    if len(veh_counts) < 4:
                        veh_counts += [0]*(4 - len(veh_counts))
                    else:
                        veh_counts = veh_counts[:4]

                    action = choose_action(veh_counts)
                    # Example: set phase timer to 15 steps
                    phase_timer[j] = 15

                    # Set traffic light to the chosen state's patterns
                    traci.trafficlight.setRedYellowGreenState(j, phase_states[action][0])
                    # You might simulate a short yellow phase or do it instantly:
                    traci.trafficlight.setRedYellowGreenState(j, phase_states[action][1])
                else:
                    phase_timer[j] -= 1

            # Update cumulative waiting
            cumulative_wait += step_wait
            per_step_cumulative.append(cumulative_wait)

        return cumulative_wait, per_step_cumulative

    finally:
        traci.close()

##############################
# Main: Compare Preset vs. RL (single-run)
##############################
def main():
    parser = optparse.OptionParser()
    parser.add_option("-c", "--config", dest="config", type="string",
                      default="configuration.sumocfg", help="SUMO config file path")
    parser.add_option("-m", "--model", dest="model_bin", type="string",
                      default="models/model.bin", help="Path to pre-trained model bin")
    parser.add_option("-s", "--steps", dest="steps", type="int",
                      default=500, help="Number of simulation steps")
    options, _ = parser.parse_args()

    config_path = os.path.abspath(options.config)
    if not os.path.exists(config_path):
        sys.exit(f"Config file not found: {config_path}")

    # 1) Run Preset Simulation
    preset_final, preset_curve = run_preset_simulation(config_path, steps=options.steps)
    
    # 2) Run RL Simulation (Test Mode)
    rl_final, rl_curve = run_RL_simulation(config_path, options.model_bin, steps=options.steps)

    # Print final total waiting times
    print(f"Preset final total waiting time: {preset_final}")
    print(f"RL final total waiting time: {rl_final}")

    # Plot on the same graph
    plt.figure(figsize=(10, 6))
    x_vals = range(options.steps)
    plt.plot(x_vals, preset_curve, label="Preset (Cumulative)", color="blue")
    plt.plot(x_vals, rl_curve, label="RL (Cumulative)", color="orange")
    plt.title("Preset vs. RL: Cumulative Waiting Time over Steps")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Cumulative Waiting Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_cumulative_waiting_time.png")
    plt.show()

if __name__ == "__main__":
    main()
