# RlTraffic

RlTraffic is a reinforcement learning-based traffic management system designed to optimize traffic flow using simulation-based training.

**Demo Videos**:  
- [![YouTube Video](https://img.youtube.com/vi/Uz8dTEscGJg/0.jpg)](https://youtu.be/Uz8dTEscGJg)
- [![AI Demo](https://github.com/user-attachments/assets/a611e9da-dffc-407b-8c8d-312d79e6afbe)](https://github.com/user-attachments/assets/a611e9da-dffc-407b-8c8d-312d79e6afbe)


---

## Prerequisites

- **Python 3.7+**  
- **SUMO (Simulation of Urban MObility)** – Install from [SUMO Official Site](https://www.eclipse.org/sumo/).

---

## Installation

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/vinayakyadav2709/RlTraffic.git  
   cd RlTraffic  
   ```

2. **Create and Activate a Virtual Environment**:  
   ```bash
   python3 -m venv env  
   source env/bin/activate    # Linux/MacOS  
   # For Windows: .\env\Scripts\activate  
   ```

3. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt  
   ```


## Setup

### 1. Add SUMO Map Files
- Create the `configs/maps/` directory if it doesn’t exist:
  ```bash
  mkdir -p configs/maps
  ```
- Place your SUMO configuration files in `configs/maps/`:
  - Network file: `jamnagar.net.xml`
  - Route files: `routesdeh.rou.xml`, `routesdeh.rou.alt.xml`

### 2. Add Pre-trained Models (Optional)
- Create the `trainedModels/` directory for local storage:
  ```bash
  mkdir -p trainedModels
  ```
- Place pre-trained model checkpoints (e.g., `dqn_jamnagar.pth`) in `trainedModels/`.

---

## Running the Project

### 1. Train the RL Model
Train the reinforcement learning agent using:
```bash
python model/train.py
```
- This script trains the RL agent using the SUMO environment defined in `configs/maps/`.
- Trained models will be saved to `trainedModels/`.

### 2. Test RL vs Preset Timing
Compare the RL agent against fixed preset timings:
```bash
python test/test.py
```
- **What it does**:
  - Runs simulations for both RL-controlled and preset-controlled traffic lights.
  - Outputs metrics like average waiting time and throughput for comparison.

### 3. Test Preset Timing Only
Run simulations with only fixed preset timings:
```bash
python test/preset.py
```
- This tests baseline performance without RL intervention.

---

## File Structure
```
RlTraffic/
├── configs/
│   └── maps/                   # SUMO configuration files (.net.xml, .rou.xml)
├── trainedModels/              # Pre-trained RL models (gitignored)
├── model/
│   └── train.py               # Training script for RL agent
├── test/
│   ├── test.py                # Compare RL vs preset performance
│   └── preset.py              # Test preset timing only
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignores trainedModels/
└── README.md
```

---

## Credits
- Inspired by [Dynamic-Traffic-light-management-system](https://github.com/Maunish-dave/Dynamic-Traffic-light-management-system).

---

## Contact
For questions or support, contact:
- **Email**: [vinayakyadav2709@gmail.com](mailto:vinayakyadav2709@gmail.com)
```
