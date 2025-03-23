# RlTraffic

RlTraffic is a reinforcement learning-based traffic management system designed to optimize traffic flow using simulation-based training. Our innovative solution reduces costs by **1000x** using cutting-edge AI and IoT technology. The project leverages SUMO (Simulation of Urban MObility) for realistic traffic simulation and RL algorithms to dynamically control traffic lights and improve traffic efficiency.

**Videos**:  
- **Presentation**: (https://youtu.be/Uz8dTEscGJg)  
- **Demo**: (https://github.com/user-attachments/assets/a611e9da-dffc-407b-8c8d-312d79e6afbe)

---

## Diagrams

**Data Flow Diagram**:  
![Data Flow Diagram](https://github.com/user-attachments/assets/8ca16317-c707-4cc9-a137-136adba59fc4)  
*Illustrates how data is processed and flows through various components of the RL traffic system.*

**Roadmap Diagram**:  
![RoadMap](https://github.com/user-attachments/assets/57a0979d-b143-4de6-891e-e4a3fe6068dc)  
*Outlines project milestones, key features, and future development plans.*

**RL vs Preset Comparison Diagram**:  
![RL vs Preset Comparison](https://github.com/user-attachments/assets/7a8dddd6-601c-4fb9-a56b-27eb8cdee9a6)  
*Shows performance differences between the RL approach and preset timings.*

**Dashboard Diagram**:  
![Dashboard](https://github.com/user-attachments/assets/9dad91a0-9db0-4852-9345-2a52a1d8b0d4)  
*Displays real-time metrics and insights for monitoring system performance.*

[Download Full Report (PDF)](https://github.com/user-attachments/files/19407084/report.pdf)

*The report provides a concise analysis of the system architecture, experimental results, and highlights the 1000x cost reduction achieved through innovative AI and IoT integration.*

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

---

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
│   └── train.py                # Training script for RL agent
├── test/
│   ├── test.py                 # Compare RL vs preset performance
│   └── preset.py               # Test preset timing only
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignores trainedModels/
└── README.md
```

---

## Credits

- [Dynamic-Traffic-light-management-system](https://github.com/Maunish-dave/Dynamic-Traffic-light-management-system).

---

## Contact

For questions or support, contact:  
- **Email**: [vinayakyadav2709@gmail.com](mailto:vinayakyadav2709@gmail.com)
```
