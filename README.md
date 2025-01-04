# Prisoner Box Strategy Simulation

This program simulates the "100 prisoners and 100 boxes" problem and evaluates various strategies to maximize the prisoners' chances of survival. The implemented strategies include:
- **Loop-following Strategy**
- **Random Box Opening Strategy**
- **Dynamic Machine Learning Strategy** (powered by a neural network that learns and adapts over time)

## Features
- **Simulation Size**: Configurable for up to 10,000 simulations.
- **Machine Learning**: A neural network dynamically updates its model to improve performance over time.
- **Visualization**: Success rates of all strategies are displayed using a bar chart for comparison.
- **Model Persistence**: The trained model is periodically saved to avoid starting from scratch each time.

## Requirements
- Python 3.8+
- Required Python libraries (see `requirements.txt`)

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/ZamoRzgar/Box-Riddle.git
   cd prisoner-box-strategy
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the program:

bash
Copy code
python prisoner_box_strategy.py
Observe the simulation results and strategy comparison in the terminal and the generated visualization.

How It Works
Loop-following Strategy: Each prisoner follows the number found in the opened box until they find their own number or exhaust their tries.
Random Strategy: Each prisoner opens random boxes without a specific plan.
Dynamic Machine Learning Strategy: Uses a neural network that predicts which box a prisoner should open next, learning from past simulation results.
Model Persistence
The machine learning model is saved periodically as prisoner_strategy_model.pkl. This allows the program to continue improving from previous runs.

Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the code, add new strategies, or enhance the visualization.

