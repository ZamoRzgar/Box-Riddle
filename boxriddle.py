import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
import os

# Simulation parameters
NUM_PRISONERS = 100
TRIES = NUM_PRISONERS // 2

# Save and load model for training memory
MODEL_FILE = 'prisoner_strategy_model.pkl'

# Strategy 1: Loop-following strategy
def loop_following_strategy(prisoner_num, boxes):
    current_box = prisoner_num
    for _ in range(TRIES):
        if boxes[current_box - 1] == prisoner_num:
            return True
        current_box = boxes[current_box - 1]
    return False

# Load or initialize the neural network model
def load_or_initialize_model():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as file:
                model = pickle.load(file)
            return model
        except (FileNotFoundError, EOFError):
            pass
    # Initialize a new model if loading fails
    model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, batch_size=500, random_state=42, verbose=True)
    return model

# Generate training data for the model
def generate_training_data():
    X = []
    y = []
    for _ in range(5000):
        boxes = list(range(1, NUM_PRISONERS + 1))
        random.shuffle(boxes)
        for prisoner_num in range(1, NUM_PRISONERS + 1):
            path = []
            current_box = prisoner_num
            for _ in range(TRIES):
                path.append(boxes[current_box - 1])
                if boxes[current_box - 1] == prisoner_num:
                    break
                current_box = boxes[current_box - 1]
            X.append(boxes)
            y.append(path[-1])
    return np.array(X), np.array(y)

# Train the neural network model
def train_model(model):
    X, y = generate_training_data()
    print("Training model...")
    for epoch in range(5):  # Train for 5 epochs
        print(f"Epoch {epoch + 1}/5")
        model.partial_fit(X, y, classes=np.arange(1, NUM_PRISONERS + 1))
    print("Model training completed.")
    save_model(model)

# Save the model periodically
def save_model(model):
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved.")

# Update model dynamically after each run
def update_model(model, X_new, y_new):
    print("Updating model...")
    model.partial_fit(X_new, y_new, classes=np.arange(1, NUM_PRISONERS + 1))
    save_model(model)  # Save periodically after updates

# Dynamic ML Strategy
def dynamic_ml_strategy(prisoner_num, boxes, model):
    current_box = prisoner_num
    for _ in range(TRIES):
        features = np.array(boxes).reshape(1, -1)
        box_to_open = model.predict(features)[0] - 1
        if boxes[box_to_open] == prisoner_num:
            return True
        current_box = boxes[box_to_open]
    return False

# Strategy 2: Random box opening strategy
def random_strategy(prisoner_num, boxes):
    opened_boxes = set()
    for _ in range(TRIES):
        box_to_open = random.randint(0, NUM_PRISONERS - 1)
        while box_to_open in opened_boxes:
            box_to_open = random.randint(0, NUM_PRISONERS - 1)
        opened_boxes.add(box_to_open)
        if boxes[box_to_open] == prisoner_num:
            return True
    return False

# Simulate the entire game with a given strategy
def simulate_game(strategy, model=None):
    boxes = list(range(1, NUM_PRISONERS + 1))
    random.shuffle(boxes)

    for prisoner in range(1, NUM_PRISONERS + 1):
        if model:
            if not strategy(prisoner, boxes, model):
                return False
        else:
            if not strategy(prisoner, boxes):
                return False
    return True

# Load or train the model
model = load_or_initialize_model()
train_model(model)

# Run the simulation multiple times
SIMULATIONS = 10000
success_counts = {"Loop-following": 0, "Random": 0, "Dynamic-ML": 0}

for i in range(SIMULATIONS):
    print(f"Running simulation {i + 1}/{SIMULATIONS}...")
    if simulate_game(loop_following_strategy):
        success_counts["Loop-following"] += 1
    if simulate_game(random_strategy):
        success_counts["Random"] += 1
    if simulate_game(dynamic_ml_strategy, model):
        success_counts["Dynamic-ML"] += 1

# Print success rates for each strategy
for strategy, count in success_counts.items():
    print(f"{strategy} strategy success rate: {(count / SIMULATIONS) * 100:.2f}%")

# Visualization
strategies = list(success_counts.keys())
success_rates = [(count / SIMULATIONS) * 100 for count in success_counts.values()]

plt.figure(figsize=(10, 6))
plt.bar(strategies, success_rates, color=['blue', 'green', 'orange'])
plt.xlabel('Strategy')
plt.ylabel('Success Rate (%)')
plt.title('Comparison of Prisoner Box Strategies')
plt.ylim(0, 100)
plt.grid(axis='y')

for i, rate in enumerate(success_rates):
    plt.text(i, rate + 1, f'{rate:.2f}%', ha='center')

plt.show()
