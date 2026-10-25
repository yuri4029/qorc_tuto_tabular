import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP & HYPERPARAMETERS
np.random.seed(42)          # Makes the random numbers reproducible
n_inputs = 1                # We are feeding in 1 value at a time (sine wave height)
n_reservoir = 100           # The "Bucket of Water": 100 randomly sloshing neurons
n_outputs = 1               # We want 1 predicted value out

# 2. CREATE THE UNTRAINED RESERVOIR (The Chaos Box)
# Randomly connect inputs to the reservoir
W_in = np.random.uniform(-0.5, 0.5, (n_reservoir, n_inputs))
# Randomly connect reservoir neurons to each other (the internal water sloshing)
W_res = np.random.uniform(-0.5, 0.5, (n_reservoir, n_reservoir))

# Scale the reservoir weights so the ripples eventually fade out (Echo State Property)
spectral_radius = np.max(np.abs(np.linalg.eigvals(W_res)))
W_res = W_res * (0.95 / spectral_radius)

# 3. GENERATE TOY DATA (A simple sine wave)
time_steps = 600
data = np.sin(np.linspace(0, 30, time_steps))

# Split into Training data (first 400 steps) and Testing data (last 200 steps)
train_in = data[:400]
train_target = data[1:401]  # The target is just the wave shifted 1 step into the future

test_in = data[400:599]
test_target = data[401:600]

# 4. RUN THE RESERVOIR (Gather the ripple patterns)
# Initialize the reservoir state (water is completely still)
x = np.zeros((n_reservoir, 1))
# Create a collector matrix to record the history of reservoir states
states_history = []

for t in range(len(train_in)):
    u = train_in[t]  # Current input step
    # Water sloshes: blend the input, the current reservoir state, and add a non-linear squish (tanh)
    x = np.tanh(np.dot(W_in, u) + np.dot(W_res, x))
    states_history.append(x.flatten())

states_history = np.array(states_history)

# 5. TRAIN THE READOUT LAYER (The Camera)
# We use standard Ridge Regression (a simple linear equation solver) 
# to map the states_history directly to the train_target.
# Notice: W_in and W_res are NEVER modified. We only compute W_out!
reg = 1e-8  # Regularization constant to prevent overfitting
W_out = np.dot(np.dot(train_target, states_history), 
               np.linalg.inv(np.dot(states_history.T, states_history) + reg * np.eye(n_reservoir)))

# 6. TEST THE TRAINED AI ON UNSEEN DATA
predictions = []
for t in range(len(test_in)):
    u = test_in[t]
    x = np.tanh(np.dot(W_in, u) + np.dot(W_res, x)) # Process through the reservoir
    y_pred = np.dot(W_out, x)                       # The trained readout makes its guess
    predictions.append(y_pred.flatten()[0])

# 7. VISUALIZE RESULTS
plt.figure(figsize=(10, 4))
plt.plot(test_target, label="Actual Future Wave", color="black", linewidth=2)
plt.plot(predictions, label="Reservoir Prediction", color="cyan", linestyle="--", linewidth=2)
plt.title("Reservoir Computing: Time-Series Prediction")
plt.legend()
plt.show()
