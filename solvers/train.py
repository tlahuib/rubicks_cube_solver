import numpy as np
import models as m
from torch import tensor


# Initialize models
value_model = m.Transformer(162, 1, 6, 6, 0.2)
value_model = value_model.to(m.device)

# Pre-train value model
n_rows = 100000
epoch = 0
while True:
    print(f"---- Starting Epoch {epoch} ----")

    # Read training data
    value_data = np.loadtxt('constructor/solves.csv', delimiter=',', dtype=int, max_rows=n_rows, skiprows=n_rows * epoch)
    value_data = tensor(value_data).float()

    # Fit model
    value_model.fit(value_data, 20000, 10, 1000)
    
    # Calculate loss
    loss = m.estimate_loss(value_model, value_data, 100, 10)

    print(f"---- Epoch {epoch} finished with loss: {loss:.2f} ----\n\n")
    if len(value_data) < n_rows:
        break
    epoch += 1



