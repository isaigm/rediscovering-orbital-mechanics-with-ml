import numpy as np
import torch
from pysr import PySRRegressor
from model import PhysicsInteractionNetwork

device = torch.device('cpu') 
model_save_path = "best_model.pth"
best_model = PhysicsInteractionNetwork().to(device)
best_model.load_state_dict(torch.load(model_save_path, map_location=device))
best_model.eval()
masses_save_path = "best_masses.pth"
best_masses_param = torch.load(masses_save_path, map_location=device)
best_learned_masses = torch.exp(best_masses_param)
fixed_mass = torch.tensor([[1.0]], dtype=torch.float32)
full_learned_masses = torch.cat([fixed_mass, best_learned_masses], dim=0)

n_points_pysr = 2000
X_pysr = [] 
y_pysr = [] 
with torch.no_grad():
    for _ in range(n_points_pysr):
        idx1, idx2 = np.random.choice(range(len(full_learned_masses)), 2, replace=False)
        m1 = full_learned_masses[idx1]
        m2 = full_learned_masses[idx2]
        distance = np.random.uniform(0.5, 10.0)
        displacement = torch.tensor([[distance, 0.0]], dtype=torch.float32)
        edge_index_dummy = torch.tensor([[idx1], [idx2]], dtype=torch.long)
        net_forces = best_model(full_learned_masses, edge_index_dummy, displacement)
        total_force = net_forces[idx2]
        magnitud_fuerza = torch.linalg.norm(total_force).item()
        X_pysr.append([m1.item(), m2.item(), distance])
        y_pysr.append(magnitud_fuerza)

X_pysr = np.array(X_pysr)
y_pysr = np.array(y_pysr)
model_pysr_full = PySRRegressor(
    niterations=100,
    binary_operators=["+", "*", "/", "-", "pow"],
    procs=0,
    model_selection="best",
)
model_pysr_full.fit(X_pysr, y_pysr, variable_names=["m1", "m2", "r"])
print(model_pysr_full)
