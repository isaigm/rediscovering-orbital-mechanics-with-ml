import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from model import PhysicsInteractionNetwork
import time
from itertools import permutations
from body import Body
from body_manager import BodyManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# --- 1. Definition of Constants and Initial Conditions ---

# The simulation uses an astronomical unit system for manageable numerical values:
#   - Mass: Solar Masses (M☉), where the Sun's mass = 1.0
#   - Distance: Astronomical Units (AU), where the Earth-Sun distance = 1.0
#   - Time: Earth Years, where Earth's orbital period = 1.0

# Gravitational constant in units of (AU^3 / (M☉ * year^2))
G = 4 * np.pi**2

# The initial state of the system is defined in a dictionary for clarity.
# Masses are in Solar Masses (M☉). Body 0 is the Sun.
# Initial positions are in Astronomical Units (AU), aligned on the x-axis for simplicity.
# Initial velocities are in AU/year, calculated to achieve stable (mostly circular) orbits.
# For a circular orbit, the gravitational force (F_g) must equal the centripetal force (F_c):
#   F_g = G * M * m / r^2
#   F_c = m * v^2 / r
# Equating these and solving for v gives: v = sqrt(G * M / r).
# In our unit system (G=4*pi^2, M_sun=1), this simplifies to: v = 2*pi / sqrt(r).
initial_state = {
    "initial_pos": [
        [0.0, 0.0],  # Sun at the origin
        [1.0, 0.0],  # Earth at 1 AU
        [5.2, 0.0],  # Jupiter at 5.2 AU
        [1.5, 0.0]   # Mars at 1.5 AU
    ],
    "initial_vel": [
        [0.0, 0.0],                                     # Sun is static
        [0.0, 2 * np.pi / np.sqrt(1.0)],                # Circular velocity for Earth
        [0.0, 2 * np.pi / np.sqrt(5.2)],                # Circular velocity for Jupiter
        [0.0, 2 * np.pi / np.sqrt(1.5) * 0.9]           # 90% of circular velocity for Mars to create an elliptical orbit
    ],
    "masses": [
        1.0,      # Sun
        3e-6,     # Earth (realistic)
        9.5e-4,   # Jupiter (realistic)
        3.3e-7,   # Mars (realistic)
    ]
}
body_manager = BodyManager()
body_manager.add_body(Body(mass=initial_state["masses"][0], pos=np.array(initial_state["initial_pos"][0]), vel=np.array(initial_state["initial_vel"][0]), name="Sun"))
body_manager.add_body(Body(mass=initial_state["masses"][1], pos=np.array(initial_state["initial_pos"][1]), vel=np.array(initial_state["initial_vel"][1]), name="Earth"))
body_manager.add_body(Body(mass=initial_state["masses"][2], pos=np.array(initial_state["initial_pos"][2]), vel=np.array(initial_state["initial_vel"][2]), name="Jupiter"))
body_manager.add_body(Body(mass=initial_state["masses"][3], pos=np.array(initial_state["initial_pos"][3]), vel=np.array(initial_state["initial_vel"][3]), name="Mars"))

dt = 0.001          # Time step in years. A small value for high accuracy.
total_time = 5.0    # Total simulation duration in years.
n_steps = int(total_time / dt)

list_bodies = body_manager.get_bodies()
start_sim_time = time.time()
history_pos    = []
history_accel  = []


for step in range(n_steps):
    current_positions = np.array([body.pos.copy() for body in list_bodies])
    history_pos.append(current_positions)
    accelerations_t = np.array([body.compute_acceleration(list_bodies, G) for body in list_bodies])
    history_accel.append(accelerations_t)
    for i, body in enumerate(list_bodies):
        body.vel += accelerations_t[i] * dt
        body.pos += body.vel * dt

history_pos = np.array(history_pos)
history_accel = np.array(history_accel)

end_sim_time = time.time()
print(f"Simulation completed in {end_sim_time - start_sim_time:.2f} seconds.\n")

n_bodies = len(list_bodies)


print("Preparing data for training the GNN model...")
print(f"Using device: {device}\n")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

permutaciones = list(permutations(range((n_bodies)), 2))
edge_index = torch.tensor(permutaciones, dtype=torch.long).t().contiguous().to(device)
pos_hist_t = torch.tensor(history_pos, dtype=torch.float32)
accel_hist_t = torch.tensor(history_accel, dtype=torch.float32)
sender_nodes = edge_index[0].cpu()
receiver_nodes = edge_index[1].cpu()
displacements_hist = pos_hist_t[:, receiver_nodes] - pos_hist_t[:, sender_nodes]
split_idx = int(n_steps * 0.8)
train_indices = torch.arange(split_idx)
val_indices = torch.arange(split_idx, n_steps)


sun_mass = initial_state["masses"][0]

gnn_model = PhysicsInteractionNetwork().to(device)
fixed_masses = torch.tensor([[sun_mass]], dtype=torch.float32).to(device)
initial_masses = np.random.uniform(1e-5, 1e-2, size=(n_bodies - 1, 1))
trainable_log_masses = torch.nn.Parameter(torch.tensor(np.log(initial_masses), dtype=torch.float32, device=device))

print("Planets:", list_bodies[1:])
print("Initial masses:", np.exp(trainable_log_masses.cpu().detach().numpy()).flatten())


# Math: \text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \frac{\|a_i^{pred} - a_i^{target}\|^2}{\|a_i^{target}\|^2 + \epsilon}

def relative_loss_fn(predicted_acc, target_acc):
    epsilon = 1e-8
    numerator = torch.sum((predicted_acc - target_acc)**2, dim=-1)
    denominator = torch.sum(target_acc**2, dim=-1) + epsilon
    return torch.mean(numerator / denominator)


lr_mlp = 1e-4
lr_mass = 1e-2 
weight_decay = 1e-5
param_groups = [{'params': gnn_model.parameters(), 'lr': lr_mlp, 'weight_decay': weight_decay}, {'params': [trainable_log_masses], 'lr': lr_mass}]
optimizer = optim.Adam(param_groups)
epochs = 200
loss_fn = relative_loss_fn
train_loss_history = []
val_loss_history = []
mass_history = []
patience = 15
patience_counter = 0
best_val_loss = float('inf')
model_save_path = "best_model.pth"
masses_save_path = "best_masses.pth"


print("Model and data prepared for training.\n")

start_time = time.time()
for epoch in range(epochs):
    gnn_model.train()
    epoch_train_loss = 0.0
    for i in torch.randperm(len(train_indices)):
        idx = train_indices[i]
        edge_features_t = displacements_hist[idx].to(device)
        target_accelerations_t = accel_hist_t[idx].to(device)
        optimizer.zero_grad()
        trainable_masses = torch.exp(trainable_log_masses)
        full_masses = torch.cat([fixed_masses, trainable_masses], dim=0)
        predicted_forces = gnn_model(full_masses, edge_index, edge_features_t)
        epsilon = 1e-8
        predicted_accelerations = predicted_forces / (full_masses + epsilon)
        loss = loss_fn(predicted_accelerations, target_accelerations_t)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    avg_train_loss = epoch_train_loss / len(train_indices)
    train_loss_history.append(avg_train_loss)
    mass_history.append(torch.exp(trainable_log_masses).cpu().detach().numpy().flatten())

    gnn_model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for i in val_indices:
            edge_features_t = displacements_hist[i].to(device)
            target_accelerations_t = accel_hist_t[i].to(device)
            trainable_masses = torch.exp(trainable_log_masses)
            full_masses = torch.cat([fixed_masses, trainable_masses], dim=0)
            predicted_forces = gnn_model(full_masses, edge_index, edge_features_t)
            epsilon = 1e-8
            predicted_accelerations = predicted_forces / (full_masses + epsilon)
            val_loss = loss_fn(predicted_accelerations, target_accelerations_t)
            epoch_val_loss += val_loss.item()
    avg_val_loss = epoch_val_loss / len(val_indices)
    val_loss_history.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
    print("Learned masses:", mass_history[-1])

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"New best val loss. Saving model to {model_save_path}")
        torch.save(gnn_model.state_dict(), model_save_path)
        torch.save(trainable_log_masses, masses_save_path)
    else:
        patience_counter += 1
        print(f"There was no improvement. Patience counter: {patience_counter}/{patience}")
    if patience_counter >= patience:
        print("\n¡Early stopping activated!")
        break
end_time = time.time()
print(f"\nTraining total time: {end_time - start_time:.2f} seconds")


best_model = PhysicsInteractionNetwork().to(device)
best_model.load_state_dict(torch.load(model_save_path))
best_model.eval() 
best_masses_param = torch.load(masses_save_path)
best_learned_masses_log = best_masses_param.to(device)
print("Real planet masses: ", list_bodies[1:])
print("Best learned masses:", torch.exp(best_learned_masses_log).cpu().detach().numpy().flatten())


print("Simulating with the trained model...")
ai_history_pos = []
current_pos = torch.tensor(initial_state['initial_pos'], dtype=torch.float32).to(device)
current_vel = torch.tensor(initial_state['initial_vel'], dtype=torch.float32).to(device)
full_learned_masses = torch.cat([fixed_masses, torch.exp(best_learned_masses_log)], dim=0)

with torch.no_grad():
    for step in range(n_steps):
        ai_history_pos.append(current_pos.cpu().numpy().copy())
        sender = edge_index[0]
        receiver = edge_index[1]
        displacements_ia = current_pos[receiver] - current_pos[sender]
        predicted_forces = best_model(full_learned_masses, edge_index, displacements_ia)
        epsilon = 1e-8
        predicted_accelerations = predicted_forces / (full_learned_masses + epsilon)
        current_vel += predicted_accelerations * dt
        current_pos += current_vel * dt

ai_history_pos = np.array(ai_history_pos)


print("Generating plots")

plt.figure(figsize=(10, 10))
planet_names = ['"Tierra"', '"Júpiter"', '"Marte"']
colors = ['#1f77b4', '#2ca02c', '#d62728']
# Graficar la estrella
plt.plot(history_pos[:, 0, 0], history_pos[:, 0, 1], 'o', color='orange', markersize=15, label='Estrella')
for i in range(n_bodies - 1):
    # Órbita real (línea continua)
    plt.plot(history_pos[:, i+1, 0], history_pos[:, i+1, 1], color=colors[i], linestyle='-', label=f'{planet_names[i]} (Real)')
    plt.plot(ai_history_pos[:, i+1, 0], ai_history_pos[:, i+1, 1], color=colors[i], linestyle='--', label=f'{planet_names[i]} (IA)')
plt.title('Comparación de Órbitas: Simulación Real vs. Predicción de IA', fontsize=16)
plt.xlabel('Posición X (AU)', fontsize=12)
plt.ylabel('Posición Y (AU)', fontsize=12)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(True)
plt.savefig("imagen1_comparacion_orbitas.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
mass_history_array = np.array(mass_history)
for i in range(n_bodies - 1):
    plt.plot(mass_history_array[:, i], label=f'Masa Aprendida ({planet_names[i]})', color=colors[i])
    plt.axhline(y=initial_state["masses"][i+1], color=colors[i], linestyle='--', label=f'Masa Real ({planet_names[i]})')
plt.title('Convergencia de las Masas Planetarias', fontsize=16)
plt.xlabel('Época de Entrenamiento', fontsize=12)
plt.ylabel('Masa (escala logarítmica)', fontsize=12)
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig("imagen2_convergencia.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Pérdida de Entrenamiento')
plt.plot(val_loss_history, label='Pérdida de Validación')
best_epoch = np.argmin(val_loss_history)
plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Mejor Época ({best_epoch+1})')
plt.title('Curvas de Pérdida de Entrenamiento y Validación', fontsize=16)
plt.xlabel('Época de Entrenamiento', fontsize=12)
plt.ylabel('Pérdida Relativa (escala logarítmica)', fontsize=12)
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig("imagen3_perdida.png", dpi=300, bbox_inches='tight')
