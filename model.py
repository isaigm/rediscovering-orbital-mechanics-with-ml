import torch
from torch.nn import Sequential, Linear, Tanh
from transforms import  cartesian_to_spherical_log, spherical_log_to_cartesian
from torch_scatter import scatter_add

class PhysicsInteractionNetwork(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        # Math: \hat{\mathbf{e}}_k = f_{\text{GN}}(v_{r_k}, v_{s_k}, \mathbf{e}_k; \theta)
        self.edge_mlp = Sequential(Linear(2, 128), Tanh(), Linear(128, 128), Tanh(), Linear(128, 2))
        
    
    def forward(self, node_attr, edge_index, edge_attr):
        spherical_edge_attr = cartesian_to_spherical_log(edge_attr)
        spherical_base_force = self.edge_mlp(spherical_edge_attr)
        base_force_per_edge = spherical_log_to_cartesian(spherical_base_force)
        sender_mass = node_attr[edge_index[0]]
        receiver_mass = node_attr[edge_index[1]]
        final_force_per_edge = base_force_per_edge * sender_mass * receiver_mass
        net_force_per_node = scatter_add(final_force_per_edge, edge_index[1], dim=0, dim_size=node_attr.size(0))
        return net_force_per_node