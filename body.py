import numpy as np
from typing import List

class Body:
    def __init__(self, mass: float, pos: np.array, vel: np.array, name: str = None):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.name = name

    def __repr__(self):
        return f"Body(name={self.name}, mass={self.mass:.2e}, pos={self.pos}, vel={self.vel})"
    
    # Computes the total incoming force on this body from all other bodies
    def compute_total_incoming_force(self, bodies: List['Body'], G: float) -> np.array:
        force = np.zeros_like(self.pos)
        for other in bodies:
            if other is not self:
                r_vec = other.pos - self.pos
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 0:
                    f_mag = (G * self.mass * other.mass) / (r_mag**2)
                    force += f_mag * (r_vec / r_mag)
        return force
    
    def compute_acceleration(self, bodies: List['Body'], G: float) -> np.array:
        force = self.compute_total_incoming_force(bodies, G)
        return force / self.mass

    def update_state(self, acceleration: np.array, dt: float):
        self.vel += acceleration * dt
        self.pos += self.vel * dt