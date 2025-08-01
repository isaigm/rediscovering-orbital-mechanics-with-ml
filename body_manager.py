import numpy as np

class BodyManager:
    def __init__(self):
        self.bodies = []
    
    def add_body(self, body):
        self.bodies.append(body)
    
    def get_bodies(self):
        return self.bodies
    
    def clear_bodies(self):
        self.bodies = []
    
    def get_masses_as_array(self):
        return np.array([body.mass for body in self.bodies])
    
    def get_positions_as_array(self):
        return np.array([body.pos for body in self.bodies])
    
    def get_velocities_as_array(self):
        return np.array([body.vel for body in self.bodies])
    

    def __repr__(self):
        return f"BodyManager(bodies={self.bodies})"