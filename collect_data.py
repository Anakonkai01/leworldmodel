import os 
# use for headless server, ssh
# import os 
# os.environ["SDL_VIDEODRIVER"] = "dummy"
from typing import Any

import pygame
import numpy as np 
import glob

rng = np.random.default_rng(seed=42)


from sim.world import (
    WIDTH, HEIGHT, make_world, draw_world, get_observation, reset_world, collect_episode
)

class ExplorePolicy: 
    """random explore"""
    def __init__(self, rng, v_max=160.0, omega_max=2.5):
        self.rng = rng
        self.v_max = v_max 
        self.omega_max = omega_max
        self.hold = 0 # hold last actions
        self.action = (0.0, 0.0) # actions was hold
    
    def __call__(self):  # when call ExplorePolicy object, this function will run 
        if self.hold <= 0:
            v = float(self.rng.uniform(0.3 * self.v_max, self.v_max)) # always forward (backward later will use)
            omega = float(self.rng.uniform(-self.omega_max, self.omega_max))
            self.action = (v, omega)
            self.hold = int(self.rng.integers(10, 40)) # hold last actions in # step 
        self.hold -= 1 
        return self.action

    

def main(): 
    pygame.init()
    # not set mode -> not render 
    surface = pygame.Surface((WIDTH, HEIGHT))
    os.makedirs("data", exist_ok=True)

    # init state 
    rng = np.random.default_rng(0)
    n_episode, n_steps = 200, 150 
    
    for ep in range(n_episode): 
        world = reset_world(rng)
        policy = ExplorePolicy(rng)
        obs, action = collect_episode(surface, world, policy, n_steps)
        np.savez_compressed(f"data/ep_{ep:04d}.npz", obs=obs, actions=action)
        if ep % 20 == 0: 
            print(f"ep {ep}, obs {obs.shape}, act {action.shape}")

    print("done")
    
    

    pygame.quit()

def inspect():
    files = sorted(glob.glob("data/ep_*.npz"))
    print("số episode:", len(files))
    d = np.load(files[0])
    print("obs:", d["obs"].shape, d["obs"].dtype, "| actions:", d["actions"].shape, d["actions"].dtype)
    # soi 1 khung bằng mắt
    frame = d["obs"][0]
    pygame.image.save(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), "loaded_frame.png")
    print("đã lưu loaded_frame.png")




if __name__ == "__main__":
    main()
    inspect()
    