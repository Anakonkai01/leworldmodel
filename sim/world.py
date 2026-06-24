import pygame
import math 
import numpy as np             

# configuration 
WIDTH = 600
HEIGHT = 600
FPS = 60
DT = 1.0 / FPS  # timestep
V_FORWARD = 160.0 # velocity (pixel / second)
OMEGA = 2.5  # gyro (radian/second)
OBSERVATION_SIZE = 96

# config color 
BACKGROUND = (20, 20, 30)
CAR = (80, 170, 255)
HEADING = (255, 255, 255)
GOAL = (90, 230, 120)
OBST = (200, 80, 80)



def collect_episode(surface, world, policy, n_steps):
    obs_list, act_list = [], []
    for t in range(n_steps): 
        draw_world(surface, world, draw_goal=False) # we don't need goal in explore 
        obs_list.append(get_observation(surface)) # o_t 
        action = policy()
        act_list.append(action)
        step(world, action, DT)
    obs = np.stack(obs_list).astype(np.uint8) # because obs_list are an array so we want to stack it to have T 
    actions = np.array(act_list, dtype=np.float32) # action is a list of tuple (v, omega) -> 
    return obs, actions


def random_obstacles(rng, n=3): 
    obs = []
    for _ in range(n): 
        w = int(rng.integers(60, 120))
        h = int(rng.integers(60, 120))
        x = int(rng.integers(0, WIDTH - w))
        y = int(rng.integers(0, HEIGHT - h))
        obs.append({"x": x, "y": y, "w": w, "h": h})
        
    return obs

def free_spot(rng, obstacles, radius): 
    # rejection sampling: return x, y without overlap 
    for _ in range(100): # try 100 times 
        x = float(rng.uniform(radius, WIDTH - radius))
        y = float(rng.uniform(radius, HEIGHT - radius))
        if not any(circle_rect_hit(x, y, radius, ob) for ob in obstacles): 
            return x, y
    return x, y # rare, return random 
        

def reset_world(rng, min_goal_dist=200.0): 
    # obstacles = random_obstacles(rng)
    obstacles = []
    car_r, goal_r = 28, 16 
    
    cx, cy = free_spot(rng, obstacles, car_r)
    
    # make sure goal is far from the car 
    for _ in range(100): 
        gx, gy = free_spot(rng, obstacles, goal_r)
        if math.hypot(gx - cx, gy - cy) >= min_goal_dist:
            break
        
    

    return {
        "obstacles": obstacles,
        "car_radius": car_r,
        "goal_radius": goal_r,
        "car":  {"x": cx, "y": cy, "theta": float(rng.uniform(0, 2 * math.pi))},
        "goal": {"x": gx, "y": gy},
    }


def car_triangle(cx, cy, theta, size):
    nose  = (cx + math.cos(theta)         * size * 1.6,
             cy + math.sin(theta)         * size * 1.6)
    left  = (cx + math.cos(theta + 2.5)   * size,
             cy + math.sin(theta + 2.5)   * size)
    right = (cx + math.cos(theta - 2.5)   * size,
             cy + math.sin(theta - 2.5)   * size)
    return [nose, left, right]


def get_observation(screen):
    "screenshot current screen -> numpy"
    # rescale 
    small_screen = pygame.transform.smoothscale(screen, (OBSERVATION_SIZE, OBSERVATION_SIZE))
   
    # -> numpy, return (W, H, 3)  = [x][y]
    arr = pygame.surfarray.array3d(small_screen)

    # transpose to standard image [y][x]
    obs = np.transpose(arr, (1, 0 , 2))
    return obs # dtype uint8, 0 -> 255

    

def get_action(keys):
    "read the key being hold -> return action (v, omega)"
    v = 0.0
    omega = 0.0
    if keys[pygame.K_UP]:
        v += V_FORWARD
    if keys[pygame.K_DOWN]:
        v -= V_FORWARD
    if keys[pygame.K_LEFT]:
        omega -= OMEGA
    if keys[pygame.K_RIGHT]:
        omega += OMEGA
    return (v, omega)


def clamp(value, lo, hi): 
    return max(lo, min(value, hi))


def circle_rect_hit(cx, cy, r, rect): 
    nx = clamp(cx, rect['x'], rect['x'] + rect['w']) # nearest x to the obs x
    ny = clamp(cy, rect['y'], rect['y'] + rect['h']) # nearest y to the obs y
    dx = cx - nx 
    dy = cy - ny
    return dx**2 + dy**2 < r**2 # compare spare 
    
    
def car_collides(world):
    car = world["car"]
    r = world["car_radius"]
    for ob in world["obstacles"]:
        if circle_rect_hit(car['x'], car['y'], r, ob):
            return True 
    return False


def step(world, action, dt): 
    "update the new state, with unicyle"
    v, omega = action 
    car = world["car"]
    r = world["car_radius"] 

    # remember last state 
    old_x, old_y = car['x'], car['y']
    

    car["theta"] += omega * dt 
    car["x"] += v * math.cos(car["theta"]) * dt 
    car["y"] += v * math.sin(car["theta"]) * dt
    
    # block edge 
    car['x'] = clamp(car['x'], r, WIDTH - r)
    car['y'] = clamp(car['y'], r , HEIGHT - r)

    # block obs 
    if car_collides(world):
        car['x'], car['y']  = old_x, old_y

def make_world():
    "return the init state"
    return { 
        "car": {"x": 300.0, "y": 300.0, "theta": 0.0}, # pos + direction (radian)
        "goal": {"x": 500.0, "y": 120.0},
        "obstacles": [
            {"x": 250, "y": 150, "w": 80, "h": 80}, 
            {"x": 550, "y": 250, "w": 80, "h": 80}, 
            {"x": 350, "y": 350, "w": 80, "h": 80}, 
        ],
        "car_radius": 28, 
        "goal_radius": 12,
    }


def draw_world(screen, world, draw_goal=False): 
    "only read world and draw, not change the world"
    # reset background
    screen.fill(BACKGROUND)

    # draw obs 
    for ob in world["obstacles"]:
        pygame.draw.rect(screen, OBST, (ob['x'], ob['y'], ob['w'], ob['h']))

    # draw goal 
    if draw_goal:
        g = world['goal']
        pygame.draw.circle(screen, GOAL, (int(g['x']), int(g['y'])), world["goal_radius"] )

    # draw the car 
    c = world["car"]
    pts = car_triangle(c["x"], c["y"], c["theta"], world["car_radius"])
    pygame.draw.polygon(screen, CAR, [(int(x), int(y)) for x, y in pts])


# main 
def main():
    # init the game 
    pygame.init()
    # create the window 
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # create caption 
    pygame.display.set_caption("RC CAR SIM")
    # clock for fps
    clock = pygame.time.Clock()

    # create state 
    world = make_world()

    running = True
    while running: 
        # read the even 
        for event in pygame.event.get():
            # catch the QUIT
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                obs = get_observation(screen)
                print("obs", obs)
                print("obs shape", obs.shape)
                
        # update the state 
        keys = pygame.key.get_pressed()
        action = get_action(keys)
        step(world, action, DT)
        
        # draw the new state 
        draw_world(screen, world)
        pygame.display.flip()

        # hold tick (without this, it will run full cpu)
        clock.tick(FPS) 
    
    pygame.quit()



if __name__ == "__main__":
    main()


    