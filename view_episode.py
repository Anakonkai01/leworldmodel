import glob
import numpy as np
import pygame

OBS_SIZE = 96
SCALE = 6                      # phóng 96 -> 576
FPS = 30

def load_episode(path):
    d = np.load(path)
    return d["obs"], d["actions"]

def frame_to_surface(frame):
    # frame: (H, W, 3) = [y][x]  ->  pygame cần (W, H, 3) = [x][y]
    arr = np.transpose(frame, (1, 0, 2))
    surf = pygame.surfarray.make_surface(arr)
    return pygame.transform.scale(surf, (OBS_SIZE * SCALE, OBS_SIZE * SCALE))

def main(idx=0):
    files = sorted(glob.glob("data/ep_[0-9]*.npz"))   # chỉ file đánh số
    
    obs, actions = load_episode(files[idx])

    pygame.init()
    screen = pygame.display.set_mode((OBS_SIZE * SCALE, OBS_SIZE * SCALE))
    clock = pygame.time.Clock()

    t = 0
    paused = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:                 # tạm dừng
                    paused = not paused
                elif event.key == pygame.K_RIGHT:               # episode kế
                    idx = (idx + 1) % len(files)
                    obs, actions = load_episode(files[idx]); t = 0
                elif event.key == pygame.K_LEFT:                # episode trước
                    idx = (idx - 1) % len(files)
                    obs, actions = load_episode(files[idx]); t = 0

        # vẽ khung hiện tại
        screen.blit(frame_to_surface(obs[t]), (0, 0))
        pygame.display.set_caption(
            f"{files[idx]}  t={t}/{len(obs)-1}  "
            f"v={actions[t,0]:.0f}  omega={actions[t,1]:.2f}"
        )
        pygame.display.flip()

        if not paused:
            t += 1
            if t >= len(obs):                       # hết episode này
                idx = (idx + 1) % len(files)        # -> tự sang episode kế
                obs, actions = load_episode(files[idx])
                t = 0
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
