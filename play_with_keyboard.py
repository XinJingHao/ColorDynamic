from Sparrow_V2 import Sparrow
import pygame
import torch


def main_dicrete_action():
    envs = Sparrow()
    envs.reset()
    while True:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: a = 0
        elif keys[pygame.K_UP]: a = 2
        elif keys[pygame.K_RIGHT]: a = 4
        elif keys[pygame.K_DOWN]: a = 5
        else: a = 7 # Stop

        a = torch.ones(envs.N, dtype=torch.long, device=envs.dvc) * a
        s_next, r, terminated, truncated, info = envs.step(a)


if __name__ == '__main__':
    main_dicrete_action()
