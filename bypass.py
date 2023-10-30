from script import BypassingEnv
from stable_baselines3 import PPO
import pygame

from model import bypass

env = BypassingEnv(bypass)
model = PPO.load("bypass_model")

obs = env.reset()
env.render()

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    print("Reward:", reward)
    print("Done:", done)

    pygame.time.wait(800)
