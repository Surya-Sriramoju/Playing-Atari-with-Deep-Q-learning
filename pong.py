import torch
import torch.optim as optim
from model import DQN
from wrappers import make_atari_env
from replay_memory import ReplayBuffer
from utils import train_func, test_func
from Params import *

from IPython.display import clear_output
import matplotlib.pyplot as plt


device = torch.device(device_use)

env_id = "PongNoFrameskip-v4"
env = make_atari_env(env_id)
print(env.action_space)
env.unwrapped.get_action_meanings()

current_model = DQN(env.observation_space.shape, env.action_space.n).to(device)    
target_model = DQN(env.observation_space.shape, env.action_space.n).to(device)    

optimizer = optim.Adam(current_model.parameters(), lr=0.0001)
replay_buffer = ReplayBuffer(MEMORY_SIZE)

model = DQN(env.observation_space.shape, env.action_space.n)
model.load_state_dict(torch.load('models/Pong400.pth', map_location='cpu'))

val = int(input("For training type 1, for testing with trained model press 2: \n"))
if val == 1:
    train_func(env, current_model,target_model, optimizer, replay_buffer, device)
elif val == 2:
    episodes = 100
    test_func(env, model, episodes, render=True, device=device, context="video")
else:
    print('Enter Correct Value!')