import gymnasium as gym
import numpy as np
from sac_agent import Agent
# from utils import plot_learning_curve
import numpy as np
from gymnasium.wrappers import RescaleAction
from env import Env
from joblib import Parallel, delayed
import copy
from buffer import ReplayBuffer
# from quad_v3 import Quad
from torch.utils.tensorboard import SummaryWriter

def rollout(env, agent, buffer):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.step_counter += 1
        buffer.store_transition(observation, action, reward, observation_, done)
        # agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        score += reward
        observation = observation_

    return 0



if __name__ == '__main__':

    writer = SummaryWriter(log_dir='runs/testing')
    env = [Env(coa=i) for i in range(3)]
    coas = [0,1,2]

    agent = Agent(env, tau=0.9, lr=3e-4, name="test")
    # agent = Agent(env=env, lr=3e-4, input_dims=env.observation_space.shape[0],
    #                 tau=0.995, n_actions=env.action_space.shape[0], name="sac_single_reward_3_input_random_init")

    # agent.load_models(best=True, chkpt_file='models/sac_single_reward_3_input', policy_only=True)

    n_episodes = 1000000

    for i in range(n_episodes):
        


    agents = []
    buffers = []
    for _ in coas:
        agents.append(copy.deepcopy(agent))
        buffers.append(ReplayBuffer(1000000, 1))
    
    

    score_history = []

    for i in range(n_episodes):
        # env.render()
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.step_counter += 1
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        
        # critic_loss, policy_loss, alpha = agent.get_stats()
        # writer.add_scalar("Critic Loss", critic_loss, i)
        # writer.add_scalar("Policy Loss", policy_loss, i)
        # writer.add_scalar("Alpha", alpha, i)
        # writer.add_scalar("Reward", score, i)

        score_history.append(score)
        agent.save_models(ep = i, best=False)
        if score == max(score_history):
            agent.save_models()
        if agent.step_counter > agent.update_after:
            critic_loss, actor_loss, alpha = agent.get_stats()
            print("Episode:"+str(i), "Reward:"+str(score), "Critic Loss:"+str(critic_loss), 
                    "Actor Loss:"+str(actor_loss), "alpha:"+str(alpha), sep="\t")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from ripser import Rips
import persim
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle

with open("graph_dict.pkl", "rb") as input_file:
   data = pickle.load(input_file)

n = 10000

data = data[:n]
inputs = []
for i in range(n):
    for j in range(i, n):
        inputs.append([i,j])
inputs = tqdm(inputs)

def compute_distance(i):
    dgm1 = data[i[0]][1][2]
    dgm2 = data[i[1]][1][2]
    if i[0] == i[1]:
        dist = 0
    else:
        dist = persim.wasserstein(dgm1, dgm2, matching=False)
    return [i[0], i[1], dist]

graph_data_list = Parallel(n_jobs=20, require='sharedmem')(delayed(compute_distance)(i) for i in inputs)

dist_matrix = np.zeros((n, n))
for i in graph_data_list:
    if i[0] != i[1]:
        dist_matrix[i[0]][i[1]] = i[2]
        dist_matrix[i[1]][i[0]] = i[2]

np.save('dist_mat.npy)', dist_matrix)