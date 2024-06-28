import gymnasium as gym
import numpy as np
from sac_agent import Agent
# from utils import plot_learning_curve
import numpy as np
from gymnasium.wrappers import RescaleAction
# from quad_v3 import Quad
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    writer = SummaryWriter(log_dir='runs/mujoco_bipedal_test')
    env = gym.make("BipedalWalker-v3", hardcore=True)
    env = RescaleAction(env, -1, 1)

    agent = Agent(env=env, lr=3e-4, input_dims=env.observation_space.shape[0],
                    tau=0.995, n_actions=env.action_space.shape[0], name="bipedal_test")

    # agent.load_models(best=True, chkpt_file='models/sac_single_reward_3_input', policy_only=True)
    n_episodes = 1000000

    score_history = []

    for i in range(n_episodes):
        # env.render()
        observation, _ = env.reset()
        done = False
        trunc = False
        score = 0
        while done == False and trunc == False:
            action = agent.choose_action(observation)
            observation_, reward, done, trunc, info = env.step(action)
            agent.step_counter += 1
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_

        score_history.append(score)
        agent.save_models(ep = i, best=False)
        if score == max(score_history):
            agent.save_models()
        if agent.step_counter > agent.update_after:
            critic_loss, policy_loss, alpha = agent.get_stats()
            writer.add_scalar("Critic Loss", critic_loss, i)
            writer.add_scalar("Policy Loss", policy_loss, i)
            writer.add_scalar("Alpha", alpha, i)
            writer.add_scalar("Reward", score, i)
            critic_loss, actor_loss, alpha = agent.get_stats()
            print("Episode:"+str(i), "Reward:"+str(score), "Critic Loss:"+str(critic_loss), 
                    "Actor Loss:"+str(actor_loss), "alpha:"+str(alpha), sep="\t")