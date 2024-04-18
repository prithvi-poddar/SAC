import os
import torch as T
import copy
import numpy as np
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self,
                 env, 
                 lr, 
                 input_dims, 
                 tau, 
                 n_actions, 
                 gamma=0.99, 
                 max_size=1000000,
                 fc1_dims=256, fc2_dims=256, fc3_dims=256, 
                 batch_size=256,
                 start_after=10000, 
                 update_after=1000,
                 chkpt_dir="models",
                 name="v1"):
        self.env = env
        self.lr = lr
        self.input_dims = input_dims
        self.tau = tau
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        os.makedirs(self.checkpoint_file, exist_ok=True)

        self.memory = ReplayBuffer(max_size, env.observation_space.shape, n_actions)

        self.actor = ActorNetwork(lr, input_dims, n_actions, fc1_dims, fc2_dims)

        self.critic1 = CriticNetwork(lr, input_dims, n_actions, fc1_dims, fc2_dims, fc3_dims)
        self.critic2 = CriticNetwork(lr, input_dims, n_actions, fc1_dims, fc2_dims, fc3_dims)

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.target_entropy = -self.env.action_space.shape[0]
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp()

        # self.update_network_parameters(tau=1)

        self.step_counter = 0
        self.start_after = start_after
        self.update_after = update_after

    def choose_action(self, observation, deterministic=False):
        if deterministic:
            with T.no_grad():
                state = T.tensor([observation], dtype=T.float).to(self.actor.device)
                actions, _ = self.actor.forward(state)
                actions = T.tanh(actions)
            return actions.detach().cpu().numpy()[0]

        rand = np.random.random()
        if self.step_counter < self.start_after and rand < 0.001:
            actions = self.env.action_space.sample()[0]
            return actions
        else:
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
            return actions.detach().cpu().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        with T.no_grad():
            for target1, target2, crit1, crit2 in zip(self.target_critic1.parameters(), self.target_critic2.parameters(), self.critic1.parameters(), self.critic2.parameters()):
                target1.data.copy_((1-tau)*crit1.data + tau*target1.data)
                target2.data.copy_((1-tau)*crit2.data + tau*target2.data)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.step_counter < self.update_after:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.unsqueeze(T.tensor(reward, dtype=T.float),1).to(self.critic1.device)
        done = T.unsqueeze(T.from_numpy(done).float(),1).to(self.critic1.device) 
        next_state = T.tensor(new_state, dtype=T.float).to(self.critic1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic1.device)

        with T.no_grad():
            next_action, logprobs_next_action = self.actor.sample_normal(next_state)
            q_t1 = self.target_critic1.forward(next_state, next_action)
            q_t2 = self.target_critic2.forward(next_state, next_action)
            q_target = T.min(q_t1, q_t2)
            critic_target = reward + (1.0 - done) * self.gamma * (q_target - self.alpha * logprobs_next_action)

        q_1 = self.critic1.forward(state, action)
        q_2 = self.critic2.forward(state, action)
        loss_1 = T.nn.MSELoss()(q_1, critic_target)
        loss_2 = T.nn.MSELoss()(q_2, critic_target)

        q_loss_step = loss_1 + loss_2
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_loss_step.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.q1_loss = loss_1.detach().item()
        self.q2_loss = loss_2.detach().item()

        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        policy_action, logprobs_policy_action = self.actor.sample_normal(state)
        p1 = self.critic1.forward(state, policy_action)
        p2 = self.critic2.forward(state, policy_action)

        target = T.min(p1, p2)
        policy_loss = (self.alpha * logprobs_policy_action - target).mean()
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()
        self.policy_loss = policy_loss.detach().item()

        temp_loss = -self.log_alpha * (logprobs_policy_action.detach() + self.target_entropy).mean()
        self.alpha_optim.zero_grad()
        temp_loss.backward()
        self.alpha_optim.step()

        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

        self.update_network_parameters()

    def get_stats(self):
        return self.q1_loss+self.q2_loss, self.policy_loss, self.alpha.item()

    def save_models(self, ep=0, best=True):
        if best:
            print("Saving best model")
            T.save(self.actor, self.checkpoint_file+"/actor")
            T.save(self.critic1, self.checkpoint_file+"/critic1")
            T.save(self.critic2, self.checkpoint_file+"/critic2")
            T.save(self.target_critic1, self.checkpoint_file+"/target_critic1")
            T.save(self.target_critic2, self.checkpoint_file+"/target_critic2")
            T.save(self.log_alpha, self.checkpoint_file+"/logalpha")
        else:
            print("Saving at ",ep," steps")
            T.save(self.actor, self.checkpoint_file+"/actor_"+str(ep))
            T.save(self.critic1, self.checkpoint_file+"/critic1_"+str(ep))
            T.save(self.critic2, self.checkpoint_file+"/critic2_"+str(ep))
            T.save(self.target_critic1, self.checkpoint_file+"/target_critic1_"+str(ep))
            T.save(self.target_critic2, self.checkpoint_file+"/target_critic2_"+str(ep))
            T.save(self.log_alpha, self.checkpoint_file+"/logalpha_"+str(ep))


    def load_models(self, ep=0, best=True, chkpt_file=None, policy_only=False):
        if chkpt_file == None:
            if best:
                print("Loading best model")
                self.actor = T.load(self.checkpoint_file+"/actor")
                self.critic1 = T.load(self.checkpoint_file+"/critic1")
                self.critic2 = T.load(self.checkpoint_file+"/critic2")
                self.target_critic1 = T.load(self.checkpoint_file+"/target_critic1")
                self.target_critic2 = T.load(self.checkpoint_file+"/target_critic2")
                self.log_alpha = T.load(self.checkpoint_file+"/logalpha")
            else:
                print("Loadong regular model")
                self.actor = T.load(self.checkpoint_file+"/actor_"+str(ep))
                self.critic1 = T.load(self.checkpoint_file+"/critic1_"+str(ep))
                self.critic2 = T.load(self.checkpoint_file+"/critic2_"+str(ep))
                self.target_critic1 = T.load(self.checkpoint_file+"/target_critic1_"+str(ep))
                self.target_critic2 = T.load(self.checkpoint_file+"/target_critic2_"+str(ep))
                self.log_alpha = T.load(self.checkpoint_file+"/logalpha_"+str(ep))

        else:
            if policy_only:
                if best:
                    print("Loading best model")
                    self.actor = T.load(chkpt_file+"/actor")
                else:
                    print("Loadong regular model")
                    self.actor = T.load(chkpt_file+"/actor_"+str(ep))

            else:
                if best:
                    print("Loading best model")
                    self.actor = T.load(chkpt_file+"/actor")
                    self.critic1 = T.load(chkpt_file+"/critic1")
                    self.critic2 = T.load(chkpt_file+"/critic2")
                    self.target_critic1 = T.load(chkpt_file+"/target_critic1")
                    self.target_critic2 = T.load(chkpt_file+"/target_critic2")
                    self.log_alpha = T.load(chkpt_file+"/logalpha")
                else:
                    print("Loadong regular model")
                    self.actor = T.load(chkpt_file+"/actor_"+str(ep))
                    self.critic1 = T.load(chkpt_file+"/critic1_"+str(ep))
                    self.critic2 = T.load(chkpt_file+"/critic2_"+str(ep))
                    self.target_critic1 = T.load(chkpt_file+"/target_critic1_"+str(ep))
                    self.target_critic2 = T.load(chkpt_file+"/target_critic2_"+str(ep))
                    self.log_alpha = T.load(chkpt_file+"/logalpha_"+str(ep))