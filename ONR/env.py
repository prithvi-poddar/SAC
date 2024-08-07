import numpy as np
import torch
n = 50
task_dim = 7
ag = 5
ag_dim = 8
class Env():
    def __init__(self, coa=0):
        self.coa = coa
        pass
    def step(self, action):
        print(action)
        state = {'nodes': np.random.rand(n, task_dim),
             'cost_adjacency': np.random.rand(n, n),
             'time_adjacency': np.random.rand(n, n),
             'degree_mat': np.random.rand(n, n),
             'agent_feats': np.random.rand(ag, ag_dim),
             'mask': np.random.choice([0, 1], size=n, p=[.5, .5]),
             'agent_index': np.random.randint(0,ag),
             'coa_num': self.coa,
             'time_elapsed': np.random.rand()}
        return state
    def reset(self):
        state = {'nodes': np.random.rand(n, task_dim),
             'cost_adjacency': np.random.rand(n, n),
             'time_adjacency': np.random.rand(n, n),
             'degree_mat': np.random.rand(n, n),
             'agent_feats': np.random.rand(ag, ag_dim),
             'mask': np.random.choice([0, 1], size=n, p=[.5, .5]),
             'agent_index': np.random.randint(0,ag),
             'coa_num': self.coa,
             'time_elapsed': np.random.rand()}
        return state