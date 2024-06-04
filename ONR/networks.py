import os
import math
from warnings import formatwarning
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.distributions import constraints
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution, Categorical
from torch.distributions.transforms import Transform
from Graph_Capsule_Convolution.Models.ONR.encoder import Task_Graph_Encoder, Context_Encoder
from Graph_Capsule_Convolution.Models.ONR.decoder import MHA_Decoder
from Graph_Capsule_Convolution.Networks.utils import normalize_edges

class ActorNetwork(nn.Module):
    def __init__(self, lr=0.9, 
                 node_feats=7,
                 encoder_out_feats=64,
                 edge_feats=2,
                 encoder_hidden_dims=[32,32,32],
                 k=2, p=3,
                 gcn_model='Edge_Laplacian',
                 context_in_feats=8,
                 context_out_feats=64,
                 context_hidden_dims=[32,32],
                 decoder_heads=8,
                 decoder_hidden_dims=[64,64],
                 decoder_out_feats=50,
                 coa_num=3,
                 latent_dim=128,
                 activation=nn.ReLU(),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.task_encoder = Task_Graph_Encoder(in_feats=node_feats,
                                               out_feats=encoder_out_feats,
                                               edge_feat=edge_feats,
                                               hidden_dims=encoder_hidden_dims,
                                               k=k, p=p,
                                               gcn_model=gcn_model,
                                               activation=activation,
                                               device=device)
        
        self.coa_keys = nn.ModuleList(nn.Linear(in_features=encoder_out_feats,
                                                out_features=latent_dim) for i in range(coa_num))
        self.coa_vals = nn.ModuleList(nn.Linear(in_features=encoder_out_feats,
                                                out_features=latent_dim) for i in range(coa_num))
        
        self.context_encoder = Context_Encoder(in_feats=context_in_feats,
                                               out_feats=context_out_feats,
                                               hidden_dims=context_hidden_dims,
                                               device=device)
        self.decoder = MHA_Decoder(context_dim=context_out_feats,
                                   key_dim=latent_dim,
                                   value_dim=latent_dim,
                                   num_heads=decoder_heads,
                                   hidden_dim=decoder_hidden_dims,
                                   out_feats=decoder_out_feats,
                                   device=device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device

        self.to(self.device)

    def forward(self, state):
        tasks = torch.tensor(state['nodes'], dtype=torch.float32, device=self.device)
        cost_adj = state['degree_mat']-state['cost_adjacency']
        time_adj = state['degree_mat']-state['time_adjacency']
        adj = normalize_edges(np.stack((cost_adj, time_adj), axis=2))
        adj = torch.tensor(adj, dtype=torch.float32, device=self.device)
        agents = torch.tensor(state['agent_feats'], dtype=torch.float32, device=self.device)
        mask = torch.unsqueeze(torch.tensor(state['mask'], dtype=torch.float32, device=self.device), 0)
        ag_idx = int(state['agent_index'])
        coa_num = state['coa_num']
        time = torch.unsqueeze(torch.tensor([state['time_elapsed']], dtype=torch.float32, device=self.device), 0)
        agent = torch.unsqueeze(agents[ag_idx,:], 0)
        peers = torch.cat((agents[:ag_idx,:],agents[ag_idx+1:,:]), dim=0)

        encoder_x = self.task_encoder(X=tasks, L=adj)
        context_x = self.context_encoder(agent=agent, peers=peers, time=time)
        key = self.coa_keys[coa_num](encoder_x)
        value = self.coa_vals[coa_num](encoder_x)
        probs = self.decoder(context=context_x, key=key, value=value)
        probs = probs*mask
        # Softmax ignoring 0 values
        x_max = torch.max(probs, dim=1, keepdim=True)[0][0]
        x_exp = torch.exp(probs-x_max)
        x_exp = x_exp * (probs!=0)
        probs_softmax = x_exp / torch.sum(x_exp, dim=1, keepdim=True)
        return probs_softmax
        
        # x = self.fc1(state)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # mu = self.mu(x)
        # logsigma = self.logsigma(x)
        # return mu, logsigma

    def sample_normal(self, state, reparameterize=True):
        probs = self.forward(state)
        probabilities = Categorical(probs)
        action = probabilities.sample()
        # mu, logsigma = self.forward(state)
        # logsigma = T.clamp(logsigma, -20, 2)
        # sigma = logsigma.exp()
        # probabilities = Normal(mu, sigma)
        # transforms = [TanhTransform(cache_size=1)]
        # probabilities = TransformedDistribution(probabilities, transforms)
        # if reparameterize:
        #     action = probabilities.rsample()
        # else:
        #     action = probabilities.sample()

        log_probs = probabilities.log_prob(action).sum(axis=-1, keepdim=True)
        log_probs.to(self.device)

        return action, log_probs

# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))



class CriticNetwork(nn.Module):
    def __init__(self, lr, obs_dims, action_dims, fc1_dims, fc2_dims, fc3_dims):
        super(CriticNetwork, self).__init__()
        self.lr = lr
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        # self.name = name
        # self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        # os.makedirs(self.checkpoint_dir, exist_ok = True)

        self.fc1 = nn.Linear(obs_dims + action_dims, fc1_dims)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        nn.init.xavier_uniform_(self.fc3.weight.data)
        self.q = nn.Linear(fc3_dims, 1)
        nn.init.xavier_uniform_(self.q.weight.data)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(T.cat((state, action), dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        q = self.q(x)

        return q