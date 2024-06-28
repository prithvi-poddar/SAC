import os
import math
from warnings import formatwarning
import numpy as np
import torch as T
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.distributions import constraints
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import Transform

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



class ActorNetwork(nn.Module):
    def __init__(self, lr, obs_dims, action_dims, fc1_dims, fc2_dims):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.obs_dims = obs_dims
        self.action_dims = action_dims

        self.fc1 = nn.Linear(self.obs_dims, fc1_dims)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        nn.init.xavier_uniform_(self.fc2.weight.data)

        self.mu = nn.Linear(fc2_dims, self.action_dims)
        self.logsigma = nn.Linear(fc2_dims, self.action_dims)
        nn.init.xavier_uniform_(self.mu.weight.data)
        nn.init.xavier_uniform_(self.logsigma.weight.data)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        return mu, logsigma

    def sample_normal(self, state, reparameterize=True):
        mu, logsigma = self.forward(state)
        logsigma = T.clamp(logsigma, -20, 2)
        sigma = logsigma.exp()
        probabilities = Normal(mu, sigma)
        transforms = [TanhTransform(cache_size=1)]
        probabilities = TransformedDistribution(probabilities, transforms)
        if reparameterize:
            action = probabilities.rsample()
        else:
            action = probabilities.sample()

        log_probs = probabilities.log_prob(action).sum(axis=-1, keepdim=True)
        log_probs.to(self.device)

        return action, log_probs


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