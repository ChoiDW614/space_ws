import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.linalg import cholesky


class GaussianSample():
    def __init__(self, n_horizon, n_action, device, seed=0, scale_tril=None, covariance_matrix=None, fixed_samples=False):
        self.device = device
        self.n_horizon = n_horizon
        self.n_action = n_action
        self.n_dims = self.n_horizon * self.n_action
        self.zero_seq = torch.zeros(self.n_horizon * self.n_action, device=self.device)
        self.zero_act_seq = torch.zeros([1, self.n_horizon, self.n_action], device=self.device)
        self.scale_tril = scale_tril
        self.covariance_matrix = covariance_matrix
        self.seed = seed
        self.fixed_samples = fixed_samples
        self.samples = None
        self.n_sample = 0


        if self.scale_tril is None:
            if covariance_matrix is None:
                self.covariance_matrix = 0.05 * torch.eye(self.n_dims, device=self.device)
            self.scale_tril = cholesky(self.covariance_matrix)

        self.mvn = MultivariateNormal(loc=self.zero_seq, scale_tril=self.scale_tril)
    
    def get_samples(self, n_sample):
        torch.manual_seed(self.seed)
        if self.n_sample != n_sample or not self.fixed_samples:
            self.n_sample = n_sample
            samples = self.mvn.sample(sample_shape=[self.n_sample])
            samples = samples.view(self.n_sample, self.n_horizon, self.n_action)
            # if filter_smooth:
            #     samples = self.filter_smooth(samples)
            # else:
            #     samples = self.filter_samples(samples)
            self.samples = samples
        return self.samples

    def get_action(self, n_sample, mu, seed=None):
        if seed is not None and seed != self.seed:
            self.seed = seed

        noise = self.get_samples(n_sample - 1)
        noise = torch.cat((noise, self.zero_act_seq), dim=0)
        
        act_seq = mu.unsqueeze(0) + noise

        # act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs)
        return act_seq

    # def scale_ctrl(ctrl, action_lows, action_highs):
    #     ctrl = torch.max(torch.min(ctrl, action_highs), action_lows)
    #     return ctrl
    
    def filter_samples(self, samples):
        pass

    def filter_smooth(self, samples):
        pass
    