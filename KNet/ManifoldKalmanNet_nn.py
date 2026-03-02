import torch
import torch.nn as nn
import torch.nn.functional as func
import pypose as pp


class ManifoldKalmanNet(nn.Module):

    def __init__(self):
        super().__init__()

    def build(self, state_dim, latent_dim, args):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.m = state_dim
        self.n = latent_dim

        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.batch_size = args.n_batch
        self.seq_len_input = 1

        in_mult = args.in_mult_KNet
        out_mult = args.out_mult_KNet

        hidden_cap = max(self.m ** 2, 128)

        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        self.d_hidden_S = min(self.n ** 2, hidden_cap)
        self.d_input_S_fc1 = min(self.n ** 2, hidden_cap)
        self.d_obs_feat = min(2 * self.n * in_mult, hidden_cap)

        self.obs_proj = nn.Sequential(
            nn.Linear(2 * self.n, self.d_obs_feat),
            nn.ReLU()).to(self.device)

        self.sigma_proj = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma, self.d_input_S_fc1),
            nn.ReLU()).to(self.device)

        self.d_input_S = self.d_input_S_fc1 + self.d_obs_feat
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.d_input_S_fc1
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU()).to(self.device)

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = min(self.d_input_FC2 * out_mult, 4096)
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)

        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU()).to(self.device)

        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU()).to(self.device)

        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU()).to(self.device)

        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * in_mult
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU()).to(self.device)

        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = self.d_obs_feat
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU()).to(self.device)

        self.prior_Q = torch.eye(self.m).to(self.device)
        self.prior_Sigma = torch.zeros(self.m, self.m).to(self.device)
        self.prior_S = torch.eye(self.d_hidden_S).to(self.device)

    def init_hidden(self):
        self.h_S = torch.zeros(
            self.seq_len_input, self.batch_size, self.d_hidden_S).to(self.device)
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(
            self.seq_len_input, self.batch_size, 1)
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(
            self.seq_len_input, self.batch_size, 1)

    def init_sequence(self, x0_pose, x0_vel):
        self.pose_posterior = x0_pose
        self.vel_posterior = x0_vel
        self.pose_posterior_prev = x0_pose
        self.vel_posterior_prev = x0_vel
        self.pose_prior_prev = x0_pose
        self.vel_prior_prev = x0_vel
        self.z_prev = None

    @staticmethod
    def manifold_diff_pose(pose_a, pose_b):
        delta = pose_a.Inv() @ pose_b
        return delta.Log().tensor()

    def state_vec(self, pose, vel):
        t = pose.translation()
        r = pose.rotation().Log().tensor()
        return torch.cat([t, r, vel], dim=-1)

    def compute_kalman_gain(self, z_obs, z_pred, pose_prior, vel_prior):
        state_posterior_vec = self.state_vec(self.pose_posterior, self.vel_posterior)
        state_posterior_prev_vec = self.state_vec(self.pose_posterior_prev, self.vel_posterior_prev)
        state_prior_prev_vec = self.state_vec(self.pose_prior_prev, self.vel_prior_prev)

        fw_evol_diff = state_posterior_vec - state_posterior_prev_vec
        fw_update_diff = state_posterior_vec - state_prior_prev_vec

        if self.z_prev is None:
            obs_diff = torch.zeros_like(z_obs)
        else:
            obs_diff = z_obs - self.z_prev
        obs_innov_diff = z_obs - z_pred

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        KG = self.kgain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        return KG.reshape(self.batch_size, self.m, self.n)

    def kgain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        out_FC5 = self.FC5(fw_update_diff)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)

        out_FC6 = self.FC6(fw_evol_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        out_FC1 = self.FC1(out_Sigma)

        obs_cat = torch.cat((obs_diff, obs_innov_diff), -1)
        if obs_cat.dim() == 2:
            obs_cat = obs_cat.unsqueeze(0)
        out_FC7 = self.FC7(obs_cat)

        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        self.h_Sigma = out_FC4

        return out_FC2

    def step(self, z_obs, z_pred, pose_prior, vel_prior):
        KGain = self.compute_kalman_gain(z_obs, z_pred, pose_prior, vel_prior)

        innovation = z_obs - z_pred
        correction = torch.bmm(KGain, innovation.unsqueeze(-1)).squeeze(-1)

        delta_pose = pp.se3(correction[:, :6]).Exp()
        delta_vel = correction[:, 6:9]

        self.pose_posterior_prev = self.pose_posterior
        self.vel_posterior_prev = self.vel_posterior

        self.pose_posterior = pose_prior @ delta_pose
        self.vel_posterior = vel_prior + delta_vel

        self.pose_prior_prev = pose_prior
        self.vel_prior_prev = vel_prior
        self.z_prev = z_obs

        return self.pose_posterior, self.vel_posterior

    def forward(self, z_obs, z_pred, pose_prior, vel_prior):
        z_obs = z_obs.to(self.device)
        z_pred = z_pred.to(self.device)
        return self.step(z_obs, z_pred, pose_prior, vel_prior)
