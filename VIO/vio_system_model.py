import torch
import pypose as pp


class VIOSystemModel:

    def __init__(self, dt=0.1, gravity=None):
        self.dt = dt
        if gravity is None:
            self.gravity = torch.tensor([0.0, 0.0, -9.81])
        else:
            self.gravity = gravity
        self.state_dim = 15
        self.pose_dim = 6
        self.m = self.state_dim
        self.n = self.state_dim

    def imu_preintegrate(self, acc, gyro, dt=None):
        if dt is None:
            dt = self.dt
        T = acc.shape[0]
        batch = acc.shape[1] if acc.dim() == 3 else 1
        if acc.dim() == 2:
            acc = acc.unsqueeze(1)
            gyro = gyro.unsqueeze(1)

        delta_R = pp.identity_SO3(batch).to(acc.device)
        delta_v = torch.zeros(batch, 3, device=acc.device)
        delta_p = torch.zeros(batch, 3, device=acc.device)

        for i in range(T):
            omega = gyro[i]
            dR = pp.so3(omega * dt).Exp()
            R_mat = delta_R.matrix()
            rotated_acc = torch.bmm(R_mat, acc[i].unsqueeze(-1)).squeeze(-1)
            delta_v = delta_v + rotated_acc * dt
            delta_p = delta_p + delta_v * dt + 0.5 * rotated_acc * dt * dt
            delta_R = delta_R * dR

        return delta_R, delta_v, delta_p

    def predict_state(self, pose_prev, vel_prev, acc_data, gyro_data, dt=None):
        if dt is None:
            dt = self.dt
        delta_R, delta_v, delta_p = self.imu_preintegrate(acc_data, gyro_data, dt)

        R_prev = pose_prev.rotation()
        t_prev = pose_prev.translation()

        R_mat_prev = R_prev.matrix()
        g = self.gravity.to(pose_prev.device)

        t_pred = t_prev + vel_prev * dt + 0.5 * g * dt * dt + torch.bmm(R_mat_prev, delta_p.unsqueeze(-1)).squeeze(-1)
        v_pred = vel_prev + g * dt + torch.bmm(R_mat_prev, delta_v.unsqueeze(-1)).squeeze(-1)
        R_pred = R_prev * delta_R

        pose_pred = pp.SE3(torch.cat([t_pred, R_pred.tensor()], dim=-1))

        return pose_pred, v_pred

    def state_to_vec(self, pose, velocity):
        if not isinstance(pose, pp.LieTensor):
            pose = pp.SE3(pose)
        t = pose.translation()
        r = pose.rotation().Log().tensor()
        return torch.cat([t, r, velocity], dim=-1)

    def vec_to_state(self, vec):
        t = vec[..., :3]
        r_log = vec[..., 3:6]
        v = vec[..., 6:9]
        R = pp.so3(r_log).Exp()
        pose = pp.SE3(torch.cat([t, R.tensor()], dim=-1))
        return pose, v
