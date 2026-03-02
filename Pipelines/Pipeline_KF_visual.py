import torch
import torch.nn as nn
import pypose as pp
import random
import time


def geodesic_loss(pose_pred, pose_gt):
    delta = pose_pred.Inv() @ pose_gt
    log_delta = delta.Log().tensor()
    return torch.mean(torch.sum(log_delta ** 2, dim=-1))


def velocity_loss(vel_pred, vel_gt):
    return torch.mean(torch.sum((vel_pred - vel_gt) ** 2, dim=-1))


def combined_loss(pose_pred, pose_gt, vel_pred, vel_gt, alpha=0.5):
    l_geo = geodesic_loss(pose_pred, pose_gt)
    l_vel = velocity_loss(vel_pred, vel_gt)
    return alpha * l_geo + (1.0 - alpha) * l_vel


class Pipeline_KF_visual:

    def __init__(self, folder_name, model_name):
        super().__init__()
        self.folder_name = folder_name + '/'
        self.model_name = model_name

    def set_components(self, encoder, obs_model, kalman_net, vio_model):
        self.encoder = encoder
        self.obs_model = obs_model
        self.kalman_net = kalman_net
        self.vio_model = vio_model

    def set_training_params(self, args):
        self.args = args
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_steps = args.n_steps
        self.n_batch = args.n_batch
        self.lr = args.lr
        self.wd = args.wd

    def _move_to_device(self):
        self.encoder = self.encoder.to(self.device)
        self.obs_model = self.obs_model.to(self.device)
        self.kalman_net = self.kalman_net.to(self.device)

    def _freeze(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def _unfreeze(self, module):
        for p in module.parameters():
            p.requires_grad = True

    def train_stage1(self, train_images, train_poses, train_vels, n_steps=None, lr=None):
        self._move_to_device()
        self._freeze(self.kalman_net)
        self._unfreeze(self.encoder)
        self._unfreeze(self.obs_model)

        if n_steps is None:
            n_steps = self.n_steps
        if lr is None:
            lr = self.lr

        params = list(self.encoder.parameters()) + list(self.obs_model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.wd)
        loss_fn = nn.MSELoss()

        losses = []
        N = len(train_images)
        for step in range(n_steps):
            optimizer.zero_grad()

            indices = random.sample(range(N), min(self.n_batch, N))

            batch_loss = torch.tensor(0.0, device=self.device)
            count = 0

            for idx in indices:
                imgs = train_images[idx]
                poses = train_poses[idx]
                vels = train_vels[idx]

                T_seq = len(imgs) - 1
                if T_seq < 1:
                    continue

                for t in range(T_seq):
                    img_prev = imgs[t].unsqueeze(0).to(self.device)
                    img_curr = imgs[t + 1].unsqueeze(0).to(self.device)

                    z_obs = self.encoder(img_prev, img_curr)

                    pose_t = poses[t + 1].to(self.device)
                    vel_t = vels[t + 1].to(self.device)
                    state_vec = self.vio_model.state_to_vec(
                        pose_t.unsqueeze(0) if pose_t.dim() == 1 else pose_t,
                        vel_t.unsqueeze(0) if vel_t.dim() == 1 else vel_t
                    )
                    z_pred = self.obs_model(state_vec)

                    batch_loss = batch_loss + loss_fn(z_obs, z_pred)
                    count += 1

            if count > 0:
                batch_loss = batch_loss / count

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            losses.append(batch_loss.item())
            if step % 50 == 0:
                print(f"Stage1 step {step}/{n_steps}, loss: {batch_loss.item():.6f}")

        return losses

    def train_stage2(self, train_images, train_poses, train_vels,
                     train_acc, train_gyro, n_steps=None, lr=None):
        self._move_to_device()
        self._freeze(self.encoder)
        self._freeze(self.obs_model)
        self._unfreeze(self.kalman_net)

        if n_steps is None:
            n_steps = self.n_steps
        if lr is None:
            lr = self.lr

        optimizer = torch.optim.Adam(
            self.kalman_net.parameters(), lr=lr, weight_decay=self.wd)

        losses = []
        N = len(train_images)

        for step in range(n_steps):
            optimizer.zero_grad()

            indices = random.sample(range(N), min(self.n_batch, N))

            batch_loss = torch.tensor(0.0, device=self.device)
            count = 0

            for idx in indices:
                imgs = train_images[idx]
                poses = train_poses[idx]
                vels = train_vels[idx]
                acc = train_acc[idx]
                gyro = train_gyro[idx]

                T_seq = len(imgs) - 1
                if T_seq < 1:
                    continue

                init_pose = poses[0].to(self.device)
                init_vel = vels[0].to(self.device)
                if init_pose.dim() == 1:
                    init_pose = init_pose.unsqueeze(0)
                if init_vel.dim() == 1:
                    init_vel = init_vel.unsqueeze(0)

                init_pose = pp.SE3(init_pose)
                self.kalman_net.batch_size = 1
                self.kalman_net.init_hidden()
                self.kalman_net.init_sequence(init_pose, init_vel)

                for t in range(T_seq):
                    img_prev = imgs[t].unsqueeze(0).to(self.device)
                    img_curr = imgs[t + 1].unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        z_obs = self.encoder(img_prev, img_curr)

                    acc_t = acc[t].unsqueeze(0).unsqueeze(0).to(self.device)
                    gyro_t = gyro[t].unsqueeze(0).unsqueeze(0).to(self.device)

                    pose_prev = self.kalman_net.pose_posterior
                    vel_prev = self.kalman_net.vel_posterior
                    pose_prior, vel_prior = self.vio_model.predict_state(
                        pose_prev, vel_prev, acc_t, gyro_t)

                    state_prior_vec = self.vio_model.state_to_vec(pose_prior, vel_prior)
                    with torch.no_grad():
                        z_pred = self.obs_model(state_prior_vec)

                    pose_post, vel_post = self.kalman_net(z_obs, z_pred, pose_prior, vel_prior)

                    gt_pose = poses[t + 1].to(self.device)
                    gt_vel = vels[t + 1].to(self.device)
                    if gt_pose.dim() == 1:
                        gt_pose = gt_pose.unsqueeze(0)
                    if gt_vel.dim() == 1:
                        gt_vel = gt_vel.unsqueeze(0)
                    gt_pose = pp.SE3(gt_pose)

                    batch_loss = batch_loss + combined_loss(pose_post, gt_pose, vel_post, gt_vel)
                    count += 1

            if count > 0:
                batch_loss = batch_loss / count

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kalman_net.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(batch_loss.item())
            if step % 50 == 0:
                print(f"Stage2 step {step}/{n_steps}, loss: {batch_loss.item():.6f}")

        return losses

    def train_stage3(self, train_images, train_poses, train_vels,
                     train_acc, train_gyro, n_steps=None, lr=None):
        self._move_to_device()
        self._unfreeze(self.encoder)
        self._unfreeze(self.obs_model)
        self._unfreeze(self.kalman_net)

        if n_steps is None:
            n_steps = self.n_steps
        if lr is None:
            lr = self.lr * 0.1

        all_params = (list(self.encoder.parameters()) +
                      list(self.obs_model.parameters()) +
                      list(self.kalman_net.parameters()))
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=self.wd)

        losses = []
        N = len(train_images)

        for step in range(n_steps):
            optimizer.zero_grad()

            indices = random.sample(range(N), min(self.n_batch, N))

            batch_loss = torch.tensor(0.0, device=self.device)
            count = 0

            for idx in indices:
                imgs = train_images[idx]
                poses = train_poses[idx]
                vels = train_vels[idx]
                acc = train_acc[idx]
                gyro = train_gyro[idx]

                T_seq = len(imgs) - 1
                if T_seq < 1:
                    continue

                init_pose = poses[0].to(self.device)
                init_vel = vels[0].to(self.device)
                if init_pose.dim() == 1:
                    init_pose = init_pose.unsqueeze(0)
                if init_vel.dim() == 1:
                    init_vel = init_vel.unsqueeze(0)

                init_pose = pp.SE3(init_pose)
                self.kalman_net.batch_size = 1
                self.kalman_net.init_hidden()
                self.kalman_net.init_sequence(init_pose, init_vel)

                for t in range(T_seq):
                    img_prev = imgs[t].unsqueeze(0).to(self.device)
                    img_curr = imgs[t + 1].unsqueeze(0).to(self.device)

                    z_obs = self.encoder(img_prev, img_curr)

                    acc_t = acc[t].unsqueeze(0).unsqueeze(0).to(self.device)
                    gyro_t = gyro[t].unsqueeze(0).unsqueeze(0).to(self.device)

                    pose_prev = self.kalman_net.pose_posterior
                    vel_prev = self.kalman_net.vel_posterior
                    pose_prior, vel_prior = self.vio_model.predict_state(
                        pose_prev, vel_prev, acc_t, gyro_t)

                    state_prior_vec = self.vio_model.state_to_vec(pose_prior, vel_prior)
                    z_pred = self.obs_model(state_prior_vec)

                    pose_post, vel_post = self.kalman_net(z_obs, z_pred, pose_prior, vel_prior)

                    gt_pose = poses[t + 1].to(self.device)
                    gt_vel = vels[t + 1].to(self.device)
                    if gt_pose.dim() == 1:
                        gt_pose = gt_pose.unsqueeze(0)
                    if gt_vel.dim() == 1:
                        gt_vel = gt_vel.unsqueeze(0)
                    gt_pose = pp.SE3(gt_pose)

                    batch_loss = batch_loss + combined_loss(pose_post, gt_pose, vel_post, gt_vel)
                    count += 1

            if count > 0:
                batch_loss = batch_loss / count

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            optimizer.step()

            losses.append(batch_loss.item())
            if step % 50 == 0:
                print(f"Stage3 step {step}/{n_steps}, loss: {batch_loss.item():.6f}")

        return losses

    def test(self, test_images, test_poses, test_vels,
             test_acc, test_gyro):
        self._move_to_device()
        self.encoder.eval()
        self.obs_model.eval()
        self.kalman_net.eval()

        N = len(test_images)
        geo_errors = []
        vel_errors = []

        start = time.time()

        with torch.no_grad():
            for idx in range(N):
                imgs = test_images[idx]
                poses = test_poses[idx]
                vels = test_vels[idx]
                acc = test_acc[idx]
                gyro = test_gyro[idx]

                T_seq = len(imgs) - 1
                if T_seq < 1:
                    continue

                init_pose = poses[0].to(self.device)
                init_vel = vels[0].to(self.device)
                if init_pose.dim() == 1:
                    init_pose = init_pose.unsqueeze(0)
                if init_vel.dim() == 1:
                    init_vel = init_vel.unsqueeze(0)

                init_pose = pp.SE3(init_pose)
                self.kalman_net.batch_size = 1
                self.kalman_net.init_hidden()
                self.kalman_net.init_sequence(init_pose, init_vel)

                seq_geo = 0.0
                seq_vel = 0.0

                for t in range(T_seq):
                    img_prev = imgs[t].unsqueeze(0).to(self.device)
                    img_curr = imgs[t + 1].unsqueeze(0).to(self.device)

                    z_obs = self.encoder(img_prev, img_curr)

                    acc_t = acc[t].unsqueeze(0).unsqueeze(0).to(self.device)
                    gyro_t = gyro[t].unsqueeze(0).unsqueeze(0).to(self.device)

                    pose_prev = self.kalman_net.pose_posterior
                    vel_prev = self.kalman_net.vel_posterior
                    pose_prior, vel_prior = self.vio_model.predict_state(
                        pose_prev, vel_prev, acc_t, gyro_t)

                    state_prior_vec = self.vio_model.state_to_vec(pose_prior, vel_prior)
                    z_pred = self.obs_model(state_prior_vec)

                    pose_post, vel_post = self.kalman_net(z_obs, z_pred, pose_prior, vel_prior)

                    gt_pose = poses[t + 1].to(self.device)
                    gt_vel = vels[t + 1].to(self.device)
                    if gt_pose.dim() == 1:
                        gt_pose = gt_pose.unsqueeze(0)
                    if gt_vel.dim() == 1:
                        gt_vel = gt_vel.unsqueeze(0)
                    gt_pose = pp.SE3(gt_pose)

                    seq_geo += geodesic_loss(pose_post, gt_pose).item()
                    seq_vel += velocity_loss(vel_post, gt_vel).item()

                geo_errors.append(seq_geo / T_seq)
                vel_errors.append(seq_vel / T_seq)

        elapsed = time.time() - start

        geo_errors = torch.tensor(geo_errors)
        vel_errors = torch.tensor(vel_errors)

        results = {
            'geo_mean': geo_errors.mean().item(),
            'geo_std': geo_errors.std(unbiased=False).item() if len(geo_errors) > 1 else 0.0,
            'vel_mean': vel_errors.mean().item(),
            'vel_std': vel_errors.std(unbiased=False).item() if len(vel_errors) > 1 else 0.0,
            'time': elapsed,
        }

        print(f"Geodesic Loss: {results['geo_mean']:.6f} +/- {results['geo_std']:.6f}")
        print(f"Velocity Loss: {results['vel_mean']:.6f} +/- {results['vel_std']:.6f}")
        print(f"Inference Time: {elapsed:.3f}s")

        return results

    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'obs_model': self.obs_model.state_dict(),
            'kalman_net': self.kalman_net.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.obs_model.load_state_dict(ckpt['obs_model'])
        self.kalman_net.load_state_dict(ckpt['kalman_net'])
