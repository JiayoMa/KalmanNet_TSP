import torch
import pypose as pp
import argparse
import os

from VIO import SpatiotemporalEncoder
from VIO.latent_observation_model import LatentObservationModel
from VIO.vio_system_model import VIOSystemModel
from KNet.ManifoldKalmanNet_nn import ManifoldKalmanNet
from Pipelines.Pipeline_KF_visual import Pipeline_KF_visual


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-End Latent VIO')

    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--state_dim', type=int, default=9)
    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--in_mult_KNet', type=int, default=5)
    parser.add_argument('--out_mult_KNet', type=int, default=40)

    parser.add_argument('--stage1_steps', type=int, default=200)
    parser.add_argument('--stage2_steps', type=int, default=200)
    parser.add_argument('--stage3_steps', type=int, default=100)
    parser.add_argument('--stage1_lr', type=float, default=1e-4)
    parser.add_argument('--stage2_lr', type=float, default=1e-4)
    parser.add_argument('--stage3_lr', type=float, default=1e-5)

    parser.add_argument('--data_dir', type=str, default='data/vio')
    parser.add_argument('--save_dir', type=str, default='results/vio')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'demo'])
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--n_train', type=int, default=20)
    parser.add_argument('--n_test', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--dt', type=float, default=0.1)

    return parser.parse_args()


def generate_synthetic_data(n_sequences, seq_len, img_h, img_w, dt):
    images_list = []
    poses_list = []
    vels_list = []
    acc_list = []
    gyro_list = []

    for _ in range(n_sequences):
        imgs = torch.randn(seq_len + 1, 3, img_h, img_w)

        t_init = torch.zeros(1, 3)
        q_init = pp.identity_SO3(1)
        pose_init = pp.SE3(torch.cat([t_init, q_init.tensor()], dim=-1))
        vel = torch.zeros(1, 3)

        poses = [pose_init.tensor().squeeze(0)]
        velocities = [vel.squeeze(0)]

        for t in range(seq_len):
            acc_t = torch.randn(3) * 0.1
            gyro_t = torch.randn(3) * 0.01

            vel = vel + acc_t.unsqueeze(0) * dt
            t_new = poses[-1][:3] + vel.squeeze(0) * dt
            R_prev = pp.SO3(poses[-1][3:7].unsqueeze(0))
            dR = pp.so3(gyro_t.unsqueeze(0) * dt).Exp()
            R_new = R_prev * dR

            pose_new = torch.cat([t_new, R_new.tensor().squeeze(0)], dim=-1)
            poses.append(pose_new)
            velocities.append(vel.squeeze(0).clone())

        acc_seq = torch.randn(seq_len, 3) * 0.1
        gyro_seq = torch.randn(seq_len, 3) * 0.01

        images_list.append(imgs)
        poses_list.append(torch.stack(poses))
        vels_list.append(torch.stack(velocities))
        acc_list.append(acc_seq)
        gyro_list.append(gyro_seq)

    return images_list, poses_list, vels_list, acc_list, gyro_list


def main():
    args = parse_args()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        args.use_cuda = False
        print("Using CPU")

    os.makedirs(args.save_dir, exist_ok=True)

    encoder = SpatiotemporalEncoder(
        latent_dim=args.latent_dim,
        backbone=args.backbone,
        pretrained=args.pretrained)

    obs_model = LatentObservationModel(
        state_dim=args.state_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim)

    vio_model = VIOSystemModel(dt=args.dt)

    kalman_net = ManifoldKalmanNet()
    kalman_net.build(args.state_dim, args.latent_dim, args)

    pipeline = Pipeline_KF_visual(args.save_dir, 'latent_vio')
    pipeline.set_components(encoder, obs_model, kalman_net, vio_model)
    pipeline.set_training_params(args)

    if args.mode == 'train':
        print("Generating synthetic training data...")
        train_imgs, train_poses, train_vels, train_acc, train_gyro = \
            generate_synthetic_data(args.n_train, args.seq_len,
                                    args.img_h, args.img_w, args.dt)

        print("Generating synthetic test data...")
        test_imgs, test_poses, test_vels, test_acc, test_gyro = \
            generate_synthetic_data(args.n_test, args.seq_len,
                                    args.img_h, args.img_w, args.dt)

        print("=" * 60)
        print("Stage 1: Pre-training visual encoder and observation model")
        print("=" * 60)
        losses_s1 = pipeline.train_stage1(
            train_imgs, train_poses, train_vels,
            n_steps=args.stage1_steps, lr=args.stage1_lr)

        print("=" * 60)
        print("Stage 2: Training KalmanNet with frozen encoder")
        print("=" * 60)
        losses_s2 = pipeline.train_stage2(
            train_imgs, train_poses, train_vels,
            train_acc, train_gyro,
            n_steps=args.stage2_steps, lr=args.stage2_lr)

        print("=" * 60)
        print("Stage 3: Joint fine-tuning (BPTT)")
        print("=" * 60)
        losses_s3 = pipeline.train_stage3(
            train_imgs, train_poses, train_vels,
            train_acc, train_gyro,
            n_steps=args.stage3_steps, lr=args.stage3_lr)

        save_path = os.path.join(args.save_dir, 'model_latent_vio.pt')
        pipeline.save(save_path)
        print(f"Model saved to {save_path}")

        print("=" * 60)
        print("Testing")
        print("=" * 60)
        results = pipeline.test(
            test_imgs, test_poses, test_vels,
            test_acc, test_gyro)

    elif args.mode == 'test':
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.save_dir, 'model_latent_vio.pt')
        pipeline.load(args.checkpoint)

        print("Generating synthetic test data...")
        test_imgs, test_poses, test_vels, test_acc, test_gyro = \
            generate_synthetic_data(args.n_test, args.seq_len,
                                    args.img_h, args.img_w, args.dt)

        results = pipeline.test(
            test_imgs, test_poses, test_vels,
            test_acc, test_gyro)

    elif args.mode == 'demo':
        print("Running demo with synthetic data...")
        demo_imgs, demo_poses, demo_vels, demo_acc, demo_gyro = \
            generate_synthetic_data(1, 5, args.img_h, args.img_w, args.dt)

        pipeline.train_stage1(
            demo_imgs, demo_poses, demo_vels, n_steps=2, lr=1e-4)
        pipeline.train_stage2(
            demo_imgs, demo_poses, demo_vels,
            demo_acc, demo_gyro, n_steps=2, lr=1e-4)
        pipeline.train_stage3(
            demo_imgs, demo_poses, demo_vels,
            demo_acc, demo_gyro, n_steps=2, lr=1e-5)

        results = pipeline.test(
            demo_imgs, demo_poses, demo_vels,
            demo_acc, demo_gyro)
        print("Demo complete.")


if __name__ == '__main__':
    main()
