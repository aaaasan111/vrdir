# pretrain_sad.py -----------------------------
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from data.dataset import make_dataloader_single
from model.SAD import SAD
from torchvision import models
import argparse
import numpy as np
import random
from SAD_pre.SADtrainer import PretrainSADTrainer
import torch.optim as optim


def parse_args_sad():
    parser = argparse.ArgumentParser('Pretrain SAD')
    parser.add_argument('--clean_dir', default="")
    parser.add_argument('--deg_dir', type=str, default='')
    parser.add_argument('--val_deg_dir', type=str, default="")
    parser.add_argument('--val_clean_dir', type=str, default='')

    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--total_epochs', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--work_dir', default='')
    parser.add_argument('--resume_from', type=str, default="")

    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)

    return parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args_sad()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader = make_dataloader_single(args.deg_dir, args.clean_dir, args.batch_size, args.num_workers, args.resize,
                                          args.shuffle, args.pin_memory)
    val_loader = make_dataloader_single(args.val_deg_dir, args.val_clean_dir, args.batch_size, args.num_workers,
                                        args.resize, False, args.pin_memory)

    # 特征提取器
    vgg = models.vgg16(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(vgg.features.children())[:8])
    for p in feature_extractor.parameters():
        p.requires_grad = False

    model = SAD(in_channels=128)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    trainer = PretrainSADTrainer(
        model=model,
        feature_extractor=feature_extractor,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        work_dir=args.work_dir,
        device=device,
        total_epochs=args.total_epochs,
        save_every=args.save_every
    )

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    trainer.train()


if __name__ == '__main__':
    main()
