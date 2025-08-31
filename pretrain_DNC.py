import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import torch
import numpy as np
import random
from torchvision import models
import torch.optim as optim
from data.dataset import make_dataloader_single
from model.ISE import DNC
from model.SAD import SAD
from DNC_pre.DNCtrainer import DNCTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain DNC Branches')
    parser.add_argument('--deg_dir', type=str, default='')
    parser.add_argument('--clean_dir', type=str, default='')
    parser.add_argument('--val_deg_dir', type=str, default='')
    parser.add_argument('--val_clean_dir', type=str, default='')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--total_epochs', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--work_dir', type=str, default='')
    parser.add_argument('--sad_ckpt', type=str, default="")
    parser.add_argument('--resume_from', type=str, default="")
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--branch_name', type=str, default="hazy")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # 加载并冻结预训练 SAD 模型
    sad = SAD(in_channels=128).to(device)
    sad_ckpt = torch.load(args.sad_ckpt, map_location=device)
    sad.load_state_dict(sad_ckpt["model"])
    sad.eval()

    dnc = DNC(in_channels=128)

    optimizer = optim.Adam(dnc.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    trainer = DNCTrainer(
        dnc=dnc,
        sad=sad,
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        work_dir=args.work_dir,
        device=device,
        total_epochs=args.total_epochs,
        save_every=args.save_every
    )

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    trainer.train_branch(args.branch_name)
