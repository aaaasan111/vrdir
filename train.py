import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import random
import argparse
import torch
from data.dataset import make_dataloader_single
from model.ISE import DNC,ISE
from model.SAD import SAD
from restore import Trainer


def create_args():
    parser = argparse.ArgumentParser('Train vrdir')
    parser.add_argument('--deg_dir', type=str, default='')
    parser.add_argument('--clean_dir', type=str, default='')
    parser.add_argument('--val_deg_dir', type=str, default='')
    parser.add_argument('--val_clean_dir', type=str, default='')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader worker threads')
    # Training parameters
    parser.add_argument('--resize', type=int, default=128, help='Input image resize size')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--total_epochs', type=int, default=1200, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--save_every', type=int, default=1, help='Save validation results and logs every N epochs')
    parser.add_argument('--work_dir', type=str, default='', help='Directory to save results and model weights')
    parser.add_argument('--resume_from', type=str, default="", help='checkpoint path')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--num_degrad', type=int, default=3)
    parser.add_argument('--pre_sad_checkpoint', type=str, default="")
    parser.add_argument('--pre_dnc1_checkpoint', type=str, default="")
    parser.add_argument('--pre_dnc2_checkpoint', type=str, default="")
    parser.add_argument('--pre_dnc3_checkpoint', type=str, default="")
    args = parser.parse_args()

    return args


def build_ise_model(args, device):
    """
    构建并返回 ISE 模型：加载预训练的多个 DNC 分支权重，并注入 ISE。
    """
    dncs = []
    for ckpt_path in [
        args.pre_dnc1_checkpoint,
        args.pre_dnc2_checkpoint,
        args.pre_dnc3_checkpoint
    ]:
        dnc = DNC(in_channels=128).to(device)
        state = torch.load(ckpt_path, map_location=device)
        dnc.load_state_dict(state['model'] if 'model' in state else state)
        dncs.append(dnc)
    ise = ISE(dncs, in_channels=128, num_branches=len(dncs)).to(device)
    return ise


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = create_args()
    os.makedirs(args.work_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader = make_dataloader_single(args.deg_dir, args.clean_dir,args.batch_size, args.num_workers,args.resize,args.shuffle,args.pin_memory)
    val_loader = make_dataloader_single(args.val_deg_dir, args.val_clean_dir,args.batch_size, args.num_workers,args.resize,False,args.pin_memory)
    test_loader=make_dataloader_single(args.val_deg_dir, args.val_clean_dir,args.batch_size, args.num_workers,args.resize,False,args.pin_memory)

    # 加载并冻结预训练 SAD 模型
    sad = SAD(in_channels=128).to(device)
    sad_ckpt = torch.load(args.pre_sad_checkpoint, map_location=device)
    sad.load_state_dict(sad_ckpt["model"])
    sad.eval()

    ise = build_ise_model(args, device)

    optimizer = torch.optim.Adam(ise.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    trainer = Trainer(device,sad,ise,optimizer,scheduler,train_loader,val_loader,test_loader,args.work_dir,args.save_every,args.total_epochs)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    trainer.train()

    # trainer.test()


if __name__ == '__main__':
    main()

