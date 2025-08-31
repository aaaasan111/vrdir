import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


class DNCTrainer:
    def __init__(
            self, dnc, sad, feature_extractor, train_loader, val_loader, optimizer, scheduler,
            work_dir, device, total_epochs, save_every
    ):
        self.dnc = dnc.to(device)
        self.sad = sad.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.work_dir = work_dir
        self.device = device
        self.save_every = save_every
        self.total_epochs = total_epochs
        self.start_epoch = 0

        # 冻结 SAD 参数，不参与反向传播
        for p in self.sad.parameters():
            p.requires_grad = False
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        os.makedirs(self.work_dir, exist_ok=True)

    def load_checkpoint(self, ckpt_path):
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.dnc.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Loaded checkpoint '{ckpt_path}' (epoch {self.start_epoch})")
        else:
            print(f"No checkpoint found at '{ckpt_path}'")

    def save_checkpoint(self, branch_idx, epoch):
        save_dir = os.path.join(self.work_dir, "checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            'model': self.dnc.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
        }
        if epoch % 10 == 0 or epoch == self.total_epochs - 1:
            path = os.path.join(save_dir, f'dnc_branch_{branch_idx}_epoch{epoch}.pth')
            torch.save(ckpt, path)
            print(f"Branch {branch_idx} Epoch {epoch} checkpoint saved to {path}")

    def train_branch(self, branch_idx):
        for epoch in range(self.start_epoch, self.total_epochs):
            self.dnc.train()
            total_loss = 0.0
            for deg, clean in self.train_loader:
                deg = deg.to(self.device)
                clean = clean.to(self.device)
                with torch.no_grad():
                    clean_feat = self.feature_extractor(clean)  # 提取特征
                    deg_feat = self.feature_extractor(deg)
                dnc_out = self.dnc(deg_feat)
                rec_deg = self.sad(dnc_out)

                # 计算损失
                dnc_feat_max = F.adaptive_max_pool2d(dnc_out, (1, 1)).view(dnc_out.size(0), -1)  # 全局最大池化到 (B,C,1,1)
                clean_feat_max = F.adaptive_max_pool2d(clean_feat, (1, 1)).view(clean_feat.size(0), -1)
                loss_smc = (1 - F.cosine_similarity(dnc_feat_max, clean_feat_max, dim=1)).mean()
                loss_rec = F.l1_loss(rec_deg, clean)
                loss = loss_rec + loss_smc

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Branch {branch_idx} Epoch {epoch}: Loss={avg_loss:.4f}, LR={self.scheduler.get_last_lr()[0]:.6f}")

            if epoch % self.save_every == 0 or epoch == self.total_epochs:
                self.save_checkpoint(branch_idx, epoch)
                self.validate(branch_idx, epoch)

    def validate(self, branch_idx, epoch, num_images=4):
        self.dnc.eval()
        val_loss = 0.0
        vis_dir = os.path.join(self.work_dir, 'val_images')
        os.makedirs(vis_dir, exist_ok=True)
        with torch.no_grad():
            for idx, (deg, clean) in enumerate(self.val_loader):
                deg, clean = deg.to(self.device), clean.to(self.device)
                clean_feat = self.feature_extractor(clean)
                deg_feat = self.feature_extractor(deg)
                dnc_out = self.dnc(deg_feat)
                rec_deg = self.sad(dnc_out)

                dnc_feat_max = F.adaptive_max_pool2d(dnc_out, (1, 1)).view(dnc_out.size(0), -1)
                clean_feat_max = F.adaptive_max_pool2d(clean_feat, (1, 1)).view(clean_feat.size(0), -1)
                loss_smc = (1 - F.cosine_similarity(dnc_feat_max, clean_feat_max, dim=1)).mean()
                loss_rec = F.l1_loss(rec_deg, clean)
                loss = loss_rec + loss_smc
                val_loss += loss.item()

                # 可视化前 num_images 样本
                if idx < num_images:
                    grid = torch.cat([clean, rec_deg], dim=0)
                    save_image(grid, os.path.join(vis_dir, f'branch{branch_idx}_ep{epoch}_idx{idx}.png'),
                               nrow=clean.size(0))
        print(f"Validation images saved for branch {branch_idx}, epoch {epoch}")
