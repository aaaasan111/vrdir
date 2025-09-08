import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import models


class Trainer:
    def __init__(self, device, SAD, ise, optimizer, scheduler,
                 train_loader, val_loader, test_loader, work_dir, save_every, total_epochs):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # 使用 torchvision 自带的 VGG16 前8层作为特征提取器
        vgg = models.vgg16(pretrained=True)
        # 冻结所有 VGG 参数
        for param in vgg.features.parameters():
            param.requires_grad = False
        self.ext = torch.nn.Sequential(*list(vgg.features.children())[:8]).to(self.device)
        self.sad = SAD.to(self.device)
        self.ise = ise.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.work_dir = work_dir
        self.save_every = save_every
        self.total_epochs = total_epochs
        self.start_epoch = 0

    def save_checkpoint(self, epoch):
        if epoch % 10 == 0 or epoch == self.total_epochs - 1:
            save_dir = os.path.join(self.work_dir, "checkpoint")
            os.makedirs(save_dir, exist_ok=True)
            ckpt = {
                'epoch': epoch,
                'ise': self.ise.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }
            path = os.path.join(save_dir, f'ckpt_epoch{epoch}.pth')
            torch.save(ckpt, path)
            print(f"Epoch{epoch} checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.ise.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint '{path}', resuming at epoch {self.start_epoch}")

    def train(self):
        self.sad.eval()
        self.ise.train()
        for epoch in range(self.start_epoch, self.total_epochs):
            total_loss = 0.0
            for deg, clean in self.train_loader:
                deg, clean = deg.to(self.device), clean.to(self.device)
                with torch.no_grad():
                    clean_feat = self.ext(clean)
                    deg_feat = self.ext(deg)
                deg_ise = self.ise(deg_feat)
                deg_recon = self.sad(deg_ise)

                # 损失
                loss_rec = F.l1_loss(deg_recon, clean)
                deg_ise_max = F.adaptive_max_pool2d(deg_ise, (1, 1)).view(deg_ise.size(0), -1)  # 全局最大池化到 (B,C,1,1)
                clean_feat_max = F.adaptive_max_pool2d(clean_feat, (1, 1)).view(clean_feat.size(0), -1)
                loss_smc = (1 - F.cosine_similarity(deg_ise_max, clean_feat_max, dim=1)).mean()
                loss = loss_rec + loss_smc

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Train ISE Epoch {epoch}, Loss: {avg_loss:.4f}, LR={self.scheduler.get_last_lr()[0]:.6f}")

            if epoch % self.save_every == 0 or epoch == self.total_epochs:
                self.save_checkpoint(epoch)
                self.validate(epoch)

    def validate(self, epoch, num_images=4):
        self.sad.eval()
        self.ise.eval()
        val_loss = 0.0
        vis_dir = os.path.join(self.work_dir, 'val_images')
        os.makedirs(vis_dir, exist_ok=True)

        with torch.no_grad():
            for idx, (deg, clean) in enumerate(self.val_loader):
                deg, clean = deg.to(self.device), clean.to(self.device)
                clean_feat = self.ext(clean)
                deg_feat = self.ext(deg)
                deg_ise = self.ise(deg_feat)
                deg_recon = self.sad(deg_ise)

                loss_rec = F.l1_loss(deg_recon, clean)
                deg_ise_max = F.adaptive_max_pool2d(deg_ise, (1, 1)).view(deg_ise.size(0), -1)  # 全局最大池化到 (B,C,1,1)
                clean_feat_max = F.adaptive_max_pool2d(clean_feat, (1, 1)).view(clean_feat.size(0), -1)
                loss_smc = (1 - F.cosine_similarity(deg_ise_max, clean_feat_max, dim=1)).mean()
                loss = loss_rec + loss_smc
                val_loss += loss.item()

                # 保存对比图像
                if idx < num_images:
                    grid = torch.cat([clean, deg_recon], dim=0)
                    save_image(grid, os.path.join(vis_dir, f'epoch{epoch}_idx{idx}.png'), nrow=clean.size(0))
        print(f"Saved validation images for epoch {epoch}")
    
    def test(self):
        self.sad.eval()
        self.ise.eval()
        
        vis_dir = os.path.join(self.work_dir, 'test_images')
        os.makedirs(vis_dir, exist_ok=True)
        saved_count = 0

        with torch.no_grad():
            for idx, (deg, clean) in enumerate(self.test_loader):
                deg, clean = deg.to(self.device), clean.to(self.device)
                # clean_feat = self.ext(clean) # 提取特征
                deg_feat = self.ext(deg)
                deg_ise = self.ise(deg_feat)
                deg_recon = self.sad(deg_ise)

                # 保存图像
                save_image(deg_recon, os.path.join(vis_dir, f'idx{idx}.png'))
                bs = deg_recon.size(0)
                for b in range(bs):
                    img = deg_recon[b].detach().cpu() 
                    save_path = os.path.join(vis_dir, f'idx{idx}_img{b}.png')
                    save_image(img, save_path)

        return None
