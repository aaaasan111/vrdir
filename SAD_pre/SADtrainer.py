import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def similarity_ranking_loss(recon_clean, recon_degraded, clean, degraded, margin=1, eps=1e-8):
    B = recon_clean.size(0)

    # 将图像展平为向量 (B, C*H*W)，以便计算相似度
    recon_clean_flat = recon_clean.view(B, -1)
    clean_flat = clean.view(B, -1)
    recon_deg_flat = recon_degraded.view(B, -1)
    degraded_flat = degraded.view(B, -1)

    A_clean = F.cosine_similarity(recon_clean_flat, clean_flat, dim=1, eps=eps)
    A_degraded = F.cosine_similarity(recon_deg_flat, degraded_flat, dim=1, eps=eps)
    loss_per = F.relu(-(A_clean - A_degraded) + margin)

    return loss_per.mean()


class PretrainSADTrainer:
    def __init__(
            self, model, feature_extractor, optimizer, scheduler,
            train_loader, val_loader, work_dir, device, total_epochs, save_every
    ):
        self.model = model.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.work_dir = work_dir
        self.device = device
        self.total_epochs = total_epochs
        self.save_every = save_every
        self.start_epoch = 0
        os.makedirs(self.work_dir, exist_ok=True)

    def load_checkpoint(self, ckpt_path: str):
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Loaded checkpoint '{ckpt_path}' (epoch {self.start_epoch})")
        else:
            print(f"No checkpoint found at '{ckpt_path}'")

    def save_checkpoint(self, epoch: int):
        save_dir = os.path.join(self.work_dir, "checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
        }
        if epoch % 10 == 0 or epoch == self.total_epochs - 1:
            path = os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save(ckpt, path)
            print(f"Checkpoint saved to {path}")

    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.total_epochs):
            total_rec_loss = 0.0
            total_sr_loss = 0.0
            for deg, clean in self.train_loader:
                deg, clean = deg.to(self.device), clean.to(self.device)
                clean_feat = self.feature_extractor(clean)
                deg_feat = self.feature_extractor(deg)
                clean_recon = self.model(clean_feat)
                deg_recon = self.model(deg_feat)

                # 计算损失
                loss_rec = F.l1_loss(clean_recon, clean) + 0.1 * F.l1_loss(deg_recon, deg)
                loss_sr = similarity_ranking_loss(clean_recon, deg_recon, clean, deg)
                loss = loss_rec + loss_sr

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_rec_loss += loss_rec.item()
                total_sr_loss += loss_sr.item()
            self.scheduler.step()

            avg_rec = total_rec_loss / len(self.train_loader)
            avg_sr = total_sr_loss / len(self.train_loader)
            print(
                f"Epoch {epoch}: total_loss={(avg_rec + avg_sr):.4f} Lrec={avg_rec:.4f}, Lsr={avg_sr:.4f}, LR={self.scheduler.get_last_lr()[0]:.6f}")

            if epoch % self.save_every == 0 or epoch == self.total_epochs:
                self.save_checkpoint(epoch)
                self.validate(epoch)

    def validate(self, epoch, num_images=4):
        self.model.eval()
        val_loss = 0.0
        vis_dir = os.path.join(self.work_dir, 'val_images')
        os.makedirs(vis_dir, exist_ok=True)
        with torch.no_grad():
            for idx, (deg, clean) in enumerate(self.val_loader):
                deg, clean = deg.to(self.device), clean.to(self.device)
                clean_feat = self.feature_extractor(clean)
                deg_feat = self.feature_extractor(deg)
                clean_recon = self.model(clean_feat)
                deg_recon = self.model(deg_feat)

                # 计算损失
                loss_rec = F.l1_loss(clean_recon, clean) + 0.1 * F.l1_loss(deg_recon, deg)
                loss_sr = similarity_ranking_loss(clean_recon, deg_recon, clean, deg)
                loss = loss_rec + loss_sr
                val_loss += loss.item()

                # 前 num_images 张图像可视化保存
                if idx < num_images:
                    grid = torch.cat([clean, clean_recon], dim=0)
                    save_image(grid, os.path.join(vis_dir, f'epoch{epoch}_idx{idx}.png'), nrow=clean.size(0))
            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {epoch}: Val Loss={avg_val_loss:.4f}")

    def test(self, feat):
        self.model.eval()
        with torch.no_grad():
            feat = feat.to(self.device)
            feat_recon = self.model(feat)

        return feat_recon
