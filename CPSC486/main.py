import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from scipy import linalg


def linear_beta_schedule(T):
    return torch.linspace(1e-4, 2e-2, T)

class DiffusionConfig:
    def __init__(self, T=200, device='cuda'):
        self.device = device
        betas = linear_beta_schedule(T).to(device)
        alphas = 1 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alphas_cum = alphas_cum
        self.sqrt_ac = torch.sqrt(alphas_cum)
        self.sqrt_omac = torch.sqrt(1 - alphas_cum)
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self.sqrt_ac[t].view(-1,1,1,1) * x0 +
            self.sqrt_omac[t].view(-1,1,1,1) * noise
        )

class UNet(nn.Module):
    def __init__(self, ch=1, base=64):
        super().__init__()
        self.enc1 = nn.Conv2d(ch, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base*2, 3, padding=1)
        self.mid  = nn.Conv2d(base*2, base*2, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(base*2, base, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Conv2d(base, ch, 3, padding=1)
        self.time_emb = nn.Linear(1, base*2)
    def forward(self, x, t):
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(F.avg_pool2d(h1,2)))
        te = self.time_emb(t.view(-1,1))[:,:,None,None]
        h2 = h2 + te
        h_mid = F.relu(self.mid(h2))
        u2 = F.relu(self.dec2(h_mid))
        return self.dec1(u2 + h1)

class Classifier(nn.Module):
    def __init__(self, classes=10, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base*2, classes)
        )
    def forward(self, x, t):
        return self.net(x)

def train_diff(model, loader, cfg, opt, device):
    mse = nn.MSELoss()
    model.train()
    for x, _ in tqdm(loader, desc="Diffusion Train", leave=False):
        x = x.to(device)
        b = x.size(0)
        t = torch.randint(0, cfg.T, (b,), device=device)
        noise = torch.randn_like(x)
        x_noisy = cfg.q_sample(x, t, noise)
        pred = model(x_noisy, t.float()/cfg.T)
        loss = mse(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()

def train_clf(clf, loader, opt, device):
    ce = nn.CrossEntropyLoss()
    clf.train()
    for x, y in tqdm(loader, desc="Classifier Train", leave=False):
        x, y = x.to(device), y.to(device)
        logits = clf(x, torch.zeros(x.size(0),device=device))
        loss = ce(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def sample_uncond(model, cfg, device, n=128):
    x = torch.randn(n,1,28,28,device=device)
    for i in reversed(range(cfg.T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = model(x, t.float()/cfg.T)
        α, α_bar, β = cfg.alphas[i], cfg.alphas_cum[i], cfg.betas[i]
        x = (1/α.sqrt())*(x - (β/(1-α_bar).sqrt())*eps)
        if i>0: x += β.sqrt()*torch.randn_like(x)
    return x.clamp(-1,1)

def sample_guided(model, clf, cfg, device, n=128, scale=2.0):
    x = torch.randn(n,1,28,28,device=device)
    for i in reversed(range(cfg.T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, t.float()/cfg.T)
        x_in = x.clone().detach().requires_grad_(True)
        logits = clf(x_in, t.float()/cfg.T)
        lp = F.log_softmax(logits, dim=1)
        topc = lp.argmax(dim=1)
        sel = lp[torch.arange(n), topc].sum()
        grad = torch.autograd.grad(sel, x_in)[0]
        eps = eps - scale * grad
        α, α_bar, β = cfg.alphas[i], cfg.alphas_cum[i], cfg.betas[i]
        x = (1/α.sqrt())*(x - (β/(1-α_bar).sqrt())*eps)
        if i>0: x += β.sqrt()*torch.randn_like(x)
    return x.clamp(-1,1)

def get_inception_features(images, model, device):
    model.eval()
    imgs = images.repeat(1,3,1,1)  # 1->3 channel
    imgs = F.interpolate(imgs, size=(299,299), mode='bilinear')
    with torch.no_grad():
        feats = model(imgs.to(device)).detach()
    return feats.cpu()

def frechet_distance(mu1, sigma1, mu2, sigma2):
    mu1, mu2 = mu1.cpu().numpy(), mu2.cpu().numpy()
    sigma1, sigma2 = sigma1.cpu().numpy(), sigma2.cpu().numpy()

    diff = mu1 - mu2

    # Matrix square root of product
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Handle imaginary components from numerical error
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)

def compute_fid_manual(real_imgs, fake_imgs, device):
    # prepare pretrained inception (without top layer)
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()

    real_feats = get_inception_features(real_imgs, inception, device)
    fake_feats = get_inception_features(fake_imgs, inception, device)

    mu1, sigma1 = real_feats.mean(0), torch.cov(real_feats.T)
    mu2, sigma2 = fake_feats.mean(0), torch.cov(fake_feats.T)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    ds = datasets.MNIST('data', train=True, download=True, transform=tf)
    subset = Subset(ds, list(range(1000)))  # smaller dataset for speed
    loader = DataLoader(subset, batch_size=128, shuffle=True)

    cfg = DiffusionConfig(T=100, device=device)
    ddpm = UNet(ch=1).to(device)
    clf  = Classifier().to(device)
    opt_d = optim.Adam(ddpm.parameters(), lr=1e-4)
    opt_c = optim.Adam(clf.parameters(), lr=2e-4)

    print("Training small diffusion and classifier...")
    for epoch in range(1):
        train_diff(ddpm, loader, cfg, opt_d, device)
        train_clf(clf, loader, opt_c, device)

    os.makedirs('samples', exist_ok=True)

    print("Generating unconditional samples...")
    uimgs = sample_uncond(ddpm, cfg, device, n=128)
    utils.save_image((uimgs[:64]+1)/2, 'samples/unconditional.png', nrow=8)

    print("Comparing guided scales...")
    scales = [0.5, 1.0, 2.0]
    fid_results = {}
    real_batch, _ = next(iter(loader))
    real_batch = real_batch.to(device)[:128]

    for s in scales:
        print(f"Scale={s}")
        gimgs = sample_guided(ddpm, clf, cfg, device, n=128, scale=s)
        utils.save_image((gimgs[:64]+1)/2, f'samples/guided_scale{s}.png', nrow=8)
        fid_val = compute_fid_manual(real_batch, gimgs, device)
        fid_results[s] = fid_val
        print(f"  Manual FID ≈ {fid_val:.2f}")

    print("\nManual FID Comparison:")
    fid_uncond = compute_fid_manual(real_batch, uimgs, device)
    print(f"  Unconditional: {fid_uncond:.2f}")
    for s, v in fid_results.items():
        print(f"  Guided scale={s}: {v:.2f}")

if __name__ == '__main__':
    main()