import torch
import torchvision

from ADM_Unet import Unet, time_embedding

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n=20, sigma_min=0.002, sigma_max=80, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

class EDM:
    def __init__(self, f_sigma=get_sigmas_karras, *args, **kwargs) -> None:
        self.sigmas = f_sigma(*args, **kwargs)
        self.sigma_data = 0.5
        self.s = 1
        self.P_mean = -1.2
        self.P_std = 1.2

        self.unet = Unet()
        self.time_embd = time_embedding(dim_emd=512, dim_map=768)
        # pass


    def plot_sigma(self, sigmas=None):
        import matplotlib.pyplot as plt
        plt.plot(sigmas)
        plt.show()

    def add_noise(self, y: torch.Tensor) -> torch.Tensor:

        batch = y.shape[0] #B, C, H, W
        sigma_train = (torch.randn(batch) * self.P_std + self.P_mean).exp()
        # self.plot_sigma(sigma_train.squeeze().sort()[0])

        return y + torch.randn_like(y)*sigma_train.view(-1, 1, 1, 1), sigma_train
    
    def compute_loss(self, y: torch.Tensor) -> torch.Tensor:

        x, sigmas = self.add_noise(y) 
        c_skip = self.sigma_data **2 / (self.sigma_data **2 + sigmas **2)
        c_out = sigmas * self.sigma_data / (self.sigma_data **2 + sigmas **2) **0.5
        c_in = 1 / (self.sigma_data **2 + sigmas **2) **0.5
        c_noise = 0.25 * torch.log2(sigmas)


        print(c_skip.shape)
    

if __name__ == '__main__':
    dataloader = torchvision.datasets.ImageFolder(root='./images', transform=torchvision.transforms.ToTensor())
    edm = EDM()
    x = torch.randn(10, 3, 28, 28)
    edm.compute_loss(x)
    # edm.plot_sigma()