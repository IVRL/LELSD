import contextlib

import numpy as np
import torch
from PIL import Image


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


class StyleGAN1SampleGenerator:
    """
    Wrapper class for generating images and intermediate features
    from the given StyleGAN generator.
    Parameters
    ----------
    G: Pytorch Module,
        Trained StyleGAN Generator model as a pytorch module.
    device: torch device used for the computation
    res: int, default=32
        Resolution at which the feature maps are evaluated.
    truncation_psi : float in [0, 1], default=0.7
        truncation value used for generating w from the mapping network
    noise_mode : {'fixed', 'random', 'zeros'}, default='fixed'
        The noise mode for generating samples in StyleGAN
    batch_size: int, default=8
        Number of samples in each generated batch
    only_w: bool, default=False
        Whether to return only w or not. If you only need w setting
        this attribute  makes the computation faster.
    return_image: bool, default=False,
        Whether to return image or not. If you don't need the generated
        image set this attribute to False to save memory
    feature_type: {'normalized', 'stylized'}, default=normalized
        Either to return normalized feature maps
        or the stylized feature maps (after AdaIN)
    return_all_layers: bool, default=False,
        Whether to return feature maps for all  of the layers or not.
        If you don't need all of the generated feature maps this
        attribute to False and to save memory
    """
    resolution_to_feature_dim = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256,
        128: 128,
        256: 64,
        512: 32,
        1024: 16,
    }

    layer_to_resolution = {
        0: 4, 1: 4,
        2: 8, 3: 8,
        4: 16, 5: 16,
        6: 32, 7: 32,
        8: 64, 9: 64,
        10: 128, 11: 128,
        12: 256, 13: 256,
        14: 512, 15: 512,
        16: 1024, 17: 1024,

    }

    def __init__(self, G, device, truncation_psi=0.7,
                 noise_mode='fixed', batch_size=8,
                 only_w=False, return_image=False,
                 feature_type='normalized',
                 return_all_layers=False):

        assert noise_mode in ['random', 'fixed', 'zeros']
        assert feature_type in ['normalized', 'stylized']
        assert 0.0 <= truncation_psi <= 1.0

        self.G = G.to(device)
        self.n_layers = 2 * len(self.G.g_synthesis.blocks.items())
        self.num_ws = len(self.G.AdaIN_layers)
        self.device = device
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        if self.noise_mode == 'random':
            self.noise_mode = None
        self.batch_size = batch_size
        self.only_w = only_w
        self.return_image = return_image
        self.feature_type = feature_type
        self.return_all_layers = return_all_layers

        self.G.g_synthesis.set_noise(mode=self.noise_mode)

    def ys_to_batch_ys(self, ys):
        return [[ys[l][b] for l in range(len(ys))] for b in range(len(ys[0]))]

    def batch_ys_to_ys(self, batch_ys):
        return [torch.stack([batch_ys[b][l] for b in range(len(batch_ys))]) for l in range(len(batch_ys[0]))]

    def generate_image_from_batch_ys(self, batch_ys, raw=False):
        ys = self.batch_ys_to_ys(batch_ys)
        return self.generate_image_from_ys(ys, raw=raw)

    @torch.no_grad()
    def generate_image_from_ys(self, ys, raw=False):
        raw_img = self.G.ys_to_rgb(ys)
        if raw:
            return raw_img
        return self.raw_image_to_pil_image(raw_img)

    def generate_image_from_z(self, z, raw=False):
        ys = self.G.z_to_ys(z, truncation=self.truncation_psi)
        return self.generate_image_from_ys(ys, raw=raw)

    def generate_image_from_ws(self, ws, raw=False):
        ys = self.G.w_to_ys(ws)
        return self.generate_image_from_ys(ys, raw=raw)

    @torch.no_grad()
    def raw_image_to_pil_image(self, raw_img):
        img = (raw_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_imgs = []
        for i in range(len(img)):
            pil_imgs.append(Image.fromarray(img[i].cpu().numpy(), 'RGB'))
        return pil_imgs

    def generate_batch_from_ys(self, ys, return_image=None, return_all_layers=None, max_resolution=1024,
                               requires_grad=False):
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            return_image = self.return_image if return_image is None else return_image
            return_all_layers = self.return_all_layers if return_all_layers is None else return_all_layers
            result = {}
            if self.only_w:
                return result

            L = -1
            for i, (res, m) in enumerate(self.G.g_synthesis.blocks.items()):
                resolution = int(res.split("x")[0])
                if i == 0:
                    style1 = ys[2 * i]
                    style2 = ys[2 * i + 1]
                    epi1 = m.epi1
                    epi2 = m.epi2
                    batch_size = style2.size(0)

                    x1 = m.const.expand(batch_size, -1, -1, -1)
                    x1 = x1 + m.bias.view(1, -1, 1, 1)
                    x1_normalized = epi1.top_epi(x1)
                    x1_stylized = x1_normalized * (style1[:, 0] + 1.) + style1[:, 1]
                    L += 1

                    x2 = m.conv(x1_stylized)
                    x2_normalized = epi2.top_epi(x2)
                    x2_stylized = x2_normalized * (style2[:, 0] + 1.) + style2[:, 1]
                    x = x2_stylized
                    L += 1
                else:
                    style1 = ys[2 * i]
                    style2 = ys[2 * i + 1]
                    epi1 = m.epi1
                    epi2 = m.epi2

                    x1 = m.conv0_up(x)
                    x1_normalized = epi1.top_epi(x1)
                    x1_stylized = x1_normalized * (style1[:, 0] + 1.) + style1[:, 1]
                    L += 1

                    x2 = m.conv1(x1_stylized)
                    x2_normalized = epi2.top_epi(x2)
                    x2_stylized = x2_normalized * (style2[:, 0] + 1.) + style2[:, 1]
                    x = x2_stylized
                    L += 1

                if return_all_layers:
                    if not return_image and resolution > max_resolution:
                        break
                    result[f'layer_{L - 1}'] = x1_normalized if self.feature_type == 'normalized' else x1_stylized
                    result[f'layer_{L}'] = x2_normalized if self.feature_type == 'normalized' else x2_stylized

            if return_image:
                raw_img = self.G.g_synthesis.torgb(x)
                pil_imgs = self.raw_image_to_pil_image(raw_img)
                result['raw_image'] = raw_img.clamp(-1.0, 1.0)
                result['image'] = pil_imgs

        return result

    def generate_batch_from_ws(self, ws, return_image=None, return_all_layers=None, return_style=True,
                               max_resolution=1024, requires_grad=False):
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            return_image = self.return_image if return_image is None else return_image
            return_all_layers = self.return_all_layers if return_all_layers is None else return_all_layers
            ys = self.G.w_to_ys(ws)
            result = self.generate_batch_from_ys(ys, return_image, return_all_layers, max_resolution, requires_grad)
            if return_style:
                batch_ys = self.ys_to_batch_ys(ys)
                result['batch_ys'] = batch_ys
                result['ys'] = ys
                result = {'batch_ys': batch_ys, 'ys': ys}

            return result

    def generate_batch_from_z(self, z, return_image=None, return_all_layers=None, return_style=True,
                              max_resolution=1024, requires_grad=False):
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            ws = self.G.z_to_w(z, truncation=self.truncation_psi)[:, :self.num_ws]
            result = self.generate_batch_from_ws(ws, return_image, return_all_layers, return_style, max_resolution,
                                                 requires_grad)
        if return_style:
            result['ws'] = ws
        return result

    def generate_batch(self, seed, batch_size=None, return_image=None, return_all_layers=None, return_style=True,
                       max_resolution=1024, requires_grad=False):
        """
        Paramteres
        ----------
        seed: int, The random seed used for generating samples.
        batch_size: int, Batch size of the generated samples.
            If not specified, the class attribute will be used.
        return_image: bool, Whether to return the generated images or not.
            If not specified, the class attribute will be used.
        return_all_layers: bool, Whether to return the generated images or not.
            If not specified, the class attribute will be used.
        return_style: bool, default=True Whether to return the style vectors (w, z, y)
            for the generated samples or not.
        max_resolution: int, default=1024, This is only important when return_all_layers is
            set to True. This resolution determines the last resolution to be returned. If you
            only need the layers up to a specific resolution setting this to True can speed up
            the computation.
        requires_grad: bool, default=False
            Whether to compute the result in no grad mode or not.
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, 512).astype(np.float32)).to(self.device)
            result = self.generate_batch_from_z(z, return_image, return_all_layers, return_style, max_resolution,
                                                requires_grad)
        if return_style:
            result['z'] = z
        return result

    def zero_grad(self):
        self.G.zero_grad()

