import contextlib

import numpy as np
import torch
from PIL import Image

from models.stylegan3.torch_utils.ops import bias_act
from models.stylegan3.torch_utils.ops import upfirdn2d
from models.stylegan3.training.networks_stylegan2 import modulated_conv2d


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


class StyleGAN2SampleGenerator:
    """
    Wrapper class for generating images and intermediate features
    from the given StyleGAN2 generator.
    Parameters
    ----------
    G: Pytorch Module,
        Trained StyleGAN2 Generator model as a pytorch module.
    device: torch device used for the computation
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
        64: 512,
        128: 256,
        256: 128,
        512: 64,
        1024: 32,
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

    layer_to_rgb_style = {
        1: 0,
        3: 1,
        5: 2,
        7: 3,
        9: 4,
        11: 5,
        13: 6,
        15: 7,
        17: 8,
    }

    def __init__(self, G, device,
                 truncation_psi=0.7,
                 noise_mode='const', batch_size=8,
                 only_w=False, return_image=False,
                 return_all_layers=False):

        assert noise_mode in ['random', 'const', 'none']
        assert 0.0 <= truncation_psi <= 1.0

        self.G = G.to(device)
        self.n_layers = 2 * len(self.G.synthesis.block_resolutions)
        self.device = device
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.only_w = only_w
        self.return_image = return_image
        self.return_all_layers = return_all_layers

    def ys_to_s(self, ys, rgb_ys):
        s = []
        ys_iter = iter(ys)
        rgb_ys_iter = iter(rgb_ys)
        s.append(next(ys_iter))
        s.append(next(rgb_ys_iter))
        while True:
            try:
                s.append(next(ys_iter))
                s.append(next(ys_iter))
                s.append(next(rgb_ys_iter))
            except StopIteration:
                break
        return s

    def s_to_ys(self, s):
        ys = []
        rgb_ys = []
        for i in [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26]:
            if i < len(s):
                ys.append(s[i])
        for i in [1, 4, 7, 10, 13, 16, 19, 22, 25]:
            if i < len(s):
                rgb_ys.append(s[i])

        return ys, rgb_ys

    def ys_to_batch_ys(self, ys):
        return [[ys[l][b] for l in range(len(ys))] for b in range(len(ys[0]))]

    def batch_ys_to_ys(self, batch_ys):
        return [torch.stack([batch_ys[b][l] for b in range(len(batch_ys))]) for l in range(len(batch_ys[0]))]

    def ws_to_ys(self, ws):
        block_ws = []
        w_idx = 0
        for res in self.G.synthesis.block_resolutions:
            block = getattr(self.G.synthesis, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

        ys = []
        rgb_ys = []
        for i, (res, cur_ws) in enumerate(zip(self.G.synthesis.block_resolutions, block_ws)):
            synthesis_block = getattr(self.G.synthesis, f'b{res}')
            cur_ws_iter = iter(cur_ws.unbind(dim=1))
            # Input Block
            if i == 0:
                y = synthesis_block.conv1.affine(next(cur_ws_iter))
                ys.append(y)
                rgb_y = synthesis_block.torgb.affine(next(cur_ws_iter)) * synthesis_block.torgb.weight_gain
                rgb_ys.append(rgb_y)
            else:
                y = synthesis_block.conv0.affine(next(cur_ws_iter))
                ys.append(y)
                y = synthesis_block.conv1.affine(next(cur_ws_iter))
                ys.append(y)
                rgb_y = synthesis_block.torgb.affine(next(cur_ws_iter)) * synthesis_block.torgb.weight_gain
                rgb_ys.append(rgb_y)

        # The last layer will get stylized by the rgb style
        ys.append(rgb_ys[-1])
        return ys, rgb_ys

    def ys_to_ws(self, ys):
        """
        Parameters
        ----------
        ys: List of tensors
        """

        def get_inverse_affine(affine_layer, y, add_bias=True):
            A = affine_layer.weight * affine_layer.weight_gain
            if A.shape[0] == A.shape[1]:
                A_inverse = torch.inverse(A)
            else:
                # Use Pseudo Inverse
                A_inverse = torch.pinverse(A)
            B = affine_layer.bias * affine_layer.bias_gain
            y_batch_last = y.permute(1, 0)
            if add_bias:
                w = torch.matmul(A_inverse, y_batch_last - B[:, None])
            else:
                w = torch.matmul(A_inverse, y_batch_last)
            return w.permute(1, 0)

        ws = []
        L = 0
        for i, res in enumerate(self.G.synthesis.block_resolutions):
            synthesis_block = getattr(self.G.synthesis, f'b{res}')
            # Input Block
            if i == 0:
                affine_layer = synthesis_block.conv1.affine
                w = get_inverse_affine(affine_layer, ys[L], add_bias=True)
                ws.append(w)
                L += 1
            else:
                affine_layer = synthesis_block.conv0.affine
                w = get_inverse_affine(affine_layer, ys[L], add_bias=True)
                ws.append(w)
                affine_layer = synthesis_block.conv1.affine
                w = get_inverse_affine(affine_layer, ys[L + 1], add_bias=True)
                ws.append(w)
                L += 2

        # The last style uses torgb layer's affine transformation
        affine_layer = synthesis_block.torgb.affine
        w = get_inverse_affine(affine_layer, ys[-1] / synthesis_block.torgb.weight_gain, add_bias=True)
        ws.append(w)

        return torch.stack(ws, dim=1)

    def raw_image_to_pil_image(self, raw_img):
        img = (raw_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_imgs = []
        for i in range(len(img)):
            pil_imgs.append(Image.fromarray(img[i].cpu().numpy(), 'RGB'))
        return pil_imgs

    def generate_image_from_ws(self, ws, raw=False):
        batch_size = len(ws)
        img = self.G.synthesis(ws, noise_mode=self.noise_mode)
        if raw:
            return img
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs = []
        for i in range(batch_size):
            imgs.append(Image.fromarray(img[i].cpu().numpy(), 'RGB'))
        return imgs

    def generate_image_from_ys(self, ys, rgb_ys, raw=False):
        def rgb_layer_forward(rgb_layer, x, rgb_y):
            x = modulated_conv2d(x=x, weight=rgb_layer.weight, styles=rgb_y, demodulate=False, fused_modconv=True)
            x = bias_act.bias_act(x, rgb_layer.bias.to(x.dtype), clamp=rgb_layer.conv_clamp)
            return x

        def synthesis_layer_forward(synthesis_layer, x, y, gain=1.0):
            in_resolution = synthesis_layer.resolution // synthesis_layer.up
            noise = None
            if synthesis_layer.use_noise and self.noise_mode == 'random':
                noise = torch.randn([x.shape[0], 1, synthesis_layer.resolution, synthesis_layer.resolution],
                                    device=x.device) * synthesis_layer.noise_strength
            if synthesis_layer.use_noise and self.noise_mode == 'const':
                noise = synthesis_layer.noise_const * synthesis_layer.noise_strength

            flip_weight = (synthesis_layer.up == 1)  # slightly faster
            x = modulated_conv2d(x=x, weight=synthesis_layer.weight, styles=y, noise=noise, up=synthesis_layer.up,
                                 padding=synthesis_layer.padding, resample_filter=synthesis_layer.resample_filter,
                                 flip_weight=flip_weight, fused_modconv=True)

            act_gain = synthesis_layer.act_gain * gain
            act_clamp = synthesis_layer.conv_clamp * gain if synthesis_layer.conv_clamp is not None else None
            x = bias_act.bias_act(x, synthesis_layer.bias.to(x.dtype), act=synthesis_layer.activation,
                                  gain=act_gain, clamp=act_clamp)
            return x

        x = img = None
        ys_iter = iter(ys)
        rgb_ys_iter = iter(rgb_ys)
        for i, res in enumerate(self.G.synthesis.block_resolutions):
            synthesis_block = getattr(self.G.synthesis, f'b{res}')
            if i == 0:
                x = synthesis_block.const.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                x = x.unsqueeze(0).repeat([len(ys[0]), 1, 1, 1])
                x = synthesis_layer_forward(synthesis_block.conv1, x, next(ys_iter))

            else:
                x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                if synthesis_block.architecture == 'resnet':
                    y = synthesis_block.skip(x, gain=np.sqrt(0.5))
                    x = synthesis_layer_forward(synthesis_block.conv0, x, next(ys_iter))
                    x = synthesis_layer_forward(synthesis_block.conv1, x, next(ys_iter), gain=np.sqrt(0.5))
                    x = y.add_(x)
                else:
                    x = synthesis_layer_forward(synthesis_block.conv0, x, next(ys_iter))
                    x = synthesis_layer_forward(synthesis_block.conv1, x, next(ys_iter))

            if img is not None:
                img = upfirdn2d.upsample2d(img, synthesis_block.resample_filter)
            if synthesis_block.is_last or synthesis_block.architecture == 'skip':
                y = rgb_layer_forward(synthesis_block.torgb, x, next(rgb_ys_iter))
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                img = img.add_(y) if img is not None else y

        if raw:
            return img

        pil_imgs = self.raw_image_to_pil_image(img)
        return pil_imgs

    def generate_image_from_batch_ys(self, batch_ys, batch_rgb_ys, raw=False):
        ys = self.batch_ys_to_ys(batch_ys)
        rgb_ys = self.batch_ys_to_ys(batch_rgb_ys)
        return self.generate_image_from_ys(ys, rgb_ys, raw)

    def generate_batch_from_ys(self, ys_tuple, return_image=None, return_all_layers=None, max_resolution=1024,
                               requires_grad=False):
        def rgb_layer_forward(rgb_layer, x, rgb_y):
            x = modulated_conv2d(x=x, weight=rgb_layer.weight, styles=rgb_y, demodulate=False, fused_modconv=True)
            x = bias_act.bias_act(x, rgb_layer.bias.to(x.dtype), clamp=rgb_layer.conv_clamp)
            return x

        def synthesis_layer_forward(synthesis_layer, x, y, gain=1.0):
            in_resolution = synthesis_layer.resolution // synthesis_layer.up
            noise = None
            if synthesis_layer.use_noise and self.noise_mode == 'random':
                noise = torch.randn([x.shape[0], 1, synthesis_layer.resolution, synthesis_layer.resolution],
                                    device=x.device) * synthesis_layer.noise_strength
            if synthesis_layer.use_noise and self.noise_mode == 'const':
                noise = synthesis_layer.noise_const * synthesis_layer.noise_strength

            flip_weight = (synthesis_layer.up == 1)  # slightly faster
            x = modulated_conv2d(x=x, weight=synthesis_layer.weight, styles=y, noise=noise, up=synthesis_layer.up,
                                 padding=synthesis_layer.padding, resample_filter=synthesis_layer.resample_filter,
                                 flip_weight=flip_weight, fused_modconv=True)

            act_gain = synthesis_layer.act_gain * gain
            act_clamp = synthesis_layer.conv_clamp * gain if synthesis_layer.conv_clamp is not None else None
            x = bias_act.bias_act(x, synthesis_layer.bias.to(x.dtype), act=synthesis_layer.activation,
                                  gain=act_gain, clamp=act_clamp)
            return x

        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            return_image = self.return_image if return_image is None else return_image
            return_all_layers = self.return_all_layers if return_all_layers is None else return_all_layers
            if return_image:
                max_resolution = 1024

            ys, rgb_ys = ys_tuple

            result = {}
            if self.only_w:
                return result

            x = img = None
            ys_iter = iter(ys)
            rgb_ys_iter = iter(rgb_ys)
            L = 0
            for i, res in enumerate(self.G.synthesis.block_resolutions):
                synthesis_block = getattr(self.G.synthesis, f'b{res}')
                if i == 0:
                    x1 = synthesis_block.const.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                    x1 = x1.unsqueeze(0).repeat([len(ys[0]), 1, 1, 1])
                    x2 = synthesis_layer_forward(synthesis_block.conv1, x1, next(ys_iter))
                    x = x2
                    if return_all_layers:
                        result[f'layer_{L}'] = x1
                        result[f'layer_{L + 1}'] = x2
                    L += 2

                else:
                    x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                    if synthesis_block.architecture == 'resnet':
                        y = synthesis_block.skip(x, gain=np.sqrt(0.5))
                        x1 = synthesis_layer_forward(synthesis_block.conv0, x, next(ys_iter))
                        x2 = synthesis_layer_forward(synthesis_block.conv1, x1, next(ys_iter), gain=np.sqrt(0.5))
                        x2 = y.add_(x2)
                        x = x2
                        if return_all_layers:
                            result[f'layer_{L}'] = x1
                            result[f'layer_{L + 1}'] = x2
                        L += 2
                    else:
                        x1 = synthesis_layer_forward(synthesis_block.conv0, x, next(ys_iter))
                        x2 = synthesis_layer_forward(synthesis_block.conv1, x1, next(ys_iter))
                        x = x2
                        if return_all_layers:
                            result[f'layer_{L}'] = x1
                            result[f'layer_{L + 1}'] = x2
                        L += 2

                if return_image:
                    if img is not None:
                        img = upfirdn2d.upsample2d(img, synthesis_block.resample_filter)
                    if synthesis_block.is_last or synthesis_block.architecture == 'skip':
                        y = rgb_layer_forward(synthesis_block.torgb, x, next(rgb_ys_iter))
                        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                        img = img.add_(y) if img is not None else y

                if res > max_resolution:
                    break

            if return_image:
                pil_imgs = self.raw_image_to_pil_image(img)
                result['raw_image'] = img.clamp(-1.0, 1.0)
                result['image'] = pil_imgs

        return result

    def generate_batch_from_ws(self, ws, return_image=None, return_all_layers=None, return_style=True,
                               max_resolution=1024, requires_grad=False):
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            return_image = self.return_image if return_image is None else return_image
            return_all_layers = self.return_all_layers if return_all_layers is None else return_all_layers
            if return_image:
                max_resolution = 1024

            block_ws = []
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.G.synthesis.block_resolutions:
                block = getattr(self.G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

            ys, rgb_ys = self.ws_to_ys(ws)
            s = self.ys_to_s(ys, rgb_ys)
            batch_ys = self.ys_to_batch_ys(ys)
            batch_rgb_ys = self.ys_to_batch_ys(rgb_ys)
            result = self.generate_batch_from_ys((ys, rgb_ys), return_image, return_all_layers, max_resolution,
                                                 requires_grad)
            if return_style:
                result['ys'] = ys
                result['rgb_ys'] = rgb_ys
                result['batch_ys'] = batch_ys
                result['batch_rgb_ys'] = batch_rgb_ys
                result['s'] = s

            # L = 0
            # x = img = None
            # for i, (res, cur_ws) in enumerate(zip(self.G.synthesis.block_resolutions, block_ws)):
            #     synthesis_block = getattr(self.G.synthesis, f'b{res}')
            #     cur_ws_iter = iter(cur_ws.unbind(dim=1))
            #     # Input Block
            #     if i == 0:
            #         x1 = synthesis_block.const.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            #         x1 = x1.unsqueeze(0).repeat([cur_ws.shape[0], 1, 1, 1])
            #         x2 = synthesis_block.conv1(x1, next(cur_ws_iter), noise_mode=self.noise_mode)
            #         x = x2
            #         if return_all_layers:
            #             result[f'layer_{L}'] = x1
            #             result[f'layer_{L + 1}'] = x2
            #         L += 2
            #
            #     else:
            #         x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            #         if synthesis_block.architecture == 'resnet':
            #             y = synthesis_block.skip(x, gain=np.sqrt(0.5))
            #             x1 = synthesis_block.conv0(x, next(cur_ws_iter), noise_mode=self.noise_mode)
            #             x2 = synthesis_block.conv1(x1, next(cur_ws_iter), gain=np.sqrt(0.5), noise_mode=self.noise_mode)
            #             x2 = y.add_(x2)
            #             x = x2
            #             if return_all_layers:
            #                 result[f'layer_{L}'] = x1
            #                 result[f'layer_{L + 1}'] = x2
            #             L += 2
            #         else:
            #             x1 = synthesis_block.conv0(x, next(cur_ws_iter), noise_mode=self.noise_mode)
            #             x2 = synthesis_block.conv1(x1, next(cur_ws_iter), noise_mode=self.noise_mode)
            #             x = x2
            #             if return_all_layers:
            #                 result[f'layer_{L}'] = x1
            #                 result[f'layer_{L + 1}'] = x2
            #             L += 2
            #
            #     if return_image:
            #         if img is not None:
            #             img = upfirdn2d.upsample2d(img, synthesis_block.resample_filter)
            #         if synthesis_block.is_last or synthesis_block.architecture == 'skip':
            #             y = synthesis_block.torgb(x, next(cur_ws_iter))
            #             y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            #             img = img.add_(y) if img is not None else y
            #
            #     if res > max_resolution:
            #         break
            #
            # if return_image:
            #     pil_imgs = self.raw_image_to_pil_image(img)
            #     result['raw_image'] = img.clamp(-1.0, 1.0)
            #     result['image'] = pil_imgs

            return result

    def generate_batch_from_z(self, z, return_image=None, return_all_layers=None,
                              return_style=True,
                              max_resolution=1024, requires_grad=False):
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            label = torch.zeros([1, self.G.c_dim], device=self.device)
            ws = self.G.mapping(z, label, truncation_psi=self.truncation_psi)
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

    def generate_batch_from_seeds(self, seeds, return_image=None, return_all_layers=None, return_style=True,
                       max_resolution=1024, requires_grad=False):
        """
        Paramteres
        ----------
        seeds: list of int, The random seeds used for generating samples.
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
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            z = []
            for seed in seeds:
                z_seed = np.random.RandomState(seed).randn(1, 512).astype(np.float32)
                z.append(z_seed)

            z = np.concatenate(z, axis=0)
            z = torch.from_numpy(z).to(self.device)
            result = self.generate_batch_from_z(z, return_image, return_all_layers, return_style, max_resolution,
                                                requires_grad)
        if return_style:
            result['z'] = z
        return result

    def zero_grad(self):
        self.G.zero_grad()

