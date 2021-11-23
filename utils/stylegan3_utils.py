import contextlib

import numpy as np
import torch
from PIL import Image

from models.stylegan3.torch_utils.ops import filtered_lrelu
from models.stylegan3.training.networks_stylegan3 import modulated_conv2d


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


class StyleGAN3SampleGenerator:
    """
    Wrapper class for generating images and intermediate features
    from the given StyleGAN2 generator.
    Parameters
    ----------
    G: Pytorch Module,
        Trained StyleGAN3 Generator model as a pytorch module.
    device: torch device used for the computation
    truncation_psi : float in [0, 1], default=0.7
        truncation value used for generating w from the mapping network
    truncation_cutoff : int, default=None
        This determines the last layer to apply truncation on W.
        If None, then truncation is applied to all layers.
    noise_mode : {'const', 'random', 'none'}, default='const'
        The noise mode for generating samples in StyleGAN
    force_fp32 : bool, default=False
        Force 32bit floating precision, results in slower
        but more precise computation
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

    def __init__(self, G, device,
                 truncation_psi=0.7, truncation_cutoff=None,
                 noise_mode='const', force_fp32=False, batch_size=8,
                 only_w=False, return_image=False,
                 return_all_layers=False):

        assert noise_mode in ['random', 'const', 'none']
        assert 0.0 <= truncation_psi <= 1.0

        self.G = G.to(device)
        self.n_layers = 1 + len(self.G.synthesis.layer_names)
        self.device = device
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.noise_mode = noise_mode
        self.force_fp32 = force_fp32
        self.batch_size = batch_size
        self.only_w = only_w
        self.return_image = return_image
        self.return_all_layers = return_all_layers

        def isPowerOfTwo(x):
            return (x and (not (x & (x - 1))))

        self.layer_to_resolution = {0: G.synthesis.input.size[0] - G.synthesis.margin_size * 2}
        self.layer_to_spatial_size = {0: G.synthesis.input.size[0]}
        self.layer_to_feature_dim = {0: G.synthesis.input.channels}
        for l, layer_name in enumerate(G.synthesis.layer_names):
            layer_spatial_size = int(layer_name.split("_")[1])
            self.layer_to_spatial_size[l + 1] = layer_spatial_size
            if isPowerOfTwo(layer_spatial_size):
                self.layer_to_resolution[l + 1] = layer_spatial_size
            else:
                self.layer_to_resolution[l + 1] = layer_spatial_size - G.synthesis.margin_size * 2

            self.layer_to_feature_dim[l + 1] = int(layer_name.split("_")[2])

    def ys_to_batch_ys(self, ys):
        return [[ys[l][b] for l in range(len(ys))] for b in range(len(ys[0]))]

    def batch_ys_to_ys(self, batch_ys):
        return [torch.stack([batch_ys[b][l] for b in range(len(batch_ys))]) for l in range(len(batch_ys[0]))]

    def ws_to_ys(self, ws):
        ys = []
        ws = ws.permute(1, 0, 2)
        ys.append(self.G.synthesis.input.affine(ws[0]))
        for name, w in zip(self.G.synthesis.layer_names, ws[1:]):
            y = getattr(self.G.synthesis, name).affine(w)
            ys.append(y)

        return ys

    def raw_image_to_pil_image(self, raw_img):
        img = (raw_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_imgs = []
        for i in range(len(img)):
            pil_imgs.append(Image.fromarray(img[i].cpu().numpy(), 'RGB'))
        return pil_imgs

    def generate_image_from_ws(self, ws, raw=False):
        raw_img = self.G.synthesis(ws, noise_mode=self.noise_mode, force_fp32=self.force_fp32)
        if raw:
            return raw_img

        return self.raw_image_to_pil_image(raw_img)

    def generate_image_from_ys(self, ys, raw=False):
        def input_layer_forward(input_layer, y):
            """
            Parameters
            ----------
            y: result of applying affine transformation of input layer on w[0]
            """
            transforms = input_layer.transform.unsqueeze(0)  # [batch, row, col]
            freqs = input_layer.freqs.unsqueeze(0)  # [batch, channel, xy]
            phases = input_layer.phases.unsqueeze(0)  # [batch, channel]

            # Apply learned transformation.
            t = y  # t = (r_c, r_s, t_x, t_y)
            t = t / t[:, :2].norm(dim=1, keepdim=True)  # t' = (r'_c, r'_s, t'_x, t'_y)
            m_r = torch.eye(3, device=y.device).unsqueeze(0).repeat(
                [y.shape[0], 1, 1])  # Inverse rotation wrt. resulting image.
            m_r[:, 0, 0] = t[:, 0]  # r'_c
            m_r[:, 0, 1] = -t[:, 1]  # r'_s
            m_r[:, 1, 0] = t[:, 1]  # r'_s
            m_r[:, 1, 1] = t[:, 0]  # r'_c
            m_t = torch.eye(3, device=y.device).unsqueeze(0).repeat(
                [y.shape[0], 1, 1])  # Inverse translation wrt. resulting image.
            m_t[:, 0, 2] = -t[:, 2]  # t'_x
            m_t[:, 1, 2] = -t[:, 3]  # t'_y
            transforms = m_r @ m_t @ transforms  # First rotate resulting image, then translate, and finally apply user-specified transform.

            # Transform frequencies.
            phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)  # The translation part
            freqs = freqs @ transforms[:, :2, :2]  # The rotation part

            # Dampen out-of-band frequencies that may occur due to the user-specified transform.
            amplitudes = (1 - (freqs.norm(dim=2) - input_layer.bandwidth) / (
                    input_layer.sampling_rate / 2 - input_layer.bandwidth)).clamp(0, 1)

            # Construct sampling grid.
            theta = torch.eye(2, 3, device=y.device)
            theta[0, 0] = 0.5 * input_layer.size[0] / input_layer.sampling_rate
            theta[1, 1] = 0.5 * input_layer.size[1] / input_layer.sampling_rate
            grids = torch.nn.functional.affine_grid(theta.unsqueeze(0),
                                                    [1, 1, input_layer.size[1], input_layer.size[0]],
                                                    align_corners=False)
            # Compute Fourier features.
            x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(
                3)  # [batch, height, width, channel]
            x = x + phases.unsqueeze(1).unsqueeze(2)
            x = torch.sin(x * (np.pi * 2))
            x = x * amplitudes.unsqueeze(1).unsqueeze(2)

            # Apply trainable mapping.
            weight = input_layer.weight / np.sqrt(input_layer.channels)
            x = x @ weight.t()

            # Ensure correct shape.
            x = x.permute(0, 3, 1, 2)  # [batch, channel, height, width]
            return x

        def synthesis_layer_forward(synthesis_layer, x, y):
            force_fp32 = self.force_fp32
            input_gain = synthesis_layer.magnitude_ema.rsqrt()

            # Execute affine layer.
            styles = y
            if synthesis_layer.is_torgb:
                weight_gain = 1 / np.sqrt(synthesis_layer.in_channels * (synthesis_layer.conv_kernel ** 2))
                styles = styles * weight_gain

            # Execute modulated conv2d.
            dtype = torch.float16 if (
                    synthesis_layer.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
            x = modulated_conv2d(x=x.to(dtype), w=synthesis_layer.weight, s=styles,
                                 padding=synthesis_layer.conv_kernel - 1, demodulate=(not synthesis_layer.is_torgb),
                                 input_gain=input_gain)

            # Execute bias, filtered leaky ReLU, and clamping.
            gain = 1 if synthesis_layer.is_torgb else np.sqrt(2)
            slope = 1 if synthesis_layer.is_torgb else 0.2
            x = filtered_lrelu.filtered_lrelu(x=x, fu=synthesis_layer.up_filter, fd=synthesis_layer.down_filter,
                                              b=synthesis_layer.bias.to(x.dtype),
                                              up=synthesis_layer.up_factor, down=synthesis_layer.down_factor,
                                              padding=synthesis_layer.padding, gain=gain,
                                              slope=slope, clamp=synthesis_layer.conv_clamp)

            # Ensure correct shape and dtype.
            return x

        x = input_layer_forward(self.G.synthesis.input, ys[0])
        for layer_name, y in zip(self.G.synthesis.layer_names, ys[1:]):
            synthesis_layer = getattr(self.G.synthesis, layer_name)
            x = synthesis_layer_forward(synthesis_layer, x, y)

        if self.G.synthesis.output_scale != 1:
            x = x * self.G.synthesis.output_scale

        img = x.to(torch.float32)
        if raw:
            return img.clamp(-1.0, 1.0)

        pil_imgs = self.raw_image_to_pil_image(img)
        return pil_imgs

    def generate_image_from_batch_ys(self, batch_ys, batch_rgb_ys, raw=False):
        ys = self.batch_ys_to_ys(batch_ys)
        return self.generate_image_from_ys(ys, raw)

    def generate_batch_from_ys(self, ys, return_image=None, return_all_layers=None, max_resolution=1024,
                               requires_grad=False):
        def input_layer_forward(input_layer, y):
            """
            Parameters
            ----------
            y: result of applying affine transformation of input layer on w[0]
            """
            transforms = input_layer.transform.unsqueeze(0)  # [batch, row, col]
            freqs = input_layer.freqs.unsqueeze(0)  # [batch, channel, xy]
            phases = input_layer.phases.unsqueeze(0)  # [batch, channel]

            # Apply learned transformation.
            t = y  # t = (r_c, r_s, t_x, t_y)
            t = t / t[:, :2].norm(dim=1, keepdim=True)  # t' = (r'_c, r'_s, t'_x, t'_y)
            m_r = torch.eye(3, device=y.device).unsqueeze(0).repeat(
                [y.shape[0], 1, 1])  # Inverse rotation wrt. resulting image.
            m_r[:, 0, 0] = t[:, 0]  # r'_c
            m_r[:, 0, 1] = -t[:, 1]  # r'_s
            m_r[:, 1, 0] = t[:, 1]  # r'_s
            m_r[:, 1, 1] = t[:, 0]  # r'_c
            m_t = torch.eye(3, device=y.device).unsqueeze(0).repeat(
                [y.shape[0], 1, 1])  # Inverse translation wrt. resulting image.
            m_t[:, 0, 2] = -t[:, 2]  # t'_x
            m_t[:, 1, 2] = -t[:, 3]  # t'_y
            transforms = m_r @ m_t @ transforms  # First rotate resulting image, then translate, and finally apply user-specified transform.

            # Transform frequencies.
            phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)  # The translation part
            freqs = freqs @ transforms[:, :2, :2]  # The rotation part

            # Dampen out-of-band frequencies that may occur due to the user-specified transform.
            amplitudes = (1 - (freqs.norm(dim=2) - input_layer.bandwidth) / (
                    input_layer.sampling_rate / 2 - input_layer.bandwidth)).clamp(0, 1)

            # Construct sampling grid.
            theta = torch.eye(2, 3, device=y.device)
            theta[0, 0] = 0.5 * input_layer.size[0] / input_layer.sampling_rate
            theta[1, 1] = 0.5 * input_layer.size[1] / input_layer.sampling_rate
            grids = torch.nn.functional.affine_grid(theta.unsqueeze(0),
                                                    [1, 1, input_layer.size[1], input_layer.size[0]],
                                                    align_corners=False)
            # Compute Fourier features.
            x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(
                3)  # [batch, height, width, channel]
            x = x + phases.unsqueeze(1).unsqueeze(2)
            x = torch.sin(x * (np.pi * 2))
            x = x * amplitudes.unsqueeze(1).unsqueeze(2)

            # Apply trainable mapping.
            weight = input_layer.weight / np.sqrt(input_layer.channels)
            x = x @ weight.t()

            # Ensure correct shape.
            x = x.permute(0, 3, 1, 2)  # [batch, channel, height, width]
            return x

        def synthesis_layer_forward(synthesis_layer, x, y):
            force_fp32 = self.force_fp32
            input_gain = synthesis_layer.magnitude_ema.rsqrt()

            # Execute affine layer.
            styles = y
            if synthesis_layer.is_torgb:
                weight_gain = 1 / np.sqrt(synthesis_layer.in_channels * (synthesis_layer.conv_kernel ** 2))
                styles = styles * weight_gain

            # Execute modulated conv2d.
            dtype = torch.float16 if (
                    synthesis_layer.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
            x = modulated_conv2d(x=x.to(dtype), w=synthesis_layer.weight, s=styles,
                                 padding=synthesis_layer.conv_kernel - 1, demodulate=(not synthesis_layer.is_torgb),
                                 input_gain=input_gain)

            # Execute bias, filtered leaky ReLU, and clamping.
            gain = 1 if synthesis_layer.is_torgb else np.sqrt(2)
            slope = 1 if synthesis_layer.is_torgb else 0.2
            x = filtered_lrelu.filtered_lrelu(x=x, fu=synthesis_layer.up_filter, fd=synthesis_layer.down_filter,
                                              b=synthesis_layer.bias.to(x.dtype),
                                              up=synthesis_layer.up_factor, down=synthesis_layer.down_factor,
                                              padding=synthesis_layer.padding, gain=gain,
                                              slope=slope, clamp=synthesis_layer.conv_clamp)

            # Ensure correct shape and dtype.
            return x

        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            return_image = self.return_image if return_image is None else return_image
            return_all_layers = self.return_all_layers if return_all_layers is None else return_all_layers
            if return_image:
                max_resolution = 1024

            result = {}
            if self.only_w:
                return result

            L = 0
            x = input_layer_forward(self.G.synthesis.input, ys[0])
            margin = self.G.synthesis.margin_size
            if return_all_layers:
                result[f'layer_{L}_extended'] = x
                if self.layer_to_resolution[L] != self.layer_to_spatial_size[L]:
                    x_center = x[:, :, margin:-margin, margin:-margin]
                else:
                    x_center = x

                result[f'layer_{L}'] = x_center

            L += 1
            for layer_name, y in zip(self.G.synthesis.layer_names, ys[1:]):
                res = self.layer_to_resolution[L]
                if res > max_resolution:
                    return result
                synthesis_layer = getattr(self.G.synthesis, layer_name)
                x = synthesis_layer_forward(synthesis_layer, x, y)
                if return_all_layers:
                    result[f'layer_{L}_extended'] = x

                    if self.layer_to_resolution[L] != self.layer_to_spatial_size[L]:
                        x_center = x[:, :, margin:-margin, margin:-margin]
                    else:
                        x_center = x

                    result[f'layer_{L}'] = x_center

                L += 1

            if self.G.synthesis.output_scale != 1:
                x = x * self.G.synthesis.output_scale

            if return_image:
                img = x.to(torch.float32)
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

            ws = ws.to(torch.float32)
            ys = self.ws_to_ys(ws)
            batch_ys = self.ys_to_batch_ys(ys)
            result = self.generate_batch_from_ys(ys, return_image, return_all_layers, max_resolution,
                                                 requires_grad)
            if return_style:
                result['ys'] = ys
                result['batch_ys'] = batch_ys

            return result

    def generate_batch_from_z(self, z, return_image=None, return_all_layers=None,
                              return_style=True,
                              max_resolution=1024, requires_grad=False):
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            label = None
            ws = self.G.mapping(z, label, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff)
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
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, self.G.z_dim).astype(np.float32)).to(
                self.device)
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
                z_seed = np.random.RandomState(seed).randn(1, self.G.z_dim).astype(np.float32)
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
