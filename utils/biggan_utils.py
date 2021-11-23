import contextlib

import torch
from PIL import Image

from models.biggan.pytorch_pretrained_biggan import GenBlock, truncated_noise_sample, one_hot_from_names


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


class BigGANSampleGenerator:
    """
    Wrapper class for generating images and intermediate features
    from the given BigGAN generator.
    Parameters
    ----------
    G: Pytorch Module,
        Trained StyleGAN2 Generator model as a pytorch module.
    device: torch device used for the computation
    class_label: string, required
        Label of the class to generate images for. Note that this
        will translate into class idx using NLTK library and wordnet
    truncation_psi : float in (0, 1], default=0.6
        truncation value used for random latent code
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

    layer_to_resolution = {
        0: 4, 1: 4,
        2: 8, 3: 8,
        4: 16, 5: 16,
        6: 32, 7: 32,
        8: 64, 9: 64, 10: 64,
        11: 128, 12: 128,
        13: 256, 14: 256,
        15: 512,

    }

    def __init__(self, G, device, class_label, truncation_psi=0.6, batch_size=8, only_w=False, return_image=False,
                 return_all_layers=False):
        assert 0.0 < truncation_psi <= 1.0
        self.G = G.to(device)
        self.config = self.G.generator.config
        # One extra layer for the self attention
        # One extra for the first layer that starts from
        # Condition vector
        self.n_layers = len(self.config.layers) + 2
        self.device = device
        self.class_label = class_label
        self.truncation_psi = truncation_psi
        self.batch_size = batch_size
        self.only_w = only_w
        self.return_image = return_image
        self.return_all_layers = return_all_layers

    def raw_image_to_pil_image(self, raw_img):
        img = (raw_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_imgs = []
        for i in range(len(img)):
            pil_imgs.append(Image.fromarray(img[i].cpu().numpy(), 'RGB'))
        return pil_imgs

    def generate_batch_from_ws(self, ws, class_label=None, return_image=None, return_all_layers=None, return_style=True,
                               max_resolution=1024, requires_grad=False):
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            return_image = self.return_image if return_image is None else return_image
            class_label = self.class_label if class_label is None else class_label
            return_all_layers = self.return_all_layers if return_all_layers is None else return_all_layers
            if return_image:
                max_resolution = 1024

            result = {}
            batch_size = len(ws)
            c = one_hot_from_names(class_label, batch_size=batch_size)
            c = torch.tensor(c, dtype=torch.float).to(self.device)
            c_embed = self.G.embeddings(c)
            if return_style:
                result["c_embed"] = c_embed

            if self.only_w:
                return result

            cond_vectors = [torch.cat((ws[:, l], c_embed), dim=1) for l in range(self.n_layers)]
            iter_cond_vector = iter(cond_vectors)

            x = self.G.generator.gen_z(next(iter_cond_vector))
            x = x.view(-1, 4, 4, 16 * self.config.channel_width)
            x = x.permute(0, 3, 1, 2).contiguous()
            L = 0
            if return_all_layers:
                result[f'layer_{L}'] = x
            L += 1
            for i, layer in enumerate(self.G.generator.layers):
                if isinstance(layer, GenBlock):
                    x = layer(x, next(iter_cond_vector), self.truncation_psi)
                else:
                    x = layer(x)

                res = x.shape[-1]
                if res > max_resolution:
                    return result

                if return_all_layers:
                    result[f'layer_{L}'] = x
                #                     print(f"Layer {L}, Resolution {res}x{res}")

                L += 1

            x = self.G.generator.bn(x, self.truncation_psi)
            x = self.G.generator.relu(x)
            x = self.G.generator.conv_to_rgb(x)
            x = x[:, :3, ...]
            img = self.G.generator.tanh(x)

            if return_image:
                pil_imgs = self.raw_image_to_pil_image(img)
                result['raw_image'] = img.clamp(-1.0, 1.0)
                result['image'] = pil_imgs

            return result

    def generate_batch_from_z(self, z, class_label=None, return_image=None, return_all_layers=None,
                              return_style=True,
                              max_resolution=1024, requires_grad=False):
        with torch.no_grad() if not requires_grad else dummy_context_mgr():
            # Slef.attention layer does not need condition vecotr
            ws = torch.stack([z.clone() for layer in range(self.n_layers)], dim=1)

            result = self.generate_batch_from_ws(ws, class_label, return_image, return_all_layers, return_style,
                                                 max_resolution,
                                                 requires_grad)
        if return_style:
            result['ws'] = ws
        return result

    def generate_batch(self, seed, class_label=None, batch_size=None, return_image=None, return_all_layers=None,
                       return_style=True,
                       max_resolution=1024, requires_grad=False):
        """
        Paramteres
        ----------
        seed: int, The random seed used for generating samples.
        class_label: string, The label of the class to generate images for
            If not specified, the class attribute will be used
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
            z = truncated_noise_sample(batch_size=batch_size, truncation=self.truncation_psi, seed=seed)
            z = torch.tensor(z, dtype=torch.float).to(self.device)

            result = self.generate_batch_from_z(z, class_label, return_image, return_all_layers, return_style,
                                                max_resolution,
                                                requires_grad)
        if return_style:
            result['z'] = z
        return result

    def generate_image_from_ws(self, ws, class_label=None, raw=False):
        class_label = self.class_label if class_label is None else class_label
        batch_size = len(ws)
        c = one_hot_from_names(class_label, batch_size=batch_size)
        c = torch.tensor(c, dtype=torch.float).to(self.device)
        c_embed = self.G.embeddings(c)

        cond_vectors = [torch.cat((ws[:, l], c_embed), dim=1) for l in range(self.n_layers)]
        iter_cond_vector = iter(cond_vectors)

        x = self.G.generator.gen_z(next(iter_cond_vector))
        x = x.view(-1, 4, 4, 16 * self.config.channel_width)
        x = x.permute(0, 3, 1, 2).contiguous()
        for i, layer in enumerate(self.G.generator.layers):
            if isinstance(layer, GenBlock):
                x = layer(x, next(iter_cond_vector), self.truncation_psi)
            else:
                x = layer(x)

        x = self.G.generator.bn(x, self.truncation_psi)
        x = self.G.generator.relu(x)
        x = self.G.generator.conv_to_rgb(x)
        x = x[:, :3, ...]
        img = self.G.generator.tanh(x)
        if raw:
            return img

        pil_imgs = self.raw_image_to_pil_image(img)
        return pil_imgs

    def zero_grad(self):
        self.G.zero_grad()
