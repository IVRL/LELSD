""" LELSD PyTorch model.
    From "Optimizing Latent Space Directions For GAN-based Local Image Editing"
    By Ehsan Pajouheshgar, Tong Zhang, and Sabine Susstrunk
    https://arxiv.org/abs/2111.12583
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm

from utils.segmentation_utils import GANLinearSegmentation

try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter


class LELSD:
    stylegan2_y_dims = {
        0: 512,  # 4 x 4
        1: 512,  # 8 x 8
        2: 512,  # 8 x 8
        3: 512,  # 16 x 16
        4: 512,  # 16 x 16
        5: 512,  # 32 x 32
        6: 512,  # 32 x 32
        7: 512,  # 64 x 64
        8: 512,  # 64 x 64
        9: 512,  # 128 x 128
        10: 256,  # 128 x 128
        11: 256,  # 256 x 256
        12: 128,  # 256 x 256
        13: 128,  # 512 x 512
        14: 64,  # 512 x 512
        15: 64,  # 1024 x 1024
        16: 32,  # 1024 x 1024
        17: 32,  # Unused
    }

    stylegan2_rgb_y_dims = {
        0: 512,  # 4 x 4
        1: 512,  # 8 x 8
        2: 512,  # 16 x 16
        3: 512,  # 32 x 32
        4: 512,  # 64 x 64
        5: 256,  # 128 x 128
        6: 128,  # 256 x 256
        7: 64,  # 512 x 512
        8: 32,  # 1024 x 1024
    }

    stylegan2_s_dims = {
        0: 512,  # 4 x 4 s1
        1: 512,  # 4 x 4 trgb
        2: 512,  # 8 x 8 s1
        3: 512,  # 8 x 8 s2
        4: 512,  # 8 x 8 trgb
        5: 512,  # 16 x 16 s1
        6: 512,  # 16 x 16 s2
        7: 512,  # 16 x 16 trgb
        8: 512,  # 32 x 32 s1
        9: 512,  # 32 x 32 s2
        10: 512,  # 32 x 32 trgb
        11: 512,  # 64 x 64 s1
        12: 512,  # 64 x 64 s2
        13: 512,  # 64 x 64 trgb
        14: 512,  # 128 x 128 s1
        15: 256,  # 128 x 128 s2
        16: 256,  # 128 x 128 trgb
        17: 256,  # 256 x 256 s1
        18: 128,  # 256 x 256 s2
        19: 128,  # 256 x 256 trgb
        20: 128,  # 512 x 512 s1
        21: 64,  # 512 x 512 s2
        22: 64,  # 512 x 512 trgb
        23: 64,  # 1024 x 1024 s1
        24: 32,  # 1024 x 1024 s2
        25: 32,  # 1024 x 1024 trgb
        26: 32  # 1024 x 1024 unused
    }

    """
    This class tries to find directions in the latent space of a pretrained GAN
    that if added to the initial latent vector, the change in the output image
    will be localized to a specific semantic part of the object.
    For example you can use this model to find latent directions that only change
    the mouth, or the nose of the generated face.
    For fitting this model, a pretrained GAN, and a pretrained segmentation model are required.
    Parameters
    ----------
    device: pytorch device
        Device used for performing the computation.
    localization_layers: list, required
        The layers that will be used in the Localization score.
        The Localozation Score tries to maximize the ratio of the
        changes in the desired part to changes outside
        the desired part. The mask determining the desired part will
        be bilinearly [up-down]-sampled to have the same shape as layer feature maps.
        Note that the layer indices should be ascendingly sorted.
    semantic_parts: list, required
        List of strings. This list will determine the semantic parts that determine the localized area
        of the output image we want to modify and edit. For fitting the model a segmentation object is
        required with an dict attribute named 'part_to_mask_idx' that will determine the mask idx corresponding
        to each semantic part in this list.
    loss_function: {'L1', 'L2', 'cos'}, default='L2'
        The loss function used to evaluate and optimize localization score     
    mode: string, default='foreground'
        Either 'foreground' or 'background'
        Wether to treat the semantic parts as foreground or background.
    mask_aggregation: {'union', 'intersection', 'average'}, default='average'
        The method to combine two masks
    n_layers: int, default=18
        Number of layers in the generator network.
    num_latent_directions: int, default=3
        Number of distinct direction that will be find. Note that the model will try to
        find perpendicular directions by minimizing the correlation between the pair of directions.
    latent_dim: int, default=512
        Dimension of the latent space of the GAN
    batch_size: int, default=4
        Batch size used for fitting the model
    learning_rate: float, default=0.001,
        The learning rate used for optimizing the latent directions
    localization_layer_weights: list, default=None
        Weight of each layer in the Localization loss term.
        If not specified, the layers will have equal weights.
    gamma_correlation: float, default=1.0
        Coefficient of correlation loss
    unit_norm: bool, default=True,
        Setting this to True will force the optimization to normalize the latent
        directions to unit norm after each step of gradient descent.
        If this is set to false then (-1.0, 1.0) will be used as the alpha range
        with min_abs_alpha_value equal to 0.2
    latent_space: {'Z', 'W', 'W+', S*'}, Type of latent space used for extracting semantics.
        Z: Normally distributed latent codes
        W: StyleGAN intermediate Latent Space
        W+: StyleGAN Intermediate Latent Space separated for each layer
        S: StyleSpace defined in paper 'https://arxiv.org/abs/2011.12799'
            Examples
            S10: 10layer according to the paper 
            S+: all layers
            S_conv: all conv layers (trgb styles not included)
    onehot_temperature: float, default=0.001
        Temperature of the RelaxedOneHotCategorical distribution used for generating
        weight of each latent direction for the linear interpolation.
        If this temperature is very small (close to zero) then  the coefficients in the
        linear interpolation will be almost onehot and if the temperature is to high, all
        of the coefficients will be equal to (1 / num_latent_dirs)
    min_alpha_value: float, default=-3.0
        The minimum alpha value used for adding the latent direction.
    min_alpha_value: float, default=-3.0
        The maximum alpha value used for adding the latent direction.
        Alpha will be sampled randomly in this range.
    min_abs_alpha_value: float, default=0.5
        If abs(alpha) for the sampled alpha is smaller than this value,
        alpha will be resampled. Notice that small values of alpha does not
        change the semantic noticeably and may cause unstable optimization.
    log_dir: str, default=log
        Directory used for saving the model, and optimization summaries
    random_seed: int, default=1234
        Random seed used for initializing the latent directions
    """

    def __init__(self, device, localization_layers, semantic_parts, loss_function="MSE",
                 mode='foreground', mask_aggregation='average', n_layers=18, num_latent_dirs=3, latent_dim=512,
                 batch_size=4,
                 learning_rate=0.001, localization_layer_weights=None, gamma_correlation=1.0,
                 unit_norm=True, latent_space='W',
                 onehot_temperature=0.001,
                 min_alpha_value=-3.0, max_alpha_value=3.0, min_abs_alpha_value=0.5,
                 log_dir='log', random_seed=1234):
        assert mask_aggregation in ['average', 'union', 'intersection']
        assert loss_function in ['L1', 'L2', 'cos']
        self.device = device
        self.localization_layers = localization_layers
        self.semantic_parts = semantic_parts
        self.loss_function = loss_function
        self.mode = mode
        self.mask_aggregation = mask_aggregation
        self.n_layers = n_layers
        self.num_latent_dirs = num_latent_dirs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = learning_rate
        if localization_layer_weights is None:
            self.localization_layer_weights = np.array([1.0] * len(self.localization_layers))
        else:
            self.localization_layer_weights = np.array(localization_layer_weights)

        self.localization_layer_weights /= self.localization_layer_weights.sum()
        self.localization_layer_weights = list(self.localization_layer_weights)
        self.gamma_correlation = gamma_correlation

        self.unit_norm = unit_norm
        self.latent_space = latent_space

        self.min_alpha_value = min_alpha_value
        self.max_alpha_value = max_alpha_value
        self.min_abs_alpha_value = min_abs_alpha_value

        self.onehot_temperature = onehot_temperature
        assert self.onehot_temperature >= 0.001
        self.coefficient_sampling_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature=torch.tensor([self.onehot_temperature]),
            probs=torch.tensor([1 / self.num_latent_dirs] * self.num_latent_dirs))

        self.init_time_signature = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(log_dir, self.init_time_signature)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        if self.latent_space == 'Z':
            self.latent_dirs = torch.randn(self.num_latent_dirs, self.latent_dim, device=self.device)
            self.latent_dirs = self.latent_dirs / torch.linalg.norm(self.latent_dirs, dim=1, keepdim=True)
        if self.latent_space == 'W':
            self.latent_dirs = torch.randn(self.num_latent_dirs, self.latent_dim, device=self.device)
            self.latent_dirs = self.latent_dirs / torch.linalg.norm(self.latent_dirs, dim=1, keepdim=True)
        if self.latent_space == 'W+':
            self.latent_dirs = torch.randn(self.num_latent_dirs, self.n_layers, self.latent_dim, device=self.device)
            self.latent_dirs = self.latent_dirs / torch.linalg.norm(self.latent_dirs, dim=2, keepdim=True)
        if self.latent_space.startswith("S"):
            # This feature is only implemented for StyleGAN2
            self.latent_dirs = []
            for layer in self.stylegan2_s_dims:
                dim = self.stylegan2_s_dims[layer]
                w = torch.randn(self.num_latent_dirs, dim, device=self.device)
                w = w / torch.linalg.norm(w, dim=1, keepdim=True)
                self.latent_dirs.append(w)
            if self.latent_space == 'S+':
                self.s_layers_to_apply = None
            elif self.latent_space == 'S_conv':
                self.s_layers_to_apply = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26]
            else:
                self.s_layers_to_apply = [int(self.latent_space[1:])]

        if not self.latent_space.startswith("S"):
            self.latent_dirs = Variable(self.latent_dirs, requires_grad=True)
        else:
            #             pass
            self.latent_dirs = [Variable(w, requires_grad=True) for w in self.latent_dirs]

    def sample_alpha(self, min_value, max_value, min_abs_value=0.5):
        L = max_value - min_value
        alpha = L * np.random.rand() + min_value
        if abs(alpha) < min_abs_value:
            return self.sample_alpha(min_value, max_value, min_abs_value)
        return alpha

    def _move_latent_codes(self, latent_codes, latent_dir, alpha, layers_to_apply=None):
        if layers_to_apply:
            # if hasattr(self, "one_for_each_layer") and self.one_for_each_layer:
            if self.latent_space.startswith("S"):
                new_latent_codes = [x.clone() for x in latent_codes]
                for l in layers_to_apply:
                    if l >= len(new_latent_codes):
                        continue
                    new_latent_codes[l] += latent_dir[l] * alpha
            else:
                new_latent_codes = latent_codes.clone()
                if self.latent_space == 'W+':
                    new_latent_codes[:, layers_to_apply, :] += latent_dir[:, layers_to_apply, :] * alpha
                elif self.latent_space == 'W':
                    new_latent_codes[:, layers_to_apply, :] += latent_dir.unsqueeze(0) * alpha
                else:
                    # Z cannot be separately applied to different layers
                    new_latent_codes += latent_codes * alpha
        else:
            if self.latent_space.startswith("S"):
                new_latent_codes = [x.clone() for x in latent_codes]
                for l in range(len(new_latent_codes)):
                    new_latent_codes[l] += latent_dir[l] * alpha
            else:
                new_latent_codes = latent_codes.clone()
                if self.latent_space == 'W+':
                    new_latent_codes += latent_dir * alpha
                elif self.latent_space == 'W':
                    new_latent_codes += latent_dir.unsqueeze(0) * alpha
                else:
                    new_latent_codes += latent_dir * alpha

        return new_latent_codes

    def move_latent_codes(self, latent_codes, latent_dir_idx, alpha, layers_to_apply=None):
        """
        Moves the latent vector ws in the direction of self.latent_dirs[latent_dir_idx] by amount of alpha

        Parameters
        ----------
        latent_codes: If latent_space is 'Z' then Tensor with shape [batch_size, latent_dim]
            If latent_space is 'W' or 'W+ then Tensor with shape [batch_size, num_layers, latent_dim]
            If latent_space is 'S' then tuple of (ys, rgb_ys)
                where ys and rgb_ys are list of tensors of shape [batch_size, latent_dim]

        latent_dir_idx: int, Index of the latent direction
        alpha: float, Determines the magnitude of the vector that will be added to ws
        layers_to_apply: list, default=None
            list of layers that will use the updated w.
            If None then all of the layers will be updated
        """
        if not self.latent_space.startswith("S"):
            latent_dir = self.latent_dirs[latent_dir_idx: latent_dir_idx + 1]
        else:
            latent_dir = [x[latent_dir_idx: latent_dir_idx + 1] for x in self.latent_dirs]

        return self._move_latent_codes(latent_codes, latent_dir, alpha, layers_to_apply)

    @staticmethod
    def correlation_loss(ws):
        """
        Parameters
        ----------
        ws: Tensor of shape [num_ws, latent_dim]
            The rows of ws should have unit length.
        """
        covariance_matrix = torch.matmul(ws, ws.T)
        diagonal = torch.diag(covariance_matrix)
        normalizer = torch.sqrt(diagonal[:, None])
        normalizer = normalizer.T * normalizer
        correlation_matrix = covariance_matrix / normalizer
        non_diagonal_elements = correlation_matrix - torch.eye(len(ws), device=ws.device)
        return torch.mean(torch.abs(non_diagonal_elements))

    @staticmethod
    def combine_mask(mask1, mask2, method='average'):
        assert method in ['average', 'union', 'intersection']
        if method == 'average':
            return 0.5 * mask1 + 0.5 * mask2
        elif method == 'intersection':
            return mask1 * mask2
        else:
            return mask1 + mask2 - mask1 * mask2

    def fit(self, gan_sample_generator, segmentation_model, num_batches, num_lr_halvings=4, batch_size=None,
            pgbar=False, summary=True, snapshot_interval=200):
        """
        This method will optimize the latent directions using samples generated by gan_sample_generator
        that will be segmented by segmentation_model.
        Parameters
        ----------
        gan_sample_generator: SampleGenerator,
            This is a wrapper class on top of the pretrained generator network.
        segmentation_model: SegmentationModel
            This is a wrapper class on top of the pretrained segmentation network.
            This class should have an attribute named 'part_to_mask_idx' that will
            be used to translate each semantic part to the corresponding mask idx.
        num_batches: int
            Number of batches used for optimizing the latent directions
        num_lr_halvings: int, default=4
            Number of times that the learning rate will be halved.
        batch_size: int, default=None
            The batch size used for the optimization.
            if not specified the class attribute will be used.
        pgbar: bool, default=False
            Whether to show the progress bar or not
        summary: bool, default=True
            Whether to write summaries or not
        snapshot_interval: int, default=200
            The interval between saving image snapshots.
        """
        if batch_size is None:
            batch_size = self.batch_size
        assert hasattr(segmentation_model, 'part_to_mask_idx')
        if summary:
            with open(os.path.join(self.log_dir, "config.json"), "w") as f:
                config_dict = self.__dict__.copy()
                del config_dict['device']
                del config_dict['latent_dirs']
                del config_dict['coefficient_sampling_dist']
                json.dump(config_dict, f, indent=4, sort_keys=True, default=lambda x: x.__name__)

            try:
                summary_writer = SummaryWriter(logdir=self.log_dir)
            except:
                summary_writer = SummaryWriter(self.log_dir)

        part_ids = [segmentation_model.part_to_mask_idx[part_name] for part_name in self.semantic_parts]
        lr = self.lr
        lr_decay_steps = num_batches // (num_lr_halvings + 1)
        min_alpha_value = self.min_alpha_value
        max_alpha_value = self.max_alpha_value
        global_step = 0
        pbar = tqdm(range(54321, 54321 + num_batches, 1), disable=not pgbar, total=num_batches)
        pil_to_tensor = torchvision.transforms.ToTensor()

        if self.latent_space.startswith("S"):
            optimizer = torch.optim.Adam(self.latent_dirs, lr=lr)
        else:
            optimizer = torch.optim.Adam([self.latent_dirs], lr=lr)
        for i, seed in enumerate(pbar):
            if i % lr_decay_steps == (lr_decay_steps - 1):
                lr /= 2
                optimizer.param_groups[0]['lr'] = lr

            batch_data = gan_sample_generator.generate_batch(batch_size=batch_size, seed=seed, requires_grad=True,
                                                             return_all_layers=True, return_image=True)
            if isinstance(segmentation_model, GANLinearSegmentation):
                old_segmentation_output = segmentation_model.predict(batch_data, one_hot=False)
            else:
                old_segmentation_output = segmentation_model.predict(batch_data['image'], one_hot=False)
            segmentation_output_res = old_segmentation_output.shape[2]

            old_mask = 0.0
            for part_idx in part_ids:
                old_mask += 1.0 * (old_segmentation_output == part_idx)

            # for w_idx in range(self.num_ws):
            gan_sample_generator.zero_grad()
            segmentation_model.zero_grad()
            alpha = self.sample_alpha(min_alpha_value, max_alpha_value, min_abs_value=self.min_abs_alpha_value)
            latent_coefs = self.coefficient_sampling_dist.sample().unsqueeze(1).to(self.device)  # [num_ws, 1]
            if not self.latent_space.startswith("S"):
                if self.latent_space.startswith("W"):
                    old_latent = batch_data['ws'].detach()
                    if self.latent_space == "W+":
                        latent_coefs = latent_coefs.unsqueeze(2)
                    latent_dir = torch.sum(self.latent_dirs * latent_coefs, dim=0, keepdim=True)
                    new_latent = self._move_latent_codes(old_latent, latent_dir, alpha)
                    new_batch_data = gan_sample_generator.generate_batch_from_ws(new_latent, requires_grad=True,
                                                                                 return_all_layers=True,
                                                                                 return_image=True)
                else:
                    old_latent = batch_data['z'].detach()
                    latent_dir = torch.sum(self.latent_dirs * latent_coefs, dim=0, keepdim=True)
                    new_latent = self._move_latent_codes(old_latent, latent_dir, alpha)
                    new_batch_data = gan_sample_generator.generate_batch_from_z(new_latent, requires_grad=True,
                                                                                return_all_layers=True,
                                                                                return_image=True)

            else:
                old_latent = [tmp.detach() for tmp in batch_data['s']]
                latent_dir = [torch.sum(u * latent_coefs, dim=0, keepdim=True) for u in self.latent_dirs]
                new_latent = self._move_latent_codes(old_latent, latent_dir, alpha,
                                                     layers_to_apply=self.s_layers_to_apply)
                ys_tuple = gan_sample_generator.s_to_ys(new_latent)
                new_batch_data = gan_sample_generator.generate_batch_from_ys(ys_tuple, requires_grad=True,
                                                                             return_all_layers=True,
                                                                             return_image=True)

            if isinstance(segmentation_model, GANLinearSegmentation):
                new_segmentation_output = segmentation_model.predict(new_batch_data, one_hot=False)
            else:
                new_segmentation_output = segmentation_model.predict(new_batch_data['image'], one_hot=False)
            new_mask = 0.0
            for part_idx in part_ids:
                new_mask += 1.0 * (new_segmentation_output == part_idx)

            combined_mask = self.combine_mask(old_mask, new_mask, self.mask_aggregation)
            combined_mask = combined_mask.detach()
            # Cloning to avoid redundant computation
            mask = combined_mask.clone()

            last_layer_res = None
            localization_loss = 0
            # To maximize the Localization Score in localization layers
            for layer, layer_weight in zip(reversed(self.localization_layers),
                                           reversed(self.localization_layer_weights)):
                layer_res = gan_sample_generator.layer_to_resolution[layer]
                if last_layer_res != layer_res:
                    if layer_res != segmentation_output_res:
                        mask = torch.nn.functional.interpolate(mask, size=(layer_res, layer_res),
                                                               mode='bilinear',
                                                               align_corners=True)
                    else:
                        mask = combined_mask.clone()
                last_layer_res = layer_res
                if layer_weight == 0:
                    continue

                x1 = batch_data[f'layer_{layer}'].detach()
                x2 = new_batch_data[f'layer_{layer}']
                if self.loss_function == 'L1':
                    diff = torch.mean(torch.abs(x1 - x2), dim=1)
                elif self.loss_function == 'L2':
                    diff = torch.mean(torch.square(x1 - x2), dim=1)
                elif self.loss_function == 'cos':
                    diff = 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8)
                else:
                    diff = torch.mean(torch.square(x1 - x2), dim=1)
                indicator = mask[:, 0]
                if self.mode == 'background':
                    indicator = 1 - indicator

                localization_loss -= layer_weight * torch.sum(diff * indicator, dim=[1, 2]) / (
                        torch.sum(diff, dim=[1, 2]) + 1e-6)

            # -1.0 means perfect localization and 0 means poor localization
            localization_loss = torch.mean(localization_loss)
            if self.latent_space == "W+":
                correlation_loss = 0
                for layer in range(self.n_layers):
                    correlation_loss += self.correlation_loss(self.latent_dirs[:, layer, :])
                correlation_loss /= self.n_layers
            elif self.latent_space.startswith("S"):
                correlation_loss = 0
                if self.s_layers_to_apply is None:
                    for latent_dir in self.latent_dirs:
                        correlation_loss += self.correlation_loss(latent_dir)
                    correlation_loss /= len(self.latent_dirs)
                else:
                    for layer in self.s_layers_to_apply:
                        correlation_loss += self.correlation_loss(self.latent_dirs[layer])
                    correlation_loss /= len(self.s_layers_to_apply)
            else:
                correlation_loss = self.correlation_loss(self.latent_dirs)
            loss = localization_loss + self.gamma_correlation * correlation_loss
            loss.backward()
            if self.unit_norm:
                optimizer.step()
                self.latent_dirs.data = self.latent_dirs.data / torch.linalg.norm(self.latent_dirs.data, dim=1,
                                                                                  keepdim=True)
                optimizer.zero_grad()
            else:
                pass
                optimizer.step()
                optimizer.zero_grad()

            if summary:
                if global_step % snapshot_interval == 0:
                    old_images = [pil_to_tensor(img.resize((256, 256))) for img in batch_data['image'][:4]]
                    old_mask_images = [
                        torch.nn.functional.interpolate(old_mask[b: b + 1], size=(256, 256), mode='bilinear',
                                                        align_corners=True)[0].repeat(3, 1, 1).detach().cpu() for b in
                        range(min(4, batch_size))]
                    new_images = [pil_to_tensor(img.resize((256, 256))) for img in new_batch_data['image'][:4]]
                    new_mask_images = [
                        torch.nn.functional.interpolate(new_mask[b: b + 1], size=(256, 256), mode='bilinear',
                                                        align_corners=True)[0].repeat(3, 1, 1).detach().cpu() for b in
                        range(min(4, batch_size))]
                    images = old_images + old_mask_images + new_images + new_mask_images
                    image_grid = torchvision.utils.make_grid(images, nrow=4)
                    torchvision.utils.save_image(image_grid,
                                                 os.path.join(self.log_dir, f"batch={i}-image_alpha={alpha:.2f}.jpg"))

                summary_writer.add_scalar("localization score", -localization_loss, global_step)
                summary_writer.add_scalar("correlation_loss", correlation_loss, global_step)
                summary_writer.add_scalar("loss", loss, global_step)
                summary_writer.add_scalar("alpha", alpha, global_step)
                summary_writer.add_scalar("lr", lr, global_step)
                global_step += 1

        if summary:
            summary_writer.close()
        if self.unit_norm:
            if not self.latent_space.startswith("S"):
                self.latent_dirs.data = self.latent_dirs.data / torch.linalg.norm(self.latent_dirs.data, dim=1,
                                                                                  keepdim=True)
            else:
                for l in range(len(self.latent_dirs)):
                    self.latent_dirs[l].data = self.latent_dirs[l].data / torch.linalg.norm(self.latent_dirs[l].data,
                                                                                            dim=1, keepdim=True)

    def edit_batch_data(self, gan_sample_generator, batch_data, latent_dir_idx, alpha, layers_to_apply=None):
        if not self.latent_space.startswith("S"):
            if self.latent_space.startswith("W"):
                old_latent = batch_data['ws'].detach()
                new_latent = self.move_latent_codes(old_latent, latent_dir_idx, alpha, layers_to_apply)
                new_batch_data = gan_sample_generator.generate_batch_from_ws(new_latent, return_style=True,
                                                                             return_image=True, return_all_layers=True)
                new_batch_data['ws'] = new_latent
            else:
                old_latent = batch_data['z'].detach()
                new_latent = self.move_latent_codes(old_latent, latent_dir_idx, alpha)
                new_batch_data = gan_sample_generator.generate_batch_from_z(new_latent, return_style=True,
                                                                            return_image=True, return_all_layers=True)
                new_batch_data['z'] = new_latent

        else:
            old_latent = [tmp.detach() for tmp in batch_data['s']]
            new_latent = self.move_latent_codes(old_latent, latent_dir_idx, alpha, layers_to_apply)
            ys_tuple = gan_sample_generator.s_to_ys(new_latent)
            new_batch_data = gan_sample_generator.generate_batch_from_ys(ys_tuple, return_image=True,
                                                                         return_all_layers=True)
            new_batch_data['s'] = new_latent

        return new_batch_data

    def save(self):
        torch.save(self, os.path.join(self.log_dir, "model.pth"))

    @staticmethod
    def load(model_path):
        return torch.load(model_path)

    def randomize_latent_dirs(self):
        if self.latent_space == 'Z':
            self.latent_dirs = torch.randn(self.num_latent_dirs, self.latent_dim, device=self.device)
            self.latent_dirs = self.latent_dirs / torch.linalg.norm(self.latent_dirs, dim=1, keepdim=True)
        if self.latent_space == 'W':
            self.latent_dirs = torch.randn(self.num_latent_dirs, self.latent_dim, device=self.device)
            self.latent_dirs = self.latent_dirs / torch.linalg.norm(self.latent_dirs, dim=1, keepdim=True)
        if self.latent_space == 'W+':
            self.latent_dirs = torch.randn(self.num_latent_dirs, self.n_layers, self.latent_dim, device=self.device)
            self.latent_dirs = self.latent_dirs / torch.linalg.norm(self.latent_dirs, dim=2, keepdim=True)
        if self.latent_space.startswith("S"):
            # This feature is only implemented for StyleGAN2
            self.latent_dirs = []
            for layer in self.stylegan2_s_dims:
                dim = self.stylegan2_s_dims[layer]
                w = torch.randn(self.num_latent_dirs, dim, device=self.device)
                w = w / torch.linalg.norm(w, dim=1, keepdim=True)
                self.latent_dirs.append(w)
            if self.latent_space == 'S+':
                self.s_layers_to_apply = None
            elif self.latent_space == 'S_conv':
                self.s_layers_to_apply = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26]
            else:
                self.s_layers_to_apply = [int(self.latent_space[1:])]

        if not self.latent_space.startswith("S"):
            self.latent_dirs = Variable(self.latent_dirs, requires_grad=True)
        else:
            #             pass
            self.latent_dirs = [Variable(w, requires_grad=True) for w in self.latent_dirs]

        return self

    def load_from_InterFaceGAN(self, npy_path):
        if "_w_" in npy_path:
            latent_space = "W"
        else:
            latent_space = "Z"
        w = np.load(npy_path)
        self.latent_space = latent_space
        self.latent_dirs = torch.from_numpy(w).to(self.device)

    def load_from_StyleSpace(self, layer_idx, channel_idx):
        self.latent_space = f"S{layer_idx}"
        self.s_layers_to_apply = [layer_idx]
        self.latent_dirs = []
        for layer in self.stylegan2_s_dims:
            dim = self.stylegan2_s_dims[layer]
            w = torch.randn(1, dim, device=self.device) * 0.0
            if layer == layer_idx:
                w[:, channel_idx] = 1.0
            self.latent_dirs.append(w)
