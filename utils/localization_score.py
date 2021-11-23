import numpy as np

import lpips
import torch


class LocalizationLPIPS:
    """
    This class uses LPIPS metric to evaluate Localization Score
    Localization score is defined as the ratio of the change in
    the desired region (given by a mask), to the whole change.
    Parameters
    ----------
    net_type: {'alex', 'vgg'}, default='alex,
        The pretrained network used for evaluating LPIPS distance
    device: pytorch device
        Device used for performing the computation.
    input_size: tuple int, default=(256, 256) The input images will be resized
        to this size before evaluating LPIPS and localization score
    min_distance: float, default=1e-4
        Distance below this will be considered as zero. This value is used to
        deal with the presence of noise.
    """

    def __init__(self, device, net_type='alex', input_size=(256, 256), min_distance=1e-6):
        assert net_type in ['alex', 'vgg']
        self.net_type = net_type
        self.device = device
        self.loss_fn = lpips.LPIPS(net=self.net_type, verbose=False, spatial=True).to(device)
        self.input_size = input_size
        self.min_distance = min_distance

    @torch.no_grad()
    def get_spatial_distance(self, x1, x2, mask=None, threshold=False):
        """
        Parameters
        ----------
        x1: Torch tensor with shape (B, C, H, W)
            The values should be in range [-1, 1]
        x2: Torch tensor with shape (B, C, H, W)
            The values should be in range [-1, 1]
        mask: Torch tensor with shape (B, H, W), default=None
            Only used when threshold is set to True
            The values should be in range [0, 1]
        threshold: bool, default=False
            If True then spatial_distance will be thresholded 
            by the threshold that maximizes localization score
        """
        x1_resized = torch.nn.functional.interpolate(x1, size=self.input_size,
                                                     mode='bilinear',
                                                     align_corners=False)
        x2_resized = torch.nn.functional.interpolate(x2, size=self.input_size,
                                                     mode='bilinear',
                                                     align_corners=False)

        spatial_distance = self.loss_fn(x1_resized, x2_resized)[:, 0]  # B * H * W
        if threshold:
            mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(1), size=self.input_size,
                                                           mode='bilinear',
                                                           align_corners=False)[:, 0]
            spatial_distance[spatial_distance < self.min_distance] = 0.0
            batch_size = len(x1_resized)
            z_hat = spatial_distance.cpu().numpy().reshape([batch_size, -1])
            y = mask_resized.cpu().numpy().reshape([batch_size, -1])
            scores = []
            for i in range(batch_size):
                _, t_idx = get_maximum_score(y[i], z_hat[i])
                t = z_hat[i][t_idx]
                spatial_distance[spatial_distance < t] = 0.0

        else:
            spatial_distance[spatial_distance < threshold] = 0.0

        spatial_distance = spatial_distance.cpu().numpy()

        return spatial_distance

    @torch.no_grad()
    def get_lpips_distance(self, x1, x2):
        x1_resized = torch.nn.functional.interpolate(x1, size=self.input_size,
                                                     mode='bilinear',
                                                     align_corners=False)
        x2_resized = torch.nn.functional.interpolate(x2, size=self.input_size,
                                                     mode='bilinear',
                                                     align_corners=False)

        spatial_distance = self.loss_fn(x1_resized, x2_resized)[:, 0]
        lpips_distance = spatial_distance.mean().cpu().numpy()

        return lpips_distance

    @torch.no_grad()
    def get_localization_score(self, x1, x2, mask):
        """
        Parameters
        ----------
        x1: Torch tensor with shape (B, C, H, W)
            The values should be in range [-1, 1]
        x2: Torch tensor with shape (B, C, H, W)
            The values should be in range [-1, 1]
        mask: Torch tensor with shape (B, H, W)
            The values should be in range [0, 1]
        """
        x1_resized = torch.nn.functional.interpolate(x1, size=self.input_size,
                                                     mode='bilinear',
                                                     align_corners=False)
        x2_resized = torch.nn.functional.interpolate(x2, size=self.input_size,
                                                     mode='bilinear',
                                                     align_corners=False)
        mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(1), size=self.input_size,
                                                       mode='bilinear',
                                                       align_corners=False)[:, 0]

        spatial_distance = self.loss_fn(x1_resized, x2_resized)[:, 0]  # B * H * W
        lpips_distance = torch.mean(spatial_distance, dim=[1, 2])

        #         if threshold:
        #         mask_ratio = torch.mean(mask_resized, dim=[1, 2])
        #         quantiles = []
        #         for i in range(len(mask_ratio)):
        #             q = torch.quantile(spatial_distance[i], 1 - mask_ratio[i]).item()
        #             q = max(q, threshold)
        #             quantiles.append(q)

        #         quantiles = torch.FloatTensor(quantiles)[:, None, None].to(self.device)
        #         thresholded_spatial_distance = (spatial_distance >= quantiles) * 1.0
        #         localization_score = torch.mean(thresholded_spatial_distance * mask_resized, dim=[1, 2]) / mask_ratio
        #         else:
        #             localization_score = torch.mean(spatial_distance * mask_resized, dim=[1, 2]) / lpips_distance

        #############################
        spatial_distance[spatial_distance < self.min_distance] = 0.0
        batch_size = len(x1_resized)
        z_hat = spatial_distance.cpu().numpy().reshape([batch_size, -1])
        y = mask_resized.cpu().numpy().reshape([batch_size, -1])
        scores = []
        for i in range(batch_size):
            score, _ = get_maximum_score(y[i], z_hat[i])
            scores.append(score)

        localization_score = torch.FloatTensor(scores).to(self.device)

        #############################

        lpips_distance = lpips_distance.cpu().numpy()
        localization_score = localization_score.cpu().numpy()
        return lpips_distance, localization_score


def get_maximum_score(y, z_hat):
    """
    Parameters
    ----------
    y: Numpy array with shape (N, )
        The values should be binary in {0, 1}
    z_hat: Numpy array with shape (N, )
        higher values represent higher confidence
    
    This function evaluates score for different thresholding
    values on z i.e.  Score(y, z > t) and returns the maximum score.
    The scoring function can be F1 (harmonic mean) or any other function 
    combining precision and recall.
    """

    #     TP = 0
    #     T = 0
    #     N = np.sum(y)

    #     F1_scores = []
    #     sorted_ids = np.argsort(-z_hat)
    #     TPs = np.cumsum(y[sorted_ids])
    #     T = np.arange(1, len(y) + 1, 1)
    #     precision = TPs / T
    #     recall = TPs / N
    #     F1_scores = precision * recall / (precision + recall + 1e-7)
    # #     F1_scores = np.power(np.power(precision, 3) * np.power(recall, 1), 1 / 4)
    #     return np.max(F1_scores), sorted_ids[np.argmax(F1_scores)]

    TP = 0
    T = 0
    N = np.sum(y)

    F1_scores = []
    sorted_ids = np.argsort(-z_hat)
    TPs = np.cumsum(y[sorted_ids])
    T = np.arange(1, len(y) + 1, 1)
    precision = TPs / T
    recall = TPs / N
    F1_scores = precision * recall / (precision + recall + 1e-7)
    #     F1_scores = np.power(np.power(precision, 3) * np.power(recall, 1), 1 / 4)
    return F1_scores[int(N)], sorted_ids[int(N)]

# import cv2
# import colour
# import numpy as np

# def color_delta_E(image1, image2, method='cie2000'):
#         assert method in ['cie1976', 'cie1994', 'cie2000', 'CMC', 'euclidean']
#         image1_rgb = np.array(image1)
#         image2_rgb = np.array(image2)
#         if method == 'euclidean':
#             x1 = image1_rgb.astype(np.float32)
#             x2 = image2_rgb.astype(np.float32)
#             delta_E = np.sqrt(np.sum(np.square(x1 - x2), axis=-1)) / np.sqrt(3.0)
#         else:
#             image1_lab = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2Lab)
#             image2_lab = cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2Lab)
#             delta_E = colour.delta_E(image1_lab, image2_lab, method=method)
#         return delta_E

# def locality_score(image1, image2, mask1, mask2, distance_metric):
#     distance = distance_metric(image1, image2)
#     mask = 0.5 * mask1  + 0.5 * mask2
#     if mask.shape != distance.shape:
#         mask = cv2.resize(mask, distance.shape)

#     sum_distance = np.sum(distance)
#     if sum_distance == 0:
#         return 0

#     return np.sum(distance * mask) / sum_distance, np.mean(distance)
