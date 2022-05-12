import torch


def _load_biggan_model(model_name='biggan-deep-256'):
    from models.biggan.pytorch_pretrained_biggan import BigGAN
    assert model_name in [
        'biggan-deep-128',
        'biggan-deep-256',
        'biggan-deep-512',
    ]
    G = BigGAN.from_pretrained(model_name).eval()
    return G


def _load_stylegan3_model(model_path):
    from models.stylegan3 import dnnlib
    from models.stylegan3 import legacy
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema']
    G = G.eval()
    return G


def _load_stylegan2_model(model_path):
    from models.stylegan3 import dnnlib
    from models.stylegan3 import legacy
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema']
    G = G.eval()
    return G


def _load_stylegan1_model(model_path):
    from models.stylegan1.stylegan1 import StyleGAN
    G = StyleGAN.load_from_pth(model_path)
    G = G.eval()
    return G


def _load_face_bisenet_model(model_path):
    """
    You can download the pretrained model from this repository
    https://github.com/zllrunning/face-parsing.PyTorch
    """
    from models.face_bisenet.model import BiSeNet
    model = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model


def _load_cocostuff_deeplabv2(model_path):
    from models.deeplab.deeplabv2 import DeepLabV2
    from models.deeplab.msc import MSC
    """
    You can download the pretrained model from this repository
    https://github.com/kazuto1011/deeplab-pytorch
    """

    def DeepLabV2_ResNet101_MSC(n_classes):
        return MSC(
            base=DeepLabV2(
                n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
            ),
            scales=[0.5, 0.75],
        )

    model = DeepLabV2_ResNet101_MSC(n_classes=182)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = model.eval()
    return model


def _load_mit_semseg_ADE20K(model_path):
    import os
    from models.mit_semseg.mit_models.models import ModelBuilder, SegmentationModule
    from models.mit_semseg.config import cfg

    cfg.merge_from_file(os.path.join(model_path, "config.yaml"))
    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    cfg.DIR = model_path
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    model = SegmentationModule(net_encoder, net_decoder, crit)
    model = model.eval()

    return model


def _load_gan_linear_segmentation(model_path):
    """
        You can download the pretrained model from this repository
        https://github.com/AtlantixJJ/LinearGAN
        Semantic segmentation using a linear transformation on GAN features.
    """
    from models.linear_segmentation.semantic_extractor import EXTRACTOR_POOL
    data = torch.load(model_path)
    model_type = data['arch']['type']
    model = EXTRACTOR_POOL[model_type](**data['arch'])
    model.load_state_dict(data["param"])
    model = model.eval()
    return model


def _load_arcface_model(model_name, model_path, **kwargs):
    def get_model_backbone():
        from models.arcface.iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
        from models.arcface.iresnet2060 import iresnet2060
        from models.arcface.mobilefacenet import get_mbf

        # resnet
        if model_name == "r18":
            return iresnet18(False, **kwargs)
        elif model_name == "r34":
            return iresnet34(False, **kwargs)
        elif model_name == "r50":
            return iresnet50(False, **kwargs)
        elif model_name == "r100":
            return iresnet100(False, **kwargs)
        elif model_name == "r200":
            return iresnet200(False, **kwargs)
        elif model_name == "r2060":
            return iresnet2060(False, **kwargs)
        elif model_name == "mbf":
            fp16 = kwargs.get("fp16", False)
            num_features = kwargs.get("num_features", 512)
            return get_mbf(fp16=fp16, num_features=num_features)
        else:
            raise ValueError()

    model = get_model_backbone()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.eval()
    return model





_model_factories = {
    'stylegan1': _load_stylegan1_model,
    'stylegan2': _load_stylegan2_model,
    'stylegan3': _load_stylegan3_model,
    'face_bisenet': _load_face_bisenet_model,
    'cocostuff_deeplab': _load_cocostuff_deeplabv2,
    "biggan": _load_biggan_model,
    "gan_linear_seg": _load_gan_linear_segmentation,
    "mit_semseg_ade20k": _load_mit_semseg_ADE20K,
    "arcface": _load_arcface_model,
}


def get_available_models():
    return _model_factories.keys()


def get_model(name, *args, **kwargs):
    return _model_factories[name](*args, **kwargs)
