import datetime
import os
import numpy as np
import streamlit as st
import torch
from PIL import ImageOps, Image, ImageFont, ImageDraw

import sys

sys.path.append("../")
import models
from lelsd import LELSD
from utils.biggan_utils import BigGANSampleGenerator
from utils.stylegan1_utils import StyleGAN1SampleGenerator
from utils.stylegan2_utils import StyleGAN2SampleGenerator

model2available_dataset = {
    "stylegan1": {
        "FFHQ": "Gs_karras2019stylegan-ffhq-1024x1024.pt",
        "LSUN Bedroom": "Gs_karras2019stylegan-bedrooms-256x256.pt",
        "WikiArt Faces": "wikiart_faces.pt",
    },
    "stylegan2": {
        "FFHQ": "ffhq.pkl",
        "MetFaces": "metfaces.pkl",
        "LSUN Car": "stylegan2-car-config-f.pkl",
        "LSUN Horse": "stylegan2-horse-config-f.pkl",
        "LSUN Church": "stylegan2-church-config-f.pkl",
    },

}


@st.cache(ttl=None, allow_output_mutation=True, max_entries=2)
def load_pretrained_model(model_name, dataset_name):
    if model_name != "biggan":
        G = models.get_model(model_name,
                             f"../pretrained/{model_name}/{model2available_dataset[model_name][dataset_name]}")
    else:
        G = models.get_model(model_name, model2available_dataset[model_name][dataset_name])
    return G


@st.cache(ttl=None, allow_output_mutation=True, max_entries=2)
def get_segmentation_model(model_name):
    if model_name == "face_bisenet":
        face_bisenet = models.get_model("face_bisenet", "../pretrained/face_bisenet/model.pth")
        segmentation_model = FaceSegmentation(face_bisenet=face_bisenet, device=device)
    elif model_name == 'deeplab':
        deeplabv2_resnet101 = models.get_model("cocostuff_deeplab",
                                               "../pretrained/cocostuff_deeplab/deeplabv2_resnet101_msc-cocostuff164k-100000.pth")
        segmentation_model = StuffSegmentation(deeplabv2_resnet101=deeplabv2_resnet101, device=device)
    else:
        segmentation_model = None
    return segmentation_model


@st.cache(ttl=None, allow_output_mutation=True, max_entries=1,
          hash_funcs={
              StyleGAN1SampleGenerator: lambda s: hash(s.truncation_psi),
              StyleGAN2SampleGenerator: lambda s: hash(s.truncation_psi),
              BigGANSampleGenerator: lambda s: hash(s.truncation_psi),
          })
def get_batch_data(sample_generator, seed, model_name, dataset_name):
    batch_data = sample_generator.generate_batch(seed, return_image=True, return_style=True, batch_size=1)
    return batch_data


exp_dir = "../out/"

stylegan1_part2lelsd_model_paths = {
    #     "FFHQ": "lelsd_stylegan1_ffhq",
    #     "WikiArt Faces": "lelsd_stylegan1_wikiart_faces",
    #     "LSUN Bedroom": "lelsd_stylegan1_lsun_bedroom",
}

stylegan2_part2lelsd_model_paths = {
    "FFHQ": "lelsd_stylegan2_ffhq",
    "LSUN Car": "lelsd_stylegan2_lsun_car",
    "MetFaces": "lelsd_stylegan2_metfaces",
    "LSUN Horse": "lelsd_stylegan2_lsun_horse",
    "LSUN Church": "lelsd_stylegan2_lsun_church",
}

model2dataset2part2lelsd = {
    "stylegan1": stylegan1_part2lelsd_model_paths,
    "stylegan2": stylegan2_part2lelsd_model_paths,
}

st.title("2D Visualization of LELSD")

model_name = st.sidebar.selectbox(
    "Choose the GAN type you want.",
    ['stylegan1', 'stylegan2'],
)

dataset_name = st.sidebar.selectbox(
    "Choose the dataset you want the pretrained model for",
    list(model2available_dataset[model_name].keys()),
)

truncation_psi = st.sidebar.slider(f'Truncation Psi', 0.01, 1.0, 0.7)  # min, max, default

model_path = os.path.join(exp_dir, model2dataset2part2lelsd[model_name][dataset_name])
latent_space = st.sidebar.selectbox("Base Model Latent Space", os.listdir(model_path))
last_path = os.path.join(model_path, latent_space, "2D")
seg_type = st.sidebar.selectbox("Base Model Segmentation Model Type", os.listdir(last_path))
last_path = os.path.join(last_path, seg_type)
semantic_edit_part = st.sidebar.selectbox("Semantic Edit Part Name", os.listdir(last_path))
last_path = os.path.join(last_path, semantic_edit_part)
avaliable_dates = os.listdir(last_path)
last_date = sorted(avaliable_dates, key=lambda x: datetime.datetime.strptime(x, '%b%d_%H-%M-%S'))[-1]
base_lelsd_path = os.path.join(last_path, f"{last_date}/model.pth")
base_lelsd = LELSD.load(base_lelsd_path)

alpha_range_type = st.sidebar.selectbox(
    "Choose the alpha range",
    ["normal", "extreme"]
)
if alpha_range_type == "normal":
    min_value = -15
    max_value = 15
    value = (-5, 5)
else:
    min_value = -75
    max_value = 75
    value = (-30, 30)

if base_lelsd.latent_space.startswith("W"):
    layers_to_apply = list(
        map(int, st.sidebar.multiselect("Base LELSD Layers to apply the change", list(range(base_lelsd.n_layers)))))
elif base_lelsd.latent_space.startswith("S"):
    choices = base_lelsd.s_layers_to_apply
    if choices:
        layers_to_apply = list(map(int, st.sidebar.multiselect("Base LELSD Layers to apply the change", choices)))
        if len(layers_to_apply) == 0:
            layers_to_apply = choices
    else:
        layers_to_apply = list(
            map(int,
                st.sidebar.multiselect("Base LELSD Layers to apply the change", range(len(base_lelsd.latent_dirs)))))
else:
    layers_to_apply = None

G = load_pretrained_model(model_name, dataset_name)
device = torch.device('cuda')
if model_name == 'stylegan1':
    sample_generator = StyleGAN1SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)
elif model_name == 'stylegan2':
    sample_generator = StyleGAN2SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)
else:
    pass

random_seed = int(st.sidebar.text_input('Random seed for generating the image', value='982'))
original_batch_data = get_batch_data(sample_generator, random_seed, model_name, dataset_name)
original_image = original_batch_data['image'][0]
original_raw_image = original_batch_data['raw_image']

k = 2 * int(st.sidebar.text_input('Number of Images', value='2')) + 1
width = 196
height = 196
img_size = (width, height)

editing_model = base_lelsd
latent_dir1_idx = st.sidebar.selectbox(
    "Choose the index of the first latent direction.",
    list(range(editing_model.num_latent_dirs))
)
remaining_w_ids = list(range(editing_model.num_latent_dirs))
remaining_w_ids.remove(latent_dir1_idx)
latent_dir2_idx = st.sidebar.selectbox(
    "Choose the index of the second latent direction.",
    remaining_w_ids
)

editing_image = np.zeros(shape=(width * k, height * k, 3), dtype=np.uint8)
alpha1_min_value, alpha1_max_value = st.sidebar.slider(f'Alpha for latent dir1', min_value=min_value,
                                                       max_value=max_value, value=value)
alpha2_min_value, alpha2_max_value = st.sidebar.slider(f'Alpha for latent dir2', min_value=min_value,
                                                       max_value=max_value, value=value)
alpha1_mesh, alpha2_mesh = np.meshgrid(np.linspace(alpha1_min_value, alpha1_max_value, k),
                                       np.linspace(alpha2_min_value, alpha2_max_value, k))
big_image = np.zeros(shape=(width * k, height * k, 3), dtype=np.uint8)

for i in range(k):
    for j in range(k):
        alpha1 = alpha1_mesh[i, j]
        alpha2 = alpha2_mesh[i, j]
        new_batch_data = editing_model.edit_batch_data(sample_generator, original_batch_data, latent_dir1_idx, alpha1,
                                                       layers_to_apply)
        new_batch_data = editing_model.edit_batch_data(sample_generator, new_batch_data, latent_dir2_idx, alpha2,
                                                       layers_to_apply)
        image = new_batch_data['image'][0]
        if i == k // 2 and j == k // 2:
            image = ImageOps.expand(image, border=int(24 / (1024 // image.size[0])), fill='red')

        image = image.resize(img_size)
        editing_image[i * width: (i + 1) * width, j * height: (j + 1) * height, :] = np.array(image)

st.image(editing_image, caption='2D Visualization of LELSD', use_column_width=False)
