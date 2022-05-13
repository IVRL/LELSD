import datetime
import os
import numpy as np
import streamlit as st
import torch
import clip
from PIL import ImageOps, Image, ImageFont, ImageDraw

import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
  sys.path.append(module_path)

import models
from cliplsd import CLIPLSD
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
def get_clip_model(model_name):
    if model_name in clip.available_models():
        clip_model, _ = clip.load(model_name)
    else:
        clip_model = None
    return clip_model


@st.cache(ttl=None, allow_output_mutation=True, max_entries=1,
          hash_funcs={
              StyleGAN2SampleGenerator: lambda s: hash(s.truncation_psi),
          })
def get_batch_data(sample_generator, seed, model_name, dataset_name):
    batch_data = sample_generator.generate_batch(seed, return_image=True, return_style=True, batch_size=1)
    return batch_data


exp_dir = "../out/"

stylegan2_part2cliplsd_model_paths = {
    "FFHQ": "cliplsd_stylegan2_ffhq",
    "LSUN Car": "cliplsd_stylegan2_lsun_car",
    "MetFaces": "cliplsd_stylegan2_metfaces",
    "LSUN Horse": "cliplsd_stylegan2_lsun_horse",
    "LSUN Church": "cliplsd_stylegan2_lsun_church",
}

model2dataset2part2cliplsd = {
#    "stylegan1": stylegan1_part2cliplsd_model_paths,
    "stylegan2": stylegan2_part2cliplsd_model_paths,
}

st.title("1D Visualization of CLIPLSD")

model_name = st.sidebar.selectbox(
    "Choose the GAN type you want.",
    ['stylegan2'],
)

dataset_name = st.sidebar.selectbox(
    "Choose the dataset you want the pretrained model for",
    list(model2available_dataset[model_name].keys()),
)

truncation_psi = st.sidebar.slider(f'Truncation Psi', 0.01, 1.0, 0.7)  # min, max, default

model_path = os.path.join(exp_dir, model2dataset2part2cliplsd[model_name][dataset_name])
latent_space = st.sidebar.selectbox("Base Model Latent Space", os.listdir(model_path))
last_path = os.path.join(model_path, latent_space)
dim = st.sidebar.selectbox("Dimension", os.listdir(last_path))
last_path = os.path.join(last_path, dim)
clip_type = st.sidebar.selectbox("Clip Model Type", os.listdir(last_path))
last_path = os.path.join(last_path, clip_type)
semantic_text_path = st.sidebar.selectbox("Semantic Text", os.listdir(last_path))
last_path = os.path.join(last_path, semantic_text_path)
avaliable_dates = os.listdir(last_path)
dates_path = st.sidebar.selectbox("Date", sorted(avaliable_dates, key=lambda x: datetime.datetime.strptime(x, '%b%d_%H-%M-%S')))
base_cliplsd_path = os.path.join(last_path, "/model.pth")
base_cliplsd = CLIPLSD.load(base_cliplsd_path)

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

if base_cliplsd.latent_space.startswith("W"):
    layers_to_apply = list(
        map(int, st.sidebar.multiselect("Base CLIPLSD Layers to apply the change", list(range(base_cliplsd.n_layers)))))
elif base_cliplsd.latent_space.startswith("S"):
    choices = base_cliplsd.s_layers_to_apply
    if choices:
        layers_to_apply = list(map(int, st.sidebar.multiselect("Base CLIPLSD Layers to apply the change", choices)))
        if len(layers_to_apply) == 0:
            layers_to_apply = choices
    else:
        layers_to_apply = list(
            map(int,
                st.sidebar.multiselect("Base CLIPLSD Layers to apply the change", range(len(base_cliplsd.latent_dirs)))))
else:
    layers_to_apply = None

G = load_pretrained_model(model_name, dataset_name)
device = torch.device('cuda')
if model_name == 'stylegan2':
    sample_generator = StyleGAN2SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)
else:
    pass

random_seed = int(st.sidebar.text_input('Random seed for generating the image', value='1'))
original_batch_data = get_batch_data(sample_generator, random_seed, model_name, dataset_name)
original_image = original_batch_data['image'][0]
original_raw_image = original_batch_data['raw_image']

k = 2 * int(st.sidebar.text_input('Number of Images', value='2')) + 1
height = 196
width = 196
img_size = (height, width)

editing_model = base_cliplsd
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

editing_image = np.zeros(shape=(height, width * k , 3), dtype=np.uint8)
alpha_min_value, alpha_max_value = st.sidebar.slider(f'Alpha', min_value=min_value,
                                                       max_value=max_value, value=value)
alpha_mesh = np.linspace(alpha_min_value, alpha_max_value, k)
big_image = np.zeros(shape=(height, width * k, 3), dtype=np.uint8)

for i, alpha in enumerate(alpha_mesh):
    new_batch_data = editing_model.edit_batch_data(sample_generator, original_batch_data, latent_dir1_idx, alpha,
                                                    layers_to_apply)
    image = new_batch_data['image'][0]
    if i == k // 2:
        image = ImageOps.expand(image, border=int(24 / (1024 // image.size[0])), fill='red')

    image = image.resize(img_size)
    editing_image[0:height, i * width: (i + 1) * width, :] = np.array(image)

st.image(editing_image, caption='1D Visualization of CLIPLSD', use_column_width=False)
text = "Hyperparameters:\n"
text += f"Learning Rate: {base_cliplsd.learning_rate}\n"
text += f"Batch Size: {base_cliplsd.batch_size}\n"
text += f"Min alpha value: {base_cliplsd.min_alpha_value}\n"
text += f"Max alpha value: {base_cliplsd.max_alpha_value}\n"
text += f"Min alpha abs value: {base_cliplsd.min_alpha_abs_value}\n"
text += f"L2 lambda: {base_cliplsd.l2_lambda}\n"
text += f"ID lambda: {base_cliplsd.id_lambda}\n"
text += f"Localization lambda: {base_cliplsd.localization_lambda}\n"
st.markdown(body=text)
