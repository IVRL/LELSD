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
from utils.localization_score import LocalizationLPIPS
from utils.segmentation_utils import FaceSegmentation, StuffSegmentation
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

st.title("Comparing LELSD with other models")

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
last_path = os.path.join(model_path, latent_space, "1D")
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

alpha_min_value, alpha_max_value = st.sidebar.slider(f'Alpha Range for Base LELSD', min_value=min_value,
                                                     max_value=max_value, value=value)  # min, max, default

comparison_dict = {
    "base LELSD": (base_lelsd, layers_to_apply, (alpha_min_value, alpha_max_value))

}

another_lelsd = st.sidebar.checkbox(label="Another LELSD", value=False, key='Another LELSD')
if another_lelsd:
    model_path = os.path.join(exp_dir, model2dataset2part2lelsd[model_name][dataset_name])
    latent_space = st.sidebar.selectbox("Another Model Latent Space", os.listdir(model_path))
    last_path = os.path.join(model_path, latent_space, "1D")
    another_lelsd_seg_type = st.sidebar.selectbox("Another Model Segmentation Model Type", os.listdir(last_path))
    last_path = os.path.join(last_path, another_lelsd_seg_type, semantic_edit_part)
    avaliable_dates = os.listdir(last_path)
    last_date = sorted(avaliable_dates, key=lambda x: datetime.datetime.strptime(x, '%b%d_%H-%M-%S'))[-1]
    another_lelsd_path = os.path.join(last_path, f"{last_date}/model.pth")
    another_lelsd = LELSD.load(another_lelsd_path)
    if another_lelsd.latent_space.startswith("S"):
        choices = another_lelsd.s_layers_to_apply
        if choices:
            another_lelsd_layers_to_apply = list(
                map(int, st.sidebar.multiselect("Another LELSD Layers to apply the change", choices)))
            if len(another_lelsd_layers_to_apply) == 0:
                another_lelsd_layers_to_apply = choices
        else:
            another_lelsd_layers_to_apply = list(
                map(int, st.sidebar.multiselect("Another LELSD Layers to apply the change",
                                                range(len(another_lelsd.latent_dirs)))))

    else:
        another_lelsd_layers_to_apply = layers_to_apply

    comparison_dict[f"{latent_space}/{another_lelsd_seg_type}"] = (
    another_lelsd, another_lelsd_layers_to_apply, (alpha_min_value, alpha_max_value))

checkbox_random_lelsd = st.sidebar.checkbox(label="Random LELSD", value=False, key='Random LELSD')
if checkbox_random_lelsd:
    random_lelsd = LELSD.load(base_lelsd_path).randomize_latent_dirs()
    comparison_dict["random LELSD"] = (random_lelsd, layers_to_apply, (alpha_min_value, alpha_max_value))

checkbox_stylespace = st.sidebar.checkbox(label="StyleSpace", value=False, key='StyleSpace')
if checkbox_stylespace:
    stylespace_model = LELSD.load(base_lelsd_path).randomize_latent_dirs()
    stylespace_layer_idx = int(st.sidebar.number_input("Layer Index", min_value=0, max_value=26, value=0))
    stylespace_channel_idx = int(st.sidebar.number_input("Channel Index", min_value=0, max_value=512, value=0))
    alpha_min_value, alpha_max_value = st.sidebar.slider(f'Alpha Range for StyleSpace', min_value=min_value,
                                                         max_value=max_value, value=value)
    stylespace_model.load_from_StyleSpace(stylespace_layer_idx, stylespace_channel_idx)
    comparison_dict["StyleSpace"] = (stylespace_model, [stylespace_layer_idx], (alpha_min_value, alpha_max_value))

G = load_pretrained_model(model_name, dataset_name)
device = torch.device('cuda')
if model_name == 'stylegan1':
    sample_generator = StyleGAN1SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)
elif model_name == 'stylegan2':
    sample_generator = StyleGAN2SampleGenerator(G=G, device=device, truncation_psi=truncation_psi)
else:
    pass

random_seed = int(st.sidebar.text_input('Random seed for generating samples', value='982'))
original_batch_data = get_batch_data(sample_generator, random_seed, model_name, dataset_name)
original_image = original_batch_data['image'][0]
original_raw_image = original_batch_data['raw_image']


def find_alpha(sample_generator, editing_model, latent_dir_idx, layers_to_apply, desired_lpips_distance,
               threshold=1e-4):
    if desired_lpips_distance == 0:
        return 0.0, 0.0
    alpha_min_value = 0
    if desired_lpips_distance < 0:
        step = -1.0
        desired_lpips_distance *= -1.0
    else:
        step = 1.0

    a = step
    last_alpha = alpha_min_value
    x1 = original_batch_data['raw_image']

    for i in range(30):
        alpha = a + alpha_min_value
        new_batch_data = editing_model.edit_batch_data(sample_generator, original_batch_data, latent_dir_idx, alpha,
                                                       layers_to_apply)
        x2 = new_batch_data['raw_image']
        new_image = new_batch_data['image']
        lpips_distance = metric.get_lpips_distance(x1, x2)
        print(alpha, lpips_distance)
        if abs(desired_lpips_distance - lpips_distance) < threshold:
            return alpha, lpips_distance * np.sign(alpha)
        elif desired_lpips_distance > lpips_distance:
            last_alpha = alpha
            a *= 2
        else:
            alpha_min_value = last_alpha
            a = a / 2

    return alpha, lpips_distance * np.sign(alpha)


k = 2 * int(st.sidebar.text_input('Number of Images', value='2')) + 1
width = 196
height = 196
img_size = (width, height)
metric = LocalizationLPIPS(device=device, net_type='alex', input_size=(1024, 1024), min_distance=1e-7)

checkbox_esp = st.sidebar.checkbox(label="Equally Spaced LPIPS", value=False, key='ESL')
if checkbox_esp:
    lpips_left_distance = float(st.sidebar.text_input('Maximum LPIPS distance in left direction', value='0.15'))
    lpips_right_distance = float(st.sidebar.text_input('Maximum LPIPS distance in right direction', value='0.15'))

for editing_model_name in comparison_dict:
    editing_model, layers_to_apply, alpha_range = comparison_dict[editing_model_name]
    latent_dir_idx = 0
    alpha_min_value, alpha_max_value = alpha_range

    editing_image = np.zeros(shape=(width * 1, height * k, 3), dtype=np.uint8)
    distance_heatmap_image = np.zeros(shape=(width * 1, height * k), dtype=np.uint8)

    max_distances = []
    for j, alpha in enumerate(np.linspace(alpha_min_value, alpha_max_value, k)):
        if checkbox_esp:
            desired_lpips_distance = np.linspace(-lpips_left_distance, lpips_right_distance, k)[j]
            alpha, lpips_distance = find_alpha(sample_generator, editing_model, latent_dir_idx, layers_to_apply,
                                               desired_lpips_distance, threshold=1e-4)

        new_batch_data = editing_model.edit_batch_data(sample_generator, original_batch_data, latent_dir_idx, alpha,
                                                       layers_to_apply)
        image = new_batch_data['image'][0]
        raw_image = new_batch_data['raw_image']

        if j == k // 2:
            image = ImageOps.expand(image, border=int(24 / (1024 // image.size[0])), fill='red')

        image = image.resize(img_size)
        if checkbox_esp:
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("./Baskerville.ttc", size=20, index=1)
            draw.text((5, 170), "d={:.3f}".format(abs(lpips_distance)), (255, 255, 255), font=font)
        else:
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("./Baskerville.ttc", size=20, index=1)
            draw.text((5, 170), "a={:.1f}".format(alpha), (255, 255, 255), font=font)

        editing_image[:width, j * height: (j + 1) * height, :] = np.array(image)

    st.image(editing_image, caption=f'Editing Model = {editing_model_name}', use_column_width=False)
