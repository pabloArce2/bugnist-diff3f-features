import os
import torch
from PIL import Image
import numpy as np
from diffusers import ControlNetModel
from unet_2d_condition import UNet2DConditionModel
from pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import cv2
from torchvision import transforms
import warnings


# The runwayml repository now redirects to stable-diffusion-v1-5/stable-diffusion-v1-5 on the Hub.
# DIFF3F_SD_MODEL can point at that mirror (or a local copy) if the redirect ever stops working.
DIFFUSION_MODEL_ID = os.environ.get("DIFF3F_SD_MODEL", "runwayml/stable-diffusion-v1-5")
ckpt = "diffusion_pytorch_model.fp16.safetensors"
repo = DIFFUSION_MODEL_ID

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


def rgb2canny(img):
    input_image = np.asarray(img)
    preprocessor = CannyDetector()
    low_threshold = 100
    high_threshold = 200
    detected_map = preprocessor(input_image, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)
    return detected_map


def sketch(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inverted_gray_image = 255 - gray_image
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    inverted_blurred_image = 255 - blurred_image
    pencil_sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
    inverted_pencil_sketch_image = 255 - pencil_sketch_image
    rgb = cv2.cvtColor(inverted_pencil_sketch_image, cv2.COLOR_GRAY2RGB) * 10
    return rgb


def rgb2normalmap(normal_map):
    normal_map = normal_map[:,:,0,:3].numpy()
    min_value = np.min(normal_map)
    max_value = np.max(normal_map)
    span = max(max_value - min_value, 1e-6)
    normalized_normal_map = np.where(normal_map != 0, (normal_map - min_value) / span, 0)
    normal_map_image = (normalized_normal_map * 255).astype(np.uint8)
    detected_map = HWC3(normal_map_image)
    return detected_map



def init_pipe(device, use_normal_map=True):
    depth_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
    )
    if use_normal_map:
        controlnet = [
            depth_controlnet,
            # ControlNetModel.from_pretrained(
            #     "lllyasviel/control_v11p_sd15_canny",
            #     torch_dtype=torch.float16,
            # ),
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_normalbae",
                torch_dtype=torch.float16,
            ),
        ]
    else:
        controlnet = depth_controlnet
    unet = UNet2DConditionModel.from_config(DIFFUSION_MODEL_ID, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo_id=repo, subfolder="unet", filename=ckpt)))
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        DIFFUSION_MODEL_ID,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_model_cpu_offload()
    except ImportError as exc:
        if "accelerate" not in str(exc):
            raise
        warnings.warn(
            "Diffusers could not enable accelerate-based CPU offload. "
            "Continuing with the full pipeline on the selected device; "
            "restart the notebook kernel if accelerate was installed recently.",
            RuntimeWarning,
        )
        pipe = pipe.to(device)
    # pipe.enable_xformers_memory_efficient_attention()
    return pipe


transform = transforms.ToPILImage()

def process_depth_map(depth):
    max_depth = depth.max()
    indices = depth == -1
    depth = max_depth - depth 
    depth[indices] = 0
    max_depth = depth.max()
    depth = depth / max_depth
    depth = transform(depth)
    return depth


def run_diffusion(
    pipe, 
    input_image,
    depth_map,
    prompt,
    normal_map_input=None,
    use_latent=False,
    num_images_per_prompt=1,
    return_image=False
):
    depth_map = process_depth_map(depth_map)
    # canny = Image.fromarray(np.uint8(rgb2canny(input_image)))
    if normal_map_input is not None:
        normal_map =  Image.fromarray(rgb2normalmap(normal_map_input))
        control_image = [depth_map, normal_map]
    else:
        control_image = depth_map
    generator = torch.manual_seed(60)
    pos_prompt = f"{prompt},best quality,highly detailed,photorealistic,photo"
    negative_prompt = "lowres,low quality,monochrome,watermark"
    output_type = "pil"
    if use_latent:
        output_type = "latent"
    output = pipe(
        pos_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        image=Image.fromarray(input_image),
        control_image=control_image,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=7,
        eta=1,
        output_type=output_type,
        return_image=return_image
        # generator=generator,
    ).images
    return output


def add_texture_to_render(
    pipe, input_image, depth_map, prompt, normal_map_input=None, use_latent=False, num_images_per_prompt=1, return_image=False
):
    return run_diffusion(
        pipe, input_image, depth_map, prompt, normal_map_input, use_latent=use_latent, num_images_per_prompt=num_images_per_prompt,return_image=return_image
    )
