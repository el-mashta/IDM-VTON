import sys
sys.path.append('./')
from PIL import Image, ImageDraw
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import cv2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# DressCode label map for category-specific masking
label_map = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3,
    "upper_clothes": 4, "skirt": 5, "pants": 6, "dress": 7,
    "belt": 8, "left_shoe": 9, "right_shoe": 10, "head": 11,
    "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
    "bag": 16, "scarf": 17,
}

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def get_agnostic_mask(parse_array, pose_data, category):
    """
    Generate category-specific agnostic mask for DressCode
    Adapted from inference_dc.py
    """
    height, width = parse_array.shape
    
    parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 3).astype(np.float32) + \
                (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                        (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["scarf"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                    (parse_array == 12).astype(np.float32) + \
                    (parse_array == 13).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                             (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                    (parse_array == 12).astype(np.float32) + \
                    (parse_array == 13).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    # Arms handling for upper body and dresses
    im_arms = Image.new('L', (width, height))
    arms_draw = ImageDraw.Draw(im_arms)
    
    if category in ['dresses', 'upper_body'] and pose_data is not None and len(pose_data) > 7:
        # Draw arms based on pose keypoints
        try:
            shoulder_right = tuple(np.multiply(pose_data[2, :2], [width/384, height/512]))
            shoulder_left = tuple(np.multiply(pose_data[5, :2], [width/384, height/512]))
            elbow_right = tuple(np.multiply(pose_data[3, :2], [width/384, height/512]))
            elbow_left = tuple(np.multiply(pose_data[6, :2], [width/384, height/512]))
            wrist_right = tuple(np.multiply(pose_data[4, :2], [width/384, height/512]))
            wrist_left = tuple(np.multiply(pose_data[7, :2], [width/384, height/512]))
            
            arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 
                          'white', 30, 'curve')
        except:
            pass

    # Dilate parse mask
    if height > 512:
        parse_mask = cv2.dilate(np.float32(parse_mask), np.ones((20, 20), np.uint16), iterations=5)
    elif height > 256:
        parse_mask = cv2.dilate(np.float32(parse_mask), np.ones((10, 10), np.uint16), iterations=5)
    else:
        parse_mask = cv2.dilate(np.float32(parse_mask), np.ones((5, 5), np.uint16), iterations=5)
    
    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    
    return Image.fromarray((parse_mask_total * 255).astype(np.uint8))

# Global variables for model caching
current_category = None
current_models = {}

def load_models_for_category(category):
    """Load appropriate models based on category"""
    global current_category, current_models
    
    if current_category == category and current_models:
        return current_models
    
    print(f"Loading models for category: {category}")
    
    if category in ['lower_body', 'dresses']:
        base_path = 'yisol/IDM-VTON-DC'
    else:
        base_path = 'yisol/IDM-VTON'
    
    # Load models
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)
    
    # For IDM-VTON-DC, we need to check if unet_encoder exists
    try:
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )
    except:
        # Fall back to main IDM-VTON for unet_encoder if not available in DC
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            'yisol/IDM-VTON',
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )
    
    unet_encoder.requires_grad_(False)
    
    # Load other components (shared between models)
    tokenizer_one = AutoTokenizer.from_pretrained(
        'yisol/IDM-VTON',  # Use main model for tokenizers
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        'yisol/IDM-VTON',
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained('yisol/IDM-VTON', subfolder="scheduler")
    
    text_encoder_one = CLIPTextModel.from_pretrained(
        'yisol/IDM-VTON',
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        'yisol/IDM-VTON',
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'yisol/IDM-VTON',
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        'yisol/IDM-VTON',
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    
    # Set requires_grad to False
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # Create pipeline
    pipe = TryonPipeline.from_pretrained(
        'yisol/IDM-VTON',
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = unet_encoder
    
    current_models = {
        'pipe': pipe,
        'unet': unet,
        'unet_encoder': unet_encoder
    }
    current_category = category
    
    return current_models

# Initialize shared components
base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

tensor_transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed, category):
    """Enhanced try-on function with category support"""
    
    # Load appropriate models for category
    models = load_models_for_category(category)
    pipe = models['pipe']
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict["background"].convert("RGB")

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        
        if category in ['lower_body', 'dresses']:
            # Use DressCode-specific masking
            parse_array = np.array(model_parse)
            pose_data = np.array(keypoints) if keypoints is not None else None
            mask = get_agnostic_mask(parse_array, pose_data, category)
        else:
            # Use original masking for upper body
            mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args((
        'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v',
        '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Category-specific prompts
                if category == 'lower_body':
                    prompt = f"model is wearing {garment_des} pants"
                    cloth_prompt = f"a photo of {garment_des} pants"
                elif category == 'dresses':
                    prompt = f"model is wearing {garment_des} dress"
                    cloth_prompt = f"a photo of {garment_des} dress"
                else:
                    prompt = f"model is wearing {garment_des}"
                    cloth_prompt = f"a photo of {garment_des}"
                
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    if not isinstance(cloth_prompt, List):
                        cloth_prompt = [cloth_prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            cloth_prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device, torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                        cloth=garm_tensor.to(device, torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768, 1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray

# Setup example paths
garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict = {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

# Gradio interface
image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚 + DressCode Support")
    gr.Markdown("Virtual Try-on with support for upper body, lower body, and dresses. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)
            with gr.Row():
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)", value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing", value=False)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            
            # Category selection dropdown
            category = gr.Dropdown(
                choices=["upper_body", "lower_body", "dresses"],
                value="upper_body",
                label="Garment Category",
                info="Select the type of garment for better results"
            )
            
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts",
                        show_label=False,
                        elem_id="prompt"
                    )
            
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path
            )
            
        with gr.Column():
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img", show_share_button=False)
        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img", show_share_button=False)

    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)

    try_button.click(
        fn=start_tryon,
        inputs=[imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed, category],
        outputs=[image_out, masked_img],
        api_name='tryon'
    )

image_blocks.launch()

