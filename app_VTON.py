import gradio as gr
import argparse, torch, os
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from typing import List
from util.common import open_folder
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc

parser = argparse.ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for optimization memory consumption")
parser.add_argument("--fixed_vae", action="store_true", default=True,  help="Use fixed vae for FP16.")
args = parser.parse_args()

load_mode = args.load_mode
fixed_vae = args.fixed_vae

dtype = torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = 'yisol/IDM-VTON'
vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

dtypeQuantize = dtype

if load_mode in ('4bit','8bit'):
    dtypeQuantize = torch.float8_e4m3fn

ENABLE_CPU_OFFLOAD = args.lowvram
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

unet = None
pipe = None
UNet_Encoder = None
example_path = os.path.join(os.path.dirname(__file__), 'example')

# --- New function for auto-cropping on upload ---
def auto_crop_upload(editor_value, crop_flag):
    """
    When a user uploads an image (EditorValue) and if auto-cropping is enabled 
    (crop_flag is True) this function performs the cropping and resizing.
    It also updates the "auto_cropped" flag within the EditorValue.
    """
    if editor_value is None:
        return None
    if editor_value.get("background") is None:
        return editor_value
    try:
        img = editor_value["background"].convert("RGB")
        if crop_flag:
            width, height = img.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = img.crop((left, top, right, bottom))
            # Resize the cropped image to the proper dimensions used by the model.
            resized_img = cropped_img.resize((768, 1024))
            editor_value["background"] = resized_img
            # If there are any drawn layers (mask layers), crop and resize them too.
            if editor_value.get("layers"):
                new_layers = []
                for layer in editor_value["layers"]:
                    if layer is not None:
                        new_layer = layer.crop((left, top, right, bottom)).resize((768, 1024))
                        new_layers.append(new_layer)
                    else:
                        new_layers.append(None)
                editor_value["layers"] = new_layers
            # Optionally update "composite"
            editor_value["composite"] = resized_img
            editor_value["auto_cropped"] = True
    except Exception as e:
        print("Error in auto crop:", e)
    return editor_value

# --- Modified try-on function to check if auto-cropping was already applied ---
def start_tryon(dict, garm_img, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    print(f"Input dict from ImageEditor: {dict}")  # Debug: Print the input dict
    print(f"Full Input dict content: {dict}")  # NEW DEBUG PRINT - Print the entire dict

    if pipe is None:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtypeQuantize,
        )
        if load_mode == '4bit':
            quantize_4bit(unet)

        unet.requires_grad_(False)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        if load_mode == '4bit':
            quantize_4bit(image_encoder)

        if fixed_vae:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_pretrained(model_id,
                                                subfolder="vae",
                                                torch_dtype=dtype,
            )

        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            model_id,
            subfolder="unet_encoder",
            torch_dtype=dtypeQuantize,
        )

        if load_mode == '4bit':
            quantize_4bit(UNet_Encoder)

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)

        pipe_param = {
            'pretrained_model_name_or_path': model_id,
            'unet': unet,
            'torch_dtype': dtype,
            'vae': vae,
            'image_encoder': image_encoder,
            'feature_extractor': CLIPImageProcessor(),
        }

        pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
        pipe.unet_encoder = UNet_Encoder
        pipe.unet_encoder.to(pipe.unet.device)

        if load_mode == '4bit':
            if pipe.text_encoder is not None:
                quantize_4bit(pipe.text_encoder)
            if pipe.text_encoder_2 is not None:
                quantize_4bit(pipe.text_encoder_2)
    else:
        if ENABLE_CPU_OFFLOAD:
            need_restart_cpu_offloading = True

    torch_gc()
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    tensor_transfrom = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if need_restart_cpu_offloading:
        restart_cpu_offload(pipe, load_mode)
    elif ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict["background"].convert("RGB")

    if is_checked_crop:
        # If the image has already been auto-cropped via the upload event, then skip cropping.
        if not dict.get("auto_cropped", False):
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
            human_img = human_img_orig
            crop_size = human_img.size
    else:
        human_img = human_img_orig.resize((768, 1024))
        crop_size = human_img.size

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
        print("Auto-masking used")
    else:
        # Manual mask branch: extract the alpha channel from the drawn layer.
        if dict.get('layers') and len(dict['layers']) > 0 and dict['layers'][0] is not None:
            mask_layer = dict['layers'][0]
            if mask_layer.mode == "RGBA":
                mask_alpha = mask_layer.split()[-1]
            else:
                mask_alpha = mask_layer.convert("L")
            mask_alpha = mask_alpha.resize((768, 1024))
            print("Manual mask alpha extracted:", type(mask_alpha), mask_alpha.mode, mask_alpha.size)
            mask = pil_to_binary_mask(mask_alpha)
            print("Manual mask binary mask:", type(mask), mask.mode, mask.size)
        else:
            mask = Image.new('L', (768, 1024), 0)
            print("No manual mask provided, using default black mask")

    print("Mask before pipe:", type(mask), mask.mode, mask.size)

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args_apply = apply_net.create_argument_parser().parse_args((
        'show', 
        './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
        './ckpt/densepose/model_final_162be9.pkl', 
        'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_img = args_apply.func(args_apply, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
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
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )
                    pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, dtype)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, dtype)
                    results = []
                    current_seed = seed
                    for i in range(number_of_images):
                        if is_randomize_seed:
                            current_seed = torch.randint(0, 2**32, size=(1,)).item()
                        generator = torch.Generator(device).manual_seed(current_seed) if seed != -1 else None
                        current_seed = current_seed + i
                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device, dtype),
                            negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=pose_img_tensor.to(device, dtype),
                            text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                            cloth=garm_tensor.to(device, dtype),
                            mask_image=mask,
                            image=human_img,
                            height=1024,
                            width=768,
                            ip_adapter_image=garm_img.resize((768, 1024)),
                            guidance_scale=2.0,
                            dtype=dtype,
                            device=device,
                        )[0]
                        if is_checked_crop:
                            out_img = images[0].resize(crop_size)
                            human_img_orig.paste(out_img, (int(left), int(top)))
                            img_path = save_output_image(human_img_orig, base_path="outputs", base_filename='img', seed=current_seed)
                            results.append(img_path)
                        else:
                            img_path = save_output_image(images[0], base_path="outputs", base_filename='img')
                            results.append(img_path)
                    return results, mask_gray

garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    if "Jensen" in ex_human or "sam1 (1)" in ex_human:
        ex_dict = {}
        ex_dict['background'] = ex_human
        ex_dict['layers'] = None
        ex_dict['composite'] = None
        human_ex_list.append(ex_dict)

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## V14 - IDM-VTON 👕👔👚 improved by SECourses : 1-Click Installers Latest Version On : https://www.patreon.com/posts/103022942")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(
                sources='upload',
                type="pil",
                label='Human. Mask with pen or use auto-masking',
                interactive=True,
                height=550
            )
            # --- Attach auto-crop event: when an image is uploaded, if auto-crop is enabled,
            # the auto_crop_upload function will process the image and update it.
            imgs.upload(auto_crop_upload, inputs=[imgs, gr.Checkbox(value=True, label="Use auto-crop & resizing")], outputs=imgs)
            with gr.Row():
                category = gr.Radio(choices=["upper_body", "lower_body", "dresses"], label="Select Garment Category", value="upper_body")
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)", value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing", value=True)
            example = gr.Examples(
                inputs=imgs,
                examples_per_page=2,
                examples=human_ex_list
            )
        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path
            )
        with gr.Column():
            with gr.Row():
                masked_img = gr.Image(label="Masked image output", elem_id="masked-img", show_share_button=False)
            with gr.Row():
                btn_open_outputs = gr.Button("Open Outputs Folder")
                btn_open_outputs.click(fn=open_folder)
        with gr.Column():
            with gr.Row():
                image_gallery = gr.Gallery(label="Generated Images", show_label=True)
            with gr.Row():
                try_button = gr.Button(value="Try-on")
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=120, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1)
                is_randomize_seed = gr.Checkbox(label="Randomize seed for each generated image", value=True)
                number_of_images = gr.Number(label="Number Of Images To Generate (it will start from your input seed and increment by 1)", minimum=1, maximum=9999, value=1, step=1)

    try_button.click(
        fn=start_tryon,
        inputs=[imgs, garm_img, prompt, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images],
        outputs=[image_gallery, masked_img],
        api_name='tryon'
    )

image_blocks.launch(inbrowser=True, share=args.share)