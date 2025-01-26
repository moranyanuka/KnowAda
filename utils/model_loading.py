import os
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
)


def load_llm(config, quantize=True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        max_new_tokens=500
    ) if quantize else None

    device_map = 'cpu' if config['model_device'] == 'cpu' else 'auto'
    model = AutoModelForCausalLM.from_pretrained(
        config["model_ckpt"],
        device_map=device_map,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        cache_dir=config["cache_dir"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_ckpt"],
        cache_dir=config["cache_dir"],
        model_max_length=1024
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer, model

def load_vlm(config):
    device_map = 'cpu' if config['device'] == 'cpu' else 'auto'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        use_flash_attention_2=True,
    ) if config['quantize'] else None

    if 'llava' in config['model_base_ckpt'] and 'tiny' not in config['model_base_ckpt']:
        processor = AutoProcessor.from_pretrained(
            config["model_base_ckpt"],
            cache_dir=config["cache_dir"]
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            config["model_ckpt"],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            use_flash_attention_2=True,
            device_map=device_map,
            cache_dir=config["cache_dir"]
        )
    elif 'paligemma' in config['model_ckpt']:
        processor = AutoProcessor.from_pretrained(
            config["model_base_ckpt"],
            cache_dir=config["cache_dir"]
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            config["model_ckpt"],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            device_map=device_map,
            cache_dir=config["cache_dir"]
        )
    elif 'tinyllava' in config['model_ckpt']:
        model = AutoModelForCausalLM.from_pretrained(
            config['model_base_ckpt'],
            trust_remote_code=True,
            cache_dir=config['cache_dir']
        ).cuda()
        tinyllava_config = model.config
        processor = AutoTokenizer.from_pretrained(
            config['model_base_ckpt'],
            use_fast=False,
            model_max_length=tinyllava_config.tokenizer_model_max_length,
            padding_side=tinyllava_config.tokenizer_padding_side,
            cache_dir=config['cache_dir']
        )
    elif 'InternVL' in config['model_ckpt']:
        model = AutoModel.from_pretrained(
            config['model_base_ckpt'],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            cache_dir=config['cache_dir']
        ).eval().cuda()
        processor = AutoTokenizer.from_pretrained(
            config['model_base_ckpt'],
            trust_remote_code=True,
            use_fast=False
        )
    else:
        raise ValueError(f"Model {config['model_base_ckpt']} not supported")

    if 'adapter_ckpt' in config:
        print("loading adapter...")
        model.load_adapter(config['adapter_ckpt'])
        print("adapter loaded\n")

    return processor, model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )) for i in range(blocks)
    ]

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def process_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values