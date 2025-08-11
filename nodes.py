import io
import os
import base64
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import folder_paths
import openai

# -------------------------------
# Load environment variables
# -------------------------------
# Detect ComfyUI root
comfyui_root = os.path.dirname(folder_paths.base_path) if hasattr(folder_paths, 'base_path') \
    else os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Potential .env locations
env_paths = [
    os.path.join(comfyui_root, '.env'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'),
    os.path.join(os.getcwd(), '.env'),
    '.env'
]

# Try loading from known locations
env_loaded = False
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path, override=True)
        print(f"[OpenAICaptionImage] Loaded .env from: {path}")
        env_loaded = True
        break

if not env_loaded:
    print(f"[OpenAICaptionImage] Warning: .env not found in: {env_paths}")
    load_dotenv()  # fallback

# -------------------------------
# Model choices
# -------------------------------
MODELS = [
    "gpt-4o",            # Vision + text
    "chatgpt-4o-latest", # ChatGPT 4o variant
    "gpt-5"        # Text-only (no vision)
]

# -------------------------------
# Node definition
# -------------------------------
class OpenAICaptionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {}),
                "model": (MODELS,),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant."}),
                "caption_prompt": ("STRING", {"default": "What's in this image?"}),
                "max_tokens": ("INT", {"default": 300}),
                "temperature": ("FLOAT", {"default": 0.5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    CATEGORY = "openai"
    FUNCTION = "caption"

    def caption(self, image_in, model, system_prompt, caption_prompt, max_tokens, temperature):
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY not found. Checked: {env_paths}"
            )

        # Set global API key (avoids proxies error)
        openai.api_key = api_key

        # Convert tensor to PIL Image
        pil_image = Image.fromarray(
            np.clip(255.0 * image_in.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

        # Convert PIL Image to Base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Make API call
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_str}"}
                            }
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

        # Validate and return caption
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("No caption text returned from OpenAI API.")

        caption = response.choices[0].message.content.strip()
        return (caption,)
