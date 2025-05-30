import random
import numpy as np
import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class LoadChatterboxTTSModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./ChatterboxTTS"}),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ChatterboxTTS"

    def load_model(self, model_path):
        model = ChatterboxTTS.from_pretrained(device="cuda")
        return (model,)


class ChatterboxTTSPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "input_text"
    CATEGORY = "ChatterboxTTS"

    def input_text(self, text):
        prompt = text
        return (prompt,)


class ChatterboxTTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("PROMPT",),
                "exaggeration": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.25,
                        "max": 2,
                        "step": 0.05,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.05,
                        "max": 5,
                        "step": 0.05,
                    },
                ),
                ""seed_num": ("INT", {"default": 0}),
                "cfgw": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.2,
                        "max": 1,
                        "step": 0.05,
                    },
                ),
            }
        }

    RETURN_TYPES = ("TTS", "SR",)
    RETURN_NAMES = ("tts", "sr",)
    FUNCTION = "generate"
    CATEGORY = "ChatterboxTTS"

    def generate(self, model, prompt, exaggeration, temperature, seed_num, cfgw):

        if model is None:
            model = ChatterboxTTS.from_pretrained(device="cuda")
    
        if seed_num != 0:
            set_seed(int(seed_num))
    
        wav = model.generate(
            prompt,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfgw,
        )
    
        return (wav, model.sr)

