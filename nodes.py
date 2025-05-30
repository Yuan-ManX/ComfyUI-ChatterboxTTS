import random
import numpy as np
import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


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
