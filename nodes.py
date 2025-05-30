import random
import numpy as np
import torch
import torchaudio

from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC


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
                "model_path": ("STRING", {"default": "./ResembleAI/chatterbox"}),
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


class LoadChatterboxVCModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./ResembleAI/chatterbox"}),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ChatterboxTTS"

    def load_model(self, model_path):
        model = ChatterboxVC.from_pretrained(device="cuda")
        return (model,)


class ChatterboxPrompt:
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


class ChatterboxAudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "input_path": ("STRING", {"default": None}),
            }
        }

    RETURN_TYPES = ("AUDIOPROMPT",)
    RETURN_NAMES = ("audio_prompt_path",)
    FUNCTION = "load_audio_prompt"
    CATEGORY = "ChatterboxTTS"

    def load_audio_prompt(self, input_path):
        if input_path is None:
            audio_prompt_path = None
        else:
            audio_prompt_path = input_path
            
        return (audio_prompt_path,)


class ChatterboxTTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("PROMPT",),
                "audio_prompt_path": ("AUDIOPROMPT",),
                ""seed_num": ("INT", {"default": 0}),
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

    RETURN_TYPES = ("WAV", "SR",)
    RETURN_NAMES = ("wav", "sr",)
    FUNCTION = "generate_tts"
    CATEGORY = "ChatterboxTTS"

    def generate_tts(self, model, prompt, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
        
        if model is None:
            model = ChatterboxTTS.from_pretrained(device="cuda")
    
        if seed_num != 0:
            set_seed(int(seed_num))
    
        wav = model.generate(
            prompt,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfgw,
        )

        sr = model.sr
        
        return (wav, sr)


class LoadChatterboxAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "YOUR_AUDIO.wav"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "input_audio"
    CATEGORY = "ChatterboxTTS"

    def input_audio(self, audio_path):
        audio = audio_path
        return (audio,)


class LoadChatterboxTargetAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_audio_path": ("STRING", {"default": "YOUR_TARGET_VOICE.wav"}),
            }
        }

    RETURN_TYPES = ("TARGETVOICE",)
    RETURN_NAMES = ("target_voice_path",)
    FUNCTION = "input_target_audio"
    CATEGORY = "ChatterboxTTS"

    def input_target_audio(self, target_audio_path):
        target_voice_path = target_audio_path
        return (target_voice_path,)


class ChatterboxVC:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "audio": ("AUDIO",),
                "target_voice_path": ("TARGETVOICE",),
            }
        }

    RETURN_TYPES = ("WAV", "SR",)
    RETURN_NAMES = ("wav", "sr",)
    FUNCTION = "generate_vc"
    CATEGORY = "ChatterboxTTS"

    def generate_vc(self, model, audio, target_voice_path):

        if model is None:
            model = ChatterboxVC.from_pretrained(device="cuda")
    
        wav = model.generate(
            audio, target_voice_path=target_voice_path,
        )

        sr = model.sr
        
        return (wav, sr)


class SaveChatterboxAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wav": ("WAV",),
                "sr": ("SR",),
                "save_path": ("STRING", {"default": "output_audio.wav"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    CATEGORY = "ChatterboxTTS"

    def save_audio(self, wav, sr, save_path):

        torchaudio.save(save_path, wav, sr)
    
        return ()

