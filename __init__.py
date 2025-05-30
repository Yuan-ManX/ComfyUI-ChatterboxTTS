from .nodes import LoadChatterboxTTSModel, LoadChatterboxVCModel, ChatterboxPrompt, ChatterboxAudioPrompt, ChatterboxTTS, LoadChatterboxAudio, LoadChatterboxTargetAudio, ChatterboxVC, SaveChatterboxAudio

NODE_CLASS_MAPPINGS = {
    "LoadChatterboxTTSModel": LoadChatterboxTTSModel,
    "LoadChatterboxVCModel": LoadChatterboxVCModel,
    "ChatterboxPrompt": ChatterboxPrompt,
    "ChatterboxAudioPrompt": ChatterboxAudioPrompt,
    "ChatterboxTTS": ChatterboxTTS,
    "LoadChatterboxAudio": LoadChatterboxAudio,
    "LoadChatterboxTargetAudio": LoadChatterboxTargetAudio,
    "ChatterboxVC": ChatterboxVC,
    "SaveChatterboxAudio": SaveChatterboxAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadChatterboxTTSModel": "Load Chatterbox TTS Model",
    "LoadChatterboxVCModel": "Load Chatterbox VC Model",
    "ChatterboxPrompt": "Chatterbox Prompt",
    "ChatterboxAudioPrompt": "Chatterbox Audio Prompt",
    "ChatterboxTTS": "Chatterbox TTS",
    "LoadChatterboxAudio": "Load Chatterbox Audio",
    "LoadChatterboxTargetAudio": "Load Chatterbox Target Audio",
    "ChatterboxVC": "Chatterbox VC",
    "SaveChatterboxAudio": "Save Chatterbox Audio",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
