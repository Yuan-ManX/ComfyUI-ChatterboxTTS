# ComfyUI-ChatterboxTTS

ComfyUI-ChatterboxTTS is now available in ComfyUI, [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) is the first production-grade open-source TTS model.



## Installation

1. Make sure you have ComfyUI installed

2. Clone this repository into your ComfyUI's custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/Yuan-ManX/ComfyUI-ChatterboxTTS.git
```

3. Install dependencies:
```
cd ComfyUI-ChatterboxTTS
pip install chatterbox-tts
pip install -r requirements.txt
```


## Model


### Download Pretrained Models

Chatterbox TTS Pretrained [Models](https://huggingface.co/ResembleAI/chatterbox)


## Tips

- **General Use (TTS and Voice Agents):**
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.

