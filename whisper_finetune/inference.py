try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import whisper
import torch
import argparse

from config import Config
from model import WhisperModelModule
from utils import load_wave



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='', help='path of checkpoint, if not set, use origin pretrained model')
    parser.add_argument('--audio_path', type=str, default='test01.wav', help='the audio file for inference')

    args = parser.parse_args()
    config = Config()
    config.checkpoint_path = args.checkpoint_path

    module = WhisperModelModule(config)
    try:
        state_dict = torch.load(config.checkpoint_path)
        state_dict = state_dict["state_dict"]
        module.load_state_dict(state_dict)
        print(f"load checkpoint successfully from {config.checkpoint_path}")
    except Exception as e:
        print(e)
        print(f"load checkpoint failt using origin weigth of {config.model_name} model")
    model = module.model
    model.to(device)

    audio = whisper.load_audio(args.audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(
        language="vi", without_timestamps=True, fp16=torch.cuda.is_available()
    )

    result = model.decode(mel, options)
    print('Predicted:', result.text)

    