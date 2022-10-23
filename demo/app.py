from flask import Flask, render_template, redirect, url_for, request, jsonify

# import cv2 as cv2
# import numpy as np
# import urllib.request
# from PIL import Image
# import io
# from scipy.io import wavfile

# from pygame import mixer
from werkzeug.utils import secure_filename
import os
import whisper
import torch
from datetime import datetime

#wav2vec
from transformers.file_utils import cached_path, hf_bucket_url
import zipfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import librosa

# from odoo.http import request
app = Flask(__name__, template_folder='templates')

# model = whisper.load_model("tiny")
model = whisper.load_model("base")
state_dict = torch.load("checkpoint-epoch=0013.ckpt", map_location="cpu")['state_dict']
# change all key of state_dict to remove "model."
new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
print(f"Model whisper base loaded")


cache_dir = './cache/'
processor_w2v = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
model_w2v = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
# lm_file = hf_bucket_url("nguyenvulebinh/wav2vec2-base-vietnamese-250h", filename='vi_lm_4grams.bin.zip')
# lm_file = cached_path(lm_file,cache_dir=cache_dir)
# with zipfile.ZipFile(lm_file, 'r') as zip_ref:
#     zip_ref.extractall(cache_dir)
lm_file = cache_dir + 'vi_lm_4grams.bin'

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    # convert ctc blank character representation
    vocab_list[tokenizer.pad_token_id] = ""
    # replace special characters
    vocab_list[tokenizer.unk_token_id] = ""
    # vocab_list[tokenizer.bos_token_id] = ""
    # vocab_list[tokenizer.eos_token_id] = ""
    # convert space character representation
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet,
                                   language_model=LanguageModel(lm_model))
    return decoder

ngram_lm_model = get_decoder_ngram_model(processor_w2v.tokenizer, lm_file)
print("Huggingface model loaded")

@app.route('/')
def home():
   return render_template('home.html')


@app.route('/transcribe-whisper', methods=['POST'])
def transcribe():   
   files = request.files
   file = files.get('file')

   file_name = secure_filename(str(datetime.now())) + ".wav"
   file_path = os.path.join("audio_folder", file_name)
   file.save(file_path)

   audio = whisper.load_audio(file_path)
   audio = whisper.pad_or_trim(audio)
   print("Audio loaded and trimmed")

   mel = whisper.log_mel_spectrogram(audio).to(model.device)
   # options = whisper.DecodingOptions(fp16 = False, withlm=F, beam_size=1, 
   #      patience=1.0, lm_path="../../dataset_tokenized_3gram.binary", lm_alpha=0.75, lm_beta=0.0,
   #      without_timestamps=True, language="vi")
   options = whisper.DecodingOptions(fp16 = False, language="vi", without_timestamps=True)
   print("Model decoding...")
   result = whisper.decode(model, mel, options)

   return jsonify(result.text)

@app.route('/transcribe-w2v', methods=['POST'])
def transcribe_w2v():   
   files = request.files
   file = files.get('file')

   file_name = secure_filename(str(datetime.now())) + ".wav"
   file_path = os.path.join("audio_folder", file_name)
   file.save(file_path)

   speech, sr = librosa.load(file_path)
   speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
   input_values = processor_w2v(
      speech, 
      sampling_rate=16000, 
      return_tensors="pt").input_values
   logits = model_w2v(input_values).logits[0]
   pred_ids = torch.argmax(logits, dim=-1)

   print("Model decoding...")
   beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)
   return jsonify(beam_search_output)
 
if __name__ == '__main__':
   app.run(debug=True) 

   