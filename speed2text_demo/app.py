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

# from odoo.http import request
app = Flask(__name__, template_folder='templates')

# model = whisper.load_model("tiny")
# model = whisper.load_model("base")
# state_dict = torch.load("checkpoint-epoch=0006.ckpt", map_location="cpu")['state_dict']
# change all key of state_dict to remove "model."
# new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict)
print(f"Model medium loaded")

@app.route('/')
def home():
   return render_template('home.html')


@app.route('/english', methods=['POST'])
def english():
   a = "haha"
   file = request.files['audio_data']
   print(file)

   filename = secure_filename(file.filename)
   filepath = os.path.join("audio_folder", filename+".wav")
   file.save(filepath)
   
   # mixer.init()
   # mixer.music.load('rXSee4hN.wav')
   # mixer.music.play()
   return "english"

@app.route('/vietnamese', methods=['POST'])
def vietnamese():
   file = request.files['audio_data']
   print(file)

   filename = secure_filename(file.filename)
   filepath = os.path.join("audio_folder", filename + ".wav")
   file.save(filepath)

   audio = whisper.load_audio(filepath)
   audio = whisper.pad_or_trim(audio)
   print("Audio loaded and trimmed")

   mel = whisper.log_mel_spectrogram(audio).to(model.device)
   # options = whisper.DecodingOptions(fp16 = False, withlm=F, beam_size=1, 
   #      patience=1.0, lm_path="../../dataset_tokenized_3gram.binary", lm_alpha=0.75, lm_beta=0.0,
   #      without_timestamps=True, language="vi")
   options = whisper.DecodingOptions(fp16 = False, language="vi", without_timestamps=True)
   print("Model decoding...")
   result = whisper.decode(model, mel, options)

   # mixer.init()
   # mixer.music.load('rXSee4hN.wav')
   # mixer.music.play()
   return result.text
 
if __name__ == '__main__':
   app.run(debug=True) 

   