from flask import Flask, render_template, redirect, url_for, request, jsonify

import cv2 as cv2
import numpy as np
import urllib.request
from PIL import Image
import io
from scipy.io import wavfile

from pygame import mixer
from werkzeug.utils import secure_filename
import os

# from odoo.http import request
app = Flask(__name__, template_folder='templates')

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
   a = "haha"
   file = request.files['audio_data']
   print(file)

   filename = secure_filename(file.filename)
   filepath = os.path.join("audio_folder", filename+".wav")
   file.save(filepath)
   
   # mixer.init()
   # mixer.music.load('rXSee4hN.wav')
   # mixer.music.play()
   return "vietnamese"
 
if __name__ == '__main__':
   app.run(debug=True) 

   