#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import click
import torch

from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline

from flask import Flask,request,send_file,send_from_directory
import os
import sys

pipe = StableDiffusionAITPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=os.environ['HF_TOKEN'],
        ).to("cuda")
app = Flask(__name__)
import threading
import time
from io import StringIO,BytesIO
sem = threading.Semaphore()
    
@app.route("/")
def render():        
        vocab = ""
        prompt = request.args.get('prompt', '')        
        steps = int(request.args.get('steps', "50"))
        strength = float(request.args.get('strength', "0.8"))
        seed = int(request.args.get('seed', "0"))
        if(seed == 0):
            seed = None        
        init_image_seed = int(request.args.get('init_image_seed', "0"))                
        
        if(vocab == ""):
            vocab = None
        sem.acquire()
        try:
            with torch.autocast("cuda"):
                init_image = None
                if(init_image_seed != 0):
                    init_image = pipe(prompt,512,512,steps,7.5,0.0,None,None,'pil',True,vocab,0.0,None,init_image_seed).images[0]                
                else:
                    strength = 0.0
                image = pipe(prompt,512,512,steps,7.5,0.0,None,None,'pil',True,vocab,strength,init_image,seed).images[0]
        finally:
            sem.release()
        img_io = BytesIO()
        image.save(img_io, 'JPEG', quality=90)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')


@app.route("/rendermany")
def rendermany():
    html = "<html><body>"    
    prompt = request.args.get('prompt', '')
    steps = int(request.args.get('steps', "50"))
    seed = int(request.args.get('seed', "1"))
    init_image_seed = int(request.args.get('init_image_seed', "0"))
    strength = float(request.args.get('strength', "0.8"))
    n = int(request.args.get('n', "3"))
    scriptStr = f"function onImageClick(e){{var img = e.target;var seed = img.getAttribute('seed');window.location.href = '/rendermany?init_image_seed=' + seed + '&strength={strength}&prompt={prompt}&steps={steps*4}&seed={seed}&n={n}';}}"
    scriptStr = f"<script>{scriptStr}</script>"
    html += scriptStr

    
    steps = int(request.args.get('steps', "50"))
    for i in range(n):
        if(init_image_seed != 0):
            html += f"<img src='/?prompt={prompt}&steps={steps}&seed={seed+i}&strength={strength}&init_image_seed={init_image_seed}' seed='{seed + i}'></img>"
        else:
            html += f"<img src='/?prompt={prompt}&steps={steps}&seed={seed+i}' seed='{seed + i}' onclick='onImageClick(event)'></img>"
    html += "</body></html>"
    return html

# static files
@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static2', path)
if __name__ == '__main__':
    with torch.autocast("cuda"):
        image = pipe("warmup",512,512,2,7.5,0.0,None,None,'pil',True,None)
    app.run(host='0.0.0.0', port=5000, debug=True)