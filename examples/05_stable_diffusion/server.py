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
from flask_caching import Cache

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}


import os
import sys
import PIL
pipe = StableDiffusionAITPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=os.environ['HF_TOKEN'],
        ).to("cuda")
app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)



import threading
import time
from io import StringIO,BytesIO
sem = threading.Semaphore()
    
@app.route("/")
def render():        
        vocab = ""
        param1 = request.args.get('param1', '')
        wildcard1 = request.args.get('wildcard1', '')
        prompt = request.args.get('prompt', '')
        steps = int(request.args.get('steps', "50"))
        strength = float(request.args.get('strength', "0.8"))
        seed = int(request.args.get('seed', "0"))
        if(seed == 0):
            seed = None        
        origImage = request.args.get('origImage', "")
        
        if(wildcard1 != ""):
            # replace all instances of wildcard1 with param1
            prompt = prompt.replace(wildcard1,param1)
            origImage = origImage.replace(wildcard1,param1)
            


        cacheKey = f"{prompt}_{steps}_{strength}_{seed}_{origImage}"
        cached = cache.get(cacheKey)
        if(cached is not None):
            return send_file(BytesIO(cached),mimetype='image/png')
        if(vocab == ""):
            vocab = None
        init_image = None
        if(origImage != ""):
            url = 'http://127.0.0.1:5000' + origImage
            init_image_data = app.test_client().get(url).data
            init_image_data = BytesIO(init_image_data)
            init_image = PIL.Image.open(init_image_data)
        sem.acquire()
        try:            
            cached = cache.get(cacheKey)
            if(cached is not None):
                return send_file(BytesIO(cached),mimetype='image/png')
            with torch.autocast("cuda"):                
                if(init_image == None):
                    strength = 0.0
                image = pipe(prompt,512,512,steps,7.5,0.0,None,None,'pil',True,vocab,strength,init_image,seed).images[0]
        finally:
            sem.release()
        img_io = BytesIO()
        image.save(img_io, 'JPEG', quality=90)
        img_io.seek(0)
        cache.set(cacheKey,img_io.getvalue())
        return send_file(img_io, mimetype='image/jpeg')

# static files
@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static2', path)


if __name__ == '__main__':
    with torch.autocast("cuda"):
        image = pipe("warmup",512,512,2,7.5,0.0,None,None,'pil',True,None)
    app.run(host='0.0.0.0', port=5000, debug=True,threaded=True)