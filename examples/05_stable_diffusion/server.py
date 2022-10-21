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

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline

from flask import Flask,request,send_file
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
sem = threading.Semaphore()

@app.route("/")
def render():
        sem.acquire()
        try:
            vocab = ""
            prompt = request.args.get('prompt', '')        
            steps = int(request.args.get('steps', "50"))
            if(vocab == ""):
                vocab = None
            with torch.autocast("cuda"):
                image = pipe(prompt,512,512,steps,7.5,0.0,None,None,'pil',True,vocab).images[0]
            image.save("/tmp/example_ait.png")
            return send_file("/tmp/example_ait.png", mimetype='image/png')            
        finally:
            sem.release()

@app.route("/rendermany")
def rendermany():
    html = "<html><body><h1>"
    prompt = request.args.get('prompt', '')
    steps = int(request.args.get('steps', "50"))
    for i in range(100):
        html += f"<img src='/?prompt={prompt}&steps={steps}&i={i}'></img>"
    html += "</h1></body></html>"
    return html
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
