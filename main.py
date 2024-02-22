import os
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import requests
import numpy as np

load_dotenv()

print(os.environ['OPENAI_KEY'])

client = OpenAI(api_key=os.environ['OPENAI_KEY'])


def generate_image(image_prompt):
    response = client.images.generate(
        model='dall-e-2',
        prompt=image_prompt,
        size='1024x1024',
        quality='standard',
        n=1
    )
    image_url = response.data[0].url
    image = Image.open(requests.get(image_url, stream=True).raw)
    return np.array(image)


gr_text_input = gr.Textbox(label="Prompt Dall-e-2", placeholder="Introduce un prompt")
gr_image_output = gr.Image(type="numpy",label="Imagen generada por Dall-e")

demo = gr.Interface(fn=generate_image, inputs=gr_text_input, outputs=gr_image_output)
demo.launch()
