from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

print(os.environ['OPENAI_KEY'])

client = OpenAI(api_key=os.environ['OPENAI_KEY'])

response = client.images.generate(
    model='dall-e-2',
    prompt='langosta',
    size='1024x1024',
    quality='standard',
    n=1
)

image_url = response.data[0].url
print(image_url)
