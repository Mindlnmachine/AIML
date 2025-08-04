from openai import OpenAI
import user_config

client = OpenAI(api_key=user_config.openai_key)

response = client.images.generate(
    model = "dall-e-2",
    prompt="An elephant riding a bike",
    n=1,
    size="1024x1024",
    quality="standard",
)

print(response.data[0].url)

from openai import OpenAI
import user_config
client = OpenAI(api_key=user_config.openai_key)

response = client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    # quality="standard",
    n=1,
)

print(response.data[0].url)


