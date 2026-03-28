import os
import io
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

person_img = Image.open("image1.png")
product_img = Image.open("image2.png")

response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[
        "This is a photo of a person:", person_img,
        "This is the product:", product_img,
        "Combine these into a new high-quality image. The person from the first image "
        "should be smiling and naturally holding the product from the second image in their hands. "
        "The background should be a clean, modern studio setting."
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
    ),
)

image_found = False
for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        generated_img = Image.open(io.BytesIO(part.inline_data.data))
        generated_img.save("result_composition.png")
        print("Success! Combined image saved as result_composition.png")
        image_found = True

if not image_found:
    print("No image was generated. Check the model response for safety filters or errors.")