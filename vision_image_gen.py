"""This script handles downloading, analyzing, and generating images using OpenAI's API with Vision and Dalle3.
You enter an image location either from local system or url, it downloads image, uses GPT Vision to look at image and create a prompt
for dalle, then based on your input of how you want to change the original image, it creates a new image and saves it."""

import os
import base64
import glob
from openai import OpenAI
import requests

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY")

def save_image(image_bytes, filename, output_dir="original_image"):
    """Saves image bytes to a file in the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'wb') as image_file:
        image_file.write(image_bytes)
    print(f"Image saved to {file_path}")

def download_and_encode_image(image_input):
    """Downloads an image from a URL or opens a local image, saves it, and returns the base64-encoded string."""
    if image_input.startswith(('http://', 'https://')):
        response = requests.get(image_input, timeout=10)
        image_bytes = response.content
        filename = image_input.split('/')[-1]
    else:
        with open(image_input, "rb") as image_file:
            image_bytes = image_file.read()
            filename = os.path.basename(image_input)

    save_image(image_bytes, filename)  # Save the image to the local directory
    return base64.b64encode(image_bytes).decode('utf-8') 


def get_image_analysis(base64_image):
    """Gets the image analysis/description using GPT-4 Vision."""
    try:
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        stream=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image as a prompt for a text to image model"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        max_tokens=250,
        )
    
        responses = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                responses += str(chunk.choices[0].delta.content)
        return responses.rstrip()
    except Exception as e:
        print(f"An error occurred during image analysis: {e}")
        return None
    
def modify_description(original_description):
    """Prompts the user for how they want to modify the image description."""
    modification = input("How would you like to modify the image? (e.g., 'add a red hat, change the background, ect..'): ").strip()
    if modification:
        new_description = f"{original_description}, modified to include {modification}"
    else:
        new_description = original_description
    return new_description

def generate_image_with_dalle(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="standard",
            response_format="b64_json",
            n=1,
        )
        b64_data = response.data[0].b64_json
        revised_prompt = response.data[0].revised_prompt
        return b64_data, revised_prompt
    except openai.BadRequestError as e:
        print(f"Error generating image: {e}")
        return None, None


def save_base64_image(b64_data, original_name, output_dir="generated_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.splitext(os.path.basename(original_name))[0] 
    pattern = os.path.join(output_dir, f"{base_filename}_generated_*.png")
    existing_files = glob.glob(pattern)
    highest_num = 0
    for f in existing_files:
        try:
            num = int(f.rsplit('_', 1)[-1].split('.')[0])
            highest_num = max(highest_num, num)
        except ValueError:
            pass  # In case of any naming anomalies, ignore and continue

    new_num = highest_num + 1
    new_filename = f"{base_filename}_generated_{new_num:02d}.png"
    file_path = os.path.join(output_dir, new_filename)

    img_data = base64.b64decode(b64_data)
    with open(file_path, 'wb') as f:
        f.write(img_data)
    print(f"Generated image saved as {file_path}")

if __name__ == "__main__":
    image_input = input("Enter the path to a local image or an image URL: ")
    base64_image = download_and_encode_image(image_input)
    
    original_description = get_image_analysis(base64_image)
    if original_description:
        print(f"\nOriginal Image description: {original_description}\n")
        
        modified_description = modify_description(original_description)
        print(f"\nModified description to generate: {modified_description}")
        
        b64_data, revised_prompt = generate_image_with_dalle(modified_description)
        if b64_data:
            print(f"Revised Prompt: {revised_prompt}")
            save_base64_image(b64_data, image_input)
        else:
            print("Unable to generate modified image due to an error.")
    else:
        print("Failed to get a valid description for the image.")
