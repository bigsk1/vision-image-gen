"""
This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation or a visible attribution in the user interface would be appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.

Creator: bigsk1
"""


import os
import glob
import base64
from PIL import Image
import requests
from openai import OpenAI
import streamlit as st


# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your API key", type="password")
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"

# Initialize the OpenAI client with the API Key provided by the user or from the environment
# client = OpenAI(api_key=api_key)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY")
st.title('AI Image Analyzer and Generator')

st.sidebar.markdown("""
    ## Instructions
    - Provide your OpenAI API key above to authenticate.
    - Upload images from a url or locally.
    - Once image is downloaded provide detailed instructions that will be added to GPT Vision's prompt of the image.
    
    ## About
    This tool uses OpenAI's services to analyze images and generate new ones based on your descriptions. Ensure your API key is valid to use these features.
    
    ## Usage Tips
    - Use clear and descriptive modifications for best results.
    - Explore different descriptions to see how they influence the generated images.
    
    created by bigsk1
""")

def save_image(image_bytes, filename, output_dir="original_image"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'wb') as image_file:
        image_file.write(image_bytes)
    st.write(f"Image saved to {file_path}")

def download_and_encode_image(image_input):
    if image_input.startswith(('http://', 'https://')):
        response = requests.get(image_input, timeout=10)
        image_bytes = response.content
        filename = image_input.split('/')[-1]
    else:
        with open(image_input, "rb") as image_file:
            image_bytes = image_file.read()
            filename = os.path.basename(image_input)
    save_image(image_bytes, filename)  # Save the image to the local directory
    return base64.b64encode(image_bytes).decode('utf-8')  ## Assuming all your defined functions are here


def get_image_analysis_streamlit(base64_image):
    """Displays image analysis/description using GPT-4 Vision in Streamlit."""
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
        print("\n" + "="*50 + "\nVision Response:\n" + "="*50)
        print(responses.rstrip())
        return responses.rstrip()
    except Exception as e:
        print(f"An error occurred during image analysis: {e}")
        return None

def modify_description_streamlit(original_description):
    """Allows users to modify the image description using a Streamlit text input."""
    st.write(f"Original Description: {original_description}")
    modification = st.text_input("How would you like to modify the image description? (e.g., 'add a red hat, change the background, etc.'): ").strip()
    if modification:
        new_description = f"{original_description}, modified to include {modification}"
    else:
        new_description = original_description
    return new_description


def generate_image_with_dalle_streamlit(prompt):
    try:
        print("\n" + "="*50 + "\nFinal Prompt Sent to DALL-E 3:\n" + "="*50)
        print(prompt)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="standard",
            response_format="b64_json",
            n=1,
        )
        if response:
            b64_data = response.data[0].b64_json
            revised_prompt = response.data[0].revised_prompt
            st.session_state['b64_image'] = b64_data  # Store in session for display
            return b64_data, revised_prompt
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None, None



def save_base64_image_streamlit(b64_data, original_name, output_dir="generated_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.splitext(os.path.basename(original_name))[0]
    # Logic to create a unique filename
    existing_files = glob.glob(os.path.join(output_dir, f"{base_filename}_generated_*.png"))
    highest_num = 0
    for f in existing_files:
        try:
            num = int(f.rsplit('_', 1)[-1].split('.')[0])
            highest_num = max(highest_num, num)
        except ValueError:
            continue

    new_num = highest_num + 1
    new_filename = f"{base_filename}_generated_{new_num:02d}.png"
    file_path = os.path.join(output_dir, new_filename)

    img_data = base64.b64decode(b64_data)
    with open(file_path, 'wb') as f:
        f.write(img_data)
    
    st.success(f"Generated image saved as {new_filename}")


def show_image_gallery(directory="generated_images"):
    if os.path.exists(directory):
        images = os.listdir(directory)
        for image in images:
            # Use a combination of directory and image name to ensure the key is unique
            unique_key = f"{directory}_{image}"
            col1, col2 = st.columns([20, 1])  # Adjust the ratio as needed
            with col1:  # Image display column
                image_path = os.path.join(directory, image)
                st.image(image_path, caption=image, use_column_width=True)
            with col2:  # Deletion button column
                # Use the unique key for the button
                if st.button("X", key=unique_key):
                    os.remove(image_path)  # Delete the image file
                    st.rerun()  
    

def clear_inputs():
    # Explicitly clear the relevant session state keys
    keys_to_clear = ['image_input', 'modification', 'last_image_input', 'original_description', 'process_image']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()  # Force a rerun of the app to reset state



if __name__ == "__main__":

    # Initialize or reset session state keys
    if 'image_input' not in st.session_state:
        st.session_state['image_input'] = ""
    if 'modification' not in st.session_state:
        st.session_state['modification'] = ""

    # Input for image URL or path with session state management
    st.session_state['image_input'] = st.text_input("Enter the path to a local image or an image URL:", value=st.session_state['image_input'])

    # Process the image input if it's provided
    if st.session_state['image_input']:
        # Check if it's a new image input or the same as before
        if 'last_image_input' not in st.session_state or st.session_state['last_image_input'] != st.session_state['image_input']:
            base64_image = download_and_encode_image(st.session_state['image_input'])
            original_description = get_image_analysis_streamlit(base64_image)
            st.session_state['original_description'] = original_description
            st.session_state['last_image_input'] = st.session_state['image_input']
            st.session_state['modification'] = ""  # Reset modification input for new image

        # Show original description to the user
        if 'original_description' in st.session_state:
            st.write(f"Description from GPT Vision: {st.session_state['original_description']}")
        
        # Modification text input
        st.session_state['modification'] = st.text_input("Enter your description to add for Dalle here:", value=st.session_state['modification'])
        if 'modification' in st.session_state and st.session_state['modification']:
            # This ensures we only print when there's an actual modification provided.
            print("\n" + "="*50 + "\nUser's Modification Input:\n" + "="*50)
            print(st.session_state['modification'])

        
        # Generate modified image on button click
        if st.button("Generate Modified Image"):
            modified_description = f"{st.session_state['original_description']} {st.session_state['modification']}" if st.session_state['modification'] else st.session_state['original_description']
            b64_data, revised_prompt = generate_image_with_dalle_streamlit(modified_description)
            if b64_data:
                img_data = base64.b64decode(b64_data)
                st.image(img_data, caption="Generated Image", use_column_width=True)
                save_base64_image_streamlit(b64_data, st.session_state['image_input'])
            else:
                st.error("Unable to generate modified image due to an error.")

    # Clear button to reset inputs and state
    if st.button("Clear & Start Over"):
        clear_inputs()

    st.title("Image Gallery")


    # Expanders for viewing images in directories
    with st.expander("View Original Images"):
        show_image_gallery(directory="original_image")
    # Display the image gallery with deletion options
    with st.expander("View Generated Images"):
        show_image_gallery(directory="generated_images")

        
st.markdown("""
    ### How it Works:
    - The app first downloads the image from the provided URL or path locally and analyzes it using the pre-trained AI model gpt-4-vision-preview to generate a description.
    - You're then given the opportunity to modify this description to guide the image generation process, the original description from the vision model and your included description are used.
    - Finally, the app uses DALL-E 3 to generate a new image 1790x1024 based on the modified description.
    - You can see the original image and then newly created image. Right click to save. 
""")