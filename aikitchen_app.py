import streamlit as st
import requests
import json
import logging
from typing import Optional
from ultralytics import YOLO
from PIL import Image
import os

#%% constants
BASE_API_URL = "https://0053-2001-e68-5456-4913-f435-54cf-71a7-8b5b.ngrok-free.app"
FLOW_ID = "ce816029-b06a-4b56-9ca4-77ef44bdc839"

TWEAKS = {
    "OpenAIModel-2w2an": {},
    "Prompt-ScINz": {},
    "TextInput-UfZrq": {},
    "ChatInput-jIiOJ": {},
    "ChatOutput-IwGvB": {},
    "Memory-7zsPb": {}
}

# initialize logging
logging.basicConfig(level=logging.INFO)

# send message to Langflow's API
def run_flow(message: str,
             endpoint: str = FLOW_ID,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
   
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"
    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    if tweaks:
        payload["tweaks"] = tweaks

    headers = {"x-api-key": api_key} if api_key else None
    response = requests.post(api_url, json=payload, headers=headers)
    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Text: {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from the server response.")
        return {}  # return response

# extract response
def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

def process_image(image):
    detected_classes = []
    
    # use best.pt model to detect ingredients
    model_path = os.path.join(os.getcwd(), 'best.pt') 
    model = YOLO(model_path)
    results = model(image)

    for result in results:
        detections = result.boxes
        for box in detections:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    detected_classes = list(set(detected_classes))  # remove duplicates
    return ", ".join(detected_classes) if detected_classes else "No ingredients detected"

# function to create a text file for download
def create_recipe_file(recipe: str) -> str:
    file_name = "recipe.txt"
    with open(file_name, "w") as file:
        file.write(recipe)
    return file_name

# function to create a downloadable link in Streamlit
def get_download_button(file_path: str, label: str) -> None:
    with open(file_path, "rb") as file:
        st.download_button(
            label=label,
            data=file,
            file_name=os.path.basename(file_path),
            mime="text/plain"
        )

def main():
    st.markdown("""
    <style>
    [data-testid=stSidebar]{
        background-color: #A47DAB;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("AI Kitchen 👩🏻‍🍳")
    st.write("AI Kitchen is an intuitive app that helps you explore recipes and cooking ideas. Whether you're looking for inspiration or need some culinary advice, AI Kitchen is here to help you cook smarter and faster!")
    st.write("It’s like having a virtual chef at your fingertips! 🍲")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "detected_ingredients" not in st.session_state:
        st.session_state.detected_ingredients = []

    with st.sidebar:
        st.markdown("### Welcome to AI Kitchen! 🍳")
        st.markdown("""
        **Use the camera or upload an image to start**
        """
        )
        
        enable_camera = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable_camera)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
        image = None
        if picture:
            image = Image.open(picture)
        elif uploaded_file:
            image = Image.open(uploaded_file)
        
        if image:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # detect ingredients
            detected_ingredients = process_image(image)
            st.session_state.detected_ingredients = detected_ingredients
            
            # send detected ingredients to Langflow for recipe suggestion
            response = run_flow(detected_ingredients, tweaks=TWEAKS)
            assistant_response = extract_message(response)
            
            # AI immediately responds when image is uploaded
            ai_message = f"Based on the ingredients you provided: {detected_ingredients}, here's a recipe suggestion: {assistant_response}"
    
            # add the AI response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_message,
                "avatar": "👩🏻‍🍳",
            })
            
            # create the recipe file and provide download button
            recipe_file = create_recipe_file(ai_message)
            get_download_button(recipe_file, "Download Recipe")

    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # chat input for user queries
    if query := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "avatar": "🎀",
        })
        with st.chat_message("user", avatar="🎀"):
            st.write(query)
        
        with st.chat_message("assistant", avatar="👩🏻‍🍳"):
            message_placeholder = st.empty()
            with st.spinner("Let me think..."):
                # use previously detected ingredients if available
                if st.session_state.detected_ingredients:
                    query_with_ingredients = f"{query}. The detected ingredients are: {st.session_state.detected_ingredients}"
                    assistant_response = extract_message(run_flow(query_with_ingredients, tweaks=TWEAKS))
                else:
                    assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "avatar": "👩🏻‍🍳",
        })

if __name__ == "__main__":
    main()
