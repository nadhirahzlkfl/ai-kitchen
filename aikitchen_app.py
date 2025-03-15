#%%
import streamlit as st
import requests
import json
import logging
from typing import Optional
from ultralytics import YOLO
from PIL import Image
import os

#%% constants
BASE_API_URL = "https://3131-2001-e68-5456-4913-3954-57ac-f3bb-c3.ngrok-free.app"
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

def run_flow(message: str,
             endpoint: str = FLOW_ID,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.
    """
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
        return {}

def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

def process_image(image):
    """
    Processes the uploaded image using YOLO model and returns detected ingredients.
    """
    detected_classes = []
    
    # use best.pt model
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

def main():
    # theme
    st.markdown(
    """
    <style>
    /* Main Background Color */
    body {
        background-color: #C8A2C8;  /* Change to your desired color */
    }

    /* Sidebar Background Color */
    .css-1d391kg {
        background-color: #C8A2C8;  /* Sidebar background color */
    }

    /* Sidebar Text Color */
    .css-1d391kg .sidebar-content {
        color: #222222;  /* Dark grey text in the sidebar */
        font-style: italic;
    }

    .css-1d391kg .sidebar .sidebar-item {
        color: #222222;  /* Dark grey text for sidebar items */
    }

    .css-1d391kg .sidebar .sidebar-item:hover {
        background-color: #FFBFF0;  /* Light pink hover effect for items */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.title("AI Kitchen 👩🏻‍🍳")
    st.write("Use the camera or upload an image to start")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("### AI Kitchen 👩🏻‍🍳")
        st.markdown(
        """
        **Welcome to AI Kitchen!**  
        🍳 **Upload a food image** or **take a picture**,  
        and AI will detect ingredients & suggest recipes!  
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
        
        # send detected ingredients to Langflow for recipe suggestion
        response = run_flow(detected_ingredients, tweaks=TWEAKS)
        assistant_response = extract_message(response)
        
        # AI immediately responds when image is uploaded
        ai_message = f"{assistant_response}"

        # add the AI response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_message,
            "avatar": "👩🏻‍🍳",
        })

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
                assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "avatar": "👩🏻‍🍳",
        })

if __name__ == "__main__":
    main()
