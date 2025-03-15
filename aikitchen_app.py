#%%
import streamlit as st
import json
import logging
from typing import Optional
from ultralytics import YOLO
from PIL import Image
import os
from langflow.load import run_flow_from_json

#%% constants
FLOW_JSON_FILE = "AI Kitchen.json"

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

def run_flow(message: str) -> str:
    """
    Run Langflow locally from a JSON file instead of using an API.
    """
    try:
        result = run_flow_from_json(flow=FLOW_JSON_FILE,
                                    input_value=message,
                                    session_id="",  # Optional session tracking
                                    fallback_to_env_vars=True, 
                                    tweaks=TWEAKS)
        return result.get("outputs", "No response received")
    except Exception as e:
        return f"Error running Langflow: {str(e)}"

def extract_message(response: dict) -> str:
    try:
        return response if isinstance(response, str) else "No valid message found in response."
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
        
        # send detected ingredients to Langflow for recipe suggestion
        response = run_flow(detected_ingredients)
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
                assistant_response = extract_message(run_flow(query))
                message_placeholder.write(assistant_response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "avatar": "👩🏻‍🍳",
        })

if __name__ == "__main__":
    main()
