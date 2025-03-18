import streamlit as st
import requests
import json
import logging
from typing import Optional
from ultralytics import YOLO
from PIL import Image
import os

# Constants
BASE_API_URL = "https://0053-2001-e68-5456-4913-f435-54cf-71a7-8b5b.ngrok-free.app"
FLOW_ID = "ce816029-b06a-4b56-9ca4-77ef44bdc839"

TWEAKS = {
    "OpenAIModel-2w2an": {},
    "Prompt-ScINz": {},
    "TextInput-UfZrq": {},
    "ChatInput-jIiOJ": {},
    "ChatOutput-IwGvB": {},
    "Memory-7zsPb": {}  # Using memory for storing ingredients
}

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Send message to Langflow's API
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
        return {}

# Extract response from Langflow API
def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

# Process image and detect ingredients using YOLO model
def process_image(image):
    detected_classes = []
    
    # Use best.pt model to detect ingredients
    model_path = os.path.join(os.getcwd(), 'best.pt') 
    model = YOLO(model_path)
    results = model(image)

    for result in results:
        detections = result.boxes
        for box in detections:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    detected_classes = list(set(detected_classes))  # Remove duplicates
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

    # Check if ingredients have already been detected and stored in session
    if "detected_ingredients" not in st.session_state:
        st.session_state.detected_ingredients = None

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
        
        # Detect ingredients
        detected_ingredients = process_image(image)
        
        # Store detected ingredients in session state (for future reference)
        st.session_state.detected_ingredients = detected_ingredients
        
        # Update Langflow memory with the detected ingredients
        TWEAKS["Memory-7zsPb"] = {"ingredients": detected_ingredients}
        
        # Send detected ingredients to Langflow for recipe suggestion
        response = run_flow(f"Give me a recipe using the ingredients: {detected_ingredients}", tweaks=TWEAKS)
        assistant_response = extract_message(response)
        
        # AI immediately responds when image is uploaded
        ai_message = f"{assistant_response}"

        # Add the AI response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_message,
            "avatar": "👩🏻‍🍳",
        })

    # Display chat history and handle further user queries
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Chat input for user queries
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
                # When ingredients are detected and stored in memory, use them in the query
                if st.session_state.detected_ingredients:
                    ingredients = st.session_state.detected_ingredients
                    query_with_ingredients = f"{ingredients}. {query}"
                    assistant_response = extract_message(run_flow(query_with_ingredients, tweaks=TWEAKS))
                else:
                    assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                
                message_placeholder.write(assistant_response)
        
        # Add the assistant's response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "avatar": "👩🏻‍🍳",
        })

if __name__ == "__main__":
    main()
