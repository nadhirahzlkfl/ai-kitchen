import os
import streamlit as st
import logging
from typing import Optional
from ultralytics import YOLO
from PIL import Image
from langflow.load import run_flow_from_json

# 💡 Force Langflow to use memory instead of an SQLite file
os.environ["LANGFLOW_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["LANGFLOW_DISABLE_DATABASE"] = "true"

# 📝 Ensure your flow JSON file exists in your project folder
FLOW_JSON_PATH = "AI Kitchen.json"
TWEAKS = {
    "OpenAIModel-2w2an": {},
    "Prompt-ScINz": {},
    "TextInput-UfZrq": {},
    "ChatInput-jIiOJ": {},
    "ChatOutput-IwGvB": {},
    "Memory-7zsPb": {}
}

logging.basicConfig(level=logging.INFO)

def run_flow(message: str, tweaks: Optional[dict] = None) -> dict:
    """Runs Langflow from JSON without a database."""
    try:
        result = run_flow_from_json(
            flow=FLOW_JSON_PATH,
            input_value=message,
            session_id="",  # ❌ No persistent session
            fallback_to_env_vars=True,
            tweaks=tweaks or {}
        )
        return result
    except Exception as e:
        logging.error(f"Error running Langflow: {str(e)}")
        return {"error": "Failed to run the AI process."}

def extract_message(response: dict) -> str:
    """Extracts AI response message."""
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

def process_image(image):
    """Processes the uploaded image using YOLO model and returns detected ingredients."""
    detected_classes = []
    model_path = os.path.join(os.getcwd(), 'best.pt') 
    model = YOLO(model_path)
    results = model(image)

    for result in results:
        detections = result.boxes
        for box in detections:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    return ", ".join(set(detected_classes)) if detected_classes else "No ingredients detected"

def main():
    st.title("AI Kitchen 👩🏻‍🍳")
    st.write("Your personal AI-powered cooking assistant!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
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
        detected_ingredients = process_image(image)
        response = run_flow(detected_ingredients, tweaks=TWEAKS)
        assistant_response = extract_message(response)

        # 📝 Save assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response, "avatar": "👩🏻‍🍳"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    if query := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": query, "avatar": "🎀"})
        with st.chat_message("user", avatar="🎀"):
            st.write(query)
        
        with st.chat_message("assistant", avatar="👩🏻‍🍳"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_response, "avatar": "👩🏻‍🍳"})

if __name__ == "__main__":
    main()
