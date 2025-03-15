# AI Kitchen App 👩🏻‍🍳

This is a Streamlit app built to assist in identifying ingredients in uploaded or camera-captured images, providing recipe suggestions based on the detected ingredients, and answering cooking-related queries. It uses YOLO (You Only Look Once) for object detection and Langflow for generating responses and recipe suggestions.

## Features

- Upload images or use camera to capture photos.
- Automatically detects ingredients using a pre-trained YOLO model (`best.pt`).
- Sends the detected ingredients to Langflow for recipe suggestions.
- Chat interface for querying the AI assistant for cooking tips and recipes.

## How It Works

### 1. Image Upload & Camera Capture

- Use the **camera input** or **file upload** option to add an image of ingredients.
- The app processes the image using a YOLO model and detects visible ingredients.

### 2. Recipe Suggestions & Assistant Chat

- Based on detected ingredients, the app sends a request to Langflow to get recipe suggestions.
- The app also has a chat feature where user can ask the AI assistant anything related to cooking or ingredients.

### 3. User Interaction

- User can interact with the assistant in the chat input field. The assistant responds with helpful suggestions and information.

## Code Explanation

### Constants

- **BASE_API_URL**: The base URL for the API used to get recipe suggestions.
- **FLOW_ID**: The ID of the flow used for querying Langflow's backend for recipe suggestions.
- **TWEAKS**: A dictionary containing optional flow tweaks.

### Key Functions

- **run_flow**: Sends a message to Langflow's API and returns the response.
- **extract_message**: Extracts the response message from Langflow's JSON response.
- **process_image**: Uses YOLO to detect ingredients in the uploaded image.

### Main Logic

1. The user uploads an image or takes a photo using the camera.
2. The app uses YOLO to detect ingredients in the image.
3. The app queries Langflow for recipe suggestions based on the detected ingredients.
4. The app displays the detected ingredients and recipe suggestions in a chat format.
5. The user can chat with the assistant for further cooking tips or queries.

## Prediction 
![Prediction](/static/output.png)

### Data Source

[Data Source](https://universe.roboflow.com/wonkeun-jung-vfcwn/ingredients-agbcq)

