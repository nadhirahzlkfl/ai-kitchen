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
        # Check if the user uploaded a picture or used the camera to take a photo
        if picture:
            image = Image.open(picture)
        elif uploaded_file:
            image = Image.open(uploaded_file)
        
        # Only process the image if it's new (not already processed)
        if image:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Only process the image and detect ingredients if it's the first time or if a new image is uploaded
            if not st.session_state.detected_ingredients:
                detected_ingredients = process_image(image)
                st.session_state.detected_ingredients = detected_ingredients
            else:
                detected_ingredients = st.session_state.detected_ingredients

            # Send detected ingredients to Langflow for recipe suggestion
            response = run_flow(detected_ingredients, tweaks=TWEAKS)
            assistant_response = extract_message(response)
            
            # AI immediately responds when image is uploaded
            ai_message = f"{assistant_response}"
    
            # Add the AI response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_message,
                "avatar": "👩🏻‍🍳",
            })

    # Display chat history
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
                # Use previously detected ingredients if available
                if st.session_state.detected_ingredients:
                    query_with_ingredients = f"{query}. The detected ingredients are: {st.session_state.detected_ingredients}"
                    assistant_response = extract_message(run_flow(query_with_ingredients, tweaks=TWEAKS))
                else:
                    assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)
        
        # Add the assistant's response to the chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "avatar": "👩🏻‍🍳",
        })

if __name__ == "__main__":
    main()
