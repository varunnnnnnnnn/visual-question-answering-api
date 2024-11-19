import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

# Set page layout to wide
st.set_page_config(layout="wide")

# Load the processor and model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image, text):
    try:
        # Load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")

        # Prepare inputs for the model
        encoding = processor(img, text, return_tensors="pt")

        # Forward pass through the model
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return answer

    except Exception as e:
        return f"Error: {str(e)}"

# Set up the Streamlit app
st.title("Visual Question Answering")
st.write("Upload an image and enter a question to get an AI-generated answer.")

# Create columns for image upload and input fields
col1, col2 = st.columns(2)

# Image upload section
with col1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    # Display the image if uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    else:
        st.warning("Please upload an image.")

# Question input and answer display section
with col2:
    question = st.text_input("Enter your question:")

    if uploaded_file and question:
        if st.button("Ask Question"):
            try:
                # Convert the image to byte array
                image_byte_array = BytesIO()
                image.save(image_byte_array, format='JPEG')
                image_bytes = image_byte_array.getvalue()

                # Get the answer from the model
                answer = get_answer(image_bytes, question)

                # Display the answer
                st.success(f"Answer: {answer}")
            except Exception as e:
                st.error(f"Error processing the question: {str(e)}")
    else:
        st.info("Please upload an image and enter a question to proceed.")
