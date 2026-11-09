from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
# Set matplotlib to use non-interactive backend before importing it
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from skimage import io
from langchain.prompts import PromptTemplate
import tempfile
from PIL import Image
import openai

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Enable CORS
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define dice coefficient function for model loading
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

# Load the segmentation model
model_path = 'liver_segmentation_model.h5'
model = load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})
print("Model loaded successfully.")

# Define base prompt
base_prompt = """
You are an advanced AI assistant specialized in providing detailed technical information to assist radiologists and healthcare providers in interpreting liver CT scans. Avoid disclaimers about being an AI or needing human professionals unless absolutely necessary. Focus on providing concise, relevant, and helpful technical information. Your task is to analyze the given liver CT scan data and present detailed information based on the detected features.

If asked about the overall quality of a liver CT scan, respond by discussing factors such as image clarity, contrast, and potential artifacts based on the data provided, without mentioning any limitations about AI's ability to visually assess images.
If a liver CT scan is provided, include the following details in your analysis:
1. The area of the detected region in square millimeters.
2. The coordinates of the bounding box around the detected region.
3. The centroid (center point) of the detected region.
4. A brief comment on the overall quality of the scan based on the data available.

If the detected region's area is very small (close to zero), indicate that the scan possibly doesn't show any tumor.

Provide clear and detailed technical information based on the provided scan data. If no liver CT scan is provided, respond to general questions about liver CT scans and their interpretation in a helpful and informative manner. Note that this information is intended to support radiologists and healthcare providers in their work.
"""

# Create LangChain prompt template
prompt_template = PromptTemplate(
    template="{base_prompt}\nUser: {user_input}\nChatbot:",
    input_variables=["base_prompt", "user_input"]
)

# Initialize conversation history
conversation_history = []

# Function to load and preprocess image
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Loads an image, resizes it to target_size, and normalizes pixel values to [0, 1].
    Converts grayscale images to RGB if needed.
    """
    if isinstance(image_path, str):
        img = Image.open(image_path).resize(target_size)
        img_np = np.array(img) / 255.0  # Normalize to [0, 1]
    else:
        # If image is already loaded as numpy array
        img = cv2.resize(image_path, target_size)
        img_np = img / 255.0 if img.max() > 1.0 else img

    # Convert grayscale to RGB if necessary
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    elif img_np.shape[-1] != 3:
        img_np = img_np[..., :3]
    
    return img_np

# Function to save the results
def save_results(img, pred_mask, save_path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(pred_mask.squeeze(), alpha=0.5, cmap='jet')
    plt.title('Image with Predicted Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to save the image with bounding box
def save_bounding_box(img, pred_mask, save_path):
    binary_mask = (pred_mask.squeeze() > 0.5).astype(np.uint8)
    # Create a copy of the image for drawing
    img_with_bbox = img.copy()
    if img_with_bbox.max() <= 1.0:
        img_with_bbox = (img_with_bbox * 255).astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    for contour in contours:
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            cv2.circle(img_with_bbox, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
            
            # Store feature information
            area = cv2.contourArea(contour)
            features.append({
                'area': area,
                'bounding_box': (x, y, w, h),
                'centroid': (centroid_x, centroid_y)
            })
    
    plt.figure()
    plt.imshow(img_with_bbox)
    plt.title('Image with Bounding Box')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    
    return features

# Function to analyze image using segmentation model
def analyze_image(image_path):
    # Load and preprocess the image
    input_image = load_and_preprocess_image(image_path)
    
    # Add batch dimension
    input_image_batch = np.expand_dims(input_image, axis=0)
    
    # Predict the segmentation mask
    pred_mask = model.predict(input_image_batch)
    
    # Apply threshold to get binary mask
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
    
    return input_image, pred_mask_binary

# Function to generate response using OpenAI GPT-4
extracted_features = ""

def generate_response(user_input, image_analysis=None):
    global extracted_features

    # Custom response for specific questions
    if "overall quality of the CT scan" in user_input.lower():
        return "Based on the analysis of the provided data, the scan appears to have been processed correctly with clear detection of regions of interest. However, for a full assessment of the scan quality, including clarity, contrast, and potential artifacts, a trained radiologist would need to visually inspect the scan."

    if image_analysis:
        # Convert extracted features to a summary string
        analysis_summary = "\n".join([
            f"Area: {feature['area']:.2f} mm², Bounding Box: {feature['bounding_box']}, Centroid: {feature['centroid']}"
            for feature in image_analysis
        ])
        # Update the global extracted_features variable
        extracted_features = f"The liver CT scan analysis indicates the following findings:\n{analysis_summary}"

    # Include the extracted features in the prompt if they exist
    prompt_content = f"{extracted_features}\n\n{user_input}" if extracted_features else user_input

    # Create the prompt using LangChain
    prompt = prompt_template.format(
        base_prompt=base_prompt,
        user_input=prompt_content
    )

    # Add user input to conversation history
    conversation_history.append({"role": "user", "content": prompt_content})

    try:
        # Send request to OpenAI GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.7,
            n=1,
            stop=None
        )

        if response:
            generated_text = response['choices'][0]['message']['content']
            # Ensure to extract the relevant part of the response
            chatbot_response = generated_text.strip()

            # Add chatbot response to conversation history
            conversation_history.append(
                {"role": "assistant", "content": chatbot_response})

            return chatbot_response
        else:
            return "Error: Could not generate response."
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return "Error: Failed to generate response."

# Endpoint to handle chatbot interactions
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = generate_response(user_input)
    return jsonify({'response': response})

# Endpoint to handle image uploads and analysis
@app.route('/upload', methods=['POST'])
def upload():
    global extracted_features

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file temporarily
        temp_path = os.path.join("/tmp", file.filename)
        file.save(temp_path)

        try:
            # Analyze the image
            input_image, pred_mask = analyze_image(temp_path)
            
            # Save results
            result_path = tempfile.mktemp(suffix='.png', dir='/tmp')
            save_results(input_image, pred_mask, result_path)
            
            # Save image with bounding box and extract features
            bbox_result_path = tempfile.mktemp(suffix='.png', dir='/tmp')
            features = save_bounding_box(input_image, pred_mask, bbox_result_path)
            
            # Generate response based on extracted features
            user_input = request.form.get('message', '')
            response = generate_response(user_input, image_analysis=features)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return jsonify({
                'response': response,
                'result_image': os.path.basename(result_path),
                'bbox_result_image': os.path.basename(bbox_result_path),
                'extracted_info': features
            })
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# Add this route to serve images
@app.route('/get_image')
def get_image():
    path = request.args.get('path')
    return send_from_directory('/tmp', path)

# Serve static files
@app.route('/')
def serve_index():
    return send_from_directory('', 'index.html')

@app.route('/styles.css')
def serve_css():
    return send_from_directory('', 'styles.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('', 'script.js')

if __name__ == '__main__':
    app.run(debug=True)
