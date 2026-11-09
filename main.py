import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# Define the custom dice coefficient function if used during training
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

# Load the trained segmentation model
model_path = "liver_segmentation_model.h5"  # Update this to your model file path
model = tf.keras.models.load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})
print("Model loaded successfully.")

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Loads an image, resizes it to target_size, and normalizes pixel values to [0, 1].
    Converts grayscale images to RGB if needed.
    """
    img = Image.open(image_path).resize(target_size)
    img_np = np.array(img) / 255.0  # Normalize to [0, 1]
    # Convert grayscale to RGB if necessary
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    elif img_np.shape[-1] != 3:
        img_np = img_np[..., :3]
    return img_np

# Path to the single JPEG image to be segmented
image_path = "volume-101_slice_204.jpg"  # Update with your image path

# Load and preprocess the image
input_image = load_and_preprocess_image(image_path)
# Expand dimensions to create a batch of size 1
input_image_batch = np.expand_dims(input_image, axis=0)

# Predict the segmentation mask
pred_mask = model.predict(input_image_batch)[0]
# Apply a threshold to get a binary mask (adjust threshold as needed)
pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)

# Display the input image and predicted segmentation mask side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_mask_binary.squeeze(), cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.show()
