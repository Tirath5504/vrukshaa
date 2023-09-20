import os
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Define the paths for your dataset and output directory
dataset_root = '/home/tirath/Documents/vruksha/vrukshaa/BangladeshiCrops/BangladeshiCrops/Crop___Disease'
preprocessed_root = '/home/tirath/Documents/vruksha/vrukshaa/preprocessed_dataset'

# Create output directories for training, validation, and test sets within preprocessed_root
os.makedirs(os.path.join(preprocessed_root, 'train'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_root, 'validation'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_root, 'test'), exist_ok=True)

# Set the percentage of data for validation and test sets
validation_split = 0.15
test_split = 0.15
min_samples_for_split = 3

# Function to resize and normalize an image
def preprocess_image(image_path, output_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    cv2.imwrite(output_path, image)

# Loop through crop folders
for crop_folder in os.listdir(dataset_root):
    if os.path.isdir(os.path.join(dataset_root, crop_folder)):
        crop_path = os.path.join(dataset_root, crop_folder)
        output_crop_path = os.path.join(preprocessed_root, 'train', crop_folder)

        # Create crop folder in the training directory
        os.makedirs(output_crop_path, exist_ok=True)

        # Loop through disease folders within the crop folder
        for disease_folder in os.listdir(crop_path):
            if os.path.isdir(os.path.join(crop_path, disease_folder)):
                disease_path = os.path.join(crop_path, disease_folder)
                output_disease_path = os.path.join(output_crop_path, disease_folder)

                # Create disease folder in the training directory
                os.makedirs(output_disease_path, exist_ok=True)

                # List all image files in the disease folder
                image_files = [f for f in os.listdir(disease_path) if f.endswith('.jpg')]

                # Split the image files into training, validation, and test sets
                if len(image_files) >= min_samples_for_split:
                    train_files, test_val_files = train_test_split(image_files, test_size=0.3, random_state=42)
                    val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)

                    # Preprocess and copy images to the corresponding directories
                    for image_file in train_files:
                        image_path = os.path.join(disease_path, image_file)
                        preprocessed_image_path = os.path.join(preprocessed_root, 'train', crop_folder, disease_folder, image_file)
                        preprocess_image(image_path, preprocessed_image_path)

                    for image_file in val_files:
                        os.makedirs(os.path.join(preprocessed_root, 'validation', crop_folder, disease_folder), exist_ok=True)
                        image_path = os.path.join(disease_path, image_file)
                        preprocessed_image_path = os.path.join(preprocessed_root, 'validation', crop_folder, disease_folder, image_file)
                        preprocess_image(image_path, preprocessed_image_path)

                    for image_file in test_files:
                        os.makedirs(os.path.join(preprocessed_root, 'test', crop_folder, disease_folder), exist_ok=True)
                        image_path = os.path.join(disease_path, image_file)
                        preprocessed_image_path = os.path.join(preprocessed_root, 'test', crop_folder, disease_folder, image_file)
                        preprocess_image(image_path, preprocessed_image_path)

                else:
                    # If there are not enough samples, copy without preprocessing
                    for image_file in image_files:
                        src_image_path = os.path.join(disease_path, image_file)
                        dst_image_path = os.path.join(preprocessed_root, 'train', crop_folder, disease_folder, image_file)
                        shutil.copy(src_image_path, dst_image_path)

print("Data preparation and preprocessing complete.")
