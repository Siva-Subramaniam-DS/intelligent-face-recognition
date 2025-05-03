import os
import shutil
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_organization.log'),
        logging.StreamHandler()
    ]
)

# Create the augmentation sequence with more realistic parameters
augmenter = iaa.Sequential([
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.3))),  # Reduced blur
    iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.02*255))),  # Reduced noise
    iaa.Sometimes(0.3, iaa.Affine(rotate=(-5, 5))),  # Reduced rotation
    iaa.Sometimes(0.3, iaa.Affine(scale=(0.95, 1.05))),  # Reduced scaling
    iaa.Sometimes(0.3, iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)})),  # Reduced translation
    iaa.Sometimes(0.3, iaa.AddToBrightness((-10, 10))),  # Reduced brightness change
    iaa.Sometimes(0.3, iaa.AddToHue((-10, 10))),  # Reduced hue change
    iaa.Sometimes(0.3, iaa.AddToSaturation((-10, 10))),  # Reduced saturation change
    iaa.Sometimes(0.3, iaa.GammaContrast((0.9, 1.1)))  # Reduced contrast change
])

def create_augmented_images(image_path, output_dir, num_augmentations=5):
    """
    Create augmented versions of a face image with realistic variations
    """
    try:
        # Load the image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Create augmented images
        augmented_images = augmenter.augment_images([image_array] * num_augmentations)
        
        # Save original image
        original_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, original_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(image_path, output_path)
        
        # Save augmented images
        for i, aug_image in enumerate(augmented_images):
            try:
                aug_image_pil = Image.fromarray(aug_image)
                aug_filename = f"aug_{i+1}_{original_filename}"
                aug_output_path = os.path.join(output_dir, aug_filename)
                aug_image_pil.save(aug_output_path)
            except Exception as e:
                logging.error(f"Error saving augmented image {i+1} for {image_path}: {str(e)}")
                continue
        
        return True
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return False

def main():
    try:
        # Directory containing the original face images
        source_dir = os.path.abspath("faces_dataset")
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")
        
        # Create a new directory for the organized dataset with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        organized_dir = os.path.abspath(f"organized_faces_dataset_{timestamp}")
        os.makedirs(organized_dir, exist_ok=True)
        
        # Get all image files in the source directory
        image_extensions = ('.jpg', '.jpeg', '.png')
        face_images = [f for f in os.listdir(source_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not face_images:
            raise FileNotFoundError("No image files found in the source directory")
        
        # Process each face image
        total_images = len(face_images)
        processed_images = 0
        
        for image_name in face_images:
            try:
                # Create person's directory
                person_name = os.path.splitext(image_name)[0]
                person_dir = os.path.join(organized_dir, person_name)
                os.makedirs(person_dir, exist_ok=True)
                
                # Create augmented images
                image_path = os.path.join(source_dir, image_name)
                if create_augmented_images(image_path, person_dir):
                    processed_images += 1
                    logging.info(f"Successfully processed {person_name} - Created augmented images")
                else:
                    logging.warning(f"Failed to process {person_name}")
                
            except Exception as e:
                logging.error(f"Error processing {image_name}: {str(e)}")
                continue
        
        logging.info(f"Processing complete. Successfully processed {processed_images}/{total_images} images")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        exit(1) 