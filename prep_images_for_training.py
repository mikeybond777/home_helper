import os
import shutil

import face_recognition
from PIL import Image, ExifTags
import pillow_heif


SOURCE_IMAGES = "./image_processing/raw"
KEY_FACES_DIR = "./image_processing/key_faces"
OUTPUT_DIR = "./image_processing/output"
DATA_INPUT_DIR = "./archive"

TRAINING = 'train'
VALIDATION = 'valid'
TESTING = 'test'


def crop_to_key_face(source_image_path, key_face_image_path, output_image_path, padding=100):
    try:
        # Load the images
        source_image = face_recognition.load_image_file(source_image_path)
        key_face_image = face_recognition.load_image_file(key_face_image_path)

        # Find all face locations and encodings in the source image
        source_face_locations = face_recognition.face_locations(source_image)
        source_face_encodings = face_recognition.face_encodings(source_image, source_face_locations)

        # Get the encoding of the key face
        # We assume the key face image contains only one face
        key_face_encoding = face_recognition.face_encodings(key_face_image)[0]

        print(f"Found {len(source_face_encodings)} face(s) in the source image.")
        print("Searching for the key face...")

        # Find the matching face in the source image
        match_found = False
        for i, source_face_encoding in enumerate(source_face_encodings):
            # If only 1 face detected we can assume it is the correct person (most likely)!
            #if len(source_face_encodings) != 1:
            matches = face_recognition.compare_faces([key_face_encoding], source_face_encoding)
            if not matches[0]:
                continue

            print("Match found!")
            # Get the coordinates of the matched face
            top, right, bottom, left = source_face_locations[i]

            # Open the original image using Pillow to crop
            pil_image = Image.open(source_image_path)

            # Define the crop box with padding
            crop_left = max(0, left - padding)
            crop_top = max(0, top - padding)
            crop_right = min(pil_image.width, right + padding)
            crop_bottom = min(pil_image.height, bottom + padding)

            # Crop the image
            cropped_image = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))

            # Save the cropped image
            cropped_image.save(output_image_path)
            print(f"Cropped image saved to {output_image_path}")
            match_found = True
            break  # Exit after finding the first match

        if not match_found:
            print("Could not find the key face in the source image.")

    except IndexError:
        print("Error: Could not find any face in the key face image. Please use a clear photo.")
    except FileNotFoundError as e:
        print(f"Error: {e}")


def list_all_file_paths(directory):
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

    return file_paths


def resize_image_to_dimensions(input_path, width=224, height=224):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    try:
        image = Image.open(input_path)

        # Resize to fixed width and height
        resized = image.resize((width, height), Image.LANCZOS)

        # Save with modified name
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_resized{ext}"
        resized.save(input_path, "JPEG", quality=90)
    except Exception as e:
        print("Unable to resize image: %s" % e)


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            for orientation_tag in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation_tag] == 'Orientation':
                    break
            orientation = exif.get(orientation_tag, None)

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Warning: Could not apply EXIF orientation. {e}")
    return image


def convert_image_to_jpg(input_path):
    """Converts a single image file (e.g., HEIC, PNG, TIFF) to a JPG file."""

    # Verify the input file path exists
    if not os.path.exists(input_path):
        print(f"Error: File not found at '{input_path}'")
        return

    # Construct the output path by replacing the extension
    base_path = os.path.splitext(input_path)[0]
    jpg_path = base_path + ".jpg"

    if jpg_path.lower().replace('\\', '/') == input_path.lower().replace('\\', '/'):
        return

    print(f"Converting '{os.path.basename(input_path)}' to '{os.path.basename(jpg_path)}'...")

    try:
        # Register the HEIF opener with Pillow.
        # This allows Pillow to open .heic files in addition to its
        # standard supported formats like PNG, BMP, etc.
        pillow_heif.register_heif_opener()

        # Open and discard alpha.
        image = Image.open(input_path)
        # Avoid unintentional rotations.
        image = apply_exif_orientation(image)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # 'quality=90' is a good balance between file size and quality.
        image.save(jpg_path, "jpeg", quality=90)
        print("Conversion successful!")
        if jpg_path.lower().replace('\\', '/') != input_path.lower().replace('\\', '/') and os.path.exists(jpg_path):
            os.remove(input_path)

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred. The file might not be a supported image format. Details: {e}")


def convert_images_to_jpg(images_dir):
    '''Convert all images in raw data to jpg.'''

    all_image_paths = list_all_file_paths(images_dir)
    for image_path in all_image_paths:
        convert_image_to_jpg(image_path)


def analyse_crop_images():
    for key_image in os.listdir(KEY_FACES_DIR):
        key_image_full_path = os.path.abspath('%s/%s' % (KEY_FACES_DIR, key_image))
        face_name = os.path.splitext(key_image)[0]

        #if face_name in ['angie', 'bond', 'cameron']:
        #    continue

        # Ensure key image output dir exists.
        output_dir = os.path.abspath('%s/%s' % (OUTPUT_DIR, face_name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        source_image_dir = os.path.abspath('%s/%s' % (SOURCE_IMAGES, face_name))
        if not os.path.exists(source_image_dir):
            print(f"WARNING, no face images found for {face_name}!")
            continue
        for source_image in os.listdir(source_image_dir):
            crop_to_key_face('%s/%s' % (source_image_dir, source_image), key_image_full_path, '%s/%s' % (output_dir, source_image))


def reformat_images(images_dir):
    all_file_paths = list_all_file_paths(images_dir)
    for file_path in all_file_paths:
        resize_image_to_dimensions(file_path)


def _split_data_training_validation_test(image_files_dir):
    image_files = ['%s/%s' % (image_files_dir, image_name) for image_name in os.listdir(image_files_dir)]
    image_files = [image_file for image_file in image_files if os.path.isfile(image_file)]

    total = len(image_files)

    # Calculate initial sizes
    two_thirds = (2 * total) // 3
    remaining = total - two_thirds
    one_sixth = remaining // 2

    # Assign leftovers (e.g. when total isn't divisible cleanly)
    remainder = total - (two_thirds + one_sixth * 2)
    data_input = {
        TRAINING: image_files[:two_thirds + remainder],
        VALIDATION: image_files[two_thirds + remainder:two_thirds + remainder + one_sixth],
        TESTING: image_files[two_thirds + remainder + one_sixth:]
    }

    return data_input

def copy_images_for_model_training(output_dir, padding=3):
    '''Copy images from cropped output folder to directories for training.'''

    for category_name in os.listdir(output_dir):
        full_output_path = '%s/%s' % (output_dir, category_name)
        print(full_output_path)
        if not os.path.isdir(full_output_path):
            continue

        # Find different types of data based on collected data.
        data_input = _split_data_training_validation_test(full_output_path)

        print(data_input)

        # Copy the files to relevant locations for the model to be trained with.
        for data_type, image_files in data_input.items():
            data_type_path = '%s/%s/%s' % (DATA_INPUT_DIR, data_type, category_name)
            if not os.path.exists(data_type_path):
                os.makedirs(data_type_path, exist_ok=True)
            for i, image_file_path in enumerate(image_files):
                number = i+1
                ext = os.path.splitext(image_file_path)[1]
                image_file_padded_name = f"{number:0{padding}d}{ext}"
                new_image_path = '%s/%s' % (data_type_path, image_file_padded_name)
                shutil.copy2(image_file_path, new_image_path)

if __name__ == "__main__":
    convert_images_to_jpg(os.path.abspath(SOURCE_IMAGES))
    convert_images_to_jpg(os.path.abspath(KEY_FACES_DIR))
    analyse_crop_images()
    reformat_images(os.path.abspath(OUTPUT_DIR))

    copy_images_for_model_training(os.path.abspath(OUTPUT_DIR).replace('\\', '/'))


