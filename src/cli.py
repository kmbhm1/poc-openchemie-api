import os
import argparse
import requests
import tempfile
import uuid
import cv2
import numpy as np
import json
from torch import cuda
from urllib.parse import urlparse
from PIL import Image
from openchemie import OpenChemIE
import shutil
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor

# Disable parallelism & huggingface warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize model
model = OpenChemIE(device="cuda" if cuda.is_available() else "cpu")


# Define custom JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_image_pillow(file_path, convert_mode='RGB'):
    """Load an image using Pillow and convert it to the specified mode."""
    with Image.open(file_path) as img:
        img_converted = img.convert(convert_mode)
    return img_converted


# Define function to process results
def process_results(results, output_dir):
    def process_item(item):
        if isinstance(item, dict):
            return process_results(item, output_dir)
        elif isinstance(item, list):
            return [process_item(sub_item) for sub_item in item]
        elif isinstance(item, np.ndarray):
            return save_image_from_ndarray(item, output_dir)
        elif isinstance(item, Image.Image):
            return save_pil_image(item, output_dir)
        return item

    if isinstance(results, dict):
        return {key: process_item(value) for key, value in results.items()}
    elif isinstance(results, list):
        return [process_item(item) for item in results]
    else:
        return results


def save_image_from_ndarray(ndarray: np.ndarray, output_dir: str) -> str:
    """Save a numpy ndarray as an image and return the file path."""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(output_dir, f"{file_id}.png")
    print(f"Saving image to {file_path}")
    cv2.imwrite(file_path, ndarray)
    return file_path


def save_pil_image(image: Image.Image, output_dir: str) -> str:
    """Save a PIL image and return the file path."""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(output_dir, f"{file_id}.png")
    print(f"Saving image to {file_path}")
    image.save(file_path)
    return file_path


# Define function to download file
def download_file(url, dest_folder):
    print(f"Downloading {url} to {dest_folder}")
    filename = os.path.basename(urlparse(url).path)
    file_path = os.path.join(dest_folder, filename)
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    # Write the content to a file
    with open(file_path, 'wb') as file:
        file.write(response.content)

    # Guess the MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        print(f"Detected MIME type: {mime_type}")
        if mime_type == 'application/pdf':
            new_file_path = os.path.join(dest_folder, f"{uuid.uuid4()}.pdf")
            os.rename(file_path, new_file_path)
            file_path = new_file_path
            print("File is a PDF.")
        elif mime_type.startswith('image/'):
            extension = mimetypes.guess_extension(mime_type)
            new_file_path = os.path.join(dest_folder, f"{uuid.uuid4()}{extension}")
            os.rename(file_path, new_file_path)
            file_path = new_file_path
            print("File is an image.")
        else:
            os.remove(file_path)
            raise ValueError("File type is not supported.")
    else:
        os.remove(file_path)
        raise ValueError("Could not determine file type.")

    return file_path


def process_pdf(file_path, output_dir):
    '''Process a PDF file using OpenChemIE.

    Args:
        file_path (str): The path to the PDF file to process.
        output_dir (str): The directory to save the processed results.
    Returns:
        None
    '''
    # Create a unique job directory
    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(output_dir, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    print(f"Processing PDF: {file_path}, results will be saved in: {job_output_dir}")

    # Copy original file to job directory
    shutil.copy(file_path, job_output_dir)

    # Define methods to run
    methods = {
        'molecules_from_figures': model.extract_molecules_from_figures_in_pdf,
        'molecules_from_text': model.extract_molecules_from_text_in_pdf,
        'reactions_from_figures': model.extract_reactions_from_figures_in_pdf,
        'reactions_from_text': model.extract_reactions_from_text_in_pdf,
        'reactions_combined': model.extract_reactions_from_text_in_pdf_combined,
        'reactions_figures_tables': model.extract_reactions_from_figures_and_tables_in_pdf,
        'molecule_corefs': model.extract_molecule_corefs_from_figures_in_pdf
    }

    # Run each method and process results
    for key, method in methods.items():
        try:
            result = method(file_path)
            processed_result = process_results({key: result}, job_output_dir)
            json_path = os.path.join(job_output_dir, f"{job_id}_{key}_results.json")
            with open(json_path, 'w') as json_file:
                json.dump(processed_result, json_file, cls=NumpyEncoder)
        except Exception as e:
            print(f"Error in {key}: {e}")


def process_image(file_path, output_dir):
    '''Process an image using OpenChemIE.

    Args:
        file_path (str): The path to the image file to process.
        output_dir (str): The directory to save the processed results.
    Returns:
        None
    '''
    # Create a unique job directory
    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(output_dir, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    print(f"Processing Image: {file_path}, results will be saved in: {job_output_dir}")

    # Copy original file to job directory
    shutil.copy(file_path, job_output_dir)

    # Define methods to run
    methods = {
        'molecules_from_figures': model.extract_molecules_from_figures,
        'reactions_from_figures': model.extract_reactions_from_figures,
        'molecule_bboxes': model.extract_molecule_bboxes_from_figures,
        'molecule_corefs': model.extract_molecule_corefs_from_figures
    }

    # Load image
    try:
        data = [load_image_pillow(file_path)]
    except Exception as e:
        print(f"Failed to load image {file_path}: {e}")
        return

    # Run each method and process results
    for key, method in methods.items():
        try:
            result = method(data)
            processed_result = process_results({key: result}, job_output_dir)
            json_path = os.path.join(job_output_dir, f"{job_id}_{key}_results.json")
            with open(json_path, 'w') as json_file:
                json.dump(processed_result, json_file, cls=NumpyEncoder)
        except Exception as e:
            print(f"Error in {key}: {e}")


def validate_args(args):
    '''Validate command line arguments.

    Args:
        args (argparse.Namespace): The command line arguments.
    Returns:
        None
    '''

    if not (args.url or args.file or args.input_dir):
        raise ValueError("You must specify either a URL, a single file, or an input directory.")

    if args.file and not os.path.isfile(args.file):
        raise ValueError(f"The specified file does not exist: {args.file}")

    if args.input_dir and not os.path.isdir(args.input_dir):
        raise ValueError(f"The specified input directory does not exist: {args.input_dir}")

    if args.output_dir and not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create or write to the output directory: {args.output_dir}. Error: {e}")


# Main function to run
def main():
    parser = argparse.ArgumentParser(description='Process scientific papers and images with OpenChemIE.')
    parser.add_argument('--url', type=str, help='URL of the file to download and process.')
    parser.add_argument('--input-dir', type=str, help='Directory containing files to process.')
    parser.add_argument('--file', type=str, help='Single file to process.')
    parser.add_argument('--output-dir', type=str, help='Directory to save processed results.')
    args = parser.parse_args()
    validate_args(args)

    def ends_with_pdf(x):
        return x.endswith(".pdf")

    def ends_with_image(x):
        return x.endswith((".png", ".jpg", ".jpeg"))

    def process_file(file_path):
        if ends_with_pdf(file_path):
            process_pdf(file_path, output_dir)
        elif ends_with_image(file_path):
            process_image(file_path, output_dir)
        else:
            print(f"Unsupported file type for {file_path}. Skipping.")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use the current working directory
        output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved in: {output_dir}")

    start_time = time.time()

    if args.url:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = download_file(args.url, temp_dir)
            if ends_with_pdf(file_path):
                process_pdf(file_path, output_dir)
            elif ends_with_image(file_path):
                process_image(file_path, output_dir)
            else:
                print(f"Unsupported file type for {file_path}. Skipping.")

    if args.file:
        if ends_with_pdf(args.file):
            process_pdf(args.file, output_dir)
        elif ends_with_image(args.file):
            process_image(args.file, output_dir)
        else:
            print(f"Unsupported file type for {args.file}. Skipping.")

    if args.input_dir:
        with ThreadPoolExecutor() as executor:
            for root, _, files in os.walk(args.input_dir):
                file_paths = [os.path.join(root, file) for file in files]
                executor.map(process_file, file_paths)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Process completed in {int(minutes)} minutes and {int(seconds)} seconds.")


if __name__ == '__main__':
    main()
