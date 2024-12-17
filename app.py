import json
import os
import uuid
from io import BytesIO
from typing import List
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import cv2
import torch
import numpy as np
from openchemie import OpenChemIE

app = FastAPI(title="OpenChemIE API")
model = OpenChemIE(device="cuda" if torch.cuda.is_available() else "cpu")

# Directory to save temporary images
TEMPDIR = tempfile.gettempdir()
TEMP_IMG_DIR = os.path.join(TEMPDIR, "images")
os.makedirs(TEMP_IMG_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Add more custom serialization logic if needed
        return super().default(obj)


def save_temp_file(upload_file: UploadFile) -> str:
    """Save uploaded file to a temporary location and return the path."""
    file_id = str(uuid.uuid4())
    assert upload_file.filename is not None, "upload_file.filename is None"
    fname: str = upload_file.filename
    ext = os.path.splitext(fname)[1]
    temp_path = os.path.join(TEMPDIR, f"{file_id}{ext}")
    with open(temp_path, "wb") as f:
        f.write(upload_file.file.read())
    return temp_path


def load_images(files: List[UploadFile]):
    """Load images from the uploaded files as PIL or cv2 images."""
    images = []
    for file in files:
        content = file.file.read()
        # Try PIL first
        try:
            img = Image.open(BytesIO(content))
            img = img.convert("RGB")
            images.append(img)
        except Exception:
            # If PIL fails, try cv2
            npimg = np.frombuffer(content, np.uint8)
            cv_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            images.append(cv_img)
    return images


def save_image_from_ndarray(ndarray: np.ndarray) -> str:
    """Save a numpy ndarray as an image and return the file path."""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_IMG_DIR, f"{file_id}.png")
    cv2.imwrite(file_path, ndarray)
    return file_path


def save_pil_image(image: Image.Image) -> str:
    """Save a PIL image and return the file path."""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_IMG_DIR, f"{file_id}.png")
    image.save(file_path)
    return file_path


def process_results(results: dict) -> dict:
    """Recursively process the results dictionary to save images and replace them with file paths."""
    def process_item(item):
        if isinstance(item, dict):
            return process_results(item)
        elif isinstance(item, list):
            return [process_item(sub_item) for sub_item in item]
        elif isinstance(item, np.ndarray):
            return save_image_from_ndarray(item)
        elif isinstance(item, Image.Image):
            return save_pil_image(item)
        return item

    for key, value in results.items():
        if key in ["image", "figure"]:
            results[key] = process_item(value)
        else:
            results[key] = process_item(value)
    return results


@app.post("/extract_molecules_from_figures_in_pdf")
async def extract_molecules_from_figures_in_pdf(pdf: UploadFile = File(...)):
    pdf_path = save_temp_file(pdf)
    results = model.extract_molecules_from_figures_in_pdf(pdf_path)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_molecules_from_text_in_pdf")
async def extract_molecules_from_text_in_pdf(pdf: UploadFile = File(...)):
    pdf_path = save_temp_file(pdf)
    results = model.extract_molecules_from_text_in_pdf(pdf_path)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_reactions_from_figures_in_pdf")
async def extract_reactions_from_figures_in_pdf(pdf: UploadFile = File(...)):
    pdf_path = save_temp_file(pdf)
    results = model.extract_reactions_from_figures_in_pdf(pdf_path)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_reactions_from_text_in_pdf")
async def extract_reactions_from_text_in_pdf(pdf: UploadFile = File(...)):
    pdf_path = save_temp_file(pdf)
    results = model.extract_reactions_from_text_in_pdf(pdf_path)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_reactions_from_pdf")
async def extract_reactions_from_pdf(pdf: UploadFile = File(...)):
    pdf_path = save_temp_file(pdf)
    results = model.extract_reactions_from_pdf(pdf_path)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_reactions_from_text_in_pdf_combined")
async def extract_reactions_from_text_in_pdf_combined(pdf: UploadFile = File(...)):
    pdf_path = save_temp_file(pdf)
    results = model.extract_reactions_from_text_in_pdf_combined(pdf_path)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_reactions_from_figures_and_tables_in_pdf")
async def extract_reactions_from_figures_and_tables_in_pdf(pdf: UploadFile = File(...)):
    pdf_path = save_temp_file(pdf)
    results = model.extract_reactions_from_figures_and_tables_in_pdf(pdf_path)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_molecule_corefs_from_figures_in_pdf")
async def extract_molecule_corefs_from_figures_in_pdf(pdf: UploadFile = File(...)):
    pdf_path = save_temp_file(pdf)
    results = model.extract_molecule_corefs_from_figures_in_pdf(pdf_path)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_molecules_from_figures")
async def extract_molecules_from_figures(files: List[UploadFile] = File(...)):
    images = load_images(files)
    results = model.extract_molecules_from_figures(images)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_reactions_from_figures")
async def extract_reactions_from_figures(files: List[UploadFile] = File(...)):
    images = load_images(files)
    results = model.extract_reactions_from_figures(images)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_molecule_bboxes_from_figures")
async def extract_molecule_bboxes_from_figures(files: List[UploadFile] = File(...)):
    images = load_images(files)
    results = model.extract_molecule_bboxes_from_figures(images)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_molecule_corefs_from_figures")
async def extract_molecule_corefs_from_figures(files: List[UploadFile] = File(...)):
    images = load_images(files)
    results = model.extract_molecule_corefs_from_figures(images)
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_figures_from_pdf")
async def extract_figures_from_pdf(
    pdf: UploadFile = File(...), output_bbox: bool = False, output_image: bool = True
):
    pdf_path = save_temp_file(pdf)
    results = model.extract_figures_from_pdf(
        pdf_path, output_bbox=output_bbox, output_image=output_image
    )
    processed_results = process_results(results)
    return processed_results


@app.post("/extract_tables_from_pdf")
async def extract_tables_from_pdf(
    pdf: UploadFile = File(...), output_bbox: bool = False, output_image: bool = True
):
    pdf_path = save_temp_file(pdf)
    results = model.extract_tables_from_pdf(
        pdf_path, output_bbox=output_bbox, output_image=output_image
    )
    processed_results = process_results(results)
    return processed_results
