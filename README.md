# Table Extraction from Images

This project extracts tabular data from images using *OpenCV* and *Tesseract* OCR, processes it into a structured *JSON* format, and packages the results into a zip file.

## Overview
The script processes an input image (image.png) to:

    - Preprocess the image for better text and table detection.
    - Detect horizontal and vertical lines to identify table cells.
    - Extract text from each cell using OCR.
    - Organize the extracted data into headers and rows.
    - Save the results as a JSON file and create a submission zip.

## Requirements
    * Python 3.x
    * Libraries:
    * opencv-python (cv2)
    * pytesseract
    * Pillow (PIL)
    * numpy


## Install dependencies:
    ```bash
    pip install opencv-python pytesseract Pillow numpy
```

## Usage
    1. Place your input image (image.png) in the same directory as the script.
    2. Run the script:

        ```bash
        python app.py
        ```
    3. Check the output:
        * output.json: Extracted table data in JSON format.
        * submission.zip: Zip file containing ``output.json`` and ``image.png``
