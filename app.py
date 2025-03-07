import cv2
import numpy as np
from PIL import Image
import pytesseract
import json
import re
import zipfile

# Set the path to the input image
IMAGE_PATH = "image.png"

def preprocess_image(image_path):
    """Preprocess the image for text and table detection."""
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 2
    )

    return image, gray, thresh

def detect_lines(image, thresh):
    """Detect horizontal and vertical lines in the image to identify table structure."""
    # Horizontal line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    thresh_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    edges_horizontal = cv2.Canny(thresh_horizontal, 100, 200, apertureSize=3)
    lines_horizontal = cv2.HoughLinesP(
        edges_horizontal, 1, np.pi / 180, 
        threshold=400, minLineLength=350, maxLineGap=100
    )

    # Vertical line detection
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    thresh_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    edges_vertical = cv2.Canny(thresh_vertical, 50, 150, apertureSize=3)
    lines_vertical = cv2.HoughLinesP(
        edges_vertical, 1, np.pi / 180, 
        threshold=100, minLineLength=200, maxLineGap=30
    )

    # Classify lines
    horizontal_lines = []
    vertical_lines = []
    min_line_length = 50

    if lines_horizontal is not None:
        for line in lines_horizontal:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 20 and abs(x2 - x1) > min_line_length:
                horizontal_lines.append([x1, y1, x2, y2])

    if lines_vertical is not None:
        for line in lines_vertical:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 20 and abs(y2 - y1) > min_line_length:
                vertical_lines.append([x1, y1, x2, y2])

    # Sort lines for consistency
    horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1])  # By y-coordinate
    vertical_lines = sorted(vertical_lines, key=lambda x: x[0])      # By x-coordinate

    return horizontal_lines, vertical_lines, image

def detect_cells(horizontal_lines, vertical_lines):
    """Detect table cells based on line intersections."""
    cells = []
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            x1 = vertical_lines[j][0]       # Left x
            y1 = horizontal_lines[i][1]     # Top y
            x2 = vertical_lines[j + 1][0]   # Right x
            y2 = horizontal_lines[i + 1][1] # Bottom y
            cells.append((x1, y1, x2, y2))
    return cells

def extract_text_from_cells(image, cells):
    """Extract text from each cell using OCR."""
    cell_text = []
    for idx, (x1, y1, x2, y2) in enumerate(cells):
        # Crop the cell
        cell_image = image[y1:y2, x1:x2]
        cell_image_rgb = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cell_image_rgb)
        
        # Extract text
        text = pytesseract.image_to_string(pil_image).strip()
        cell_text.append({"text": text, "coords": (x1, y1, x2, y2)})
    
    return cell_text

def process_table_data(cell_text):
    """Process cell text into headers and data rows, then format as JSON."""
    # Sort by y-coordinate to ensure row-wise order
    cell_text = sorted(cell_text, key=lambda x: x["coords"][1])
    
    # Extract headers (first 9 cells)
    headers_raw = [cell["text"] for cell in cell_text[:9]]
    headers = []
    for header in headers_raw:
        # Clean and format headers
        processed = re.sub(r'[^a-z0-9]+', '_', header.lower()).strip('_')
        headers.append(processed if processed else "unknown")
    
    # Extract data rows
    data_cells = cell_text[9:]
    rows = [data_cells[i:i + 9] for i in range(0, len(data_cells), 9)]
    
    # Create JSON structure
    json_data = []
    for row in rows:
        row_dict = {header: cell["text"] for header, cell in zip(headers, row)}
        json_data.append(row_dict)
    
    return json_data

def save_output(json_data, image_path):
    """Save the JSON data and create a submission zip file."""
    # Save JSON
    with open("output.json", "w") as f:
        json.dump(json_data, f, indent=4)
    
    # Create zip file
    with zipfile.ZipFile("treeleaf_task.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write("output.json")
        zipf.write(image_path)
    
    print("Output saved as 'output.json' and zipped as 'treeleaf_task.zip'")

def main():
    """Main function to execute the table extraction pipeline."""
    # Step 1: Preprocess the image
    image, gray, thresh = preprocess_image(IMAGE_PATH)
    
    # Step 2: Detect table lines
    horizontal_lines, vertical_lines, image_with_lines = detect_lines(image, thresh)
    
    # Step 3: Detect cells
    cells = detect_cells(horizontal_lines, vertical_lines)
    
    # Step 4: Extract text from cells
    cell_text = extract_text_from_cells(image, cells)
    
    # Step 5: Process table data into JSON
    json_data = process_table_data(cell_text)
    
    # Step 6: Save the output
    save_output(json_data, IMAGE_PATH)

if __name__ == "__main__":
    main()