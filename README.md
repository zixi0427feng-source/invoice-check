# ReceiptFlow

ReceiptFlow is an OCR-based receipt and invoice recognition prototype for the AIT103 group project. The application lets users scan receipts from an image file or camera, extracts key information using Tesseract OCR, displays structured results, and exports records for later analysis.

## Main Features

- Receipt image upload and camera capture
- Adjustable app layout with a draggable workspace/dashboard splitter
- OpenCV preprocessing for grayscale conversion, upscaling, denoising, and thresholding
- Tesseract OCR text recognition with English and simplified Chinese support
- Rule-based field extraction for store name, date, time, receipt number, total, tax, payment method, and item lines
- Summary, item, JSON, raw OCR, and analytics views
- CSV export for scanned receipt records

## Requirements

- Python 3.10 or later
- Tesseract OCR installed locally
- Python packages listed in `requirements.txt`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Tesseract OCR from:

```text
https://github.com/UB-Mannheim/tesseract/wiki
```

If Tesseract is installed in a different folder, update the Tesseract path in the application sidebar before scanning.

## Run

```bash
python "invoice check.py"
```

## Suggested Task 4 Screenshots

- Main application window
- Selected receipt preview
- Scan result in the Summary tab
- Item extraction in the Items tab
- Structured output in the JSON tab
- Raw OCR output
- Daily spending analytics
- CSV export result

## Known Limitations

- OCR accuracy depends on image quality, lighting, angle, and receipt layout.
- Rule-based extraction may not handle every invoice format.
- Handwritten receipts and very blurry receipts are difficult to recognize.
- Current data is stored in memory during runtime unless exported to CSV.
