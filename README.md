# ReceiptFlow

ReceiptFlow is an OCR-based receipt and invoice recognition prototype for the AIT103 group project. The application lets users scan receipts from an image file or camera, extracts key information using Tesseract OCR, displays structured results, and exports records for later analysis.

## Project Structure

```
ReceiptFlow/
├── app.py            # Tkinter GUI: camera/file input, results views, CSV export
├── ocr.py            # OCR + rule-based field extraction (UI-independent)
└── README.md         # This file (setup, usage, requirements)
```

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
- Python packages:

  ```
  opencv-python
  pytesseract
  Pillow
  ```

Install the Python packages with:

```bash
pip install opencv-python pytesseract Pillow
```

### Installing Tesseract OCR

- **Windows:** download and run the installer from
  `https://github.com/UB-Mannheim/tesseract/wiki`
- **macOS:** `brew install tesseract`
- **Linux (Debian/Ubuntu):** `sudo apt install tesseract-ocr`

`ocr.py` auto-detects the Tesseract binary in this order:

1. the `TESSERACT_CMD` environment variable, if set;
2. `tesseract` on the system `PATH`;
3. a few common per-OS install locations (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe` on Windows, `/opt/homebrew/bin/tesseract` on macOS, `/usr/bin/tesseract` on Linux).

This means the app should find Tesseract automatically on most machines without any code changes. If it still can't be found, type the full path into the **Tesseract Path** field in the app sidebar before scanning — no need to edit the source.

## Run

```bash
python app.py
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
