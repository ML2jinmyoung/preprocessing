## Pre-processing for vectorDB

### 1. Function:

- Before converting to vectorDB, extract TXT from various file formats: PDF, Excel, PPT, Word, PNG, JPG.
- When converting to TXT, display the file title and saved path.

### 2. OCR:

- check the installed path of pytesseract
```
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```