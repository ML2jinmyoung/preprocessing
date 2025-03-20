## Pre-processing for vectorDB

### 1. Function:

- Before converting to vectorDB, extract TXT from various file formats: PDF, Excel, PPT, Word, PNG, JPG.
- When converting to TXT, display the file title and saved path.

### 2. tesseract:

brew install tesseract

- Check the installed path first.
- `/usr/local/bin/tesseract` or `/opt/homebrew/bin/tesseract`

```
echo 'export TESSERACT_PATH="/opt/homebrew/bin/tesseract"' >> ~/.zshrc
echo 'export PATH="$TESSERACT_PATH:$PATH"' >> ~/.zshrc
source ~/.zshrc

```

And then use like below:

```
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")
print(pytesseract.get_tesseract_version())

```
