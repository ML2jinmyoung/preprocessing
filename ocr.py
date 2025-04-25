import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Union
from dataclasses import dataclass
import easyocr

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Initialize EasyOCR readers for each language combination
READERS = {
    'ko': easyocr.Reader(['ko', 'en']),       # Korean + English
    'ja': easyocr.Reader(['ja', 'en']),       # Japanese + English
    'en': easyocr.Reader(['en'])              # English only
}

# Tesseract config presets
default_config = '--oem 1 --dpi 300'
config_presets = {
    'accurate': '--psm 6',
    'mixed': '--psm 3',
    'sparse': '--psm 1'
}

@dataclass
class OCRResult:
    text: str
    score: float
    method: str


def ensure_numpy(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.array(image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    return image


def calc_score(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(words) * sum(len(w) for w in words) / len(words)


def preprocess_for_tess(image: np.ndarray, lang: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if lang == 'ko':
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    else:
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out


def get_tess_langs(lang: str) -> str:
    if lang == 'ko':
        return 'kor+eng'
    if lang == 'ja':
        return 'jpn+eng'
    return 'eng'


def get_tess_config(mode: str, lang: str) -> str:
    preset = config_presets.get(mode, config_presets['accurate'])
    langs = get_tess_langs(lang)
    return f"{default_config} {preset} -l {langs}"


def ocr_tesseract(image: Union[Image.Image, np.ndarray], mode: str, lang: str) -> OCRResult:
    arr = ensure_numpy(image)
    cfg = get_tess_config(mode, lang)
    text = pytesseract.image_to_string(arr, config=cfg)
    return OCRResult(text=text, score=calc_score(text), method=f"Tesseract-{mode}")


def ocr_easyocr(image: Union[Image.Image, np.ndarray], lang: str) -> OCRResult:
    arr = ensure_numpy(image)
    reader = READERS.get(lang, READERS['en'])
    results = reader.readtext(arr)
    texts = [r[1] for r in results]
    confs = [r[2] for r in results]
    text = ' '.join(texts)
    score = (sum(confs) / len(confs)) * len(texts) if confs else 0.0
    return OCRResult(text=text, score=score, method=f"EasyOCR-{lang}")


def detect_lang(image: Union[Image.Image, np.ndarray]) -> str:
    """
    이미지에서 텍스트 샘플을 추출한 후 유니코드 스크립트 기반으로 언어를 판별합니다.
    - 한글 유니코드(0xAC00–0xD7A3) 발견 시 'ko'
    - 히라가나(0x3040–0x309F) 또는 가타카나(0x30A0–0x30FF) 발견 시 'ja'
    - CJK 통합 한자(0x4E00–0x9FFF)만 있고 모음 스크립트 없으면 'ja'
    - 그 외 라틴 알파벳이 포함되면 'en'
    """
    arr = ensure_numpy(image)
    # 영문 OCR로 샘플 텍스트 추출
    sample = pytesseract.image_to_string(arr, config='--psm 6 -l eng')
    has_cjk = False
    for ch in sample:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            return 'ko'
        if 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
            return 'ja'
        if 0x4E00 <= code <= 0x9FFF:
            has_cjk = True
    if has_cjk:
        return 'ja'
    # 라틴 알파벳 포함 여부
    if any(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in sample):
        return 'en'
    # fallback
    return 'en'

def try_multiple_ocr_approaches(
    image: Union[Image.Image, np.ndarray],
    file_format: str
) -> str:(
    image: Union[Image.Image, np.ndarray],
    file_format: str
) -> str:
    arr = ensure_numpy(image)
    # PDF: EasyOCR만 사용
    if file_format.lower() == 'pdf':
        lang = detect_lang(arr)
        return ocr_easyocr(arr, lang).text

    # 자동 언어 감지
    lang = detect_lang(arr)

    # 다양한 OCR 시도
    results = []
    results.append(ocr_tesseract(arr, 'accurate', lang))
    results.append(ocr_tesseract(preprocess_for_tess(arr, lang), 'mixed', lang))
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    results.append(ocr_tesseract(cv2.equalizeHist(gray), 'sparse', lang))
    results.append(ocr_easyocr(arr, lang))

    best = max(results, key=lambda r: r.score)
    print(f"Detected lang: {lang}, Selected method: {best.method} (score: {best.score:.2f})")
    return best.text
