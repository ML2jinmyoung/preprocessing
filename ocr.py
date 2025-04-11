import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass
import easyocr

# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = '/opt/anaconda3/envs/pdf2txt/bin/pytesseract'

reader = easyocr.Reader(['en', 'ko'])

OCR_CONFIGS = {
    'default': '--oem 1 --dpi 300',
    'accurate': '--psm 6',  # 균일한 블록의 텍스트
    'mixed': '--psm 3',     # 컬럼이 있는 텍스트
    'sparse': '--psm 1'     # 방향 감지와 함께 OSD
}

PREPROCESSING_PARAMS = {
    'gaussian_blur': (3, 3),
    'adaptive_threshold': {
        'block_size': 11,
        'C': 2
    },
    'morph_kernel': (1, 1)
}

@dataclass
class OCRResult:
    text: str
    score: float
    method: str

def ensure_numpy_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """
    이미지를 numpy array로 변환
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA to RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def get_ocr_config(mode: str, lang: str) -> str:
    """
    OCR 설정 생성
    """
    base_config = OCR_CONFIGS['default']
    specific_config = OCR_CONFIGS.get(mode, OCR_CONFIGS['accurate'])
    return f"{base_config} {specific_config} -l {lang}"

def calculate_text_score(text: str) -> float:
    """
    텍스트 품질 점수 계산
    """
    words = text.split()
    if not words:
        return 0.0
    return len(words) * sum(len(w) for w in words) / len(words)

def preprocess_image_for_ocr(image: Union[Image.Image, np.ndarray], lang: str = 'kor') -> Image.Image:
    """
    OCR 정확도를 높이기 위해 이미지 전처리
    """
    # numpy array로 변환
    image = ensure_numpy_array(image)
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if lang == 'kor':
        # 한국어 텍스트 최적화 처리
        blurred = cv2.GaussianBlur(gray, PREPROCESSING_PARAMS['gaussian_blur'], 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            PREPROCESSING_PARAMS['adaptive_threshold']['block_size'],
            PREPROCESSING_PARAMS['adaptive_threshold']['C']
        )
        kernel = np.ones(PREPROCESSING_PARAMS['morph_kernel'], np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    else:
        # 기타 언어 처리 (OTSU 이진화)
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(processed)

def perform_ocr(image: Union[Image.Image, np.ndarray], config: str, format: str, method: str) -> OCRResult:
    """
    단일 OCR 시도 수행
    """
    try:
        if format == 'pdf':
            img_array = ensure_numpy_array(image)
            detection = reader.readtext(img_array)
            text = ' '.join([item[1] for item in detection])
        else:
            text = pytesseract.image_to_string(image, config=config)
        
        score = calculate_text_score(text)
        return OCRResult(text=text, score=score, method=method)
    except Exception as e:
        print(f"OCR failed with method {method}: {str(e)}")
        return OCRResult(text="", score=0.0, method=method)

def try_multiple_ocr_approaches(image: Union[Image.Image, np.ndarray], format: str, lang: str = 'kor+eng') -> str:
    """
    다양한 OCR 방식 시도
    """
    # 이미지 준비
    image_array = ensure_numpy_array(image)
    
    # OCR 시도 설정
    attempts = [
        (image_array, 'accurate', '원본'),
        (preprocess_image_for_ocr(image_array, lang), 'mixed', '전처리'),
        (cv2.equalizeHist(cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)), 'sparse', '대비향상')
    ]

    # 각 방식으로 OCR 수행
    results = []
    for img, mode, method in attempts:
        config = get_ocr_config(mode, lang)
        result = perform_ocr(img, config, format, method)
        results.append(result)

    # 최상의 결과 선택
    best_result = max(results, key=lambda x: x.score)
    print(f"Selected OCR method: {best_result.method} (score: {best_result.score:.2f})")
    return best_result.text
