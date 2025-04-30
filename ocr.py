import re
import cv2
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass
from paddleocr import PaddleOCR
import gc
import logging

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    text: str
    score: float
    method: str

_ocr_instance = None

def get_ocr_instance():
    """
    새로운 PaddleOCR 인스턴스 생성 또는 기존 인스턴스 반환
    """
    global _ocr_instance
    try:
        if _ocr_instance is None:
            logger.info("새로운 OCR 인스턴스 생성 중...")
            _ocr_instance = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=True)
            logger.info("OCR 인스턴스 생성 완료")
        return _ocr_instance
    except Exception as e:
        logger.error(f"OCR 인스턴스 생성 실패: {str(e)}")
        raise

def reset_ocr_instance():
    """
    OCR 인스턴스 초기화
    """
    global _ocr_instance
    if _ocr_instance is not None:
        del _ocr_instance
        _ocr_instance = None
        gc.collect()
        logger.info("OCR 인스턴스 초기화됨")

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

def calculate_text_score(text: str) -> float:
    """
    텍스트 품질 점수 계산
    """
    words = text.split()
    if not words:
        return 0.0
    return len(words) * sum(len(w) for w in words) / len(words)

def is_korean_dominant(text: str, threshold: float = 0.3) -> bool:
    """
    주어진 텍스트에서 한글 문자 비율이 threshold 이상이면 True
    """
    total_chars = len(text)
    korean_chars = len(re.findall(r'[\uac00-\ud7a3]', text))  # 한글 유니코드 범위
    if total_chars == 0:
        return False
    korean_ratio = korean_chars / total_chars
    return korean_ratio >= threshold

def preprocess_image_for_ocr(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """
    OCR 정확도를 높이기 위해 이미지 전처리 (글자는 살리고 연한 도장은 제거)
    """
    image = ensure_numpy_array(image)
    
    # 이미지 크기 조정 (해상도 개선)
    height, width = image.shape[:2]
    if width < 1000:
        scale = 1000 / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 이미지 선명도 개선
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 그레이 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 대비 개선 (CLAHE 사용 → 기존 equalizeHist보다 지역 대비도 개선)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 블러 (가우시안)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Global Threshold 적용
    ret, thresh = cv2.threshold(
        blurred,
        170,  # 임계값
        255,
        cv2.THRESH_BINARY
    )

    if ret:  # threshold 적용이 성공했으면
        # 모폴로지 연산 (글자 두껍게하기 위해 (1,1) 커널 사용함. 아니라면 (2,2) 커널 사용)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        # fallback 처리 (threshold 실패한 경우 gray 이미지를 그대로 반환)
        processed = gray

    return processed

def perform_ocr_with_paddle(image: Union[Image.Image, np.ndarray], method: str) -> OCRResult:
    """
    PaddleOCR로 OCR 수행
    """
    try:
        ocr = get_ocr_instance()
        if ocr is None:
            raise ValueError("OCR 인스턴스가 초기화되지 않았습니다.")
            
        img_array = ensure_numpy_array(image)
        result = ocr.ocr(img_array, cls=True)
        
        # 메모리 해제를 위해 이미지 배열 삭제
        del img_array
        gc.collect()
        
        texts = []
        scores = []
        
        if result and result[0]:
            for line in result[0]:
                texts.append(line[1][0])
                scores.append(line[1][1])
                
        text = ' '.join(texts)
        score = float(np.mean(scores)) if scores else 0.0
        
        # 결과 데이터 정리
        del texts
        del scores
        gc.collect()
        
        return OCRResult(text=text, score=score, method=method)
    except Exception as e:
        logger.error(f"PaddleOCR failed with method {method}: {str(e)}")
        return OCRResult(text="", score=0.0, method=method)

def try_multiple_ocr_approaches(image: Union[Image.Image, np.ndarray]) -> str:
    """
    다양한 전처리 방식으로 PaddleOCR 시도
    """
    try:
        image_array = ensure_numpy_array(image)
        attempts = [
            (image_array, '원본'),
            (preprocess_image_for_ocr(image_array), '전처리')
        ]
        
        results = []
        for img, method in attempts:
            result = perform_ocr_with_paddle(img, method)
            results.append(result)
            
            # 각 시도 후 메모리 정리
            del img
            gc.collect()
        
        best = max(results, key=lambda x: x.score)
        logger.info(f"Selected OCR method: {best.method} (score: {best.score:.2f})")
        
        # 결과 반환 전 메모리 정리
        del results
        gc.collect()
        
        return best.text
    except Exception as e:
        logger.error(f"OCR 처리 중 오류 발생: {str(e)}")
        return ""
