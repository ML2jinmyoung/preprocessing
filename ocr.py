import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image_for_ocr(image, lang='kor'):
    """OCR 정확도를 높이기 위해 이미지 전처리"""
    # PIL 이미지를 OpenCV 이미지로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
        # 이미지가 RGBA인 경우 RGB로 변환
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 원본 이미지 복사
    original = image.copy()

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 한국어 텍스트에 최적화된 처리
    if lang == 'kor':
        # 노이즈 제거 (가우시안 블러)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # 적응형 임계값 처리
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

        # 모폴로지 연산 (노이즈 제거 및 텍스트 선명화)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 대비 조정
        processed = opening

    else:
        # 영어 및 기타 언어에 대한 기본 처리
        # 이진화 (OTSU 방식)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed = thresh

    # 결과 이미지를 PIL 형식으로 변환
    return Image.fromarray(processed)

def try_multiple_ocr_approaches(image, lang='kor+eng'):
    """여러 OCR 접근 방식을 시도하여 최상의 결과 반환"""
    results = []
    scores = []

    # 원본 이미지 사용
    original_img = image
    custom_config1 = f'--oem 1 --psm 6 -l {lang} --dpi 300'

    # 첫 번째 시도 - 원본 이미지
    try:
        text1 = pytesseract.image_to_string(original_img, config=custom_config1)
        results.append(text1)
        # 간단한 텍스트 품질 점수 계산 (단어 수, 평균 단어 길이 등)
        words = text1.split()
        score1 = len(words) * sum(len(w) for w in words) / max(1, len(words))
        scores.append(score1)

    except:
        results.append("")
        scores.append(0)

    # 두 번째 시도 - 전처리된 이미지
    try:
        processed_img = preprocess_image_for_ocr(original_img, 'kor')
        custom_config2 = f'--oem 1 --psm 3 -l {lang} --dpi 300'
        text2 = pytesseract.image_to_string(processed_img, config=custom_config2)
        results.append(text2)

        words = text2.split()
        score2 = len(words) * sum(len(w) for w in words) / max(1, len(words))
        scores.append(score2)

    except:
        results.append("")
        scores.append(0)

    # 세 번째 시도 - 대비 향상
    try:
        img_array = np.array(original_img)
        if len(img_array.shape) == 3:  # 컬러 이미지
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # 히스토그램 평활화 적용
        enhanced = cv2.equalizeHist(img_array)
        enhanced_img = Image.fromarray(enhanced)

        custom_config3 = f'--oem 1 --psm 1 -l {lang} --dpi 300'
        text3 = pytesseract.image_to_string(enhanced_img, config=custom_config3)
        results.append(text3)

        words = text3.split()
        score3 = len(words) * sum(len(w) for w in words) / max(1, len(words))
        scores.append(score3)

    except:
        results.append("")
        scores.append(0)

    # 최상의 결과 선택
    best_index = scores.index(max(scores)) if scores else 0
    return results[best_index]
