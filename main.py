import os
import PyPDF2
import pptx
import openpyxl
import subprocess
import tempfile
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

# 경로 설정
INPUT_DIR = "../test"
OUTPUT_DIR = "../result"

# PaddleOCR 초기화 (한국어+영어 지원)
ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False)

"""
1. OCR 실행 함수 (PaddleOCR 사용)
"""
def perform_ocr_with_paddle(image):
    """PaddleOCR을 사용하여 이미지에서 텍스트 추출"""
    # PIL 이미지를 numpy 배열로 변환
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # BGR -> RGB 변환 (OpenCV와 PIL 이미지 형식 차이 처리)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        if isinstance(image, np.ndarray):  # OpenCV 이미지인 경우
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # PaddleOCR 실행
    result = ocr.ocr(img_array, cls=True)
    
    # 결과 텍스트 추출
    text = ""
    if result:
        for idx, line in enumerate(result):
            if line:
                for word_info in line:
                    if len(word_info) >= 2 and isinstance(word_info[1], tuple):
                        text += word_info[1][0] + " "
                text += "\n"
    
    return text

def preprocess_image_for_ocr(image):
    """OCR 정확도를 높이기 위해 이미지 전처리"""
    # PIL 이미지를 OpenCV 이미지로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
        # 이미지가 RGBA인 경우 RGB로 변환
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거 (가우시안 블러)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 적응형 임계값 처리
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 대비 향상
    enhanced = cv2.equalizeHist(gray)
    
    # 결과 이미지들을 반환 (여러 전처리 방식)
    return [
        image,  # 원본
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB),  # 이진화
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)  # 대비 향상
    ]

def try_multiple_ocr_approaches(image):
    """여러 OCR 접근 방식을 시도하여 최상의 결과 반환"""
    results = []
    scores = []
    
    # 여러 전처리 이미지 생성
    processed_images = preprocess_image_for_ocr(image)
    
    for img in processed_images:
        try:
            text = perform_ocr_with_paddle(img)
            results.append(text)
            
            # 간단한 텍스트 품질 점수 계산 (단어 수와 평균 단어 길이)
            words = text.split()
            score = len(words) * sum(len(w) for w in words) / max(1, len(words))
            scores.append(score)
        except:
            results.append("")
            scores.append(0)
    
    # 최상의 결과 선택
    best_index = scores.index(max(scores)) if scores else 0
    return results[best_index]

"""
2. 각 파일 포맷에서 텍스트 추출하는 함수들
"""

def convert_image_to_txt(file_path):
    """이미지 파일(PNG, JPG 등)에서 텍스트 추출"""
    try:
        # PIL 이미지 열기
        image = Image.open(file_path)
        
        # DPI가 낮으면 업스케일링 (OCR 정확도 향상을 위해)
        try:
            dpi = image.info.get('dpi', (72, 72))
            if isinstance(dpi, tuple) and min(dpi) < 200:
                scale_factor = 300 / min(72, min(dpi))
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                image = image.resize(new_size, Image.LANCZOS)
        except:
            pass
        
        # PaddleOCR로 텍스트 추출
        text = try_multiple_ocr_approaches(image)
        
        if not text.strip():
            return "[이미지에서 텍스트를 추출할 수 없습니다]"
        return text
    except Exception as e:
        return f"[⚠️ 이미지 텍스트 추출 중 에러: {e}]"

def convert_pdf_to_txt(file_path):
    """PDF 파일에서 텍스트 추출 (일반 PDF와 스캔된 PDF 모두 지원)"""
    text = ""
    
    # 일반적인 PDF 텍스트 추출 시도
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        text = f"[⚠️ PyPDF2 추출 중 에러: {e}]"

    # 텍스트가 없거나 너무 짧으면 스캔된 PDF로 간주하고 OCR 시도
    if len(text.strip()) < 100:
        try:
            print(f"PDF에서 텍스트가 충분히 추출되지 않아 OCR 방식으로 시도합니다: {file_path}")
            ocr_text = ""
            
            # PDF2Image 라이브러리를 사용하여 PDF를 이미지로 변환
            try:
                # 고품질 설정
                images = convert_from_path(
                    file_path,
                    dpi=300,   
                    thread_count=4,   
                    use_cropbox=True,   
                    strict=False,
                    grayscale=False   
                )
                
                for i, img in enumerate(images):
                    print(f"  페이지 {i+1}/{len(images)} OCR 처리 중...")
                    
                    # PaddleOCR로 텍스트 추출
                    page_text = try_multiple_ocr_approaches(img)
                    
                    ocr_text += f"--- 페이지 {i + 1} ---\n{page_text}\n\n"
                
            except Exception as pdf2image_err:
                print(f"PDF2Image 변환 실패, PyMuPDF로 대체합니다: {pdf2image_err}")
                # 대체 방법: PyMuPDF 사용
                pdf_document = fitz.open(file_path)
                
                # 고해상도 설정
                zoom_factor = 4  # 이미지 해상도 크게 향상
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                
                for page_number in range(pdf_document.page_count):
                    print(f"  페이지 {page_number+1}/{pdf_document.page_count} 대체 OCR 처리 중...")
                    page = pdf_document.load_page(page_number)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # pixmap을 PIL 이미지로 변환
                    img_data = pix.samples
                    img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                    
                    # OCR 시도
                    page_text = try_multiple_ocr_approaches(img)
                    ocr_text += f"--- 페이지 {page_number + 1} ---\n{page_text}\n\n"
            
            if ocr_text.strip():
                return ocr_text
            else:
                return text if text.strip() else "[PDF에서 텍스트를 추출할 수 없습니다]"
                
        except Exception as e:
            if text.strip():
                return text + f"\n\n[OCR 추출 중 에러: {e}]"
            else:
                return f"[⚠️ PDF 텍스트 추출 실패: {e}]"
    
    return text

def convert_doc_to_txt(file_path):
    """DOC/DOCX 파일에서 텍스트 추출"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.docx':
            # python-docx 사용
            from docx import Document
            try:
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            except Exception as e:
                return f"[⚠️ DOCX 추출 중 에러: {e}]"
        elif ext == '.doc':
            # antiword 사용
            try:
                result = subprocess.run(['antiword', file_path], capture_output=True, text=True, check=True)
                return result.stdout
            except:
                # catdoc 대체 시도
                try:
                    result = subprocess.run(['catdoc', file_path], capture_output=True, text=True, check=True)
                    return result.stdout
                except Exception as e:
                    return f"[⚠️ DOC 추출 중 에러: {e}]"
        else:
            return f"[⚠️ 지원되지 않는 형식: {ext}]"
    except Exception as e:
        return f"[⚠️ DOC 추출 중 에러: {e}]"



def convert_pptx_to_txt(file_path):
    """PPT/PPTX 파일에서 텍스트 추출"""
    text = ""
    try:
        prs = pptx.Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    except Exception as e:
        # 리눅스에서는 대체 방법 시도
        try:
            # pptx2txt 또는 다른 커맨드라인 도구 사용
            result = subprocess.run(['pptx2txt', file_path], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                text = result.stdout
            else:
                text = f"[⚠️ PPT 추출 중 에러: {e}, 대체 방법 실패]"
        except:
            text = f"[⚠️ PPT 추출 중 에러: {e}]"
    return text

def convert_xlsx_to_txt(file_path):
    """XLSX/XLS 파일에서 텍스트 추출"""
    text = ""
    try:
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        for sheet in wb.worksheets:
            text += f"Sheet: {sheet.title}\n"
            for row in sheet.iter_rows(values_only=True):
                row_data = "\t".join([str(cell) if cell is not None else "" for cell in row])
                text += row_data + "\n"
            text += "\n"
    except Exception as e:
        # 리눅스에서는 대체 방법 시도
        try:
            # xlsx2csv 사용
            tmp_output = tempfile.mktemp(suffix=".csv")
            cmd = ['xlsx2csv', file_path, tmp_output]
            subprocess.run(cmd, check=True)
            
            with open(tmp_output, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # 임시 파일 삭제
            os.remove(tmp_output)
            
            if not text.strip():
                text = f"[⚠️ XLSX 추출 중 에러: {e}, 대체 방법 결과 없음]"
        except:
            text = f"[⚠️ XLSX 추출 중 에러: {e}]"
    return text

"""
3. 파일 확장자에 따라 함수 실행
"""
def convert_file_to_txt(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
        return convert_image_to_txt(file_path)
    elif ext == ".pdf":
        return convert_pdf_to_txt(file_path)
    elif ext in [".docx", ".doc"]:
        return convert_doc_to_txt(file_path)
    elif ext in [".pptx", ".ppt"]:
        return convert_pptx_to_txt(file_path)
    elif ext in [".xlsx", ".xls"]:
        return convert_xlsx_to_txt(file_path)
    else:
        try:
            # 리눅스에서는 file 명령어로 파일 타입 확인 후 적절한 처리
            result = subprocess.run(['file', '-b', file_path], capture_output=True, text=True, check=True)
            file_type = result.stdout.lower()
            
            if 'text' in file_type:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            else:
                return f"⚠️ 지원되지 않는 파일 형식입니다: {ext} (파일 타입: {file_type.strip()})"
        except:
            return f"⚠️ 지원되지 않는 파일 형식입니다: {ext}"

"""
4. main()
"""
def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, os.path.dirname(input_dir))    
            
            if file.startswith('.'):
                continue
                
            print(f"변환 중: {file_path}")
            
            ext = os.path.splitext(file)[1].lower()
            
            if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf"]:
                print(f"  OCR 처리가 필요할 수 있어 시간이 오래 걸릴 수 있습니다...")
            
            txt_content = convert_file_to_txt(file_path)
            header = f"File: {file}\nPath: {relative_path}\n{'='*40}\n"
            full_text = header + txt_content
            
            # 출력 폴더 경로 생성
            relative_dir = os.path.dirname(relative_path)
            out_folder = os.path.join(output_dir, relative_dir)
            os.makedirs(out_folder, exist_ok=True)
            
            # 출력 파일 경로 생성
            out_file_name = os.path.splitext(file)[0] + ".txt"
            out_file_path = os.path.join(out_folder, out_file_name)
            
            try:
                with open(out_file_path, "w", encoding="utf-8") as f_out:
                    f_out.write(full_text)
                print(f"성공: {out_file_path}")
            except Exception as e:
                print(f"⚠️ 에러 ({out_file_path}): {e}")


if __name__ == "__main__":
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    
    # PaddleOCR 모델이 처음 실행 시 자동으로 다운로드됨
    print("PaddleOCR 초기화 중...")
    
    main(input_dir, output_dir)