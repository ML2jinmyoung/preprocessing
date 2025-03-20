import os
import PyPDF2  
import pptx  
import openpyxl   
import subprocess   
import textract  
import pytesseract   
from PIL import Image  
import fitz  
import tempfile
import cv2
import numpy as np
from pdf2image import convert_from_path   
from paddleocr import PaddleOCR

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")
INPUT_DIR = "/Users/jm/Desktop/test"
OUTPUT_DIR ="/Users/jm/Desktop/result"

"""
1. OCR 실행 함수 (전처리 및 paddleOCR사용)
"""
def deskew_image(image):
    """이미지 기울기 보정"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    angle = 0
    if lines is not None:
        angles = [np.arctan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lines[:, 0]]
        angle = np.median(angles) * (180 / np.pi)

    if abs(angle) > 1:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return image

def enhance_contrast(image):
    """CLAHE 적용"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def preprocess_image_for_ocr(image_path):
    """전처리 - 기울기 보정, 대비 향상"""
    image = cv2.imread(image_path)
    image = deskew_image(image)  
    image = enhance_contrast(image)   
    return image

def perform_ocr(image_path):
    image = preprocess_image_for_ocr(image_path)
    ocr = PaddleOCR(lang='korean')
    results = ocr.ocr(image, cls=True)
    
    extracted_text = "\n".join([res[1][0] for line in results for res in line if res[1][1] > 0.5])
    return extracted_text

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
        
        # 여러 OCR 접근 방식 시도
        text = perform_ocr(image)
        
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
                    
                    # 여러 OCR 접근 방식 시도하여 최상의 결과 얻기
                    page_text = perform_ocr(img)
                    
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
                    page_text = perform_ocr(img)
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
    try:
        text = textract.process(file_path).decode('utf-8')
        return text
    except Exception as e:
        # textract 실패 시 textutil 명령어 사용 시도
        try:
            tmp_output = tempfile.mktemp(suffix=".txt")
            cmd = ['textutil', '-convert', 'txt', '-output', tmp_output, file_path]
            subprocess.run(cmd, check=True)
            
            with open(tmp_output, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 임시 파일 삭제
            os.remove(tmp_output)
            return text
        except Exception as inner_e:
            return f"[⚠️ DOC 추출 중 에러: {e}, 대체 방법 에러: {inner_e}]"

def convert_pptx_to_txt(file_path):
    text = ""
    try:
        prs = pptx.Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    except Exception as e:
        # textract로 대체 시도
        try:
            text = textract.process(file_path).decode('utf-8')
        except:
            text = f"[⚠️ PPT 추출 중 에러: {e}]"
    return text

def convert_xlsx_to_txt(file_path):
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
            text = textract.process(file_path).decode('utf-8')
            return text
        except:
            return f"⚠️ 지원되지 않는 파일 형식입니다: {ext}"

"""
4. main()
"""
def main(input_dir, output_dir):
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
            
            out_folder = os.path.join(output_dir, relative_path)
            os.makedirs(out_folder, exist_ok=True)
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
    
    main(input_dir, output_dir)