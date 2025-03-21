import os
import PyPDF2
import pptx
import openpyxl
import subprocess
import tempfile
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
from ocr import try_multiple_ocr_approaches

# 경로 설정
INPUT_DIR = "../test"
OUTPUT_DIR = "../result"


"""
2. 각 파일 포맷에서 텍스트 추출하는 함수들
"""

def convert_image_to_txt(file_path):
    """이미지 파일(PNG, JPG 등)에서 텍스트 추출"""
    try:
        # PIL 이미지 열기
        image = Image.open(file_path).convert("RGB")
        
        # DPI가 낮으면 업스케일링 (OCR 정확도 향상을 위해)
        try:
            dpi = image.info.get('dpi', (72, 72))
            if isinstance(dpi, tuple) and min(dpi) < 200:
                scale_factor = 300 / min(72, min(dpi))
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                image = image.resize(new_size, Image.LANCZOS)
        except Exception as e:
            print(f"[⚠️ DPI 확인 오류: {e}]")
        
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