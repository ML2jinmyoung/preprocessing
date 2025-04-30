import fitz
import PyPDF2
import os

# PDF 파일 경로
pdf_path = "./demo/210317-2직장 내 괴롭힘으로 인한 건강장해 예방 매뉴얼(책자).pdf"


# PyMuPDF(fitz)를 사용한 텍스트 추출
def extract_text_with_pymupdf():
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    # 결과를 파일로 저장
    with open('./demo_result/210317-2직장 내 괴롭힘으로 인한 건강장해 예방 매뉴얼(책자)_raw1', 'w', encoding='utf-8') as f:
        f.write(text)

# PyPDF2를 사용한 텍스트 추출
def extract_text_with_pypdf2():
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    # 결과를 파일로 저장
    with open('./demo_result/210317-2직장 내 괴롭힘으로 인한 건강장해 예방 매뉴얼(책자)_raw2', 'w', encoding='utf-8') as f:
        f.write(text)

# 두 방식으로 텍스트 추출 실행
if __name__ == "__main__":
    try:
        print("PyMuPDF로 텍스트 추출 중...")
        extract_text_with_pymupdf()
        print("PyMuPDF 텍스트 추출 완료: test/pymupdf_result.txt")
        
        print("\nPyPDF2로 텍스트 추출 중...")
        extract_text_with_pypdf2()
        print("PyPDF2 텍스트 추출 완료: test/pypdf2_result.txt")
    except Exception as e:
        print(f"에러 발생: {str(e)}")

