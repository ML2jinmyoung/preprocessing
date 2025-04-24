import os
import subprocess

def convert_ppt_to_pdf_in_dir(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith((".ppt", ".pptx")):
                ppt_path = os.path.join(dirpath, filename)
                print(f"🔄 변환 중: {ppt_path}")
                try:
                    subprocess.run([
                        "libreoffice",
                        "--headless",
                        "--convert-to", "pdf",
                        "--outdir", dirpath,
                        ppt_path
                    ], check=True)
                    print(f"✅ 완료: {os.path.join(dirpath, filename[:-5] + '.pdf')}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ 오류 발생: {ppt_path}\n{e}")


convert_ppt_to_pdf_in_dir("./web_disk")
