def _chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> str:
	"""
	텍스트를 chunk_size 만큼 나누고, overlap만큼 간격 주고 <Chunk> 태그로 감싸고 <Content>로 전체 감쌈
	"""
	if chunk_size is None:
		chunk_size = 1000
	if overlap is None:
		overlap = 50
		
	chunks = []
	start = 0
	while start < len(text):
		end = start + chunk_size
		chunks.append(text[start:end])
		start += chunk_size - overlap  # 겹치도록 다음 시작 위치 설정

	chunked_text = "<Content>\n"
	for chunk in chunks:
		chunked_text += f"<Chunk>\n{chunk.strip()}\n</Chunk>\n\n"
	chunked_text += "</Content>\n"
	return chunked_text

# 텍스트 파일 읽기
with open('./demo_result/210317-2직장 내 괴롭힘으로 인한 건강장해 예방 매뉴얼(책자)_raw1', 'r', encoding='utf-8') as f:
	text = f.read()

# 청킹 결과를 raw.txt 파일로 저장
result = _chunk_text(text, 1000, 50)
with open('raw.txt', 'w', encoding='utf-8') as f:
	f.write(result)
	
	
	
	
	
