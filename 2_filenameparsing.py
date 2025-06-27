import re
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class ParliamentFileParser:
    """국회 회의록 파일명 및 내용 파서"""
    
    def __init__(self):
        # 파일명 패턴: 국회본회의 회의록_052588_제21대_제400회_제14차_20221208.json
        self.filename_pattern = r'국회본회의\s회의록_(\d+)_제(\d+)대_제(\d+)회_제(\d+)차_(\d{8})\.json'
        
        # 발언자 유형 분류 패턴
        self.speaker_patterns = {
            '의장': r'의장\s+([가-힣]+)',
            '부의장': r'부의장\s+([가-힣]+)', 
            '위원장': r'([가-힣]+위원회)?.*위원장(?:대리)?\s+([가-힣]+)',
            '위원': r'([가-힣]+위원회)?\s*([가-힣]+)\s*위원',
            '의원': r'([가-힣]+)\s*의원',
            '국무위원': r'(.*부(?:총리|장관)?)\s+([가-힣]+)',
            '투표결과': r'투표|명단|찬성|반대|기권'
        }
    
    def parse_filename(self, filename: str) -> Dict:
        """파일명에서 회의 정보 추출"""
        match = re.search(self.filename_pattern, filename)
        if not match:
            return {}
        
        session_id, assembly_num, session_num, meeting_num, date_str = match.groups()
        
        # 날짜 파싱
        meeting_date = datetime.strptime(date_str, '%Y%m%d')
        
        return {
            'session_id': session_id,
            'assembly_number': int(assembly_num),  # 제21대
            'session_number': int(session_num),    # 제400회
            'meeting_number': int(meeting_num),    # 제14차
            'meeting_date': meeting_date,
            'date_string': date_str,
            'meeting_type': '본회의',
            'filename': filename
        }
    
    def parse_speaker(self, speaker_text: str) -> Dict:
        """발언자 정보 파싱"""
        speaker_info = {
            'original_text': speaker_text,
            'speaker_type': '기타',
            'name': '',
            'position': '',
            'committee': '',
            'party': ''
        }
        
        # 각 패턴별로 매칭 시도
        for speaker_type, pattern in self.speaker_patterns.items():
            match = re.search(pattern, speaker_text)
            if match:
                speaker_info['speaker_type'] = speaker_type
                
                if speaker_type in ['의장', '부의장']:
                    speaker_info['name'] = match.group(1)
                    speaker_info['position'] = speaker_type
                    
                elif speaker_type == '위원장':
                    speaker_info['committee'] = match.group(1) if match.group(1) else ''
                    speaker_info['name'] = match.group(2)
                    speaker_info['position'] = '위원장'
                    
                elif speaker_type == '위원':
                    speaker_info['committee'] = match.group(1) if match.group(1) else ''
                    speaker_info['name'] = match.group(2)
                    speaker_info['position'] = '위원'
                    
                elif speaker_type == '의원':
                    speaker_info['name'] = match.group(1)
                    speaker_info['position'] = '의원'
                    
                    # 정당 정보 추출 (있는 경우)
                    party_match = re.search(r'(더불어민주당|국민의힘|정의당|민주당|국민의당)', speaker_text)
                    if party_match:
                        speaker_info['party'] = party_match.group(1)
                
                break
        
        return speaker_info

# 파서 인스턴스 생성
parser = ParliamentFileParser()

# 파일명 파싱 테스트
filename = "국회본회의 회의록_052588_제21대_제400회_제14차_20221208.json"
file_info = parser.parse_filename(filename)
print("📁 파일 정보:")
print(f"  - 제{file_info['assembly_number']}대 국회")
print(f"  - 제{file_info['session_number']}회 {file_info['meeting_number']}차 {file_info['meeting_type']}")
print(f"  - 날짜: {file_info['meeting_date'].strftime('%Y년 %m월 %d일')}")