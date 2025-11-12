"""
OCR 텍스트 추출 모듈

이미지에서 OCR을 사용해 텍스트를 추출하고 특징을 생성합니다.
완전히 독립적으로 동작하며 기존 프로젝트 코드를 import하지 않습니다.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from collections import Counter


class OCRExtractor:
    """OCR 텍스트 추출 및 특징 생성 클래스"""

    def __init__(self, engine: str = "easyocr", languages: List[str] = ["ko", "en"]):
        """
        Args:
            engine: "easyocr" 또는 "pytesseract"
            languages: 인식할 언어 리스트
        """
        self.engine = engine
        self.languages = languages
        self.reader = None

        self._init_ocr_engine()

    def _init_ocr_engine(self):
        """OCR 엔진 초기화"""
        if self.engine == "easyocr":
            try:
                import easyocr
                print(f"✅ EasyOCR 초기화 중... (언어: {self.languages})")
                self.reader = easyocr.Reader(self.languages, gpu=True)
                print("✅ EasyOCR 초기화 완료")
            except ImportError:
                print("❌ EasyOCR가 설치되지 않았습니다. pip install easyocr")
                raise

        elif self.engine == "pytesseract":
            try:
                import pytesseract
                self.reader = pytesseract
                print("✅ Pytesseract 초기화 완료")
            except ImportError:
                print("❌ Pytesseract가 설치되지 않았습니다. pip install pytesseract")
                raise
        else:
            raise ValueError(f"지원하지 않는 OCR 엔진: {self.engine}")

    def extract_text(self, image_path: str) -> Dict:
        """
        이미지에서 텍스트 추출

        Args:
            image_path: 이미지 파일 경로

        Returns:
            {
                'text': 추출된 전체 텍스트,
                'words': 단어 리스트,
                'boxes': 텍스트 박스 위치 정보,
                'confidence': 평균 confidence
            }
        """
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            return self._empty_result()

        # OCR 실행
        if self.engine == "easyocr":
            return self._extract_with_easyocr(image)
        else:
            return self._extract_with_pytesseract(image)

    def _extract_with_easyocr(self, image: np.ndarray) -> Dict:
        """EasyOCR로 텍스트 추출"""
        try:
            results = self.reader.readtext(image)

            if not results:
                return self._empty_result()

            # 결과 파싱
            texts = []
            boxes = []
            confidences = []

            for (bbox, text, conf) in results:
                texts.append(text)
                boxes.append(bbox)
                confidences.append(conf)

            full_text = " ".join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            return {
                'text': full_text,
                'words': texts,
                'boxes': boxes,
                'confidence': avg_confidence,
                'num_words': len(texts)
            }

        except Exception as e:
            print(f"⚠️ EasyOCR 추출 실패: {e}")
            return self._empty_result()

    def _extract_with_pytesseract(self, image: np.ndarray) -> Dict:
        """Pytesseract로 텍스트 추출"""
        try:
            import pytesseract

            # 텍스트 추출
            text = pytesseract.image_to_string(image, lang='kor+eng')

            # 상세 정보 추출
            data = pytesseract.image_to_data(image, lang='kor+eng', output_type=pytesseract.Output.DICT)

            # 유효한 단어만 필터링
            words = [data['text'][i] for i in range(len(data['text']))
                    if data['conf'][i] > 0 and data['text'][i].strip()]

            confidences = [data['conf'][i] for i in range(len(data['conf']))
                          if data['conf'][i] > 0 and data['text'][i].strip()]

            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

            return {
                'text': text,
                'words': words,
                'boxes': None,
                'confidence': avg_confidence,
                'num_words': len(words)
            }

        except Exception as e:
            print(f"⚠️ Pytesseract 추출 실패: {e}")
            return self._empty_result()

    def _empty_result(self) -> Dict:
        """빈 결과 반환"""
        return {
            'text': '',
            'words': [],
            'boxes': [],
            'confidence': 0.0,
            'num_words': 0
        }

    def extract_features(self, ocr_result: Dict) -> Dict:
        """
        OCR 결과에서 특징 추출

        Args:
            ocr_result: extract_text()의 결과

        Returns:
            특징 딕셔너리
        """
        text = ocr_result['text']
        words = ocr_result['words']

        features = {
            # 기본 특징
            'text_length': len(text),
            'num_words': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'confidence': ocr_result['confidence'],

            # 문자 타입
            'num_digits': sum(c.isdigit() for c in text),
            'num_alpha': sum(c.isalpha() for c in text),
            'num_special': sum(not c.isalnum() and not c.isspace() for c in text),

            # 패턴 매칭
            'has_date': self._has_date_pattern(text),
            'has_phone': self._has_phone_pattern(text),
            'has_email': self._has_email_pattern(text),
            'has_amount': self._has_amount_pattern(text),

            # 키워드 (문서 타입 구분용)
            'has_invoice_keywords': self._has_keywords(text, ['invoice', '청구서', '계산서']),
            'has_receipt_keywords': self._has_keywords(text, ['receipt', '영수증', '받음']),
            'has_form_keywords': self._has_keywords(text, ['form', '양식', '신청서']),
            'has_report_keywords': self._has_keywords(text, ['report', '보고서', '리포트']),
            'has_contract_keywords': self._has_keywords(text, ['contract', '계약서', '약정']),

            # 레이아웃 특징
            'num_boxes': len(ocr_result['boxes']) if ocr_result['boxes'] else 0,
        }

        return features

    def _has_date_pattern(self, text: str) -> bool:
        """날짜 패턴 존재 여부"""
        # YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY 등
        date_patterns = [
            r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}',
            r'\d{1,2}[-/.]\d{1,2}[-/.]\d{4}',
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일'
        ]
        return any(re.search(pattern, text) for pattern in date_patterns)

    def _has_phone_pattern(self, text: str) -> bool:
        """전화번호 패턴 존재 여부"""
        phone_patterns = [
            r'\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4}',
            r'\(\d{2,3}\)\s?\d{3,4}[-.\s]?\d{4}'
        ]
        return any(re.search(pattern, text) for pattern in phone_patterns)

    def _has_email_pattern(self, text: str) -> bool:
        """이메일 패턴 존재 여부"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return bool(re.search(email_pattern, text))

    def _has_amount_pattern(self, text: str) -> bool:
        """금액 패턴 존재 여부"""
        amount_patterns = [
            r'[$₩€¥]\s*\d+[,.\d]*',
            r'\d+[,.\d]*\s*[$₩€¥원]',
            r'\d+[,.\d]*원'
        ]
        return any(re.search(pattern, text) for pattern in amount_patterns)

    def _has_keywords(self, text: str, keywords: List[str]) -> bool:
        """키워드 존재 여부"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def process_batch(self, image_paths: List[str], verbose: bool = True) -> List[Dict]:
        """
        여러 이미지를 배치로 처리

        Args:
            image_paths: 이미지 경로 리스트
            verbose: 진행 상황 출력 여부

        Returns:
            특징 딕셔너리 리스트
        """
        results = []

        if verbose:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="OCR 추출 중")
        else:
            iterator = image_paths

        for img_path in iterator:
            ocr_result = self.extract_text(img_path)
            features = self.extract_features(ocr_result)
            features['image_path'] = str(img_path)
            features['image_name'] = Path(img_path).name
            results.append(features)

        return results


if __name__ == "__main__":
    # 테스트 코드
    print("OCR Extractor 테스트")

    # EasyOCR로 테스트
    extractor = OCRExtractor(engine="easyocr", languages=["ko", "en"])

    # 샘플 이미지 테스트
    test_image = "datasets_fin/test/0008fdb22ddce0ce.jpg"

    if Path(test_image).exists():
        result = extractor.extract_text(test_image)
        print(f"\n추출된 텍스트: {result['text'][:100]}...")
        print(f"단어 수: {result['num_words']}")
        print(f"Confidence: {result['confidence']:.3f}")

        features = extractor.extract_features(result)
        print(f"\n특징:")
        for key, value in features.items():
            print(f"  {key}: {value}")
