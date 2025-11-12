"""
OCR 기반 문서 분류 모듈

이 모듈은 기존 프로젝트와 완전히 독립적으로 동작합니다.
"""

from .ocr_extractor import OCRExtractor
from .ocr_classifier import OCRClassifier

__all__ = ['OCRExtractor', 'OCRClassifier']
