"""
OCR 기반 문서 분류기

OCR에서 추출한 특징을 사용하여 문서를 분류합니다.
두 가지 방식 지원:
1. 규칙 기반 (Rule-based)
2. ML 기반 (TF-IDF + Classifier)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle


class OCRClassifier:
    """OCR 기반 문서 분류기"""

    def __init__(self, method: str = "rule", num_classes: int = 17):
        """
        Args:
            method: "rule" (규칙 기반) 또는 "ml" (머신러닝)
            num_classes: 클래스 개수
        """
        self.method = method
        self.num_classes = num_classes
        self.model = None
        self.vectorizer = None

        # 클래스별 규칙 (수동으로 정의 - 데이터 분석 후 업데이트 필요)
        self.class_rules = self._define_class_rules()

    def _define_class_rules(self) -> Dict:
        """
        클래스별 규칙 정의 (예시)

        실제 사용시 클래스 3, 7의 특징을 분석해서 업데이트 필요!
        """
        rules = {
            # 클래스 3 규칙 예시 (실제 데이터 보고 수정 필요)
            3: {
                'keywords': ['특정키워드1', '특정키워드2'],
                'min_text_length': 100,
                'has_date': True,
                'has_amount': False,
            },
            # 클래스 7 규칙 예시 (실제 데이터 보고 수정 필요)
            7: {
                'keywords': ['다른키워드1', '다른키워드2'],
                'min_text_length': 50,
                'has_phone': True,
                'has_email': False,
            },
        }
        return rules

    def predict_with_rules(self, features: Dict) -> Tuple[int, float]:
        """
        규칙 기반 예측

        Args:
            features: OCRExtractor.extract_features()의 결과

        Returns:
            (predicted_class, confidence)
        """
        scores = defaultdict(float)

        # 클래스별 규칙 체크
        for class_id, rules in self.class_rules.items():
            score = 0.0
            max_score = 0.0

            # 키워드 체크
            if 'keywords' in rules:
                max_score += 1.0
                text = features.get('text', '').lower()
                if any(kw.lower() in text for kw in rules['keywords']):
                    score += 1.0

            # 텍스트 길이 체크
            if 'min_text_length' in rules:
                max_score += 0.5
                if features.get('text_length', 0) >= rules['min_text_length']:
                    score += 0.5

            # 패턴 체크
            for pattern_key in ['has_date', 'has_phone', 'has_email', 'has_amount']:
                if pattern_key in rules:
                    max_score += 0.5
                    if features.get(pattern_key, False) == rules[pattern_key]:
                        score += 0.5

            # 정규화된 점수
            if max_score > 0:
                scores[class_id] = score / max_score

        # 가장 높은 점수의 클래스 선택
        if scores:
            best_class = max(scores.items(), key=lambda x: x[1])
            return best_class[0], best_class[1]
        else:
            # 규칙에 매칭되지 않으면 -1 반환 (다른 모델 사용)
            return -1, 0.0

    def train_ml_model(self, train_data: pd.DataFrame, train_labels: np.ndarray):
        """
        ML 모델 학습 (TF-IDF + Logistic Regression)

        Args:
            train_data: OCR 추출 결과 DataFrame
            train_labels: 레이블 (0~16)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        print("📚 ML 모델 학습 중...")

        # 텍스트 특징
        texts = train_data['text'].fillna('')

        # TF-IDF 벡터화
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2
        )

        # 파이프라인 생성
        self.model = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=42
            ))
        ])

        # 학습
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, train_labels)

        print("✅ ML 모델 학습 완료")

    def predict_with_ml(self, features: Dict) -> Tuple[int, float]:
        """
        ML 모델로 예측

        Args:
            features: OCRExtractor.extract_features()의 결과

        Returns:
            (predicted_class, confidence)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("모델이 학습되지 않았습니다. train_ml_model()을 먼저 실행하세요.")

        text = features.get('text', '')

        # TF-IDF 변환
        X = self.vectorizer.transform([text])

        # 예측
        pred_class = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]
        confidence = pred_proba[pred_class]

        return int(pred_class), float(confidence)

    def predict(self, features: Dict) -> Tuple[int, float]:
        """
        예측 (method에 따라 자동 선택)

        Args:
            features: OCRExtractor.extract_features()의 결과

        Returns:
            (predicted_class, confidence)
        """
        if self.method == "rule":
            return self.predict_with_rules(features)
        elif self.method == "ml":
            return self.predict_with_ml(features)
        else:
            raise ValueError(f"지원하지 않는 method: {self.method}")

    def predict_batch(self, features_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 예측

        Args:
            features_list: 특징 딕셔너리 리스트

        Returns:
            (predictions, confidences)
        """
        predictions = []
        confidences = []

        for features in features_list:
            pred, conf = self.predict(features)
            predictions.append(pred)
            confidences.append(conf)

        return np.array(predictions), np.array(confidences)

    def save(self, path: str):
        """모델 저장"""
        model_data = {
            'method': self.method,
            'num_classes': self.num_classes,
            'model': self.model,
            'vectorizer': self.vectorizer,
            'class_rules': self.class_rules
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ 모델 저장 완료: {path}")

    @classmethod
    def load(cls, path: str):
        """모델 로드"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        classifier = cls(
            method=model_data['method'],
            num_classes=model_data['num_classes']
        )
        classifier.model = model_data['model']
        classifier.vectorizer = model_data['vectorizer']
        classifier.class_rules = model_data['class_rules']

        print(f"✅ 모델 로드 완료: {path}")
        return classifier


def create_ocr_predictions_csv(
    test_dir: str,
    output_path: str,
    extractor,
    classifier,
    sample_submission_path: str = "datasets_fin/sample_submission.csv"
):
    """
    OCR 기반 예측 CSV 생성

    Args:
        test_dir: 테스트 이미지 디렉토리
        output_path: 출력 CSV 경로
        extractor: OCRExtractor 인스턴스
        classifier: OCRClassifier 인스턴스
        sample_submission_path: 샘플 submission 경로
    """
    from tqdm import tqdm

    # 샘플 submission 로드
    sample_df = pd.read_csv(sample_submission_path)

    # 결과 저장용
    predictions = []
    confidences = []

    print(f"🔍 OCR 예측 시작 (총 {len(sample_df)}개 이미지)")

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        image_name = row['ID']
        image_path = Path(test_dir) / image_name

        if not image_path.exists():
            print(f"⚠️ 이미지 없음: {image_name}")
            predictions.append(0)
            confidences.append(0.0)
            continue

        # OCR 추출
        ocr_result = extractor.extract_text(str(image_path))

        # 특징 추출
        features = extractor.extract_features(ocr_result)
        features['text'] = ocr_result['text']

        # 예측
        pred, conf = classifier.predict(features)

        predictions.append(pred)
        confidences.append(conf)

    # DataFrame 생성
    result_df = pd.DataFrame({
        'ID': sample_df['ID'],
        'target': predictions,
        'confidence': confidences
    })

    # 저장
    result_df.to_csv(output_path, index=False)
    print(f"✅ OCR 예측 저장 완료: {output_path}")

    # 통계 출력
    print(f"\n📊 예측 통계:")
    print(f"  평균 confidence: {np.mean(confidences):.3f}")
    print(f"  예측 불가 (-1): {sum(np.array(predictions) == -1)}개")
    print(f"\n클래스별 분포:")
    print(result_df['target'].value_counts().sort_index())

    return result_df


if __name__ == "__main__":
    # 테스트 코드
    print("OCR Classifier 테스트")

    # 규칙 기반 분류기
    classifier = OCRClassifier(method="rule")

    # 테스트 특징
    test_features = {
        'text': 'Invoice 2024-01-01 Total: $100',
        'text_length': 30,
        'has_date': True,
        'has_amount': True,
        'has_invoice_keywords': True
    }

    pred, conf = classifier.predict(test_features)
    print(f"예측: 클래스 {pred}, 신뢰도 {conf:.3f}")
