# -*- coding: utf-8 -*-
# filename: ocr_test_pipeline.py
"""
흐림 + 회전 + 좌우반전이 섞인 test 데이터에서 OCR 리콜을 끌어올리는 '빠른' 멀티패스 파이프라인 (최소 증강).

변경 요약 (속도 최적화)
- 스케일 업(1.5/2.0), CLAHE+Unsharp, Bilateral, Adaptive 제거
- 전체 패스: (0°, 180°) × (원본, 좌우반전) = 4회만
- 필요 시에만 Otsu 이진화 1회 추가 시도
- 상단 타이틀 크롭 1회만 (ratio=0.22), 위와 동일한 최소 패스
- 초대형 이미지는 max_side 기준 한 번만 다운스케일

사용 예시
python test_subclass.py \
  --img_dir datasets_fin/test \
  --out_csv datasets_fin/test_ocr.csv \
  --make_parent_scores \
  --make_subclass \
  --keep_ext
"""

import sys
from pathlib import Path
import argparse
import unicodedata

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 외부
try:
    import easyocr
    import regex as cregex
    from rapidfuzz import fuzz
except Exception as e:
    print("❌ 필요한 패키지 설치:", e)
    print("pip install easyocr opencv-python numpy pandas regex rapidfuzz tqdm")
    sys.exit(1)

# =========================
# OCR 초기화
# =========================
def init_reader():
    try:
        # paragraph=False가 조금 더 빠른 편
        rdr = easyocr.Reader(['ko', 'en'], gpu=True)
        print("✅ EasyOCR 초기화 (GPU=True)")
        return rdr
    except Exception as e:
        print(f"⚠️ GPU 모드 실패: {e}\n   GPU=False로 재시도합니다.")
        rdr = easyocr.Reader(['ko', 'en'], gpu=False)
        print("✅ EasyOCR 초기화 (GPU=False)")
        return rdr

reader = init_reader()

# =========================
# 텍스트 정규화 유틸
# =========================
def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower()
    t = cregex.sub(r"[\p{Cc}\p{Cf}]", " ", t)              # 제어문자 제거
    t = cregex.sub(r"[^\p{Hangul}\p{Latin}\p{Nd}\s\.\-_/·:;]", " ", t)  # 허용문자만
    t = cregex.sub(r"\s+", " ", t).strip()
    return t

def nospace(s: str) -> str:
    return cregex.sub(r"[\s\.\-_/·:;]+", "", s or "")

# =========================
# (참고) 전처리 함수들 - 현재 빠른 파이프라인에서는 사용하지 않음
# =========================
def enhance_clahe_unsharp(img):
    """대비 향상 + 샤프닝 (미사용: 속도 최적화로 제외)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl,a,b))
    y = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bl = cv2.GaussianBlur(y,(0,0),1.0)
    sharp = cv2.addWeighted(y, 1.25, bl, -0.25, 0)
    return sharp

def to_otsu(img):
    """Otsu 이진화 (필요한 경우에만 1회 적용)"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def to_adaptive(img):
    """미사용: 속도 최적화로 제외"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(g, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def bilateral(img):
    """미사용: 속도 최적화로 제외"""
    return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

# =========================
# 멀티패스 OCR (빠른 버전)
# =========================
TITLE_RATIOS = [0.22]     # 상단 타이틀 크롭 비율 (1회만)
ROTATIONS    = [0, 180]   # 회전
FLIPS        = [0, 1]     # 0: none, 1: 좌우반전

def crop_title(img, ratio):
    H, W = img.shape[:2]
    h = max(10, int(H * ratio))
    return img[:h, :]

def rotate_image(img, deg):
    if deg == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _resize_cap(img, max_side=1600):
    """이미지가 너무 큰 경우만 다운스케일 (연산량 절감)"""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    r = max_side / float(m)
    return cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)

def _try_ocr_once(img) -> str:
    """원본(또는 변형본)에 한 번 읽기"""
    try:
        res = reader.readtext(img, detail=0, paragraph=False)
        return normalize_text(" ".join(res))
    except Exception:
        return ""

def _try_ocr_otsu(img) -> str:
    """Otsu 한 번만 (필요할 때만)"""
    try:
        g  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
        th3 = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        res = reader.readtext(th3, detail=0, paragraph=False)
        return normalize_text(" ".join(res))
    except Exception:
        return ""

EARLY_KWS = [
    "입원확인","입퇴원","입원사실","입원증명","입원 요약","입원요약",
    "통원확인","통원진료","외래진료","진료사실","치료확인","통원치료사실",
    "소견서","진료소견","의사소견","진단소견"
]

def extract_text_multi(img_path: Path) -> str:
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    img = _resize_cap(img)  # 과대해상도만 다운스케일

    texts = []

    # 1) 전체 이미지: 최소 패스 (원본 → 180°) × (좌우반전 Off/On)
    for rot in ROTATIONS:
        rotated = img if rot == 0 else rotate_image(img, rot)
        for flip in FLIPS:
            base = rotated if flip == 0 else cv2.flip(rotated, 1)

            # (a) 원본 한 번
            t = _try_ocr_once(base)
            if t:
                texts.append(t)
                if any(kw in t for kw in EARLY_KWS):
                    return t

    # 2) 전체 이미지: 안 잡히면 Otsu 한 번만 전체에 시도
    for rot in ROTATIONS:
        rotated = img if rot == 0 else rotate_image(img, rot)
        for flip in FLIPS:
            base = rotated if flip == 0 else cv2.flip(rotated, 1)

            t = _try_ocr_otsu(base)
            if t:
                texts.append(t)
                if any(kw in t for kw in EARLY_KWS):
                    return t

    # 3) 상단 타이틀 크롭: 1회만 (원본 → 180°) × (좌우반전 Off/On)
    for ratio in TITLE_RATIOS:
        top = crop_title(img, ratio)

        # (a) 원본 한 번
        for rot in ROTATIONS:
            rotated = top if rot == 0 else rotate_image(top, rot)
            for flip in FLIPS:
                base = rotated if flip == 0 else cv2.flip(rotated, 1)

                t = _try_ocr_once(base)
                if t:
                    texts.append(t)
                    if any(kw in t for kw in EARLY_KWS):
                        return t

        # (b) 필요시에만 Otsu 한 번
        for rot in ROTATIONS:
            rotated = top if rot == 0 else rotate_image(top, rot)
            for flip in FLIPS:
                base = rotated if flip == 0 else cv2.flip(rotated, 1)

                t = _try_ocr_otsu(base)
                if t:
                    texts.append(t)
                    if any(kw in t for kw in EARLY_KWS):
                        return t

    # 텍스트 중복 제거 후 합치기
    uniq, seen = [], set()
    for t in texts:
        k = nospace(t)
        if k and k not in seen:
            seen.add(k)
            uniq.append(t)
    return normalize_text(" ".join(uniq))

# =========================
# (옵션) 3/7/14 부모·서브클래스 매핑
# =========================
# OCR 흔오탈 치환은 정규화와 퍼지로 흡수하므로 단어 원형만 유지
CLASS_KEYWORDS = {
    3: { 0: ['입퇴원사실확인서', '입퇴원확인서', '입원확인서'],
         1: ['입원사실증명서', '입원사실증명원', '입퇴원사실증명서'],
         2: ['입원증명서', '입원퇴원증명서'],
         3: ['입원진료확인서', '입원 진료확인서'],
         4: ['입원요약지', '입원 요약지'] },
    7: { 0: ['통원확인서', '통원진료확인서'],
         1: ['진료확인서', '진료사실확인서'],
         2: ['외래진료사실확인서', '치료확인서'],
         3: ['통원치료사실확인서', '진료사실증명서'],
         4: ['외래진료확인서', '외래 진료 확인서'],
         5: ['치료사실확인서', '치료 사실 확인서'] },
    14:{ 0: ['소견서', '진료소견서', '의사소견서', '진단소견서', '임상소견서'] }
}

PARENT_RULES = {
    3: {"title": ["입원확인서", "입원사실", "입퇴원", "입원증명", "입원 진료확인", "입원 요약"],
        "pos":   ["입원", "퇴원", "입퇴원", "요약지", "진료요약"],
        "neg":   ["통원", "외래"]},
    7: {"title": ["통원확인서", "진료확인서", "통원진료", "외래진료사실", "치료확인", "진료사실", "통원치료사실",
                  "외래진료확인", "치료사실확인"],
        "pos":   ["통원", "외래", "진료사실", "치료사실", "치료확인", "내원", "외래진료", "통원치료"],
        "neg":   ["입원", "퇴원", "입퇴원", "입원진료", "입원요약"]},
    14:{"title": ["소견서", "진료소견서", "의사소견서", "진단소견서"],
        "pos":   ["소견"],
        "neg":   ["진단서", "확인서", "진료확인", "사실확인"]}
}

def contains_keyword(text: str, keyword: str, fuzzy_threshold: int = 82) -> bool:
    t = normalize_text(text)
    # 간격/점 허용 정규식
    gap = r"[\s\.\-_/·:;]*"
    pat = gap.join([cregex.escape(ch) for ch in keyword])
    rx = cregex.compile(pat, cregex.IGNORECASE)
    if rx.search(t):
        return True
    if nospace(keyword) in nospace(t):
        return True
    return fuzz.partial_ratio(keyword, t) >= fuzzy_threshold

def score_parent_class(text: str, class_id: int, fuzzy_threshold: int = 82) -> int:
    rules = PARENT_RULES[class_id]
    s = 0
    for kw in rules["title"]:
        if contains_keyword(text, kw, fuzzy_threshold): s += 3
    for kw in rules["pos"]:
        if contains_keyword(text, kw, fuzzy_threshold): s += 1
    for kw in rules["neg"]:
        if contains_keyword(text, kw, fuzzy_threshold): s -= 3
    return s

def match_subclass_specific(class_id: int, text: str, fuzzy_threshold: int = 82) -> int | None:
    if class_id not in CLASS_KEYWORDS:
        return None
    for sub_id, kws in CLASS_KEYWORDS[class_id].items():
        for kw in kws:
            if contains_keyword(text, kw, fuzzy_threshold):
                return sub_id
    return None

DEFAULT_SUBCLASS = {3: 9, 7: 9, 14: 9}  # OTHER로 폴백

def decide_parent_and_subclass(text: str, fuzzy_threshold: int = 82, parent_tau: int = 1):
    scores = {c: score_parent_class(text, c, fuzzy_threshold) for c in (3,7,14)}
    best_parent = max(scores, key=scores.get)
    best = scores[best_parent]
    second = sorted(scores.values(), reverse=True)[1]
    if best >= parent_tau and best >= second + 1:
        sub_id = match_subclass_specific(best_parent, text, fuzzy_threshold)
        if sub_id is None:
            sub_id = DEFAULT_SUBCLASS[best_parent]
        return best_parent, sub_id, scores, best-second
    return None, None, scores, 0

# =========================
# 메인 실행
# =========================
def run(
    img_dir: str,
    out_csv: str,
    keep_ext: bool = True,
    make_parent_scores: bool = False,
    make_subclass: bool = False,
    fuzzy_threshold: int = 82,
    parent_tau: int = 1
):
    img_dir = Path(img_dir)
    paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}])
    print(f"📂 Images: {len(paths)}개")

    rows = []
    for p in tqdm(paths):
        img_id = p.name if keep_ext else p.stem
        text = extract_text_multi(p)

        row = {"ID": img_id, "ocr_text": text}

        if make_parent_scores:
            s3 = score_parent_class(text, 3, fuzzy_threshold)
            s7 = score_parent_class(text, 7, fuzzy_threshold)
            s14 = score_parent_class(text, 14, fuzzy_threshold)
            scores = {3:s3, 7:s7, 14:s14}
            best_parent = max(scores, key=scores.get)
            best = scores[best_parent]
            second = sorted(scores.values(), reverse=True)[1]
            row.update({
                "parent_3": s3, "parent_7": s7, "parent_14": s14,
                "parent_best": best_parent if best >= parent_tau and best >= second + 1 else None,
                "parent_margin": best - second
            })

        if make_subclass:
            parent, sub, scores, margin = decide_parent_and_subclass(text, fuzzy_threshold, parent_tau)
            subclass_code = parent*10 + sub if (parent is not None and sub is not None) else None
            row.update({
                "target": subclass_code  # 서브클래스 코드(31/71/141...) 또는 None
            })

        rows.append(row)

    df = pd.DataFrame(rows)

    # 서브클래스 모드: ID와 target만 출력, target은 정수로 변환
    if make_subclass:
        df_out = df[["ID", "target"]].copy()
        # NaN을 유지하면서 정수로 변환 (Int64는 nullable integer)
        df_out["target"] = df_out["target"].astype("Int64")
    else:
        df_out = df

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"✅ 저장 완료: {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="멀티패스 OCR 파이프라인 (흐림/회전/좌우반전 대응, 빠른 버전)")
    ap.add_argument("--img_dir", required=True, help="이미지 폴더 (예: datasets_fin/test)")
    ap.add_argument("--out_csv", required=True, help="저장 경로 (예: datasets_fin/test_ocr.csv)")
    ap.add_argument("--keep_ext", action="store_true", help="ID에 확장자 포함 (SOTA와 동일 형식이면 권장)")
    ap.add_argument("--make_parent_scores", action="store_true", help="3/7/14 부모 점수/마진 저장")
    ap.add_argument("--make_subclass", action="store_true", help="서브클래스 코드(target=31/71/141...) 저장")
    ap.add_argument("--fuzzy_threshold", type=int, default=82, help="키워드 퍼지 임계")
    ap.add_argument("--parent_tau", type=int, default=1, help="부모 선택 임계")
    args = ap.parse_args()

    run(
        img_dir=args.img_dir,
        out_csv=args.out_csv,
        keep_ext=args.keep_ext,
        make_parent_scores=args.make_parent_scores,
        make_subclass=args.make_subclass,
        fuzzy_threshold=args.fuzzy_threshold,
        parent_tau=args.parent_tau
    )
