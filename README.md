# 영화 추천 시스템 (Streamlit)

사용자–영화 평점 데이터를 기반으로 **협업 필터링 + SVD**로 평점을 예측하고, **사용자 군집(클러스터)** 별 추천을 만든 뒤, **Streamlit 앱**에서 결과를 탐색할 수 있는 프로젝트입니다.

---

## 디렉터리 구조

```
.
├── app/
│   └── app.py                # Streamlit 앱
├── data/                     # (선택) 보조 데이터 폴더
├── dm_model.ipynb            # 전처리·모델링 노트북
├── movie_genre_df.csv        # 영화 메타(제목, movie_id, 장르 원-핫)
├── result_df.csv             # 클러스터별 추천 결과
├── requirements.txt
├── .gitignore
└── README.md
```

> `predicted_matrix.csv`(SVD 예측 평점 행렬, 약 437MB)는 **GitHub의 파일당 100MB 제한**으로 인해 저장소에 포함되어 있지 않습니다. 아래 [데이터 준비](#데이터-준비) 절차로 직접 생성하면 됩니다.

---

## 모델링 (`dm_model.ipynb`)

Google Colab 또는 로컬 Jupyter에서 실행할 수 있으며, 노트북 내 `read_csv` / `to_csv` 경로는 **본인 환경에 맞게 수정**해서 사용합니다.

### 파이프라인

1. **데이터 준비**
   - 사용자×영화 희소 평점 행렬 로드
   - 평가 수가 적은 사용자를 분위수 기준으로 필터링
   - `movie_genre_df`와 결합해 EDA(장르 분포 등)

2. **협업 필터링 & 행렬 분해**
   - `scikit-surprise`로 **user-based / item-based KNN** 비교
   - **SVD** (`n_factors=50`, biased) 학습 → RMSE / MAE로 검증
   - 관측되지 않은 셀까지 예측해 **`predicted_matrix.csv`** 생성

3. **사용자 클러스터링**
   - 예측 행렬을 **표준화 → PCA(2D) → K-Means**로 군집화 (예: k=10)
   - **Elbow / Silhouette**로 클러스터 수 검증

4. **클러스터별 추천 (`result_df.csv`)**
   - 군집별 영화 평균 예측 평점 계산
   - 전체 평균 대비 차이를 **특이성 점수**로 정의
   - 각 클러스터에 대해 **특화 추천 / 추천 / 비추천**으로 구분

---

## Streamlit 앱 (`app/app.py`)

### 4개 탭

1. **내 취향 추천** — 좋아하는 영화 + 장르 가중치로 **취향 벡터 유사도** 기반 추천 (회원가입·`userId` 불필요)
2. **클러스터 추천** — `result_df`에서 군집을 골라 **특화 추천 / 추천 / 비추천** 확인
3. **장르로 찾기** — 장르·제목 필터로 카탈로그 탐색
4. **사용자 맞춤 (예측 행렬)** — `predicted_matrix`의 `userId` 행에서 **예측 평점 상위** 영화 표시 (대용량이라 버튼 클릭 시 로드)

사이드바에 **보고 싶은 목록(워치리스트)** 을 모아둘 수 있습니다.

### 앱이 읽는 CSV 위치

`app/app.py`는 **자기 폴더(`app/`) 안**의 다음 파일을 읽습니다.

- `app/result_df.csv`
- `app/movie_genre_df.csv`
- `app/predicted_matrix.csv`

저장소 루트에만 CSV가 있다면 실행 전에 `app/`로 복사하세요.

```bash
cp movie_genre_df.csv result_df.csv predicted_matrix.csv app/
```

---

## 데이터 준비

`predicted_matrix.csv`는 저장소에 포함되어 있지 않으므로, **직접 생성**하거나 별도로 받아 넣어야 합니다.

1. `dm_model.ipynb`을 열고 셀의 데이터 경로를 본인 환경에 맞게 수정합니다.
2. SVD 학습 단계까지 실행하면 `predicted_matrix.csv`가 생성됩니다.
3. 생성된 파일을 프로젝트 루트(또는 `app/`)에 복사합니다.

> 4번 탭(**사용자 맞춤**)은 이 파일이 있어야 동작합니다. 파일이 없으면 1~3번 탭만 사용할 수 있습니다.

---

## 실행 방법

Python **3.10+** 권장.

```bash
git clone https://github.com/<계정>/<저장소>.git
cd <저장소>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# CSV를 app/ 폴더로 옮긴 뒤
cp movie_genre_df.csv result_df.csv app/
# predicted_matrix.csv도 생성 후 app/에 두면 4번 탭까지 사용 가능
streamlit run app/app.py
```

브라우저에서 기본 `http://localhost:8501` 로 접속합니다.

---

## 의존성

### 앱 실행용 (`requirements.txt`)

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
```

### 노트북 추가 의존성

노트북에서는 다음 패키지가 추가로 필요합니다(별도 설치).

```
scikit-surprise
scikit-learn
matplotlib
seaborn
koreanize-matplotlib
adjustText
```

> `scikit-surprise`는 환경에 따라 NumPy 버전 충돌이 생길 수 있어, 노트북 상단의 설치 셀을 참고하세요.

---

## 라이선스 및 데이터 출처

- 평점·장르 데이터는 **MovieLens** 형식과 호환되는 구조입니다. 공개 시 사용한 데이터 세트의 **라이선스 / 인용 조건**을 확인하세요. ([GroupLens MovieLens](https://grouplens.org/datasets/movielens/))
- 저장소 코드의 라이선스는 필요에 따라 별도 `LICENSE` 파일로 명시할 수 있습니다.
