# 영화 추천 시스템 (Streamlit 앱)

사용자–영화 평점 데이터를 바탕으로 **협업 필터링·행렬 분해(SVD)** 로 평점을 예측하고, **사용자 군집(클러스터)** 별 추천을 정리한 뒤, 그 결과를 **Streamlit 웹 앱**에서 탐색하는 프로젝트입니다.

## 디렉터리 구조

| 경로 | 설명 |
|------|------|
| `app/app.py` | Streamlit 영화 추천 UI |
| `dm_model.ipynb` | 전처리, EDA, CF/SVD 학습, 클러스터링, `result_df`·`predicted_matrix` 생성 |
| `requirements.txt` | 앱 실행용 Python 의존성 (Streamlit, pandas, numpy) |
| `movie_genre_df.csv`, `result_df.csv`, `predicted_matrix.csv` | 학습·앱용 데이터 (루트 또는 `app/`에 두는 방식은 아래 참고) |
| `data/` | (선택) `movie_genre_df`, `result_df` 등 보조 복사본을 둘 수 있는 폴더 |

### 앱이 읽는 CSV 위치

`app/app.py`는 **`app/` 폴더와 같은 디렉터리**에 있는 아래 파일을 읽습니다.

- `app/result_df.csv`
- `app/movie_genre_df.csv`
- `app/predicted_matrix.csv`

저장소 루트에만 CSV가 있다면, 실행 전에 `app/`으로 복사하거나 심볼릭 링크로 맞추면 됩니다.

## 모델링 (`dm_model.ipynb`)

Google Colab 또는 로컬 Jupyter에서 실행할 수 있습니다. 노트북 안의 **`read_csv` / `to_csv` 경로는 본인 환경에 맞게 수정**하면 됩니다. (개인 Google Drive 전체 경로에 의존하지 않도록 정리한 상태를 기준으로 설명합니다.)

### 파이프라인 요약

1. **데이터**  
   사용자×영화 **희소 평점 행렬**을 불러오고, 평가 수가 적은 사용자를 분위수 등으로 필터링해 학습 집합을 만듭니다. 장르 메타데이터(`movie_genre_df`)와 맞춰 EDA합니다.

2. **협업 필터링·SVD**  
   **scikit-surprise**로 user/item 기반 KNN 실험 후, **SVD**(`n_factors=50`, 바이어스 항)으로 학습합니다. 관측되지 않은 평점까지 채운 **예측 행렬**이 `predicted_matrix.csv`의 원천입니다.

3. **클러스터링·`result_df`**  
   예측 행렬을 **표준화 → PCA(2차원) → K-Means** 등으로 사용자를 군집화하고, 군집별 평균 예측과 전체 대비 **특이성**을 해석해 **특화 추천 / 추천 / 비추천** 목록을 만듭니다. 그 결과가 `result_df.csv`입니다.

Colab에서는 파일 업로드, 런타임 임시 경로(`/content/...`), 또는 필요 시 드라이브 마운트 후 **자신이 쓰는 단일 기준 폴더**만 노트북에 적어 사용하면 됩니다.

## 앱 (`app/app.py`)

**Streamlit** 단일 앱입니다. 회원가입 없이 CSV 기반으로 추천을 둘러볼 수 있습니다.

### 탭 구성

1. **내 취향 추천** — 좋아하는 영화(장르 원-핫)와 장르 가중으로 **취향 벡터 유사도** 기반 추천  
2. **클러스터 추천** — `result_df`의 군집별 **특화 추천 / 추천 / 비추천**  
3. **장르로 찾기** — 장르·제목 필터  
4. **사용자 맞춤 (예측 행렬)** — `predicted_matrix`에서 `userId` 행을 읽어 예측 평점 상위 표시 (파일이 크므로 앱에서 필요할 때 로드)

사이드바 **보고 싶은 목록**에 영화를 모을 수 있습니다.

## 실행 방법

Python 3.10+ 권장.

```bash
cd /path/to/this/repo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# CSV가 루트에만 있다면:
# cp movie_genre_df.csv result_df.csv predicted_matrix.csv app/
streamlit run app/app.py
```

브라우저에서 기본 `http://localhost:8501` 로 접속합니다.

### 데이터·GitHub

- `predicted_matrix.csv`는 **수백 MB**일 수 있어 앱 첫 로드가 오래 걸릴 수 있습니다.  
- GitHub 일반 저장소는 **파일당 100MB 초과**를 막으므로, 큰 파일은 [Git LFS](https://git-lfs.github.com/) 또는 클라우드 링크로 두는 방식을 검토하세요.

### 노트북 의존성

노트북은 **Surprise**, **scikit-learn**, **seaborn** 등 추가 패키지를 사용합니다. Colab이면 셀에서 설치하고, 로컬이면 해당 환경에 맞게 `pip install` 하면 됩니다. Surprise 사용 시 **NumPy 버전**을 맞출 필요가 있을 수 있습니다(노트북 상단 설치 셀 참고).

## 라이선스 및 데이터 출처

- 평점·장르 구조는 **MovieLens 계열**과 같은 형태로 쓰인 경우가 많습니다. 공개 시 원 데이터 세트 **라이선스**를 확인하세요. ([GroupLens MovieLens](https://grouplens.org/datasets/movielens/))  
- 저장소 코드·문서의 라이선스는 필요 시 별도 `LICENSE` 파일로 명시하면 됩니다.
