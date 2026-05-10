"""
영화 추천 앱 — result_df(클러스터별 추천), movie_genre_df(장르), predicted_matrix(사용자별 예측 평점)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_result_df() -> pd.DataFrame:
    path = BASE_DIR / "result_df.csv"
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


def _read_movie_genres() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "movie_genre_df.csv")


def _read_predicted_matrix() -> pd.DataFrame:
    path = BASE_DIR / "predicted_matrix.csv"
    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()
    dtypes: dict[str, type] = {}
    for c in cols:
        if c == "userId":
            dtypes[c] = np.int32
        else:
            dtypes[c] = np.float32
    return pd.read_csv(path, dtype=dtypes)


@st.cache_data(show_spinner=False)
def load_result_df() -> pd.DataFrame:
    return _read_result_df()


@st.cache_data(show_spinner=False)
def load_movie_genres() -> pd.DataFrame:
    return _read_movie_genres()


@st.cache_resource(show_spinner="예측 행렬을 불러오는 중입니다(최초 1회, 수십 초 걸릴 수 있음)…")
def load_predicted_matrix() -> pd.DataFrame:
    return _read_predicted_matrix()


GENRE_COLS = [
    "(no genres listed)",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def genres_for_row(row: pd.Series) -> list[str]:
    tags: list[str] = []
    for g in GENRE_COLS:
        if g in row.index and int(row[g]) == 1:
            tags.append(g)
    return tags


@st.cache_data(show_spinner=False)
def movie_title_to_id_map() -> dict[str, int]:
    mg = load_movie_genres()
    out: dict[str, int] = {}
    for _, row in mg.iterrows():
        t = str(row["title"])
        if t not in out:
            out[t] = int(row["movie_id"])
    return out


@st.cache_data(show_spinner=False)
def movie_id_to_meta() -> tuple[dict[int, str], dict[int, list[str]]]:
    mg = load_movie_genres()
    titles: dict[int, str] = {}
    genres: dict[int, list[str]] = {}
    for _, row in mg.iterrows():
        mid = int(row["movie_id"])
        titles[mid] = str(row["title"])
        genres[mid] = genres_for_row(row)
    return titles, genres


def top_movies_for_user_row(
    row: pd.Series,
    movie_cols: list[str],
    titles: dict[int, str],
    genres: dict[int, list[str]],
    top_n: int,
) -> list[dict]:
    scores = row[movie_cols].astype(np.float32)
    order = np.argsort(-scores.values)
    out: list[dict] = []
    for idx in order:
        if len(out) >= top_n:
            break
        mid = int(movie_cols[idx])
        score = float(scores.iloc[idx])
        out.append(
            {
                "movie_id": mid,
                "title": titles.get(mid, f"(ID {mid})"),
                "genres": genres.get(mid, []),
                "predicted_rating": round(score, 2),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def catalog_genre_matrix() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """movie_id 배열, 장르 원-핫 행렬 (float32), 제목 리스트 (인덱스 정렬 일치)."""
    mg = load_movie_genres()
    mids = mg["movie_id"].to_numpy(np.int32)
    mat = mg[GENRE_COLS].to_numpy(np.float32)
    titles = mg["title"].astype(str).tolist()
    return mids, mat, titles


def recommend_from_taste(
    liked_ids: list[int],
    boost_genres: list[str],
    avoid_genres: list[str],
    top_n: int,
    mids: np.ndarray,
    mat: np.ndarray,
    titles: list[str],
) -> list[dict]:
    """좋아하는 영화·장르 선호로 장르 벡터 유사도 추천."""
    n_genres = mat.shape[1]
    user = np.zeros(n_genres, dtype=np.float64)
    id_to_row = {int(mids[i]): i for i in range(len(mids))}
    for mid in liked_ids:
        r = id_to_row.get(int(mid))
        if r is not None:
            user += mat[r].astype(np.float64)
    for g in boost_genres:
        if g in GENRE_COLS:
            user[GENRE_COLS.index(g)] += 1.5
    if user.sum() < 1e-6:
        user[:] = 1.0 / n_genres
    else:
        user /= np.linalg.norm(user) + 1e-9

    scores = mat.astype(np.float64) @ user
    for g in avoid_genres:
        if g in GENRE_COLS:
            gi = GENRE_COLS.index(g)
            scores -= mat[:, gi] * 0.35

    liked_set = {int(x) for x in liked_ids}
    order = np.argsort(-scores)
    out: list[dict] = []
    for idx in order:
        if len(out) >= top_n:
            break
        mid = int(mids[idx])
        if mid in liked_set:
            continue
        vec = mat[idx]
        gtags = [GENRE_COLS[i] for i in range(len(GENRE_COLS)) if float(vec[i]) >= 0.5]
        out.append(
            {
                "movie_id": mid,
                "title": titles[idx],
                "score": round(float(scores[idx]), 3),
                "genres": gtags,
            }
        )
    return out


def _init_session() -> None:
    if "liked_movie_ids" not in st.session_state:
        st.session_state["liked_movie_ids"] = []
    if "watchlist_ids" not in st.session_state:
        st.session_state["watchlist_ids"] = []


def main() -> None:
    _init_session()
    st.set_page_config(page_title="영화 추천", page_icon="🎬", layout="wide")
    st.title("🎬 영화 추천")
    st.caption("내 취향 맞춤 · 클러스터 추천 · 장르 탐색 · 사용자 ID 예측 행렬")

    with st.sidebar:
        st.header("나만의 목록")
        st.caption("아래 탭에서 영화를 **보고 싶은 목록**에 넣을 수 있습니다.")
        wl = st.session_state["watchlist_ids"]
        titles_map, _ = movie_id_to_meta()
        if not wl:
            st.write("목록이 비어 있어요.")
        else:
            for i, wid in enumerate(wl):
                t = titles_map.get(wid, str(wid))
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"· {t}")
                with c2:
                    if st.button("✕", key=f"wl_rm_{wid}_{i}", help="목록에서 제거"):
                        st.session_state["watchlist_ids"] = [
                            x for x in wl if x != wid
                        ]
                        st.rerun()
        if st.button("보고 싶은 목록 비우기"):
            st.session_state["watchlist_ids"] = []
            st.rerun()

    tab_taste, tab_cluster, tab_genre, tab_user = st.tabs(
        [
            "내 취향 추천",
            "클러스터 추천",
            "장르로 찾기",
            "사용자 맞춤 (예측 행렬)",
        ]
    )

    with tab_taste:
        st.markdown(
            "**좋아하는 영화**를 고르고, 필요하면 **더 끌리는 장르**·**덜 보고 싶은 장르**를 지정하면 "
            "같은 데이터로 취향 벡터를 만들어 추천합니다. (별도 회원가입·userId 없이 이용)"
        )
        mids, mat, title_list = catalog_genre_matrix()
        mg = load_movie_genres()
        titles_map, _ = movie_id_to_meta()

        st.subheader("1) 좋아하는 영화 담기")
        search_q = st.text_input("제목으로 검색 후 목록에 추가", placeholder="예: Matrix, Toy Story …")
        if search_q.strip():
            candidates = mg[
                mg["title"].str.contains(search_q.strip(), case=False, na=False)
            ].head(80)
        else:
            candidates = mg.head(60)

        opt_labels: list[str] = []
        opt_ids: list[int] = []
        for _, row in candidates.iterrows():
            mid = int(row["movie_id"])
            opt_ids.append(mid)
            opt_labels.append(f"{row['title']}  (id {mid})")

        if not opt_labels:
            st.warning("검색 결과가 없어요. 다른 키워드를 써 보세요.")
        else:
            pick_label = st.selectbox(
                "검색 결과에서 선택",
                range(len(opt_labels)),
                format_func=lambda i: opt_labels[i],
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("내 취향 목록에 추가", type="primary"):
                    chosen_id = opt_ids[pick_label]
                    if chosen_id not in st.session_state["liked_movie_ids"]:
                        st.session_state["liked_movie_ids"].append(chosen_id)
                        st.success("추가했어요.")
                        st.rerun()
                    else:
                        st.warning("이미 들어 있는 영화예요.")
            with b2:
                if st.button("보고 싶은 목록에 넣기"):
                    chosen_id = opt_ids[pick_label]
                    if chosen_id not in st.session_state["watchlist_ids"]:
                        st.session_state["watchlist_ids"].append(chosen_id)
                        st.success("사이드바 목록에 넣었어요.")
                        st.rerun()

        liked = st.session_state["liked_movie_ids"]
        if liked:
            st.write("**현재 취향 목록** (제거하려면 옆 버튼)")
            for i, mid in enumerate(liked):
                pos_arr = np.where(mids == mid)[0]
                if len(pos_arr):
                    t = title_list[int(pos_arr[0])]
                else:
                    t = titles_map.get(mid, str(mid))
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.caption(f"· {t}")
                with c2:
                    if st.button("제거", key=f"like_rm_{mid}_{i}"):
                        st.session_state["liked_movie_ids"] = [
                            x for x in liked if x != mid
                        ]
                        st.rerun()
        else:
            st.info("영화를 한 편 이상 추가하면 추천이 더 정확해져요. 없어도 장르 가중치만으로 동작합니다.")

        st.subheader("2) 장르 조정 (선택)")
        boost = st.multiselect(
            "이런 장르를 더 반영하고 싶어요",
            GENRE_COLS,
            default=[],
        )
        avoid = st.multiselect(
            "이런 장르는 덜 추천해 주세요",
            GENRE_COLS,
            default=[],
        )
        rec_n = st.slider("추천 개수", 5, 40, 15)

        c_go, c_clr = st.columns([1, 1])
        with c_go:
            run_rec = st.button("이 설정으로 추천 받기", type="primary")
        with c_clr:
            if st.button("추천 결과 지우기"):
                st.session_state.pop("last_taste_recs", None)
                st.rerun()

        if run_rec:
            if not liked and not boost:
                st.warning("좋아하는 영화를 하나 이상 넣거나, ‘더 반영’ 장르를 골라 주세요.")
            else:
                st.session_state["last_taste_recs"] = recommend_from_taste(
                    liked,
                    boost,
                    avoid,
                    rec_n,
                    mids,
                    mat,
                    title_list,
                )

        last_recs = st.session_state.get("last_taste_recs")
        if last_recs:
            st.subheader("추천 결과")
            df_rec = pd.DataFrame(last_recs)
            df_rec["장르"] = df_rec["genres"].apply(lambda g: ", ".join(g) if g else "—")
            df_rec = df_rec.drop(columns=["genres"])
            df_rec = df_rec.rename(
                columns={"score": "취향 점수", "title": "제목", "movie_id": "영화 ID"}
            )
            st.dataframe(df_rec, use_container_width=True, hide_index=True)

            st.markdown("**마음에 들면** 보고 싶은 목록에 추가:")
            add_opts = {
                f"{r['title']} ({r['movie_id']})": int(r["movie_id"]) for r in last_recs
            }
            pick_wl = st.selectbox(
                "추천 중에서 선택",
                list(add_opts.keys()),
                key="taste_pick_wl",
            )
            if st.button("선택한 영화를 보고 싶은 목록에 추가", key="taste_btn_wl"):
                wid = add_opts[pick_wl]
                if wid not in st.session_state["watchlist_ids"]:
                    st.session_state["watchlist_ids"].append(wid)
                st.success("사이드바에서 확인할 수 있어요.")
                st.rerun()

    with tab_cluster:
        df = load_result_df()
        clusters = sorted(df["Cluster"].unique().tolist())
        c1, c2 = st.columns([1, 2])
        with c1:
            cluster = st.selectbox("사용자 그룹(클러스터)", clusters, format_func=lambda x: f"클러스터 {x}")
        subset = df[df["Cluster"] == cluster]
        with c2:
            st.info(
                "모델이 비슷한 취향으로 묶은 그룹입니다. **특화 추천**은 그룹에서 두드러지는 작품, "
                "**추천**은 무난한 긍정 후보, **비추천**은 기대가 낮은 편입니다."
            )

        t2id = movie_title_to_id_map()
        for label, title_ko in [
            ("특화 추천", "특화 추천"),
            ("추천", "추천"),
            ("비추천", "비추천"),
        ]:
            block = subset[subset["구분"] == label]
            if block.empty:
                continue
            st.subheader(title_ko)
            display = block[["영화 제목", "예상 평점", "특이성 점수"]].copy()
            display = display.rename(
                columns={
                    "영화 제목": "영화",
                    "예상 평점": "예상 평점",
                    "특이성 점수": "특이성",
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)
            pick_titles = block["영화 제목"].astype(str).tolist()
            st.caption("표에 나온 제목을 골라 **보고 싶은 목록**(사이드바)에 넣을 수 있어요.")
            c_p, c_b = st.columns([3, 1])
            with c_p:
                sel_title = st.selectbox(
                    "영화 선택",
                    pick_titles,
                    key=f"cl_pick_{cluster}_{label}",
                )
            with c_b:
                if st.button("목록에 추가", key=f"cl_add_{cluster}_{label}"):
                    mid = t2id.get(sel_title)
                    if mid is None:
                        st.warning("카탈로그에서 같은 제목을 찾지 못했어요.")
                    elif mid in st.session_state["watchlist_ids"]:
                        st.info("이미 보고 싶은 목록에 있어요.")
                    else:
                        st.session_state["watchlist_ids"].append(mid)
                        st.success("사이드바 목록에 넣었어요.")
                        st.rerun()

    with tab_genre:
        mg = load_movie_genres()
        selected = st.multiselect(
            "포함할 장르(하나 이상)",
            GENRE_COLS,
            default=["Drama"],
        )
        q = st.text_input("제목 검색(부분 일치, 비우면 전체)", "")
        limit = st.slider("표시 개수", 10, 200, 40)

        def match(row: pd.Series) -> bool:
            if selected:
                for g in selected:
                    if int(row.get(g, 0)) != 1:
                        return False
            if q.strip():
                return q.strip().lower() in str(row["title"]).lower()
            return True

        mask = mg.apply(match, axis=1)
        filtered = mg.loc[mask].head(limit)
        rows_out = []
        for _, row in filtered.iterrows():
            rows_out.append(
                {
                    "movie_id": int(row["movie_id"]),
                    "title": row["title"],
                    "장르": ", ".join(genres_for_row(row)) or "—",
                }
            )
        st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)
        if rows_out:
            st.divider()
            g_opts = {r["title"]: int(r["movie_id"]) for r in rows_out}
            g_pick = st.selectbox(
                "위 표의 영화 중 보고 싶은 목록에 넣기",
                list(g_opts.keys()),
                key="genre_wl_pick",
            )
            if st.button("선택한 영화를 보고 싶은 목록에 추가", key="genre_wl_btn"):
                gid = g_opts[g_pick]
                if gid not in st.session_state["watchlist_ids"]:
                    st.session_state["watchlist_ids"].append(gid)
                st.success("사이드바에서 확인하세요.")
                st.rerun()

    with tab_user:
        st.markdown(
            "학습된 **예측 행렬**에서 해당 사용자 행을 읽어, 예측 평점이 높은 순으로 영화를 보여줍니다."
        )
        st.caption(
            "파일이 수백 MB이므로, 아래에서 필요할 때만 불러옵니다. 한 번 로드하면 세션 동안 캐시됩니다."
        )
        if st.button("예측 행렬 불러오기", type="primary"):
            st.session_state["matrix_loaded"] = True
        if not st.session_state.get("matrix_loaded"):
            st.info('위 버튼을 누르면 `predicted_matrix.csv`를 메모리에 올립니다.')
        else:
            pm = load_predicted_matrix()
            u_min = int(pm["userId"].min())
            u_max = int(pm["userId"].max())
            uid = st.number_input(
                "userId",
                min_value=u_min,
                max_value=u_max,
                value=u_min,
                step=1,
            )
            top_n = st.slider("추천 개수", 5, 30, 15)
            row_df = pm.loc[pm["userId"] == int(uid)]
            if row_df.empty:
                st.warning("해당 userId가 행렬에 없습니다.")
            else:
                row = row_df.iloc[0]
                movie_cols = [c for c in pm.columns if c != "userId"]
                titles, genres = movie_id_to_meta()
                items = top_movies_for_user_row(
                    row, movie_cols, titles, genres, top_n
                )
                st.subheader(f"사용자 {int(uid)} — 예측 평점 상위")
                table = pd.DataFrame(items)
                table["장르"] = table["genres"].apply(
                    lambda g: ", ".join(g) if g else "—"
                )
                table = table.drop(columns=["genres"])
                table = table.rename(
                    columns={
                        "predicted_rating": "예측 평점",
                        "title": "제목",
                        "movie_id": "영화 ID",
                    }
                )
                st.dataframe(table, use_container_width=True, hide_index=True)
                uid_opts = {
                    f"{it['title']} ({it['movie_id']})": int(it["movie_id"])
                    for it in items
                }
                u_pick = st.selectbox(
                    "추천 중 보고 싶은 목록에 넣기",
                    list(uid_opts.keys()),
                    key=f"pm_wl_{uid}",
                )
                if st.button("선택한 영화를 보고 싶은 목록에 추가", key=f"pm_wl_btn_{uid}"):
                    wid = uid_opts[u_pick]
                    if wid not in st.session_state["watchlist_ids"]:
                        st.session_state["watchlist_ids"].append(wid)
                    st.success("사이드바에서 확인하세요.")
                    st.rerun()


if __name__ == "__main__":
    main()
