import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import networkx as nx
import tempfile
import streamlit.components.v1 as components
import re
import spacy
from spacy.cli import download as spacy_download


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def mask_named_entities_simple(text: str) -> str:
    if not isinstance(text, str):
        return text  # Handles NaN or non-string inputs

    # Remove parenthetical content like (Actor names)
    text = re.sub(r'\([a-zA-Z\s]*\)', '', text)
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Lightweight placeholder masking (no heavy NER)
    # Replace sequences of capitalized words that look like names with a simple placeholder
    # This is conservative and intentionally lightweight; for better masking enable spaCy below.
    text = re.sub(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", "[PERSON_NAME]", text)
    return text


@st.cache_resource
def get_spacy_model(prefer: str = "en_core_web_md"):
    """Try to load a spaCy model. Falls back to `en_core_web_sm` and will download it
    if allowed and missing.
    """
    try:
        return spacy.load(prefer)
    except Exception:
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            # Attempt to download the small model and load it
            try:
                spacy_download("en_core_web_sm")
                return spacy.load("en_core_web_sm")
            except Exception as e:
                raise RuntimeError(
                    "spaCy language model not available and could not be downloaded: " + str(e)
                )


def mask_with_spacy(texts, nlp):
    """Mask PERSON entities in a list of texts using spaCy and return the masked texts list."""
    out = []
    for doc in nlp.pipe(texts, batch_size=32):
        t = doc.text
        replacements = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                replacements.append((ent.start_char, ent.end_char, "[PERSON_NAME]"))
        replacements.sort(key=lambda x: x[0], reverse=True)
        masked_text = list(t)
        for start, end, repl in replacements:
            masked_text[start:end] = repl
        out.append("".join(masked_text))
    return out


@st.cache_resource
def get_embedding_model(name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(name)


def compute_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)


def build_similarity_dataframe(df: pd.DataFrame, embeddings: np.ndarray):
    sims = cosine_similarity(embeddings)
    rows = []
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sims[i, j])
            rows.append({
                "Source": df.iloc[i]["Title"],
                "Target": df.iloc[j]["Title"],
                "Weight": score,
            })
            rows.append({
                "Source": df.iloc[j]["Title"],
                "Target": df.iloc[i]["Title"],
                "Weight": score,
            })
    return pd.DataFrame(rows)


def build_graph(similarity_df: pd.DataFrame, all_titles, threshold: float = 0.3, top_k: int = 3):
    filtered_df = similarity_df[similarity_df["Weight"] >= threshold].copy()
    filtered_df = filtered_df.sort_values(by=["Source", "Weight"], ascending=[True, False]).groupby("Source").head(top_k)

    G = nx.Graph()
    G.add_nodes_from(all_titles)
    for _, row in filtered_df.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=row["Weight"])
    return G


def pyvis_from_nx(G: nx.Graph, height: str = "700px") -> str:
    net = Network(height=height, width="100%", bgcolor="#111111", font_color="white")
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.1)
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    net.save_graph(tmp.name)
    return tmp.name


def main():
    st.set_page_config(layout="wide", page_title="Movies Graph Explorer")
    st.title("Movies Graph Explorer")

    st.sidebar.header("Data / Filters")
    data_path = st.sidebar.text_input("CSV path", value="wiki_movie_plots_deduped.csv")
    df = load_data(data_path)

    origins = sorted(df["Origin/Ethnicity"].dropna().unique().tolist())
    selected_origins = st.sidebar.multiselect("Origin/Ethnicity", options=origins, default=origins[:4])

    min_year = int(df["Release Year"].min())
    max_year = int(df["Release Year"].max())
    year_range = st.sidebar.slider("Release Year range", min_year, max_year, (2010, max_year))

    sample_size = st.sidebar.number_input("Sample size (for speed)", min_value=50, max_value=5000, value=500, step=50)

    st.sidebar.header("Graph Options")
    threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.35)
    top_k = st.sidebar.slider("Top K neighbors per node", 1, 10, 3)

    st.sidebar.header("Processing")
    use_spacy = st.sidebar.checkbox("Use spaCy NER masking (slower, optional)", value=False)
    auto_download_spacy = st.sidebar.checkbox("Auto-download spaCy small model if missing", value=True)
    run_btn = st.sidebar.button("Build Graph")

    st.markdown("Select a movie to highlight neighbors, or build the full graph.")

    if run_btn:
        with st.spinner("Filtering data and computing embeddings..."):
            sub = df[df["Origin/Ethnicity"].isin(selected_origins)]
            sub = sub[(sub["Release Year"] >= year_range[0]) & (sub["Release Year"] <= year_range[1])]
            sub = sub.sample(min(sample_size, len(sub)), random_state=0).reset_index(drop=True)

            # Mask plots: either lightweight regex masking or spaCy NER masking
            if use_spacy:
                try:
                    if auto_download_spacy:
                        nlp = get_spacy_model()
                    else:
                        # Try only local models when auto-download disabled
                        try:
                            nlp = spacy.load("en_core_web_md")
                        except Exception:
                            nlp = spacy.load("en_core_web_sm")

                    plots = sub["Plot"].astype(str).tolist()
                    sub["Plot_Masked"] = mask_with_spacy(plots, nlp)
                except Exception as e:
                    st.warning(f"spaCy model unavailable: {e}. Falling back to lightweight masking.")
                    sub["Plot_Masked"] = sub["Plot"].apply(mask_named_entities_simple)
            else:
                sub["Plot_Masked"] = sub["Plot"].apply(mask_named_entities_simple)

            model = get_embedding_model()
            embeddings = compute_embeddings(model, sub["Plot_Masked"].tolist())

            similarity_df = build_similarity_dataframe(sub, embeddings)
            G = build_graph(similarity_df, sub["Title"].tolist(), threshold=threshold, top_k=top_k)

        st.success("Graph built — rendering...")

        html_path = pyvis_from_nx(G)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=800)

    else:
        st.info("Adjust filters and click 'Build Graph' in the sidebar.")


if __name__ == "__main__":
    main()
