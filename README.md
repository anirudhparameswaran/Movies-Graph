# Movies Graph Explorer

A concise Streamlit app that visualizes movie similarities as an interactive graph using Sentence Transformers, PyVis, and optional BERTopic/spaCy preprocessing.

## Quick Start

1. Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2. Install dependencies:

```bash
pip install -r requirements-streamlit.txt
```

3. Ensure the dataset `wiki_movie_plots_deduped.csv` is in the project root (download from Kaggle if needed), then run:

```bash
python -m streamlit run streamlit_app.py
```

## Screenshots

Interactive graph view:

![Graph view](https://github.com/anirudhparameswaran/Movies-Graph/blob/main/images/Graph.png?raw=True)

Filter and build controls:

![Graph view](https://github.com/anirudhparameswaran/Movies-Graph/blob/main/images/Interface.png?raw=True)

## Notes
- The app computes embeddings live; reduce `Sample size` in the sidebar for faster runs.
- If you want topic modeling, run `bertopic_analysis.ipynb` to generate topics and save outputs for the app to consume.

For full setup and troubleshooting, see `requirements-streamlit.txt` and the notebooks.
### 3. Download the Dataset

The project uses `wiki_movie_plots_deduped.csv`. You can either:

**Option A: Use the existing file** (if you already have it in the project directory)

**Option B: Download from Kaggle**
```bash
source .venv-streamlit/bin/activate
python -c "import kagglehub; path = kagglehub.dataset_download('jrobischon/wikipedia-movie-plots'); print('Dataset downloaded to:', path)"
```

Then copy or symlink the CSV to the project root:
```bash
cp /path/to/wiki_movie_plots_deduped.csv ./
```

### 4. Run the Streamlit App

```bash
source .venv-streamlit/bin/activate
python -m streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

### 5. Use the App

**Sidebar Controls:**
- **CSV path**: Path to the data file (default: `wiki_movie_plots_deduped.csv`).
- **Origin/Ethnicity**: Select one or more movie origins (e.g., American, Bollywood, Tamil).
- **Release Year range**: Filter by year (slider).
- **Sample size**: Number of movies to sample (50–5000; smaller = faster).
- **Similarity threshold**: Minimum cosine similarity to include edges (0.0–1.0).
- **Top K neighbors per node**: Maximum edges per movie (1–10).
- **Use spaCy NER masking**: Toggle optional person-name masking (slower, requires spaCy model).
- **Auto-download spaCy small model**: Allow downloading `en_core_web_sm` if missing.
- **Build Graph**: Button to compute embeddings and render the graph.

**Graph Interaction:**
- Click and drag nodes to move them.
- Scroll to zoom in/out.
- Hover over nodes/edges to see metadata.
- Double-click a node to focus on it.

## References

- [Streamlit Docs](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [PyVis Documentation](https://pyvis.readthedocs.io/)
- [NetworkX](https://networkx.org/)
- [BERTopic](https://maartengr.github.io/BERTopic/)
