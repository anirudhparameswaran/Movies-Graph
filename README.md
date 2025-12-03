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

## Project Structure

```
.
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
├── requirements-streamlit.txt        # Python dependencies
├── streamlit_app.py                  # Main Streamlit web app
├── analysis.ipynb                    # Exploratory notebook (legacy)
├── bertopic_analysis.ipynb           # BERTopic topic modeling notebook
├── wiki_movie_plots_deduped.csv      # Movie data (not in repo; download separately)
├── movie_graph.html                  # Output HTML graph (generated)
├── .venv-streamlit/                  # Virtual environment (auto-created)
└── moviesenv/                        # Legacy environment (can be removed if not needed)
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'sentence_transformers'`

**Cause**: You're using the wrong Python interpreter or the venv is not activated.

**Fix**:
```bash
source .venv-streamlit/bin/activate
python --version  # Should show Python 3.11+ (not 3.14)
which python      # Should point to .venv-streamlit/bin/python
pip list | grep sentence-transformers  # Should be installed
```

### `RuntimeError: Cannot install on Python version 3.14.0; only versions >=3.10,<3.14 are supported`

**Cause**: The venv is using Python 3.14, which is not yet supported by some packages (e.g., `numba`).

**Fix**: Create a new venv using Python 3.11 or 3.12:
```bash
python3.11 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
pip install -r requirements-streamlit.txt
```

### `OSError: [E050] Can't find model 'en_core_web_md'. It doesn't exist or wasn't installed for spacy==3.x.x`

**Cause**: spaCy NER model is not installed.

**Fix**:
- In the Streamlit app sidebar, enable `Use spaCy NER masking` and ensure `Auto-download spaCy small model` is checked. The app will attempt to download and install the small model on first use.
- Or manually install ahead of time:
  ```bash
  source .venv-streamlit/bin/activate
  python -m spacy download en_core_web_sm
  ```

### App is slow during graph computation

**Cause**: Computing embeddings for large samples takes time (especially with spaCy NER enabled).

**Fix**:
- Reduce `Sample size` in the sidebar (e.g., 100–200 instead of 500).
- Disable `Use spaCy NER masking` (regex-based lightweight masking is much faster).
- Use a smaller `Similarity threshold` and smaller `Top K` to reduce graph complexity.

## Advanced Usage

### Using the BERTopic Notebook

For topic modeling with `BERTopic`:

```bash
source .venv-streamlit/bin/activate
jupyter notebook bertopic_analysis.ipynb
```

This notebook:
1. Loads and samples the CSV.
2. Masks person names using spaCy NER.
3. Trains a BERTopic model to discover latent topics in plots.
4. Builds a similarity graph using hierarchical scoring (origin, genre, year, topic).
5. Exports the graph as HTML.

### Precomputing Embeddings (Optional)

For faster interactive sessions, you can precompute and cache embeddings:

```python
# Inside .venv-streamlit
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv('wiki_movie_plots_deduped.csv')
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df['Plot'].tolist())
pd.DataFrame(embeddings).to_csv('plot_embeddings.csv', index=False)
```

Then modify `streamlit_app.py` to load precomputed embeddings instead of computing them live.

## Environment Notes

- **Python 3.14**: Not yet supported by packages like `numba`. Use Python 3.10–3.13.
- **GPU acceleration**: If you have CUDA installed, PyTorch will use it automatically for faster embeddings.
- **Virtual environment**: Always activate `.venv-streamlit` before running the app or installing packages.

## Contributing & Development

To modify the app or add features:

1. Activate the venv and install dev tools (if needed):
   ```bash
   source .venv-streamlit/bin/activate
   pip install -r requirements-streamlit.txt
   ```

2. Edit `streamlit_app.py` or create new modules.

3. Test by running:
   ```bash
   python -m streamlit run streamlit_app.py
   ```

4. Streamlit auto-reloads on file changes; refresh your browser to see updates.

## License

This project uses the Kaggle Wikipedia Movie Plots dataset. See the [dataset page](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) for licensing details.

## References

- [Streamlit Docs](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [PyVis Documentation](https://pyvis.readthedocs.io/)
- [NetworkX](https://networkx.org/)
- [BERTopic](https://maartengr.github.io/BERTopic/)
