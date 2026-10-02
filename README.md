# EuroMatrix

EuroMatrix is an interactive Streamlit application designed to analyze, visualize, and explore voting behavior and political alignments in the European Parliament. Using open data from [HowTheyVote.eu](https://howtheyvote.eu), the app applies dimensionality reduction (PCA) and clustering algorithms (KMeans) to map how MEPs actually vote, independent of official party lines.

---

## Features

* **🤝 Comparator:** Compare any MEP with others to find their political allies (highest agreement) and opponents (highest divergence).
* **🧭 Map of Political Ideas:** An ideological proximity map based on Principal Component Analysis (PCA). MEPs who vote similarly are positioned closer together, color-coded by political group.
* **🤖 AI Clusters:** Unsupervised machine learning (KMeans clustering) that groups MEPs based purely on their voting patterns rather than formal party labels. Includes a breakdown of cluster compositions.
* **🔥 Topics:** Search and filter European Parliament votes by specific keywords (e.g., "Ukraine", "Green Deal") to generate dynamic topic-specific voting maps.
* **🌐 Multilingual Interface:** Fully supported in both Polish and English.

---

## Tech Stack & Libraries

* **Python**
* **Streamlit** (Interactive web app framework)
* **Pandas & NumPy** (Data manipulation and processing)
* **Scikit-Learn** (PCA & KMeans machine learning models)
* **Plotly** (Interactive scatter plots and visualizations)

---

## Data Source

The application pulls live open data releases from [HowTheyVote/data](https://github.com/HowTheyVote/data):
* Member votes (`member_votes.csv.gz`)
* MEP profiles (`members.csv.gz`)
* Roll-call votes / Metadata (`votes.csv.gz`)

---

## License

This project is open-source under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
