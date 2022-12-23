
# How to run

1. Run `pip install -r requirements.txt` to install the dependencies.
2. Run 'streamlit run webapp.py' to run the webapp.

# About the project

- When search a author name in the webapp, will map to the author id.
- Use GraphSage model to get link prediction of the author with other authors, which is not in previous dataset.
- Use the link prediction result to get the author's co-author score. Get 10 co-authors with the highest score.
- Show the co-authors network graph.
