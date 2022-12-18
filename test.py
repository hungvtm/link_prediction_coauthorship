import pandas as pd

# Authors df
authors_df = pd.read_csv('data/final_3_pca.csv', index_col=0)

Topic_dict = authors_df['topic'].to_dict()

print(Topic_dict) 