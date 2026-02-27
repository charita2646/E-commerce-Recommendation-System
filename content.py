# content.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# LOAD DATA
# -----------------------
data = pd.read_csv("clean_data.csv")

# clean columns
data['Name'] = data['Name'].astype(str).str.lower().str.strip()
data['Tags'] = data['Tags'].astype(str).str.lower().str.strip()
data['Tags'] = data['Tags'].fillna('')


# -----------------------
# BUILD TF-IDF MODEL
# -----------------------
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(data['Tags'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# -----------------------
# WRITE ITEM NAME HERE
# -----------------------
item_name = "Nail polish"


# convert to lowercase
item_name = item_name.lower().strip()


# find partial match
matches = data[data['Name'].str.contains(item_name, case=False, na=False)]


if matches.empty:

    print("Item NOT found in dataset")

else:

    # take first matched item
    idx = matches.index[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:6]

    indices = [i[0] for i in similarity_scores]

    print("\nRecommended Items:\n")

    print(data.iloc[indices]['Name'])