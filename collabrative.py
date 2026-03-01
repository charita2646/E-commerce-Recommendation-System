# collaborative.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# load dataset directly
data = pd.read_csv("clean_data.csv")   # change filename if needed

# create user-item matrix
user_item_matrix = data.pivot_table(
    index="User's ID",
    columns="Name",
    values="Rating",
    fill_value=0
)

# compute similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# function for recommendation
def collaborative_filtering_recommendations(user_id, top_n=5):

    if user_id not in user_item_matrix.index:
        return "User not found in dataset"

    user_index = list(user_item_matrix.index).index(user_id)

    similarity_scores = list(enumerate(user_similarity[user_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similar_users = [user_item_matrix.index[i[0]]
                     for i in similarity_scores[1:top_n+1]]

    return similar_users


# main execution
if __name__ == "__main__":

    print("=== COLLABORATIVE RECOMMENDATION SYSTEM ===")

    target_user_id = 10

    recommendations = collaborative_filtering_recommendations(
        target_user_id,
        top_n=5
    )

    print("\nRecommended similar users:")
    print(recommendations)
    print("COLLABRATIVE FILTERING IS WORKING")