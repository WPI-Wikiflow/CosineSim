import numpy as np
import pandas as pd

#Define cosine similarity operation
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":
    # Read in the data
    data = pd.read_csv("data.csv")

    #Request input article
    article = input("Enter article name: ")

    #Find article vector by index
    article_vector = data.loc[data['article'] == article].iloc[0, 1:].values

    #Find cosine similarity between article and all other articles
    data['similarity'] = data.iloc[:, 1:].apply(lambda x: cosine_similarity(x, article_vector), axis=1)

    # Collect the results
    results = data[['article', 'similarity']].sort_values(by='similarity', ascending=False)

    # Parse out the largest 5 results
    recs = results.head(6)
    #take out the first result because it is the same article
    recs = recs.iloc[1:, :]

    # return index of those results with titles of recommended articles
    recommendations = results.index.values
    print("Recommended articles: ")
    for i in recommendations:
        print(data.iloc[i, 0])
