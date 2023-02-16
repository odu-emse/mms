from sklearn.metrics import pairwise

from utils.helper import convertDfToMatrix


def getSimilarityMatrix(df):
    # convert to matrix
    conv_df = convertDfToMatrix(df)

    print(conv_df.info())

    # calculate similarity
    sim_matrix = pairwise.cosine_similarity(conv_df)

    return sim_matrix
