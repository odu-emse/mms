def getRecommendations(sim_matrix, df, user_id):
    # get user index
    user_index = df.index.get_loc(user_id)

    # get similarity scores
    sim_scores = sim_matrix[user_index]

    # sort the similarity scores
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get top 5 scores
    sim_scores = sim_scores[1:6]

    # get the module indices
    module_indices = [i[0] for i in sim_scores]

    # get the module ids
    module_ids = df.index[module_indices]

    return module_ids
