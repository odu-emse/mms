from requests import post, get
import pandas as pd
from sklearn.metrics import pairwise
import seaborn as sns
import matplotlib.pyplot as plt


# custom modules
from utils.seed import seedDbFeedback, getModuleFeedback
from utils.sdl import createMutationString


def main():
    print('Starting application...')
    # seedDbFeedback()
    getReviews()

def convertDfToMatrix(df):
    df = df.pivot_table(index='userID', columns='moduleID', values='rating')
    df = df.fillna(0)
    return df


def getSimilarityMatrix(df):
    similarity_matrix = pairwise.cosine_similarity(df)
    return similarity_matrix


def visualizeMatrix(sim_matrix):
    sns.heatmap(sim_matrix, annot=True)
    plt.show()


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

def convertReviewsFromDbToDf(module):
    df = pd.DataFrame(columns=['userID', 'moduleID', 'rating'])
    for feedback in module['feedback']:
        df = df.append({
            'userID': module['members'][0]['id'],
            'moduleID': module['id'],
            'rating': feedback['rating'],
        }, ignore_index=True)
    return df

def getReviews(userID="63ee40b8fc13ae480100008c"):
    # read data
    mod_data = getModuleFeedback()
    df = convertReviewsFromDbToDf(mod_data['module'][0])

    # df = pd.read_json('./input/ratings.json', orient='records')

    # take $oid property from userID and moduleID objects to string
    df['userID'] = df['userID'].apply(lambda x: x['$oid'])
    df['moduleID'] = df['moduleID'].apply(lambda x: x['$oid'])

    print(df.head())

    # convert to matrix
    conv_df = convertDfToMatrix(df)

    print(conv_df.head())

    # get similarity matrix
    # sim_matrix = getSimilarityMatrix(conv_df)

    # visualize matrix
    # visualizeMatrix(sim_matrix)

    # get the highest similarity
    # recommendations = getRecommendations(sim_matrix, conv_df, userID)
    # print(recommendations)


def getMouleReviews():
    print('Fetching module data...')
    mod_df = pd.DataFrame(columns=['userID', 'moduleID', 'rating'])
    mod_data = post('http://localhost:4000/graphql', {}, {
        'query': """query{
          module(input:{}){
            id
            moduleName
            moduleNumber
            feedback{
              rating
              feedback
            }
            members{
              id
            }
          }
        }"""
    })
    mods = mod_data.json()
    for mod in mods['data']['module']:
        for feedback in mod['feedback']:
            print(feedback['rating'])
            print(feedback['feedback'])

            mod_df = mod_df.append({
                'userID': mod['members'][0]['id'],
                'moduleID': mod['id'],
                'rating': feedback['rating'],
            }, ignore_index=True)

    print(mod_df.head())
    print(mod_df.groupby(['userID', 'moduleID']).sum().sort_values('rating', ascending=False).head())
    return mod_df


def getUserProfile():
    print('Fetching user data...')


if __name__ == '__main__':
    main()
