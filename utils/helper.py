import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def cleanMongoID(df):
    df['userID'] = df['userID'].apply(lambda x: x['$oid'])
    df['moduleID'] = df['moduleID'].apply(lambda x: x['$oid'])
    return df


def convertReviewsFromDbToDf(arr, userID):
    mod_df = pd.DataFrame(columns=['userID', 'moduleID', 'rating'])
    mod_df = cleanMongoID(mod_df)
    for mod in arr:
        for feedback in mod['feedback']:
            data = {
                'userID': '%s' % userID,
                'moduleID': mod['id'],
                'rating': feedback['rating'],
            }
            data = pd.DataFrame(data, index=[0])
            mod_df = pd.concat([data, mod_df], ignore_index=True)

    return mod_df


def convertDfToMatrix(df):
    df = df.pivot_table(index='userID', columns='moduleID', values='rating')
    df = df.fillna(0)
    return df


def visualizeMatrix(sim_matrix):
    sns.heatmap(sim_matrix, annot=True)
    plt.show()
