from utils.seed import seedDbFeedback, getModuleFeedback, seedModuleModel
from utils.helper import convertReviewsFromDbToDf


def main():
    print('Starting application...')
    # seedModuleModel()
    # seedDbFeedback()
    getReviews(userID="63da9e40020a625cc55f64c5")


def getReviews(userID):
    # read data
    mod_data = getModuleFeedback()
    df = convertReviewsFromDbToDf(mod_data['module'], userID)

    # print(df)
    # print(df.groupby(['userID', 'moduleID']).sum().sort_values('rating', ascending=False).head())
    # print(df.groupby('moduleID')['rating'].sum().sort_values(ascending=False).head())

    # get highest rated modules
    print(df.groupby('moduleID')['rating'].sum().sort_values(ascending=False).head())


def getUserProfile():
    print('Fetching user data...')
    # get user from ID
    #  find enrolled modules
    #  remove modules that already enrolled in
    #  get feedback for each module
    #  get similarity matrix
    #  get recommendations
    #  return recommendations


if __name__ == '__main__':
    main()
