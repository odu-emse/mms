import enum

from pandas import DataFrame
from requests import post
import asyncio
from prisma import Prisma

from utils.response import getModuleFromID
from utils.seed import seedDbFeedback, getModuleFeedback, seedModuleModel, seedUserDB, \
    seedPlanOfStudyDB, Seeder, Skipper
from utils.helper import convertReviewsFromDbToDf


async def main() -> None:
    print('Starting application...')

    # await seedUserDB()
    # seedModuleModel()
    # await seedPlanOfStudyDB()
    # await seedEnrollmentDB()

    await Seeder(
        skip=Skipper.user
    ).seedAll()


    # seedModuleModel()
    # seedDbFeedback()
    testConnection()
    # getReviews(userID="63da9e40020a625cc55f64c5")


def testConnection():
    print('Testing connection...')
    # test connection to API
    post('http://localhost:4000/graphql', {}, {
        'query': """query{
              module(input:{
                id: "%s"
              }){
                id
                moduleName
                moduleNumber
              }
        }""" % '63da9e40020a625cc55f64c5'
    })
    # test connection to DB
    # test connection to Redis
    # test connection to ElasticSearch


def getReviews(userID):
    # read data
    mod_data = getModuleFeedback()
    df = convertReviewsFromDbToDf(mod_data['module'], userID)

    # print(df)
    # print(df.groupby(['userID', 'moduleID']).sum().sort_values('rating', ascending=False).head())
    # print(df.groupby('moduleID')['rating'].sum().sort_values(ascending=False).head())

    # get highest rated modules
    top_mods: DataFrame = df.groupby('moduleID')['rating'].sum().sort_values(ascending=False)

    # run response for each row of the highest rated modules
    print(top_mods)
    res_top_mods = top_mods.reset_index()
    res_top_mods.apply(lambda row: getModuleFromID(row), axis=1)


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
    asyncio.run(main())
