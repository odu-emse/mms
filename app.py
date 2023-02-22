from pandas import DataFrame
import asyncio
from prisma import Prisma

from utils.db import testConnection
from utils.response import getModuleFromID
from utils.seed import getModuleFeedback, Seeder, Skipper
from utils.helper import convertReviewsFromDbToDf


async def main() -> None:
    print('Starting application...')
    testConnection()
    # await seedUserDB()
    # seedModuleModel()
    # await seedPlanOfStudyDB()
    # await seedEnrollmentDB()

    db = Seeder(
        skip=[Skipper.all],
        cleanup=[Skipper.all]
    )

    await db.seedAll()
    await db.cleanupAll()

    exit(0)
    # seedModuleModel()
    # seedDbFeedback()
    # getReviews(userID="63da9e40020a625cc55f64c5")


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


async def getUserProfile(ID):
    print('Fetching user data...')
    # get user from ID
    #  find enrolled modules
    #  remove modules that already enrolled in
    #  get feedback for each module
    #  get similarity matrix
    #  get recommendations
    #  return recommendations
    prisma = Prisma()
    await prisma.connect()
    account = await prisma.user.find_unique(
        where={
            'id': ID
        }
    )
    return account.dict()


if __name__ == '__main__':
    asyncio.run(main())
