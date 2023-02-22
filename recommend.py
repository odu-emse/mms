# get a set of modules the user has reviewed
# get the rating by the user for each module
# multiply the rating by the weight of the module
# divide the sum of the weighted ratings by the sum of the ratings
import asyncio

import pandas as pd

from prisma import Prisma


async def getRecsForUser(userID: str):
    prism = Prisma()
    await prism.connect()

    # get a set of modules the user has reviewed
    user_reviews = await prism.modulefeedback.find_many(where={'student': {'id': userID}})

    reviews = list(map(lambda x: x.dict(), user_reviews))

    # all the reviews by the user
    df = pd.DataFrame(reviews)

    all_reviews = await prism.modulefeedback.find_many()

    all_reviews = list(map(lambda x: x.dict(), all_reviews))

    df_all_reviews = pd.DataFrame(all_reviews)

    # get the weighted average of the reviews per module
    df['weighted_rating'] = df['rating'] * df_all_reviews['rating'].mean()

    df.drop(columns=['module', 'student'], inplace=True)
    print(df.info())
    print(df.head())



async def main():
    await getRecsForUser(userID='63f3b1cb9422322eb675292f')
    pass


if __name__ == '__main__':
    asyncio.run(main())
