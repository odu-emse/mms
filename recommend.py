# get a set of modules the user has reviewed
# get the rating by the user for each module
# multiply the rating by the weight of the module
# divide the sum of the weighted ratings by the sum of the ratings
from prisma import Prisma


async def getRecsForUser(userID: str):
    prism = Prisma()
    await prism.connect()

    # get a set of modules the user has reviewed
    user_reviews = await prism.modulefeedback.find_many(where={'student': {'id': userID}})
