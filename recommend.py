# get a set of modules the user has reviewed
# get the rating by the user for each module
# multiply the rating by the weight of the module
# divide the sum of the weighted ratings by the sum of the ratings
import asyncio
import random

import pandas as pd

from prisma import Prisma


async def main(target: str):
    rec = Recommender(target)
    await rec.recommend()


class Recommender:
    def __init__(self, target: str):
        self.target = target
        self.sample = []
        self.modules = []
        self.recs = []

    async def _sampleModules(self, skip: bool = True):
        prisma = Prisma()
        await prisma.connect()

        modules = await prisma.module.find_many(include={'feedback': True})

        modules = list(map(lambda x: x.dict(), modules))

        rand = random.sample(modules, len(modules) // 2)

        # create module feedback for sample modules for target user
        if not skip:
            await self._seedTargetFeedback(rand)

        await prisma.disconnect()

        self.sample = rand

    async def _seedTargetFeedback(self, sample: list):
        prisma = Prisma()
        await prisma.connect()
        for module in sample:
            await prisma.modulefeedback.create(
                data={
                    'feedback': 'This is a sample review',
                    'rating': random.randint(1, 5),
                    'module': {
                        'connect': {
                            'id': module
                        }
                    },
                    'student': {
                        'connect': {
                            'id': self.target
                        }
                    }
                }
            )
            print(f'Created feedback for module {module}')

        await prisma.disconnect()

    async def recommend(self):
        """
        At this point our target user has reviewed half of the modules in the database.
        We have access to the modules that the user has reviewed through the modules parameter, and now
        we can now get recommendations for the user
        """
        prism = Prisma()
        await prism.connect()
        await self._sampleModules(skip=True)

        df = pd.DataFrame(self.sample)

        # convert our review data into a user x module matrix
        # find cosine similarity between the target user and all other users
        # get the top 5 users with the highest similarity
        # get the modules that the top 5 users have reviewed
        # get the modules that the target user has not reviewed
        # get the modules that the top 5 users have reviewed that the target user has not reviewed
        # get the average rating for each module
        # get the top 5 modules with the highest average rating
        # return the top 5 modules with the highest average rating as recommendations

        await prism.disconnect()


if __name__ == '__main__':
    asyncio.run(main(target="63f7a3068b546b91eadb20a6"))
