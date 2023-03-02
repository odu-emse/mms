# get a set of modules the user has reviewed
# get the rating by the user for each module
# multiply the rating by the weight of the module
# divide the sum of the weighted ratings by the sum of the ratings
import asyncio
import random

import pandas as pd
from math import sqrt

from pandas import DataFrame

from prisma import Prisma


async def main(target: str):
    # rec = Recommender(target)
    # await rec.recommend()
    rec = Recs()
    await rec.run()


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


class Recs:
    def __init__(self, target=None):
        self.prisma = Prisma()
        if target is None:
            target = [
                {
                    "id": "63f4ee98ece0495cbb312604",
                    'title': 'orm,',
                    'rating': 5
                },
                {
                    'id': '63f4ee98ece0495cbb312608',
                    'title': 'me',
                    'rating': 3.5
                },
                {
                    'id': '63f4ee98ece0495cbb3125f5',
                    'title': '2017',
                    'rating': 2
                },
                {
                    "id": "63f4ee98ece0495cbb3125f9",
                    'title': 'Souppe',
                    'rating': 5
                },
                {
                    "id": "63f4ee98ece0495cbb3125fe",
                    'title': 'Frams.',
                    'rating': 4.5
                }
            ]
            self.inputMovies = pd.DataFrame(target)

        self.movies_df = pd.read_csv('input/movies.csv')
        self.modules_df = None
        self.ratings_df = pd.read_csv('input/ratings.csv')
        self.feedbacks_df = None

        self.userSubsetGroup = None
        self.pearsonCorrelationDict = dict()
        self.tempTopUsersRating = None

    def sampleModules(self):
        """
            - get a random set of 10 modules
            - create a ratings for each module
            - return as a list of dicts with id, title and rating
        """

    def cleanData(self):
        """
        Removes all the columns that are not needed for the recommendation engine. This is done to reduce the size of
        the dataset and reduce overall complexity in our data.
        """
        modules_df: DataFrame = self.modules_df.drop([
            'description',
            'duration',
            'intro',
            'numSlides',
            'keywords',
            'objectives',
            'createdAt',
            'updatedAt',
            'members',
            'assignments',
            'parentModules',
            'parentModuleIDs',
            'subModules',
            'subModuleIDs',
            'collections',
            'course',
            'courseIDs',
            'feedback',
            'moduleName'
        ], axis=1)

        feedbacks_df: DataFrame = self.feedbacks_df.drop(['student', 'module'], axis=1)

        self.feedbacks_df = feedbacks_df

        self.modules_df = modules_df

        pd.options.display.max_columns = 60

    async def __get_module_data(self):
        await self.prisma.connect()

        modules = await self.prisma.module.find_many()

        modules = list(map(lambda x: x.dict(), modules))

        self.modules_df = pd.DataFrame(modules)

        await self.prisma.disconnect()

    async def __get_feedback_data(self):
        await self.prisma.connect()

        feedbacks = await self.prisma.modulefeedback.find_many()

        feedbacks = list(map(lambda x: x.dict(), feedbacks))

        self.feedbacks_df = pd.DataFrame(feedbacks)

        await self.prisma.disconnect()

    def handleUserInput(self):
        inputID = self.modules_df[self.modules_df['id'].isin(self.inputMovies['id'].tolist())]

        inputMovies = pd.merge(inputID, self.inputMovies)

        self.inputMovies = inputMovies

    def createSubset(self):
        userSubset = self.feedbacks_df[self.feedbacks_df['moduleId'].isin(self.inputMovies['id'].tolist())]

        userSubsetGroup = userSubset.groupby(['studentId'])

        userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

        self.userSubsetGroup = userSubsetGroup

    def createSimilarityMatrix(self):
        pearsonCorrelationDict = {}

        for name, group in self.userSubsetGroup:
            group = group.sort_values(by='id')

            inputMovies = self.inputMovies.sort_values(by='rating')

            nRatings = len(group)

            temp_df = inputMovies[inputMovies['id'].isin(group['moduleId'].tolist())]

            tempRatingList = temp_df['rating'].tolist()

            tempGroupList = group['rating'].tolist()

            Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
            Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
            Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(
                tempGroupList) / float(nRatings)

            if Sxx != 0 and Syy != 0:
                pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
            else:
                pearsonCorrelationDict[name] = 0

        self.pearsonCorrelationDict = pearsonCorrelationDict

    def topUser(self):
        pearsonDF = pd.DataFrame.from_dict(self.pearsonCorrelationDict, orient='index')
        pearsonDF.columns = ['similarityIndex']
        pearsonDF['studentId'] = pearsonDF.index
        pearsonDF.index = range(len(pearsonDF))

        topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

        print(topUsers.head())

        topUsersRating = topUsers.merge(self.feedbacks_df, left_on='studentId', right_on='studentId', how='inner')

        topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']

        tempTopUsersRating = topUsersRating.groupby('moduleId').sum()[['similarityIndex', 'weightedRating']]

        tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']

        self.tempTopUsersRating = tempTopUsersRating

    def recommend(self):
        recommendation_df = pd.DataFrame()

        recommendation_df['w-AVG score'] = self.tempTopUsersRating['sum_weightedRating'] / self.tempTopUsersRating['sum_similarityIndex']
        recommendation_df['moduleId'] = self.tempTopUsersRating.index

        recommendation_df = recommendation_df.sort_values(by='w-AVG score', ascending=False)

        print(recommendation_df.head(10))

        mods_df = self.modules_df.loc[self.modules_df['id'].isin(recommendation_df.head(20)['moduleId'].tolist())]

        self.modules_df = mods_df

    def convertResultToJSON(self):
        modules = self.modules_df

        return modules.to_json(orient="records")

    async def run(self):
        await self.__get_module_data()
        await self.__get_feedback_data()
        self.cleanData()
        self.handleUserInput()
        self.createSubset()
        self.createSimilarityMatrix()
        self.topUser()
        self.recommend()
        return self.convertResultToJSON()


if __name__ == '__main__':
    asyncio.run(main(target="63f7a3068b546b91eadb20a6"))
