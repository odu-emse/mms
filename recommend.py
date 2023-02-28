# get a set of modules the user has reviewed
# get the rating by the user for each module
# multiply the rating by the weight of the module
# divide the sum of the weighted ratings by the sum of the ratings
import asyncio
import random

import pandas as pd
from math import sqrt

from prisma import Prisma


async def main(target: str):
    # rec = Recommender(target)
    # await rec.recommend()
    rec = Recs()
    rec.run()


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
        if target is None:
            target = [
                {
                    'title': 'Breakfast Club, The',
                    'rating': 5
                },
                {
                    'title': 'Toy Story',
                    'rating': 3.5
                },
                {
                    'title': 'Jumanji',
                    'rating': 2
                },
                {
                    'title': 'Pulp Fiction',
                    'rating': 5
                },
                {
                    'title': 'Akira',
                    'rating': 4.5
                }
            ]
            self.inputMovies = pd.DataFrame(target)
        self.movies_df = pd.read_csv('input/movies.csv')
        self.ratings_df = pd.read_csv('input/ratings.csv')
        self.userSubsetGroup = None
        self.pearsonCorrelationDict = dict()
        self.tempTopUsersRating = None

    def cleanData(self):
        # Using regular expressions to find a year stored between parentheses
        # We specify the parentheses, so we don’t conflict with movies that have years in their titles

        print(self.movies_df.head())

        self.movies_df['year'] = self.movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)

        # Removing the parentheses

        self.movies_df['year'] = self.movies_df.year.str.extract('(\d\d\d\d)', expand=False)

        # Removing the years from the ‘title’ column

        self.movies_df['title'] = self.movies_df.title.str.replace('(\(\d\d\d\d\))', '')
        # Applying the strip function to get rid of any ending whitespace characters that may have appeared

        self.movies_df['title'] = self.movies_df['title'].apply(lambda x: x.strip())

        print(self.movies_df.head())

        self.movies_df = self.movies_df.drop('genres', 1)

    def handleUserInput(self):
        inputID = self.movies_df[self.movies_df['title'].isin(self.inputMovies['title'].tolist())]

        inputMovies = pd.merge(inputID, self.inputMovies)

        inputMovies = inputMovies.drop('year', 1)

        print(inputMovies)
        self.inputMovies = inputMovies

    def createSubset(self):
        userSubset = self.ratings_df[self.ratings_df['movieId'].isin(self.inputMovies['movieId'].tolist())]

        userSubset.head()

        userSubsetGroup = userSubset.groupby(['userId'])

        userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

        print(userSubsetGroup[0:3])

        self.userSubsetGroup = userSubsetGroup

    def createSimilarityMatrix(self):
        pearsonCorrelationDict = {}

        for name, group in self.userSubsetGroup:
            group = group.sort_values(by='movieId')

            inputMovies = self.inputMovies.sort_values(by='movieId')

            nRatings = len(group)

            temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
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

        print(pearsonCorrelationDict.items())
        self.pearsonCorrelationDict = pearsonCorrelationDict

    def topUser(self):
        pearsonDF = pd.DataFrame.from_dict(self.pearsonCorrelationDict, orient='index')
        pearsonDF.columns = ['similarityIndex']
        pearsonDF['userId'] = pearsonDF.index
        pearsonDF.index = range(len(pearsonDF))

        print(pearsonDF.head())

        topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

        print(topUsers.head())

        topUsersRating = topUsers.merge(self.ratings_df, left_on='userId', right_on='userId', how='inner')

        print(topUsersRating.head())

        topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']

        print(topUsersRating.head())

        tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]

        tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']

        print(tempTopUsersRating.head())

        self.tempTopUsersRating = tempTopUsersRating

    def recommend(self):
        recommendation_df = pd.DataFrame()

        recommendation_df['weighted average recommendation score'] = self.tempTopUsersRating['sum_weightedRating'] / \
                                                                     self.tempTopUsersRating['sum_similarityIndex']
        recommendation_df['movieId'] = self.tempTopUsersRating.index

        print(recommendation_df.head())

        recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

        print(recommendation_df.head(10))

        movies_df = self.movies_df.loc[self.movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]

        print(movies_df.head(10))

        self.movies_df = movies_df

    def convertResultToJSON(self):
        movies = self.movies_df

        return movies.to_json(orient="records")

    def run(self):
        self.cleanData()
        self.handleUserInput()
        self.createSubset()
        self.createSimilarityMatrix()
        self.topUser()
        self.recommend()
        return self.convertResultToJSON()


if __name__ == '__main__':
    asyncio.run(main(target="63f7a3068b546b91eadb20a6"))
