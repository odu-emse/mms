import asyncio
import logging
from typing import Union
import pandas as pd
from prisma import Prisma


class Fetcher:
    def __init__(self, url: str):
        self.url = url
        self.response = None
        self.prisma = Prisma()

        self.logger = logging.getLogger('__fetch__')
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    def _createModuleMutation(self):
        self.response = """query{
                          module(input:{}){
                            id
                            moduleName
                            moduleNumber
                          }
                        }"""
        return self.response

    def _createModuleFeedbackMutation(self):
        self.response = """query{
                          moduleFeedback(input:{}){
                            id
                            rating
                            feedback
                            student{
                              id
                            }
                            module{
                              id
                            }
                          }
                        }"""
        return self.response

    async def _getModules(self):
        """
        Gets all modules from the database and returns them as a list of dictionaries.
        :return: class<list> of class<dict> of type<Module>
        """
        await self.prisma.connect()
        self.modules = await self.prisma.module.find_many()

        mods = []

        for i in range(len(self.modules)):
            mods.append(self.modules[i].dict())

        await self.prisma.disconnect()

        return mods

    async def getModules(self):
        self.logger.info('Fetching modules...')

        res = await self._getModules()

        if res:
            self.logger.info('Modules successfully fetched')
            df = pd.DataFrame(res)
            print(df.info())
        else:
            self.logger.error('Failed to fetch modules')
            return None

        return self.response

    async def _getModuleFeedback(self):
        """
        Gets all feedbacks from the database and returns them as a list of dictionaries.
        :return: class<list> of class<dict> of type<ModuleFeedback>
        """
        await self.prisma.connect()
        self.feedbacks = await self.prisma.modulefeedback.find_many()

        feedbacks = []

        for i in range(len(self.feedbacks)):
            feedbacks.append(self.feedbacks[i].dict())
        await self.prisma.disconnect()
        return feedbacks

    async def getModuleFeedback(self):
        self.logger.info('Fetching module feedback...')

        res = await self._getModuleFeedback()

        if res:
            self.logger.info('Module feedback successfully fetched')
            df = pd.DataFrame(res)
            print(df.info())
        else:
            self.logger.error('Failed to fetch module feedback')
            return None

        return self.response

    def convertJSONasDataFrame(self, model: str):
        df = pd.DataFrame(self.response['data'][model])
        return df

    def convertObjectTOColumn(self, model: str, column: Union[str, list, None]):
        df = self.convertJSONasDataFrame(model)
        if isinstance(column, list):
            for col in column:
                conv = df[col].apply(lambda x: x['id'] if x else None)
                df[col] = conv
            print(df.info())
        elif isinstance(column, str):
            conv = df[column].apply(lambda x: x['id'] if x else None)
            df[column] = conv
            print(df.head())
        else:
            print(df.head())


async def main():
    fetcher = Fetcher('http://client:4000/graphql')
    await fetcher.getModules()
    # fetcher.convertObjectTOColumn('module', None)
    await fetcher.getModuleFeedback()
    # fetcher.convertObjectTOColumn('moduleFeedback', ['module', 'student'])


if __name__ == '__main__':
    asyncio.run(main())
