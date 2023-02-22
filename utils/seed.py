from enum import Enum
import os
from typing import Union

from essential_generators import DocumentGenerator
from numpy import random
from prisma import Prisma
import logging
from requests import post
from utils.sdl import createMutationString, createModuleMutationString


async def seedUserDB():
    print('Seeding user model...')
    prisma = Prisma()
    await prisma.connect()
    gen = DocumentGenerator()
    iterations = 25
    accounts = []

    for i in range(iterations):
        accounts.append({
            'firstName': gen.name(),
            'lastName': gen.name(),
            'email': gen.email(),
            'openID': str(gen.integer()),
        })

    users = await prisma.user.create_many(data=accounts)

    print('User model seeded successfully with %d documents!' % users)
    await prisma.disconnect()


async def seedPlanOfStudyDB():
    print('Seeding plan of study model...')
    gen = DocumentGenerator()
    prisma = Prisma()
    await prisma.connect()

    modules = await prisma.module.find_many()
    accounts = await prisma.user.find_many()

    for account in accounts:
        for module in modules:
            # create plan of study for each module 25 times
            await prisma.planofstudy.create(
                data={
                    "modules": {
                        "connect": {
                            "id": module.get('id')
                        }
                    },
                    "student": {
                        "connect": {
                            "id": account['id']
                        }
                    }
                }
            )

    print('Plan of study model seeded successfully!')


def seedDbFeedback():
    print('Seeding feedback model...')
    gen = DocumentGenerator()
    iterations = 25
    modules = getModules()['module']
    for module in modules:
        # create feedback for each module 25 times
        for i in range(iterations):
            res = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"),
                       {},
                       {'query': createMutationString(
                           comment=gen.sentence(),
                           rating=random.randint(1, 6),
                           moduleID=module['id'],
                       )
                       })
            print(res.json())

    print('Feedback model seeded successfully!')


def seedModuleModel():
    print('Seeding module model...')
    gen = DocumentGenerator()
    iterations = 25
    for i in range(iterations):
        key_length = random.randint(1, 11)
        res = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"),
                   {},
                   {'query': createModuleMutationString(
                       moduleName=gen.word(),
                       moduleNumber=random.randint(1, 1000),
                       description=gen.sentence(),
                       duration=random.randint(1, 100),
                       intro=gen.sentence(),
                       numSlides=random.randint(1, 100),
                       keywords=[gen.word() for i in range(key_length)]
                   )
                   })
        print(res.json())

    print('Module model seeded successfully!')


def getModules():
    mods = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"), {}, {
        'query': """query{
          module(input:{}){
            id
            moduleName
            moduleNumber
          }
        }"""
    })
    return mods.json()['data']


def getModuleFeedback():
    mods = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"), {}, {
        'query': """query{
          module(input:{}){
            id
            moduleName
            moduleNumber
            feedback{
              rating
              feedback
            }
            members{
              id
            }
          }
        }"""
    })
    return mods.json()['data']


class Skipper(Enum):
    user = 1
    module = 2
    feedback = 3
    plan = 4
    enrollment = 5
    all = 6


class Seeder:
    """
    Seeder class for seeding the database with dummy data. Makes use of the prisma python client to connection to the
    database and the essential_generators library to create document templates.
    """

    def __init__(self, skip: [Skipper] = None, cleanup: [Skipper] = None, iterations: int = 25):
        self.gen = DocumentGenerator()
        self.gen.init_word_cache(5000)
        self.gen.init_sentence_cache(5000)
        self.prisma = Prisma()
        self.iterations = iterations
        self.accounts = []
        self.modules = []
        self.enrollments = []
        self.feedbacks = []
        self.plans = []
        self.skip = skip
        self.cleanup = cleanup
        self.logger = logging.getLogger('__seed__')
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    async def connect(self):
        """
        Connects to the database using prisma client.
        """
        await self.prisma.connect()

    async def disconnect(self):
        """
        Disconnects from the database using prisma client.
        """
        await self.prisma.disconnect()

    async def _getUserAccounts(self):
        """
        Gets all user accounts from the database and returns them as a list of dictionaries.
        :return: class<list> of class<dict> of type<User>
        """
        accounts = await self.prisma.user.find_many()

        acc = []

        for i in range(len(accounts)):
            acc.append(accounts[i].dict())

        return acc

    async def _getModules(self):
        """
        Gets all modules from the database and returns them as a list of dictionaries.
        :return: class<list> of class<dict> of type<Module>
        """
        self.modules = await self.prisma.module.find_many()

        mods = []

        for i in range(len(self.modules)):
            mods.append(self.modules[i].dict())

        return mods

    async def _getEnrollments(self):
        """
        Gets all enrollments from the database and returns them as a list of dictionaries.
        :return: class<list> of class<dict> of type<ModuleEnrollment>
        """
        self.enrollments = await self.prisma.moduleenrollment.find_many()

        enrollments = []

        for i in range(len(self.enrollments)):
            enrollments.append(self.enrollments[i].dict())

        return enrollments

    async def _getFeedbacks(self):
        """
        Gets all feedbacks from the database and returns them as a list of dictionaries.
        :return: class<list> of class<dict> of type<ModuleFeedback>
        """
        self.feedbacks = await self.prisma.modulefeedback.find_many()

        feedbacks = []

        for i in range(len(self.feedbacks)):
            feedbacks.append(self.feedbacks[i].dict())

        return feedbacks

    async def _getPlans(self) -> list[dict]:
        self.plans = await self.prisma.planofstudy.find_many()

        plans = []

        for i in range(len(self.plans)):
            plans.append(self.plans[i].dict())

        return plans

    async def _seedUserDB(self):
        """
        Seeds the user model with dummy data as many times as described by the iterations' member.
        :return: None
        """
        self.logger.info('Seeding user model...')
        accounts = []

        for i in range(self.iterations):
            accounts.append({
                'firstName': self.gen.name(),
                'lastName': self.gen.name(),
                'email': self.gen.email(),
                'openID': str(self.gen.integer()),
            })

        users = await self.prisma.user.create_many(data=accounts)

        self.accounts = accounts
        self.logger.info('User model seeded successfully with %d documents!' % users)

    async def _seedModuleDB(self):
        """
        Seeds the module model with dummy data as many times as described by the iterations' member.
        :return: None
        """
        self.logger.info('Seeding module model...')
        modules = []

        template = {
            'moduleName': {
                'typemap': 'sentence',
                'unique': True,
                'tries': 100
            },
            'moduleNumber': {
                'typemap': 'integer',
                'unique': True,
                'tries': 100
            },
            'description': 'paragraph',
            'duration': 'small_int',
            'intro': 'sentence',
            'numSlides': 'small_int',
            'keywords': {
                'set': ['engineering', 'mathematics', 'physics', 'chemistry', 'biology', 'computer science',
                        'economics']
            }
        }

        self.gen.set_template(template)
        docs = self.gen.documents(self.iterations)

        module_res = await self.prisma.module.create_many(data=docs)

        self.modules = modules
        self.logger.info('Module model seeded successfully with %d documents!' % module_res)

    async def _seedPlanOfStudyDB(self):
        """
        Seeds the plan of study model with dummy data for each account already present in the DB.
        :return:
        """
        self.logger.info('Seeding plan of study model...')
        plans = []

        for account in self.accounts:
            plans.append({
                'studentID': account['id'],
            })

        plan_res = await self.prisma.planofstudy.create_many(data=plans)

        self.plans = plans
        self.logger.info('Plan of study model seeded successfully with %d documents!' % plan_res)

    async def _seedEnrollmentDB(self):
        """
        Enroll each account in each module with a random role. The number of documents that will be created is equal
        to the number of plan of studies multiplied by the number of modules.
        :return: None
        """
        self.logger.info('Seeding enrollment model...')
        enrollments = []

        class EnrollmentRole(Enum):
            STUDENT = 1
            TEACHER = 2
            GRADER = 3

        for pos in self.plans:
            for module in self.modules:
                enrollments.append({
                    'planID': pos['id'],
                    'moduleId': module['id'],
                    'role': random.choice(list(EnrollmentRole)).name
                })

        enrollment_res = await self.prisma.moduleenrollment.create_many(data=enrollments)

        self.enrollments = enrollments
        self.logger.info('Enrollment model seeded successfully with %d documents!' % enrollment_res)

    async def _seedFeedbackDB(self):
        """
        Seed the feedback model with dummy data for each enrollment. The number of documents that will be created is
        equal to the number of enrollments.
        :return: None
        """
        self.logger.info('Seeding feedback model...')
        feedbacks = []

        # find the user's ID given a plan ID
        plans = await self._getPlans()

        for enrollment in self.enrollments:
            feedbacks.append({
                'moduleId': enrollment['moduleId'],
                'rating': random.randint(1, 6),
                'feedback': self.gen.sentence(),
                'studentId': list(map(lambda x: x['studentID'], filter(lambda x: x['id'] == enrollment['planID'], plans))).pop()
            })

        feedback_res = await self.prisma.modulefeedback.create_many(data=feedbacks)
        self.feedbacks = feedbacks
        self.logger.info('Feedback model seeded successfully with %d documents!' % feedback_res)

    async def seedAll(self):
        """
        Calls all the seeding methods in the correct order. If the skip member is set to Skipper.all, then all seeding
        operations will be skipped. If the skip member is set to Skipper.<modelName>, then the <modelName> model seeding
        will be skipped.
        :return: None
        """
        await self.connect()
        self.logger.info('Seeding operation started...')
        if Skipper.all in self.skip:
            self.logger.info('Skipping all model seeding...')
        else:
            if Skipper.user in self.skip:
                self.logger.info('Skipping user model seeding...')
                self.accounts = await self._getUserAccounts()
            else:
                await self._seedUserDB()

            if Skipper.module in self.skip:
                self.logger.info('Skipping module model seeding...')
                self.modules = await self._getModules()
            else:
                await self._seedModuleDB()

            if Skipper.plan in self.skip:
                self.logger.info('Skipping plan of study model seeding...')
                self.plans = await self._getPlans()
            else:
                await self._seedPlanOfStudyDB()

            if Skipper.enrollment in self.skip:
                self.logger.info('Skipping enrollment model seeding...')
                self.enrollments = await self._getEnrollments()
            else:
                await self._seedEnrollmentDB()

            if Skipper.feedback in self.skip:
                self.logger.info('Skipping feedback model seeding...')
                self.feedbacks = await self._getFeedbacks()
            else:
                await self._seedFeedbackDB()

            self.logger.info('All models seeded successfully!')
        await self.disconnect()

    async def cleanupAll(self):
        """
        Calls all the cleanup methods in the correct order. If the cleanup member is set to Skipper.all, then all
        cleanup operations will be skipped. If the cleanup member is set to Skipper.<modelName>, then the <modelName>
        model cleanup will be skipped.
        :return:
        """
        await self.connect()
        self.logger.info('Cleanup operation started...')

        if Skipper.all in self.cleanup:
            self.logger.info('Skipping all model cleanup...')
            return
        else:
            # Deleting all users
            if Skipper.user in self.cleanup:
                self.logger.info('Skipping user model cleanup...')
            else:
                await self._cleanupUserDB()

            # Deleting all plans of study
            if Skipper.plan in self.cleanup:
                self.logger.info('Skipping plan of study model cleanup...')
            else:
                await self._cleanupPlanOfStudyDB()

            # Deleting all enrollments
            if Skipper.enrollment in self.cleanup:
                self.logger.info('Skipping enrollment model cleanup...')
            else:
                await self._cleanupEnrollmentDB()

            # Deleting all modules
            if Skipper.module in self.cleanup:
                self.logger.info('Skipping module model cleanup...')
            else:
                await self._cleanupModuleDB()

            # Deleting all module feedback
            if Skipper.feedback in self.cleanup:
                self.logger.info('Skipping feedback model cleanup...')
            else:
                await self._cleanupFeedbackDB()

            self.logger.info('All models cleaned up successfully!')
        await self.disconnect()

    async def _cleanupUserDB(self):
        count = await self.prisma.user.delete_many()
        self.logger.info('User model cleaned up successfully! (%d documents deleted)' % count)

    async def _cleanupModuleDB(self):
        count = await self.prisma.module.delete_many()
        self.logger.info('Module model cleaned up successfully! (%d documents deleted)' % count)

    async def _cleanupFeedbackDB(self):
        count = await self.prisma.modulefeedback.delete_many()
        self.logger.info('Feedback model cleaned up successfully! (%d documents deleted)' % count)

    async def _cleanupPlanOfStudyDB(self):
        count = await self.prisma.planofstudy.delete_many()
        self.logger.info('Plan of study model cleaned up successfully! (%d documents deleted)' % count)

    async def _cleanupEnrollmentDB(self):
        count = await self.prisma.moduleenrollment.delete_many()
        self.logger.info('Enrollment model cleaned up successfully! (%d documents deleted)' % count)
