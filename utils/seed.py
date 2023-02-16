from essential_generators import DocumentGenerator
from numpy import random
from requests import post
from utils.sdl import createMutationString


def seedDbFeedback():
    print('Seeding database...')
    gen = DocumentGenerator()
    iterations = 50
    for i in range(iterations):
        res = post('http://localhost:4000/graphql',
                   {},
                   {'query': createMutationString(
                       gen.sentence(),
                       random.randint(1, 6)
                   )
                   })
        print(res.json())

    print('Database seeded successfully!')


def getModuleFeedback():
    mods = post('http://localhost:4000/graphql', {}, {
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
