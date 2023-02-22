import json
import os
from requests import post


def getModuleFromID(modules_df):
    # take in each row through a lamda function
    # get module from ID
    # send request to graphql
    # add score to res
    # append to array in json file

    print('Getting module from ID: %s' % modules_df['moduleID'])
    res = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"), {}, {
        'query': """query{
              module(input:{
                id: "%s"
              }){
                id
                moduleName
                moduleNumber
              }
        }""" % modules_df['moduleID']
    })
    res = res.json()['data']['module'][0]
    res['__score'] = modules_df['rating']

    # write to file
    os.makedirs('data', exist_ok=True)
    with open('data/response.json', 'a') as f:
        f.write(json.dumps(res) + ',')
        print('Wrote to file: %s' % res['moduleName'])
