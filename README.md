
# Module Management System

_Engineering Management & Systems Engineering - ODU_


State of the art API that utilizes Machine Learning models to generate personalized degree paths for each student based on their respective exeperiences and learning styles.

> Throughout this document we refer to one of the supplementary services as *"client"*. To avoid confusion, treat the client service as a seperate application that interacts with this API. The *"client"* application both consumes and calls this API to present students with a user friendly way of seeing their calculated degree.
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file. To run the application in isolation, you only have to define the four variables below. 

`DATABASE_URL` _mongodb+srv://<admin>:<password>@<atlas-path>.mongodb.net/<database>_

The URI that Prisma is going to use to access your databse. Currently only MongoDB is supported but it can be either a managed cluster by Mongo Atlas, or a self hosted service. 

`DIRECT_URL` _prisma://aws-us-east-1.prisma-data.com/?api_key=<API-KEY>_

The URI that Prisma provides you after creating a connection pool using their [Data Browser](https://www.prisma.io/docs/data-platform/data-browser).

`REDIS_BASE_URL`

The name of the Docker container that is running the Redis instance. If all configurations are left as is, this value in most cases will be **mms_redis**.

`API_URL`

The name of the Docker container that is running the client instance. If all configurations are left as is, this value in most cases will be **client**.

---

The varibles below are used by the client part of the API to and are not neccessary to run the application in isolation. If they are filled out, Docker will import the .env file from the root of your directory to build and launch the API client.


`GOOGLE_REFRESH_TOKEN`

`GOOGLE_SECRET`

`jwtSecret`

`jwtExpire`

`PORT`

`DB_USER`

`DB_PASSWORD`

## Installation

Building the image using Docker compose and the contexts defined in the Dockerfile.

```bash
docker-compose build .
```

Launching the application and it's related services

```bash
docker-compose up -d
```
    
## API Reference

#### Get module recommendations for a user

```http
  GET /recommend/${userID}
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `userID` | `string` | **Required**. The user's document ID |


## Roadmap

Recommending learning modules using differnt methods is our main goal with this project. Outlined below are all the methods that we are currently working on, or planning to implement before our initial launch. 

- Recommend modules based on public reviews

- Recommend modules based on keyword similary between the learner and module objectives

- Recommend modules based on learner profile similary 

- Recommend modules based on the student's learning preferences

- Recommend modules based on the learner's prior professional experiences

- Combining recommendations and drawing the most viable path for the student's degree


## Tech Stack

**Client:** GraphQL, TypeScript

**API:** Python, FastAPI, Prisma

**Server:** MongoDB, Redis

