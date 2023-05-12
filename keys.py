import pandas as pd
import json

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Keys:
    def __init__(self, target, user_input="input/user_input.json"):
        self.similarity_matrix = list()
        self.mapping = list()
        self.target = target
        self.modules = json.loads(open("input/modules.json").read())
        self.user_input = json.loads(open(user_input).read())
        self.user_df = pd.DataFrame()
        self.tf_idf = TfidfVectorizer(stop_words="english")

    def __convert_json_scalars__(self, scalar: list) -> str:
        """
        Converts a list of scalars into a comma separated string. This is used to convert the skills, education, etc.
        into a single string that can be appended to a DataFrame row.
        :param scalar: A list of strings to be converted into a comma separated string.
        :rtype: str
        """
        temp_string = ""
        for index, value in enumerate(scalar):
            if index == len(scalar) - 1:
                temp_string += value
            else:
                temp_string += value + ", "

        return temp_string

    def __read_input(self):
        df = pd.DataFrame(
            columns=[
                "id",
                "skills",
                "education",
                "work_experience",
                "courses",
                "focus",
                "degree",
            ],
            index=[0],
        )

        df["id"] = self.user_input["id"]

        df["skills"] = self.__convert_json_scalars__(scalar=self.user_input["skills"])

        df["education"] = self.__convert_json_scalars__(
            scalar=self.user_input["education"]
        )

        df["work_experience"] = self.__convert_json_scalars__(
            scalar=self.user_input["work_experience"]
        )

        df["courses"] = self.user_input["courses"]

        df["focus"] = self.user_input["focus"]

        df["degree"] = self.user_input["degree"]

        self.user_df = df

    def __read_modules(self):
        df = pd.DataFrame(
            columns=[
                "id",
                "moduleName",
                "description",
                "intro",
                "keywords",
                "objectives",
            ],
            index=[i for i in range(len(self.modules))],
        )

        for index, module in enumerate(self.modules):
            df.iloc[index]["id"] = module["id"]["$oid"]
            df.iloc[index]["moduleName"] = module["moduleName"]
            df.iloc[index]["description"] = module["description"]
            df.iloc[index]["intro"] = module["intro"]
            df.iloc[index]["keywords"] = self.__convert_json_scalars__(
                scalar=module["keywords"]
            )
            df.iloc[index]["objectives"] = self.__convert_json_scalars__(
                scalar=module["objectives"]
            )

        self.modules_df = df

    def __analyze_modules(self):
        keyword_matrix = self.tf_idf.fit_transform(self.modules_df["keywords"])

        similarity_matrix = linear_kernel(keyword_matrix, keyword_matrix)

        mapping = pd.Series(self.modules_df.index, index=self.modules_df["moduleName"])

        self.similarity_matrix = similarity_matrix

        self.mapping = mapping

    def recommend_movies_based_on_plot(self, query, mapping, matrix):
        movie_index = mapping[query]

        # get similarity values with other movies
        # similarity_score is the list of index and similarity matrix
        similarity_score = list(enumerate(matrix[movie_index]))

        # sort in descending order the similarity score of movie inputted with all the other movies
        for index, s in enumerate(similarity_score):
            id = s[0]
            scores = s[1]

            # sort scores in descending order
            scores = sorted(scores, key=lambda x: x, reverse=True)

            similarity_score[index] = (id, scores)

        # Get the scores of the 15 most similar movies. Ignore the first movie.
        similarity_score = similarity_score[1:15]

        # return movie names using the mapping series
        movie_indices = [i[0] for i in similarity_score]

        return self.modules_df.iloc[movie_indices]

    def run(self) -> str:
        """
        Runs the entire system in a sequential order.
        """
        self.__read_input()
        self.__read_modules()
        self.__analyze_modules()
        recommendation: DataFrame = self.recommend_movies_based_on_plot(
            self.target, mapping=self.mapping, matrix=self.similarity_matrix
        )
        return recommendation.to_json(orient="records")


def main():
    """
    Main function to run the system. Instantiates the Keys class and runs the entry method.
    """
    keys = Keys(target="Namfix")
    keys.run()


if __name__ == "__main__":
    main()
