import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MetaRecommender:
    """
    This class is used to recommend modules based on the module name, description, keywords and objectives. This is commonly
    referred to as a Meta-Based Recommender System. The class takes in a path to a json file, a target module name and a
    size of the recommendation list. The class will then return a list of
    recommended modules based on the target module name.
    """

    def __init__(self, path: str, target: str, size: int = 10):
        self.df = pd.read_json(path)
        self.featureDf = pd.DataFrame()
        self.target = target
        nltk.download("stopwords")
        self.stop = stopwords.words("english")
        self.cv = CountVectorizer()
        self.similarity = None
        self.size = size

    @staticmethod
    def _convert_module_id(df: pd.DataFrame()):
        """
        This function is used to convert the module id from a mongo Object ID to a string.
        :param df: the dataframe to be modified by the apply function
        :return: the modified dataframe
        """
        temp_id = df["id"]
        temp_id = dict(temp_id)
        temp_id = temp_id["$oid"]
        df["_id"] = temp_id
        return df

    @staticmethod
    def _scalar_to_str(df: pd.DataFrame()):
        lst_k = list(df["keywords"])
        lst_o = list(df["objectives"])

        str_key_feat = ""
        str_obj_feat = ""

        for key in lst_k:
            str_key_feat += str(key) + " "

        for obj in lst_o:
            str_obj_feat += str(obj) + " "

        df["keywords"] = str_key_feat
        df["objectives"] = str_obj_feat
        return df

    def _clean_data(self, cleanID, cleanScalar):
        self.df = self.df.apply(cleanID, axis=1)
        self.df = self.df.drop(["id"], axis=1)
        self.df = self.df.reset_index(drop=True)

        self.df = self.df.apply(cleanScalar, axis=1)

    def _create_features(self):
        features = []

        for i in range(0, self.df.shape[0]):
            features.append(
                self.df["moduleName"][i]
                + " "
                + self.df["intro"][i]
                + " "
                + self.df["description"][i]
                + " "
                + self.df["keywords"][i]
                + " "
                + self.df["objectives"][i]
            )
        self.df["features"] = features
        self.df.insert(1, "id", list(range(1, len(self.df) + 1)), True)

    def _text_processing(self, col: pd.DataFrame()):
        column = col.str.lower()
        column = column.str.replace("[^a-z ]", "")
        word_tokens = column.str.split()
        keys = word_tokens.apply(
            lambda x: [item for item in x if item not in self.stop]
        )
        for i in range(len(keys)):
            keys[i] = " ".join(keys[i])
            column = keys
        return column

    def _get_cosine(self):
        count_matrix = self.cv.fit_transform(self.featureDf["cleaned_features"])
        self.similarity = cosine_similarity(count_matrix)

    def _get_recommendations(self):
        try:
            module_id = self.df[
                self.df["moduleName"].str.lower() == str(self.target).lower()
            ]["id"].values[0]
            score = list(enumerate(self.similarity[module_id]))

            sorted_score = sorted(score, key=lambda x: x[1], reverse=True)

            sorted_score = sorted_score[1:]

            if self.size > len(sorted_score):
                self.size = len(sorted_score)

            results = []

            i = 0
            for item in sorted_score:
                module_title = self.df[self.df["id"] == item[0]]["moduleName"].values[0]
                module_oid = self.df[self.df["id"] == item[0]]["_id"].values[0]
                print(i + 1, module_title, module_oid, item[1])
                results.append(
                    {"title": module_title, "id": module_oid, "score": item[1]}
                )
                i += 1
                if i > self.size - 1:
                    break

            return results
        except IndexError:
            return {"error": "Module not found"}

    def run(self):
        """
        This function is used to run the recommender system. It recommends modules based on the
        target module name passed in. The list will be sorted by the cosine similarity score.
        The list will be in the format of

        *[{"title": module_title, "id": module_oid, "score": cosine_similarity_score}, ...]*

        .. todo:: Add a check to see if the target module name is in the database.

        .. return:: a list of recommended modules or a dictionary an error message
        """
        self._clean_data(self._convert_module_id, self._scalar_to_str)
        self._create_features()
        self.featureDf = self.df[["id", "features"]]
        self.featureDf["cleaned_features"] = self._text_processing(
            self.featureDf["features"]
        )

        self._get_cosine()

        return self._get_recommendations()
