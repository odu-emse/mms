import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MetaRecommender:
    def __init__(self, path: str, target: str):
        self.df = pd.read_json(path)
        self.featureDf = pd.DataFrame()
        self.target = target
        nltk.download("stopwords")
        self.stop = stopwords.words("english")
        self.cv = CountVectorizer()
        self.similarity = None

    def _convert_module_id(self, df: pd.DataFrame()):
        temp_id = df["id"]
        temp_id = dict(temp_id)
        temp_id = temp_id["$oid"]
        df["_id"] = temp_id
        return df

    def _scalar_to_str(self, df: pd.DataFrame()):
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
        module_id = self.df[self.df["moduleName"] == self.target]["id"].values[0]

        score = list(enumerate(self.similarity[module_id]))

        sorted_score = sorted(score, key=lambda x: x[1], reverse=True)

        sorted_score = sorted_score[1:]

        results = []

        i = 0
        for item in sorted_score:
            module_title = self.df[self.df["id"] == item[0]]["moduleName"].values[0]
            module_oid = self.df[self.df["id"] == item[0]]["_id"].values[0]
            print(i + 1, module_title, module_oid, item[1])
            results.append({"title": module_title, "id": module_oid, "score": item[1]})
            i += 1
            if i > 10:
                break

        return results

    def run(self):
        self._clean_data(self._convert_module_id, self._scalar_to_str)
        self._create_features()
        self.featureDf = self.df[["id", "features"]]
        self.featureDf["cleaned_features"] = self._text_processing(
            self.featureDf["features"]
        )

        print(self.featureDf.head(10))

        self._get_cosine()

        print(self.similarity)

        return self._get_recommendations()
