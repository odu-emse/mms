import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score


class Classify:
    def __init__(
        self,
        path: str = "input/603_num.tsv",
        toDownload: bool = True,
        verbose: bool = True,
    ) -> None:
        import logging

        logging.basicConfig(level=logging.INFO)

        self.logger = logging.getLogger(__name__)
        self.filePath = path
        self.toDownload = toDownload
        self.data = None
        self.verbose = verbose
        self.stop_words = set(stopwords.words("english"))
        self.N_CLUSTER = 0

        self.download()
        self.__configure__()

    def __configure__(self) -> None:
        self.stop_words.add("module")
        self.stop_words.add("problem")
        self.stop_words.add("use")
        self.stop_words.add("model")
        self.stop_words.add("solution")
        self.stop_words.add("solve")
        self.stop_words.add("analyze")

    def download(self):
        """
        Download the required libraries for the classification.
        """
        import nltk

        if self.toDownload:
            nltk.download("stopwords", quiet=not self.verbose)
            nltk.download("wordnet", quiet=not self.verbose)
            nltk.download("omw-1.4", quiet=not self.verbose)
        else:
            self.logger.info("Library downloads skipped")

    def read(self, sep="\t") -> None:
        """
        Read the data from the file path. The default separator is tab.
        """
        self.data = pd.read_csv(self.filePath, sep=sep)

        if self.verbose:
            self.logger.info("Data read successfully")
            self.logger.info(self.data.head())
        else:
            self.logger.info("Data read successfully")

    def __scalar_to_string__(self, df: DataFrame, col: str) -> DataFrame:
        """
        Convert the scalar values for each row of the DataFrame column to a string.
        """
        lst = list(df[col])

        scalar_to_string = []

        for obj in lst:
            print(obj)
            scalar_to_string.append(str(obj))

        df[col] = scalar_to_string

        if self.verbose:
            self.logger.info("Scalar values converted to string successfully")
            self.logger.info(df.head())
        else:
            self.logger.info("Scalar values converted to string successfully")

        return df

    def __clean_text__(self, input: str) -> str:
        """
        Clean the text by removing special characters.
        """
        import re

        return re.sub(
            "([A-Z][a-z]+)",
            r" \1",
            re.sub(
                "([A-Z]+)",
                r" \1",
                input.replace("\\'", ""),
            ),
        )

    def __split_camel_case__(self, df: DataFrame, col: str) -> DataFrame:
        """
        Split the words in the DataFrame column which are in camel case.
        """
        lst = list(df[col])

        split_words = []

        for obj in lst:
            split_words.append(self.__clean_text__(obj).split())

        payload = [" ".join(entry) for entry in split_words]

        df[col] = payload
        if self.verbose:
            self.logger.info("Camel case text split successfully")
            self.logger.info(df.head())
        else:
            self.logger.info("Camel case text split successfully")

        return df

    def __stemmer__(self, df: DataFrame, col: str) -> DataFrame:
        """
        Stem the words in the DataFrame column.
        """
        from collections import defaultdict

        data = df.copy()

        tag_map = defaultdict(lambda: wn.NOUN)

        tag_map["J"] = wn.ADJ
        tag_map["V"] = wn.VERB
        tag_map["R"] = wn.ADV

        data["tokens"] = [word_tokenize(entry) for entry in data[col]]

        for index, entry in enumerate(data["tokens"]):
            final_words = []
            stemmer = WordNetLemmatizer()

            for word, tag in pos_tag(entry):
                word_final = stemmer.lemmatize(word, tag_map[tag[0]])
                if word_final not in self.stop_words and word_final.isalpha():
                    final_words.append(word_final)
                else:
                    self.logger.debug("Removed word from corpus: " + word)

            data.loc[index, "target"] = " ".join(final_words)

        return data

    def __preprocess_features__(self, col: str) -> DataFrame:
        """
        1. turn entries to lower case
        2. remove special characters
        4. remove stop words
        5. stem words
        6. tokenize words
        7. return feature column
        """
        # create a copy of the data
        df = self.data.copy()

        # convert camel case text to separate words
        df = self.__split_camel_case__(df, col)

        # for each row of the feature column, turn text to lowercase
        df[col] = [entry.lower() for entry in df[col]]

        # tokenize entries, stem words and remove stop words
        df = self.__stemmer__(df, col)

        return df

    def prepare(self, col: str) -> DataFrame:
        """
        Prepare the data for the classification.
        """
        import numpy as np

        df = self.__preprocess_features__(col)

        df["label"] = ""

        if self.verbose:
            self.logger.info("Data prepared successfully")
            self.logger.info(df.head())
        else:
            self.logger.info("Data prepared successfully")

        self.data = df
        self.N_CLUSTER = int(np.sqrt(len(df)))

    def __create_model__(self, size=0.3):
        """
        Create the classification model.
        """
        from sklearn import model_selection
        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer

        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
            self.data["target"], self.data["label"], test_size=size
        )

        Encoder = LabelEncoder()

        Encoder.fit(Train_Y)

        Train_Y = Encoder.transform(Train_Y)

        Test_Y = Encoder.transform(Test_Y)

        Tfidf_vect = TfidfVectorizer(max_features=5000)

        Tfidf_vect.fit(self.data["target"])

        Train_X_Tfidf = Tfidf_vect.transform(Train_X)

        Test_X_Tfidf = Tfidf_vect.transform(Test_X)

        train_df = pd.DataFrame(
            Train_X, columns=["target", "label"], index=Train_X.index
        )

        self.__create_clusters__(
            Train_X_Tfidf,
            train_df,
        )

        self.__run_pca__(Train_X_Tfidf, train_df)

        self.logger.info(train_df.head())

        self.__run_naive_bayes__(
            X_train=Train_X_Tfidf,
            Y_train=Train_Y,
            X_test=Test_X_Tfidf,
            Y_test=Test_Y,
        )

        self.__run_svm__(
            X_train=Train_X_Tfidf,
            Y_train=Train_Y,
            X_test=Test_X_Tfidf,
            Y_test=Test_Y,
        )

        self.logger.info("Model created successfully")

        return train_df

    def __train_model__(self):
        """
        Train the classification model.
        """
        pass

    def __evaluate_model__(self):
        """
        Evaluate the classification model.
        """
        pass

    def __predict__(self):
        """
        Predict the classification model.
        """
        pass

    def __save_model__(self):
        """
        Save the classification model.
        """
        pass

    def __create_clusters__(self, X, df: DataFrame):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.N_CLUSTER, random_state=0, n_init="auto").fit(X)

        df["label"] = kmeans.predict(X)

        self.logger.info("Clusters created successfully")

    def __run_pca__(self, X, df):
        import numpy as np
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(np.sqrt(len(df))) // 2, random_state=42)

        red_feat = pca.fit_transform(X.toarray())

        x = red_feat[:, 0]
        y = red_feat[:, 1]

        df["x"] = x
        df["y"] = y

        self.logger.info("PCA run successfully")

    def __run_naive_bayes__(self, X_train, Y_train, X_test, Y_test):
        """
        Run the naive bayes classification model.
        """
        from sklearn.naive_bayes import MultinomialNB

        Naive = MultinomialNB()

        Naive.fit(X_train, Y_train)

        predictions_NB = Naive.predict(X_test)

        print(
            "Naive Bayes Accuracy Score -> ",
            accuracy_score(predictions_NB, Y_test) * 100,
        )

        self.logger.info("Naive Bayes run successfully")

    def __run_svm__(self, X_train, Y_train, X_test, Y_test):
        """
        Run the svm classification model.
        """
        from sklearn import svm

        SVM = svm.SVC(C=1.0, kernel="poly", degree=3, gamma="auto")

        SVM.fit(X_train, Y_train)

        predictions_SVM = SVM.predict(X_test)

        print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Y_test) * 100)

        self.logger.info("SVM run successfully")

    def run(self) -> None:
        """
        Run the classification model.
        """
        df = self.__create_model__()
        self.__train_model__()
        self.__evaluate_model__()
        self.__predict__()
        self.__save_model__()
        # self.generate_word_cloud(corpus=" ".join(list(self.data["target"])))
        # self.generate_scatter_plot(df)

        # self.data.to_csv("output/603_processed.csv", index=False)

    def generate_word_cloud(self, corpus: str):
        """
        Generate the word cloud for the data.
        """
        from wordcloud import WordCloud
        from matplotlib import pyplot as plt

        wordcloud = WordCloud(
            max_font_size=80, max_words=300, background_color="white"
        ).generate(corpus)
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def generate_scatter_plot(self, data):
        """
        Generate the scatter plot for the data.
        """
        import seaborn as sns
        from matplotlib import pyplot as plt

        sns.scatterplot(
            data=data,
            x="x",
            y="y",
            hue="label",
            palette="tab10",
        )
        plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classification of text")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="input/603_num.tsv",
        help="Path to the processed data",
    )
    parser.add_argument(
        "--download", "-d", type=bool, help="Download the required libraries"
    )
    parser.add_argument("--verbose", "-v", type=bool, help="Print the logs")
    parser.add_argument(
        "--sep", "-s", type=str, default="\t", help="Separator for the data"
    )
    args = parser.parse_args()

    classify = Classify(path=args.path, toDownload=False, verbose=False)
    classify.read()
    classify.prepare(col="features")
    classify.run()


if __name__ == "__main__":
    main()
