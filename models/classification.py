import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class Classify:
    """
    Classify the data into different clusters based on text content using embedding techniques and K means clustering.
    """

    def __init__(
        self,
        path: str = "input/603_trans_3.tsv",
        outputPath: str = "output/603_clean.csv",
        toDownload: bool = True,
        verbose: bool = True,
    ) -> None:
        import logging

        logging.basicConfig(level=logging.INFO)

        self.logger = logging.getLogger(__name__)
        self.filePath = path
        self.outputPath = outputPath
        self.toDownload = toDownload
        self.data = None
        self.verbose = verbose
        self.stop_words = set(stopwords.words("english"))
        self.N_CLUSTER = 0
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.train_x_vector = None
        self.test_x_vector = None
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.encoder = LabelEncoder()

        self.download()
        self._configure()

    def _configure(self) -> None:
        self.stop_words.add("module")
        self.stop_words.add("problem")
        self.stop_words.add("use")
        self.stop_words.add("model")
        self.stop_words.add("solution")
        self.stop_words.add("solve")
        self.stop_words.add("analyze")
        self.stop_words.add("example")
        self.stop_words.add("application")
        self.stop_words.add("computer")
        self.stop_words.add("computers")
        self.stop_words.add("one")
        self.stop_words.add("two")
        self.stop_words.add("three")
        self.stop_words.add("four")
        self.stop_words.add("five")
        self.stop_words.add("six")
        self.stop_words.add("seven")
        self.stop_words.add("eight")
        self.stop_words.add("x")
        self.stop_words.add("c")
        self.stop_words.add("go")
        self.stop_words.add("constraint")
        self.stop_words.add("get")

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
            row, col = self.data.shape
            self.logger.info(f"Rows: {row}, Columns: {col}")
        else:
            self.logger.info("Data read successfully")

    def _scalar_to_string(self, df: DataFrame, col: str) -> DataFrame:
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
            print(df.head())
        else:
            self.logger.debug("Scalar values converted to string successfully")

        return df

    def _merge_columns(self, df: DataFrame, col1: str, col2: str) -> DataFrame:
        """
        Merge the values of the two DataFrame columns.
        """
        lst1 = list(df[col1])
        lst2 = list(df[col2])

        merged = []

        for i in range(len(lst1)):
            cleaned_transcript = self._clean_transcript(str(lst2[i]))
            merged.append(lst1[i] + " " + cleaned_transcript)

        df[col1] = merged

        if self.verbose:
            self.logger.info("Columns merged successfully")
            print(df.head())
        else:
            self.logger.debug("Columns merged successfully")

        return df

    def _clean_transcript(self, input: str) -> DataFrame:
        """
        Clean the transcript by removing special characters and numbers from text.
        """
        import re
        from string import digits

        if input == "nan":
            return ""

        else:
            cleaned = re.sub(
                "([A-Z][a-z]+)",
                r" \1",
                re.sub(
                    "([A-Z]+)",
                    r" \1",
                    re.sub(
                        r"[^a-zA-Z0-9]+",
                        " ",
                        input.replace("\\'", ""),
                    ),
                ),
            )

            remove_digits = str.maketrans("", "", digits)

            return cleaned.translate(remove_digits)

    def _clean_text(self, input: str) -> str:
        """
        Clean the text by removing special characters, and digits from the input string.
        """
        import re

        if input == "nan":
            return ""
        return re.sub(
            r"[^a-zA-Z]+",
            " ",
            input.replace("\\'", ""),
        )

    def _split_camel_case(self, df: DataFrame, col: str) -> DataFrame:
        """
        Split the words in the DataFrame column which are in camel case.
        """
        lst = list(df[col])

        split_words = []

        for obj in lst:
            split_words.append(self._clean_text(obj).split())

        payload = [" ".join(entry) for entry in split_words]

        df[col] = payload
        if self.verbose:
            self.logger.info("Camel case text split successfully")
            print(df.head())
        else:
            self.logger.info("Camel case text split successfully")

        return df

    def _stemmer(self, df: DataFrame, col: str) -> DataFrame:
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

        stemmer = self.lemmatizer
        for index, entry in enumerate(data["tokens"]):
            final_words = []
            for word, tag in pos_tag(entry):
                word_final = stemmer.lemmatize(word, tag_map[tag[0]])
                if word_final not in self.stop_words and word_final.isalpha():
                    final_words.append(word_final)
                else:
                    self.logger.debug("Removed word from corpus: " + word)

            data.loc[index, "target"] = " ".join(final_words)

        return data

    def _preprocess_features(self, col: str) -> DataFrame:
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
        df = self._split_camel_case(df, col)

        # for each row of the feature column, turn text to lowercase
        df[col] = [entry.lower() for entry in df[col]]

        # tokenize entries, stem words and remove stop words
        df = self._stemmer(df, col)

        return df

    def prepare(self, col: str) -> DataFrame:
        """
        Prepare the data for the classification.
        """
        import numpy as np

        df = self._merge_columns(self.data, "features", "transcript")
        df = self._preprocess_features(col)

        df["label"] = ""

        if self.verbose:
            self.logger.info("Data prepared successfully")
            print(df.head())
        else:
            self.logger.info("Data prepared successfully")

        self.data = df
        self.N_CLUSTER = int(np.sqrt(len(df)))
        self.generate_count_plot(data=df)
        self._save_data_frame(df)

    def _data_transformer(self, df: DataFrame, size: float = 0.3) -> tuple:
        """
        Splits, fits, and transforms the data for the classification.
        """
        from sklearn.model_selection import train_test_split

        Train_X, Test_X, Train_Y, Test_Y = train_test_split(
            df["target"], df["label"], test_size=size
        )

        Encoder = self.encoder

        Train_Y = Encoder.fit_transform(Train_Y)

        Test_Y = Encoder.fit_transform(Test_Y)

        Tfidf_vect = self.vectorizer

        Tfidf_vect.fit(df["target"])

        Train_X_Tfidf = Tfidf_vect.transform(Train_X)

        Test_X_Tfidf = Tfidf_vect.transform(Test_X)

        self.train_x = Train_X
        self.test_x = Test_X
        self.train_y = Train_Y
        self.test_y = Test_Y
        self.train_x_vector = Train_X_Tfidf
        self.test_x_vector = Test_X_Tfidf

        # self._predict_clusters(Tfidf_vect.transform(df["target"]), df)

        # self.generate_heat_map(X=Train_X_Tfidf, df=)

        # self.generate_elbow_plot(X=Train_X_Tfidf)

        return Tfidf_vect

    def _create_model(self):
        """
        Create the classification model.
        """

        Tfidf_vect = self._data_transformer(self.data)

        train_df = pd.DataFrame(
            self.train_x, columns=["target", "label"], index=self.train_x.index
        )

        self._create_clusters(
            self.train_x_vector,
            train_df,
        )

        self._print_top_words_per_cluster(Tfidf_vect, train_df, self.train_x_vector)

        self._run_pca(self.train_x_vector, train_df)

        self._run_naive_bayes()

        self.logger.info("Model created successfully")

        return train_df

    def _print_top_words_per_cluster(self, vectorizer, df: DataFrame, X: list, n=10):
        """
        This function returns the keywords for each centroid of the KMeans
        """
        import numpy as np

        data = pd.DataFrame(X.toarray()).groupby(df["label"]).mean()
        terms = vectorizer.get_feature_names_out()

        self.logger.info("Top keywords per cluster in training set:")

        for i, r in data.iterrows():
            print("\nCluster {}".format(i))
            print(", ".join([terms[t] for t in np.argsort(r)[-n:]]))

    def _train_model(self):
        """
        Train the classification model.
        """
        pass

    def _evaluate_model(self):
        """
        Evaluate the classification model.
        """
        pass

    def _predict(self):
        """
        Predict the classification model.
        """
        pass

    def _save_model(self):
        """
        Save the classification model.
        """
        pass

    def _predict_clusters(self, X, df: DataFrame):
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=7, random_state=0, n_init="auto", init="k-means++", max_iter=500
        ).fit(X)

        title = df["name"]
        labels = kmeans.labels_

        wiki_cl = pd.DataFrame(list(zip(title, labels)), columns=["title", "cluster"])

        print(wiki_cl.sort_values(by=["cluster"]))
        print(
            "K-Means Accuracy Score -> ",
            accuracy_score(y_true=df["prefix"], y_pred=kmeans.predict(X)) * 100,
        )

    def _create_clusters(self, X, df: DataFrame):
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=self.N_CLUSTER,
            random_state=0,
            n_init="auto",
            init="k-means++",
            max_iter=500,
        ).fit(X)

        df["label"] = kmeans.predict(X)

        if self.verbose:
            self.logger.info("Clusters created successfully")
            print(
                "K-Means Accuracy Score -> ",
                accuracy_score(self.train_y, y_pred=kmeans.predict(X)) * 100,
            )
        else:
            self.logger.debug("Clusters created successfully")

    def _run_pca(self, X, df):
        import numpy as np
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(np.sqrt(len(df))) // 2, random_state=42)

        red_feat = pca.fit_transform(X.toarray())

        x = red_feat[:, 0]
        y = red_feat[:, 1]

        df["x"] = x
        df["y"] = y

        self.logger.info("PCA run successfully")

    def _run_naive_bayes(self):
        """
        Run the naive bayes classification model.
        """
        from sklearn.naive_bayes import MultinomialNB

        Naive = MultinomialNB()

        Naive.fit(self.train_x_vector, self.train_y)

        predictions_NB = Naive.predict(self.test_x_vector)

        if self.verbose:
            self.logger.info("Naive Bayes run successfully")
            print(
                "Naive Bayes Accuracy Score -> ",
                accuracy_score(predictions_NB, self.test_y) * 100,
            )
        else:
            self.logger.debug("Naive Bayes run successfully")

    def _run_svm(self, X_train, Y_train, X_test, Y_test):
        """
        Run the svm classification model.
        """
        from sklearn import svm

        SVM = svm.SVC(C=1.0, kernel="poly", degree=3, gamma="auto")

        SVM.fit(X_train, Y_train)

        predictions_SVM = SVM.predict(X_test)

        print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Y_test) * 100)

        if self.verbose:
            self.logger.info("SVM run successfully")
            print(predictions_SVM)
        else:
            self.logger.debug("SVM run successfully")

    def _run_word_cloud_per_cluster(self, df: DataFrame):
        """
        Run the word cloud per cluster.
        """

        for i in sorted(df["label"].array.unique()):
            corpus = " ".join(list(df[df["label"] == i]["target"]))
            self.generate_word_cloud(corpus=corpus)

    def _save_data_frame(self, df: DataFrame):
        """
        Save the data frame as a CSV file.
        """
        df.to_csv(self.outputPath, index=False)

    def generate_word_cloud(self, corpus: str):
        """
        Generate the word cloud for the data.
        """
        from wordcloud import WordCloud
        from matplotlib import pyplot as plt

        wordcloud = WordCloud(
            max_font_size=80,
            max_words=700,
            background_color="white",
            stopwords=self.stop_words,
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

    def generate_bar_plot(self, data):
        """
        Generate a bar plot that shows the number of documents per cluster.
        """
        import seaborn as sns
        from matplotlib import pyplot as plt

        sns.countplot(x="label", data=data)
        plt.show()

    def generate_elbow_plot(self, X):
        """
        Generate the elbow plot for the data that shows the most optimal number of clusters that should be used based on sum of squared distances.
        """
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans

        Sum_of_squared_distances = []
        K = range(2, self.N_CLUSTER * 2)

        for k in K:
            km = KMeans(n_clusters=k, max_iter=200, n_init=10)
            km = km.fit(X)
            Sum_of_squared_distances.append(km.inertia_)

        plt.plot(K, Sum_of_squared_distances, "bx-")
        plt.xlabel("k")
        plt.ylabel("Sum_of_squared_distances")
        plt.title("Elbow Method For Optimal k")
        plt.show()

    def generate_count_plot(self, data):
        """
        Generate a bar plot that sums the number of rows that share the same prefix value.
        """
        import seaborn as sns
        from matplotlib import pyplot as plt

        sns.countplot(x="prefix", data=data)
        plt.show()

    def run(self) -> None:
        """
        Run the classification model.
        """
        df = self._create_model()
        self._train_model()
        self._evaluate_model()
        self._predict()
        self._save_model()
        self.generate_scatter_plot(df)
        self.generate_bar_plot(df)
        self._run_word_cloud_per_cluster(df)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classification of text")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="input/603_trans_3.tsv",
        help="Path to the processed data",
    )
    parser.add_argument(
        "--download", "-d", type=bool, help="Download the required libraries"
    )
    parser.add_argument(
        "--verbose", "-v", type=bool, help="Print the logs", default=True
    )
    parser.add_argument(
        "--sep", "-s", type=str, default="\t", help="Separator for the data"
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="output/603_clean.csv",
        help="Path to the output file",
    )
    args = parser.parse_args()

    classify = Classify(
        path=args.path,
        toDownload=args.download,
        verbose=args.verbose,
        outputPath=args.out,
    )
    classify.read()
    classify.prepare(col="features")
    classify.run()


if __name__ == "__main__":
    main()
