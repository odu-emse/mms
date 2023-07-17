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
from typing import Union
import numpy
from pandas.core.series import Series


class Classify:
    """
    Predict the class for the given Module dataset based on TF-IDF vectorization and KNN (supervised) clustering algorithm.

    Args:
        path (str): Path to the training dataset.
        testPath (str, None): Path to the test dataset.
        outputPath (str): Path to the output directory.
        toDownload (bool): Whether to download the nltk corpus or not.
        verbose (bool): Whether to print the logs or not.
        visualize (bool): Whether to show EDA visualizations or not.
    """

    def __init__(
        self,
        path: str = "input/603_trans_3.tsv",
        testPath: Union[str, None] = None,
        outputPath: str = "output/",
        toDownload: bool = False,
        verbose: bool = False,
        visualize: bool = False,
    ) -> None:
        import logging

        self.logger = logging.getLogger(__name__)
        self.filePath: str = path
        self.testPath: str = testPath
        self.outputPath: str = outputPath
        self.logPath: str = "logs/classification.log"
        self.toDownload: bool = toDownload
        self.data: Union[DataFrame, None] = None
        self.testData: Union[DataFrame, None] = None
        self.verbose: bool = verbose
        self.viz: bool = visualize
        self.stop_words = set(stopwords.words("english"))
        self.N_CLUSTER: int = 0
        self.N_TEST_CLUSTER: int = 0
        self.train_x: Union[Series, None] = None
        self.train_y: Union[Series, None] = None
        self.test_x: Union[Series, None] = None
        self.test_y: Union[Series, None] = None
        self.train_x_vector: Union[None, numpy.ndarray] = None
        self.test_x_vector: Union[None, numpy.ndarray] = None
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.encoder = LabelEncoder()

        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.DEBUG)

        self.download()
        self._configure()

    def _configure(self) -> None:
        """
        Configure the logger and the stop words set
        """
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
        self.stop_words.add("ok")
        self.stop_words.add("uh")
        self.stop_words.add("shift")

    def download(self):
        """
        Download the required libraries for the classification.
        """
        import nltk

        if self.toDownload:
            nltk.download("stopwords", quiet=not self.verbose)
            nltk.download("wordnet", quiet=not self.verbose)
            nltk.download("omw-1.4", quiet=not self.verbose)
            self._log("Library downloads complete")
        else:
            self._log("Library downloads skipped")

    def read(self, sep="\t") -> None:
        """
        Read the data from the file path. The default separator is tab.
        """
        self.data = pd.read_csv(self.filePath, sep=sep)

        if self.testPath is not None:
            self.testData = pd.read_csv(self.testPath, sep=sep)

        if self.verbose:
            row, col = self.data.shape
            self._log(f"Shape of train data read \nRows: {row}, Columns: {col}")
            if self.testPath is not None:
                row, col = self.testData.shape
                self._log(f"Shape of test data read \nRows: {row}, Columns: {col}")
            self._log("Data read successfully")
        else:
            self._log("Data read successfully")
            if self.testPath is not None:
                self._log("Test data read successfully")

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
            self._log("Scalar values converted to string successfully")
            print(df.head())
        else:
            self._log("Scalar values converted to string successfully")

        return df

    def _merge_columns(
        self, df: DataFrame, destinationCol: str, originCol: str
    ) -> DataFrame:
        """
        Merge the values of the two DataFrame columns.
        """
        lst1 = list(df[destinationCol])
        lst2 = list(df[originCol])

        merged = []

        for i in range(len(lst1)):
            cleaned_transcript = self._clean_transcript(str(lst2[i]))
            merged.append(lst1[i] + " " + cleaned_transcript)

        df[destinationCol] = merged

        if self.verbose:
            self._log("Columns merged successfully")
            print(df.head())
        else:
            self._log("Columns merged successfully")

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
            self._log("Camel case text split successfully")
            print(df.head())
        else:
            self._log("Camel case text split successfully")

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

            data.loc[index, "target"] = " ".join(final_words)

        return data

    def _preprocess_features(self, col: str, data: DataFrame) -> DataFrame:
        """
        1. turn entries to lower case
        2. remove special characters
        4. remove stop words
        5. stem words
        6. tokenize words
        7. return feature column
        """
        # create a copy of the data
        df = data.copy()

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

        if self.testPath is None:
            df = self._merge_columns(
                self.data, destinationCol="features", originCol="transcript"
            )
            df = self._preprocess_features(col, self.data)
            df = df.drop(
                ["features", "tokens", "hours", "prefix", "transcript", "number"],
                axis=1,
            )

            if self.verbose:
                self._log("Data prepared successfully")
                print(df.head())
            else:
                self._log("Data prepared successfully")

            self.data = df
            self.N_CLUSTER = int(np.sqrt(len(df)))
            if self.viz == True:
                self.generate_count_plot(data=df)
            self._save_data_frame(df, fileName="603_clean.csv")

        else:
            dfTrain = self._merge_columns(
                self.data, destinationCol="features", originCol="transcript"
            )
            dfTrain = self._preprocess_features(col, self.data)
            dfTrain = dfTrain.drop(
                ["features", "tokens", "hours", "prefix", "transcript", "number"],
                axis=1,
            )

            dfTest = self._merge_columns(
                self.testData, destinationCol="features", originCol="transcript"
            )
            dfTest = self._preprocess_features(col, self.testData)
            dfTest = dfTest.drop(
                ["features", "tokens", "hours", "prefix", "transcript", "number"],
                axis=1,
            )

            if self.verbose:
                self._log("Train data prepared successfully")
                print(dfTrain.head())
                self._log("Test data prepared successfully")
                print(dfTest.head())
            else:
                self._log("Data prepared successfully")

            self.data = dfTrain
            self.testData = dfTest
            self.N_CLUSTER = int(np.sqrt(len(dfTrain)))
            self.N_TEST_CLUSTER = int(np.sqrt(len(dfTest)))
            if self.viz == True:
                self.generate_count_plot(data=dfTrain)
                self.generate_count_plot(data=dfTest)
            self._save_data_frame(dfTrain, fileName="603_clean.csv")
            self._save_data_frame(dfTest, fileName="614_test.csv")

    def _create_tf_idf(self, train, test) -> tuple:
        """
        Create the TF-IDF vectorizer.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            max_features=10000,
            stop_words=list(self.stop_words),
        )

        tfidf_train = self.vectorizer.fit_transform(train).toarray()
        tfidf_test = self.vectorizer.transform(test).toarray()

        return (tfidf_train, tfidf_test)

    def _data_transformer(self, size: float = 0.3):
        """
        Splits, fits, and transforms the data for the classification.
        """
        from sklearn.model_selection import train_test_split

        if self.testPath is None:
            df = self._data_encoder(df=self.data, col="cluster")

            Train_X, Test_X, Train_Y, Test_Y = train_test_split(
                df["target"],
                df["cluster"],
                test_size=size,
                random_state=42,
                shuffle=True,
                stratify=None,
            )

            Train_X_Tfidf, Test_X_Tfidf = self._create_tf_idf(Train_X, Test_X)

            self.data = df
            self.train_x = Train_X
            self.test_x = Test_X
            self.train_y = Train_Y
            self.test_y = Test_Y
            self.train_x_vector = Train_X_Tfidf
            self.test_x_vector = Test_X_Tfidf

        else:
            dfTrain = self._data_encoder(df=self.data, col="cluster")
            dfTest = self._data_encoder(df=self.testData, col="cluster")

            Train_X_Tfidf, Test_X_Tfidf = self._create_tf_idf(
                dfTrain["target"], dfTest["target"]
            )

            self.data = dfTrain
            self.testData = dfTest
            self.train_x = dfTrain["target"]
            self.test_x = dfTest["target"]
            self.train_y = dfTrain["cluster"]
            self.test_y = dfTest["cluster"]
            self.train_x_vector = Train_X_Tfidf
            self.test_x_vector = Test_X_Tfidf

        if self.verbose:
            self._log("Data transformed successfully")
            if self.testPath is not None:
                print(dfTrain.head())
                print(dfTest.head())
            else:
                print(df.head())
        else:
            self._log("Data transformed successfully")

    def _data_encoder(self, df: DataFrame, col: str = "cluster") -> DataFrame:
        """
        Encode the data for the classification.
        """
        from sklearn.preprocessing import LabelEncoder

        df[col] = self.encoder.fit_transform(df[col])

        return df

    def _create_model(self):
        """
        Create the classification model.
        """

        self._data_transformer()

        self._run_nearest_neighbors(
            Train_X_Tfidf=self.train_x_vector,
            Test_X_Tfidf=self.test_x_vector,
            Train_Y=self.train_y,
            Test_Y=self.test_y,
            algo="brute",
            metric="cosine",
            weights="distance",
        )

        self._print_top_words_per_cluster(
            vectorizer=self.vectorizer, df=self.data, X=self.train_x_vector
        )

        sim, mask = self._calculate_similarity(
            X=self.train_x_vector,
        )

        self.generate_heat_map(
            arr=sim,
            mask=mask,
            fileName="heatmap_train.png",
        )

        self._print_sorted_similarities(sim_arr=sim)

        self._run_pca(X=self.train_x_vector, df=self.data, fileName="pca_train.png")

        self._run_naive_bayes(
            X_vector=self.train_x_vector,
            X_test_vector=self.test_x_vector,
            Y_test=self.test_y,
            Y_train=self.train_y,
        )

        if self.testPath is not None:
            self._print_top_words_per_cluster(
                vectorizer=self.vectorizer,
                df=self.testData,
                X=self.test_x_vector,
                train=False,
            )

            simTest, maskTest = self._calculate_similarity(
                X=self.test_x_vector,
            )

            self.generate_heat_map(
                arr=simTest,
                mask=maskTest,
                fileName="heatmap_test.png",
            )

            self._print_sorted_similarities(sim_arr=simTest)

            self._run_pca(
                X=self.test_x_vector, df=self.testData, fileName="pca_test.png"
            )

        self._log("Model created successfully")

    def _print_top_words_per_cluster(
        self, vectorizer, df: DataFrame, X: list, n=10, train: bool = True
    ):
        """
        This function returns the keywords for each centroid of the KNN clustering algorithm.

        Parameters
        ----------
        vectorizer : TfidfVectorizer
            The TF-IDF vectorizer.
        df : DataFrame
            The data frame.
        X : list
            The list of TF-IDF vectors.
        n : int, optional
            The number of keywords to return, by default 10
        train : bool, optional
            Whether the data is training or test data, by default True
        """
        import numpy as np

        data = pd.DataFrame(X).groupby(df["cluster"]).mean()
        terms = vectorizer.get_feature_names_out()

        print("\n")

        if train:
            self._log("Top keywords per cluster in training set:")
        else:
            self._log("Top keywords per cluster in test set:")

        for i, r in data.iterrows():
            self._log(
                "Cluster {} keywords: {}".format(
                    i, ", ".join([terms[t] for t in np.argsort(r)[-n:]])
                )
            )
            self._log("Mean TF-IDF score -> %0.4f" % np.max(r))
        print("\n")

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

    def _run_nearest_neighbors(
        self,
        Train_X_Tfidf: numpy.ndarray,
        Test_X_Tfidf: numpy.ndarray,
        Train_Y: Series,
        Test_Y: Series,
        algo: str,
        metric: str,
        weights: str,
        n_neighbors: int = 5,
    ):
        """
        Run the nearest neighbors algorithm.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algo,
            metric=metric,
            n_jobs=1,
            metric_params=None,
            leaf_size=30,
            p=2,
        )

        knn.fit(Train_X_Tfidf, Train_Y)

        predicted = knn.predict(Test_X_Tfidf)

        self._log("KNN Predictions -> %s" % predicted)

        self.testData["cluster"] = predicted

        self._save_data_frame(
            df=self.testData,
            fileName="614_pred.csv",
        )

        acc = accuracy_score(Test_Y, predicted)

        self._log("KNN Accuracy -> %0.4f" % (acc * 100))
        print("\n")

        best_cv = 2
        best_score = 0
        best_cv_index = 0
        y = []
        x = []
        max_range: int = 10

        if self.testPath is None:
            max_range = 7

        for i in range(2, max_range):
            scores = cross_val_score(knn, Train_X_Tfidf, Train_Y, cv=i)

            self._log(
                "Cross Validation Accuracy: %0.2f (+/- %0.2f)"
                % (scores.mean(), scores.std() * 2)
            )

            self._log("Number of predicted classes -> %s" % len(predicted))

            print("\n")

            if scores.mean() > best_score:
                best_cv = i
                best_score = scores.mean()
                best_cv_index = scores

            y.append(scores.mean())
            x.append(i)

        self._log(
            "Best Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (best_score, best_cv_index.std() * 2)
        )
        self._log("Best Cross Validation Number of Folds: %s" % best_cv)

        self.generate_cross_validation_plot(x, y)

    def _run_pca(
        self, X: numpy.ndarray, df: DataFrame, fileName: str = "pca_scatter.png"
    ):
        """
        Applies Principal Component Analysis (PCA) to the input data X and generates a scatter plot of the reduced features.

        Args:
            X (numpy.ndarray): The input data to be reduced.
            df (pandas.DataFrame): The dataframe containing the data to be plotted.

        Returns:
            None
        """
        import numpy as np
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)

        red_feat = pca.fit_transform(X)

        x = red_feat[:, 0]
        y = red_feat[:, 1]

        df["x"] = x
        df["y"] = y

        self.generate_scatter_plot(data=df, fileName=fileName)

        self._log("PCA run successfully")

    def _run_naive_bayes(self, X_vector, Y_train, X_test_vector, Y_test):
        """
        Run the naive bayes classification model.
        """
        from sklearn.naive_bayes import MultinomialNB

        Naive = MultinomialNB()

        Naive.fit(X_vector, Y_train)

        predictions_NB = Naive.predict(X_test_vector)

        if self.verbose:
            self._log("Naive Bayes run successfully")
            print(
                "Naive Bayes Accuracy Score -> ",
                accuracy_score(predictions_NB, Y_test) * 100,
            )
            print("\n")
        else:
            self._log("Naive Bayes run successfully")

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
            self._log("SVM run successfully")
            print(predictions_SVM)
        else:
            self._log("SVM run successfully")

    def _run_word_cloud_per_cluster(self, df: DataFrame):
        """
        Run the word cloud per cluster.
        """

        for i in sorted(df["cluster"].array.unique()):
            corpus = " ".join(list(df[df["cluster"] == i]["target"]))
            self.generate_word_cloud(corpus=corpus, fileName="cluster_%s.png" % i)

    def _save_data_frame(self, df: DataFrame, fileName: str):
        """
        Save the data frame as a CSV file.
        """
        df.to_csv(str(self.outputPath + fileName), index=False)

    def generate_word_cloud(self, corpus: str, fileName: str = "word_cloud.png"):
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
        if self.viz:
            plt.show()
        else:
            plt.savefig(str(self.outputPath + fileName))

    def generate_scatter_plot(
        self, data: DataFrame, fileName: str = "scatter_plot.png"
    ):
        """
        Generate the scatter plot for the data.
        """
        import seaborn as sns
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=data, x="x", y="y", hue="cluster", palette="tab10")
        if self.viz:
            plt.show()
        else:
            plt.savefig(str(self.outputPath + fileName))

    def generate_elbow_plot(self, X: numpy.ndarray):
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
        plt.figure(figsize=(10, 10))
        plt.plot(K, Sum_of_squared_distances, "bx-")
        plt.xlabel("k")
        plt.ylabel("Sum_of_squared_distances")
        plt.title("Elbow Method For Optimal k")
        if self.viz:
            plt.show()
        else:
            plt.savefig(str(self.outputPath + "kmeans_elbow_plot.png"))

    def generate_count_plot(self, data: DataFrame, countCol: str = "cluster"):
        """
        Generate a bar plot that sums the number of rows that share the same prefix value.

        Args:
            data (DataFrame): The data to plot.
            countCol (str, optional): The name of the column to count. Defaults to "cluster".
        """
        import seaborn as sns
        from matplotlib import pyplot as plt

        sns.countplot(x=countCol, data=data)
        if self.viz:
            plt.show()
        else:
            plt.savefig(str(self.outputPath + "count_plot.png"))

    def generate_cross_validation_plot(self, x: list, y: list):
        """
        Generate a cross validation plot that shows the accuracy of the model.

        Args:
            x (list): The list of x values.
            y (list): The list of y values.
        """
        import matplotlib.pyplot as plt

        plt.plot(x, y, "bx-")
        plt.xlabel("fold")
        plt.ylabel("Accuracy")
        plt.title("Cross Validation Accuracy over 10 folds")
        if self.viz:
            plt.show()
        else:
            plt.savefig(str(self.outputPath + "cross_validation_plot.png"))

    def _calculate_similarity(self, X) -> tuple[numpy.ndarray, list]:
        """
        Calculate the similarity between the documents.

        Args:
            X (numpy.ndarray): The array of documents.

        Returns:
            tuple: The similarity array and the mask to apply to the array.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        sim_arr = cosine_similarity(X)

        mask = np.triu(np.ones_like(sim_arr, dtype=bool))

        return sim_arr, mask

    def generate_heat_map(
        self, arr: numpy.ndarray, mask: list, fileName: str = "heat_map.png"
    ):
        """
        Generate a heat map that shows the correlation between the documents, using the name column of the data frame as the tick label.

        Args:
            arr (numpy.ndarray): The similarity array.
            mask (list): The mask to apply to the array. This is used to remove the diagonal and upper duplicate values.
        """
        import seaborn as sns
        from matplotlib import pyplot as plt

        plt.figure(figsize=(20, 20))
        sns.heatmap(
            arr,
            mask=mask,
            square=False,
            robust=True,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
        )
        if self.viz:
            plt.show()
        else:
            plt.savefig(str(self.outputPath + fileName))

    def _print_sorted_similarities(self, sim_arr, threshold=0) -> DataFrame:
        """
        Store the similarities between the documents in a data frame that is sorted by the similarity score in descending order. Removing the diagonal values.

        Args:
            sim_arr (numpy.ndarray): The similarity array.
            threshold (int, optional): The threshold to filter the similarity scores by. Defaults to 0.
        """
        import pandas as pd

        df = pd.DataFrame(sim_arr)
        df = df.stack().reset_index()
        df.columns = ["Document 1", "Document 2", "Similarity Score"]
        df = df.sort_values(by=["Similarity Score"], ascending=False)
        filtered_df = df[df["Document 1"] != df["Document 2"]]
        top = filtered_df[filtered_df["Similarity Score"] > threshold]

        print(top.head(10))

        return top

    def _log(self, text: str):
        """
        Append the text to the log file.

        Args:
            text (str): The text to append to the log file.
        """
        import time

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)

        if self.verbose:
            self.logger.info(f"\n[{current_time}]: {text}")
        else:
            self.logger.debug(f"\n[{current_time}]: {text}")

        with open(self.logPath, "a") as f:
            f.write(f"\n[{current_time}]: {text}")

    def run(self) -> None:
        """
        Run the classification model.
        """
        self.read()
        self.prepare(col="features")
        self._create_model()
        if self.viz == True:
            self._run_word_cloud_per_cluster(df=self.data)
            if self.testPath is not None:
                # TODO: Fix test data not having x and y columns
                # self.generate_scatter_plot(data=self.testData)
                pass
        if self.verbose == True:
            self._log("Successfully ran the classification model")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classification of text")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="input/603_trans_3.tsv",
        help="Path to the training dataset",
        required=True,
    )
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        default="input/614_trans.tsv",
        help="Path to the testing dataset",
    )
    parser.add_argument(
        "--download",
        help="Download the required libraries",
        required=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--verbose",
        help="Print the logs",
        required=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--sep", "-s", type=str, default="\t", help="Separator for the data"
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="output/",
        help="File name of the output file",
    )
    parser.add_argument(
        "--viz",
        help="Decide whether to visualize the EDA process and display classification visualization.",
        required=True,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    classify = Classify(
        path=args.path,
        toDownload=args.download,
        verbose=args.verbose,
        outputPath=args.out,
        visualize=args.viz,
        testPath=args.test,
    )
    classify.run()


if __name__ == "__main__":
    main()
