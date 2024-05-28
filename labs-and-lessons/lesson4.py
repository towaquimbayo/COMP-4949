def ex1():
    from nltk.tokenize import RegexpTokenizer

    reviewSentence = (
        "Parents need to know that this classic 1908 children's novel by L.M. "
        + "Montgomery remains a perennial favorite thanks to its memorable heroine: "
        + "irrepressible red-headed orphan Anne Shirley. Anne's adventures are full of "
        + "amusing (and occasionally mildly dangerous) scrapes, but she's quick to learn "
        + "from her mistakes and usually has only the best of intentions. Although Anne "
        + "gets her best friend drunk in one episode (it's an honest mistake), there's very "
        + "little here that's at all iffy for kids -- though younger readers might get a "
        + "bit bogged down in the many descriptions of Anne's Prince Edward Island, Canada, "
        + "home. A sad death may hit some kids hard, but the book's messages about the "
        + "importance of love, friendship, family, and ambition are worth it."
    )

    sentences = [reviewSentence]

    # -------------------------------------------------------------
    # Create lower case array of words with no punctuation.
    # -------------------------------------------------------------
    def createTokenizedArray(sentences):
        # Initialize tokenizer and empty array to store modified sentences.
        tokenizer = RegexpTokenizer(r"\w+")
        tokenizedArray = []
        for i in range(0, len(sentences)):
            # Convert sentence to lower case.
            sentence = sentences[i].lower()

            # Split sentence into array of words with no punctuation.
            words = tokenizer.tokenize(sentence)

            # Append word array to list.
            tokenizedArray.append(words)

        print(tokenizedArray)
        return tokenizedArray  # send modified contents back to calling function.

    tokenizedList = createTokenizedArray(sentences)


# Exercise 2, 3
def ex2():
    import nltk
    from nltk.tokenize import RegexpTokenizer

    reviewSentence = (
        "Parents need to know that this classic 1908 children's novel by L.M. "
        + "Montgomery remains a perennial favorite thanks to its memorable heroine: "
        + "irrepressible red-headed orphan Anne Shirley. Anne's adventures are full of "
        + "amusing (and occasionally mildly dangerous) scrapes, but she's quick to learn "
        + "from her mistakes and usually has only the best of intentions. Although Anne "
        + "gets her best friend drunk in one episode (it's an honest mistake), there's very "
        + "little here that's at all iffy for kids -- though younger readers might get a "
        + "bit bogged down in the many descriptions of Anne's Prince Edward Island, Canada, "
        + "home. A sad death may hit some kids hard, but the book's messages about the "
        + "importance of love, friendship, family, and ambition are worth it."
    )

    sentences = [reviewSentence]

    # -------------------------------------------------------------
    # Create lower case array of words with no punctuation.
    # -------------------------------------------------------------
    def createTokenizedArray(sentences):
        # Initialize tokenizer and empty array to store modified sentences.
        tokenizer = RegexpTokenizer(r"\w+")
        tokenizedArray = []
        for i in range(0, len(sentences)):
            # Convert sentence to lower case.
            sentence = sentences[i].lower()

            # Split sentence into array of words with no punctuation.
            words = tokenizer.tokenize(sentence)

            # Append word array to list.
            tokenizedArray.append(words)

        print("\nTokenized array:")
        print(tokenizedArray)
        return tokenizedArray  # send modified contents back to calling function.

    tokenizedList = createTokenizedArray(sentences)

    from nltk.corpus import stopwords

    # To get stop words.
    nltk.download("stopwords")

    # -------------------------------------------------------------
    # Create array of words with no punctuation or stop words.
    # -------------------------------------------------------------
    def removeStopWords(tokenList):
        stopWords = set(stopwords.words("english"))
        print("\nStop words:")
        print(stopWords)
        shorterSentences = []  # Declare empty array of sentences.

        for sentence in tokenList:
            shorterSentence = []  # Declare empty array of words in single sentence.
            for word in sentence:
                if word not in stopWords:
                    # Remove leading and trailing spaces.
                    word = word.strip()

                    # Ignore single character words and digits.
                    if len(word) > 1 and word.isdigit() == False:
                        # Add remaining words to list.
                        shorterSentence.append(word)
            shorterSentences.append(shorterSentence)
        return shorterSentences

    sentenceArrays = removeStopWords(tokenizedList)
    print("\nTokenized array with stop words removed:")
    print(sentenceArrays)


def ex4():
    from nltk.tokenize import RegexpTokenizer

    sentence1 = (
        "Despite its fresh perspective, Banks's Charlie's Angels update "
        + "fails to capture the energy or style that made it the beloved phenomenon it is."
    )

    sentence2 = (
        "This 2019 Charlie's Angels is stupefyingly entertaining and "
        + "hilarious. It is a stylish alternative to the current destructive blockbusters."
    )

    sentences = [sentence1, sentence2]

    # -------------------------------------------------------------
    # Create lower case array of words with no punctuation.
    # -------------------------------------------------------------
    def createTokenizedArray(sentences):
        # Initialize tokenizer and empty array to store modified sentences.
        tokenizer = RegexpTokenizer(r"\w+")
        tokenizedArray = []
        for i in range(0, len(sentences)):
            # Convert sentence to lower case.
            sentence = sentences[i].lower()

            # Split sentence into array of words with no punctuation.
            words = tokenizer.tokenize(sentence)

            # Append word array to list.
            tokenizedArray.append(words)

        print("\nTokenized array:")
        print(tokenizedArray)
        return tokenizedArray  # send modified contents back to calling function.

    tokenizedList = createTokenizedArray(sentences)

    from nltk.corpus import stopwords

    # -------------------------------------------------------------
    # Create array of words with no punctuation or stop words.
    # -------------------------------------------------------------
    def removeStopWords(tokenList):
        stopWords = set(stopwords.words("english"))
        shorterSentences = []  # Declare empty array of sentences.

        for sentence in tokenList:
            shorterSentence = []  # Declare empty array of words in single sentence.
            for word in sentence:
                if word not in stopWords:
                    # Remove leading and trailing spaces.
                    word = word.strip()

                    # Ignore single character words and digits.
                    if len(word) > 1 and word.isdigit() == False:
                        # Add remaining words to list.
                        shorterSentence.append(word)
            shorterSentences.append(shorterSentence)
        return shorterSentences

    sentenceArrays = removeStopWords(tokenizedList)
    print("\nTokenized array with stop words removed:")
    print(sentenceArrays)

    def modifiedStopWords(sentenceLists):
        updatedList = []
        customStopWords = ["charlie", "angels"]
        counter = 0
        # Loop through list of sentences (there are two sentences)
        # Loop through words in the sentence
        # Check for words that are not in customStopWords
        # append to list if words are not in customStopWords
        for sentence in sentenceLists:
            updatedSentence = []
            for word in sentence:
                if word not in customStopWords:
                    updatedSentence.append(word)
            updatedList.append(updatedSentence)
        return updatedList

    output = modifiedStopWords(sentenceArrays)
    print("\nThe final answer is:")
    print(output)


def ex5():
    from nltk.tokenize import RegexpTokenizer

    reviewSentence = (
        "Parents need to know that this classic 1908 children's novel by L.M. "
        + "Montgomery remains a perennial favorite thanks to its memorable heroine: "
        + "irrepressible red-headed orphan Anne Shirley. Anne's adventures are full of "
        + "amusing (and occasionally mildly dangerous) scrapes, but she's quick to learn "
        + "from her mistakes and usually has only the best of intentions. Although Anne "
        + "gets her best friend drunk in one episode (it's an honest mistake), there's very "
        + "little here that's at all iffy for kids -- though younger readers might get a "
        + "bit bogged down in the many descriptions of Anne's Prince Edward Island, Canada, "
        + "home. A sad death may hit some kids hard, but the book's messages about the "
        + "importance of love, friendship, family, and ambition are worth it."
    )

    sentences = [reviewSentence]

    # -------------------------------------------------------------
    # Create lower case array of words with no punctuation.
    # -------------------------------------------------------------
    def createTokenizedArray(sentences):
        # Initialize tokenizer and empty array to store modified sentences.
        tokenizer = RegexpTokenizer(r"\w+")
        tokenizedArray = []
        for i in range(0, len(sentences)):
            # Convert sentence to lower case.
            sentence = sentences[i].lower()

            # Split sentence into array of words with no punctuation.
            words = tokenizer.tokenize(sentence)

            # Append word array to list.
            tokenizedArray.append(words)

        print("\nTokenized array:")
        print(tokenizedArray)
        return tokenizedArray  # send modified contents back to calling function.

    tokenizedList = createTokenizedArray(sentences)

    from nltk.corpus import stopwords

    # -------------------------------------------------------------
    # Create array of words with no punctuation or stop words.
    # -------------------------------------------------------------
    def removeStopWords(tokenList):
        stopWords = set(stopwords.words("english"))
        shorterSentences = []  # Declare empty array of sentences.

        for sentence in tokenList:
            shorterSentence = []  # Declare empty array of words in single sentence.
            for word in sentence:
                if word not in stopWords:
                    # Remove leading and trailing spaces.
                    word = word.strip()

                    # Ignore single character words and digits.
                    if len(word) > 1 and word.isdigit() == False:
                        # Add remaining words to list.
                        shorterSentence.append(word)
            shorterSentences.append(shorterSentence)
        return shorterSentences

    sentenceArrays = removeStopWords(tokenizedList)
    print("\nTokenized array with stop words removed:")
    print(sentenceArrays)

    from nltk.stem import PorterStemmer

    # -------------------------------------------------------------
    # Removes suffixes and rebuids the sentences.
    # -------------------------------------------------------------
    def stemWords(sentenceArrays):
        ps = PorterStemmer()
        stemmedSentences = []
        for sentenceArray in sentenceArrays:
            stemmedArray = []  # Declare empty array of words.
            for word in sentenceArray:
                stemmedArray.append(ps.stem(word))  # Add stemmed word.

            # Convert array back to sentence of stemmed words.
            delimeter = " "
            sentence = delimeter.join(stemmedArray)

            # Append stemmed sentence to list of sentences.
            stemmedSentences.append(sentence)
        return stemmedSentences

    stemmedSentences = stemWords(sentenceArrays)
    print("\nStemmed sentences:")
    print(stemmedSentences)


# Exercise 6, 7
def ex6():
    from sklearn.feature_extraction.text import CountVectorizer

    # -------------------------------------------------------------
    # Creates a matrix of word vectors.
    # -------------------------------------------------------------
    def vectorizeList(stemmedList):
        # cv  = CountVectorizer(binary=True, ngram_range=(1, 4))
        cv = CountVectorizer(binary=True)

        cv.fit(stemmedList)
        features = cv.get_feature_names_out()
        print("\nFeatures: " + str(features))
        X = cv.transform(stemmedList)
        print("\nNumber vector size: " + str(X.shape))
        return X

    sampleSentences = ["the sky is blue", "the day is bright"]
    sampleOutput = vectorizeList(sampleSentences)

    # Assigns numbers to words.
    print("\nTransformed words: \n" + str(sampleOutput))

    # Shows number of times each word appears in the list.
    print("Encoded list: \n" + str(sampleOutput.toarray()))


def ex8():
    from nltk.tokenize import RegexpTokenizer

    sentence1 = (
        "Despite its fresh perspective, Banks's Charlie's Angels update "
        + "fails to capture the energy or style that made it the beloved phenomenon it is."
    )

    sentence2 = (
        "This 2019 Charlie's Angels is stupefyingly entertaining and "
        + "hilarious. It is a stylish alternative to the current destructive blockbusters."
    )

    sentences = [sentence1, sentence2]

    # -------------------------------------------------------------
    # Create lower case array of words with no punctuation.
    # -------------------------------------------------------------
    def createTokenizedArray(sentences):
        # Initialize tokenizer and empty array to store modified sentences.
        tokenizer = RegexpTokenizer(r"\w+")
        tokenizedArray = []
        for i in range(0, len(sentences)):
            # Convert sentence to lower case.
            sentence = sentences[i].lower()

            # Split sentence into array of words with no punctuation.
            words = tokenizer.tokenize(sentence)

            # Append word array to list.
            tokenizedArray.append(words)

        print(tokenizedArray)
        return tokenizedArray  # send modified contents back to calling function.

    tokenizedList = createTokenizedArray(sentences)

    import nltk
    from nltk.corpus import stopwords

    # To get stop words.
    nltk.download("stopwords")

    # -------------------------------------------------------------
    # Create array of words with no punctuation or stop words.
    # -------------------------------------------------------------
    def removeStopWords(tokenList):
        stopWords = set(stopwords.words("english"))
        shorterSentences = []  # Declare empty array of sentences.

        for sentence in tokenList:
            shorterSentence = []  # Declare empty array of words in single sentence.
            for word in sentence:
                if word not in stopWords:
                    # Remove leading and trailing spaces.
                    word = word.strip()

                    # Ignore single character words and digits.
                    if len(word) > 1 and word.isdigit() == False:
                        # Add remaining words to list.
                        shorterSentence.append(word)
            shorterSentences.append(shorterSentence)
        return shorterSentences

    sentenceArrays = removeStopWords(tokenizedList)
    print(sentenceArrays)

    from nltk.stem import PorterStemmer

    # -------------------------------------------------------------
    # Removes suffixes and rebuids the sentences.
    # -------------------------------------------------------------
    def stemWords(sentenceArrays):
        ps = PorterStemmer()
        stemmedSentences = []
        for sentenceArray in sentenceArrays:
            stemmedArray = []  # Declare empty array of words.
            for word in sentenceArray:
                stemmedArray.append(ps.stem(word))  # Add stemmed word.

            # Convert array back to sentence of stemmed words.
            delimeter = " "
            sentence = delimeter.join(stemmedArray)

            # Append stemmed sentence to list of sentences.
            stemmedSentences.append(sentence)
        return stemmedSentences

    stemmedSentences = stemWords(sentenceArrays)
    print(stemmedSentences)

    from sklearn.feature_extraction.text import CountVectorizer

    # -------------------------------------------------------------
    # Creates a matrix of word vectors.
    # -------------------------------------------------------------
    def vectorizeList(stemmedList):
        # cv  = CountVectorizer(binary=True, ngram_range=(1, 4))
        cv = CountVectorizer(binary=True)

        cv.fit(stemmedList)
        features = cv.get_feature_names_out()
        print("\nFeatures: " + str(features))
        X = cv.transform(stemmedList)
        print("\nNumber vector size: " + str(X.shape))
        return X

    vectorizedSentences = vectorizeList(stemmedSentences)

    # Assigns numbers to words.
    print("\nTransformed words: \n" + str(vectorizedSentences))

    # Shows number of times each word appears in the list.
    print("Encoded list: \n" + str(vectorizedSentences.toarray()))


# Exercise 9, 10
def ex9():
    import nltk
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from nltk.tokenize import RegexpTokenizer
    from sklearn.metrics import accuracy_score

    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    import math

    # To get stop words.
    nltk.download("stopwords")

    # -------------------------------------------------------------
    # Create lower case array of words with no punctuation.
    # -------------------------------------------------------------
    def createTokenizedArray(sentences):
        # Initialize tokenizer and empty array to store modified sentences.
        tokenizer = RegexpTokenizer(r"\w+")
        tokenizedArray = []
        for i in range(0, len(sentences)):
            # Convert sentence to lower case.
            sentence = sentences[i].lower()

            # Split sentence into array of words with no punctuation.
            words = tokenizer.tokenize(sentence)

            # Append word array to list.
            tokenizedArray.append(words)

        print(tokenizedArray)
        return tokenizedArray  # send modified contents back to calling function.

    # -------------------------------------------------------------
    # Create array of words with no punctuation or stop words.
    # -------------------------------------------------------------
    def removeStopWords(tokenList):
        stopWords = set(stopwords.words("english"))
        shorterSentences = []  # Declare empty array of sentences.

        for sentence in tokenList:
            shorterSentence = []  # Declare empty array of words in single sentence.
            for word in sentence:
                if word not in stopWords:
                    # Remove leading and trailing spaces.
                    word = word.strip()

                    # Ignore single character words and digits.
                    if len(word) > 1 and word.isdigit() == False:
                        # Add remaining words to list.
                        shorterSentence.append(word)
            shorterSentences.append(shorterSentence)
        return shorterSentences

    # -------------------------------------------------------------
    # Removes suffixes and rebuids the sentences.
    # -------------------------------------------------------------
    def stemWords(sentenceArrays):
        ps = PorterStemmer()
        stemmedSentences = []
        for sentenceArray in sentenceArrays:
            stemmedArray = []  # Declare empty array of words.
            for word in sentenceArray:
                stemmedArray.append(ps.stem(word))  # Add stemmed word.

            # Convert array back to sentence of stemmed words.
            delimeter = " "
            sentence = delimeter.join(stemmedArray)

            # Append stemmed sentence to list of sentences.
            stemmedSentences.append(sentence)
        return stemmedSentences

    # -------------------------------------------------------------
    # Creates a matrix of word vectors.
    # -------------------------------------------------------------
    def vectorizeList(stemmedList):
        # cv  = CountVectorizer(binary=True, ngram_range=(1, 4))
        cv = CountVectorizer(binary=True)

        cv.fit(stemmedList)
        X = cv.transform(stemmedList)
        print("\nNumber vector size: " + str(X.shape))
        return X

    import pandas as pd
    from sklearn import metrics
    from sklearn.metrics import classification_report

    # -------------------------------------------------------------
    # Build model and predict scores.
    #
    # Parameters:
    # X         - X contains the stemmed and vectorized sentences.
    # target    - The target is the known rating (0 to 4).

    # Returns X_test, y_test, and y_predicted values.
    # -------------------------------------------------------------
    def modelAndPredict(X, target):
        # Create training set with 75% of data and test set with 25% of data.
        X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.75)

        # Build the model with the training data.
        model = LogisticRegression(solver="newton-cg").fit(X_train, y_train)

        # Predict target values.
        y_prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_prediction)

        print("\n\n*** The accuracy score is: " + str(accuracy))

        print(classification_report(y_test, y_prediction))

        rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
        print("RMSE: " + str(rmse2))

        # Your Python functions can return multiple values.
        return X_test, y_test, y_prediction

    # Read in the file.
    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    CLEAN_DATA = "cleanedMovieReviews.tsv"
    df = pd.read_csv(
        PATH + CLEAN_DATA,
        skiprows=1,
        sep="\t",
        names=("PhraseId", "SentenceId", "Phrase", "Sentiment"),
    )

    # Prepare the data.
    df["PhraseAdjusted"] = createTokenizedArray(df["Phrase"])
    df["PhraseAdjusted"] = removeStopWords(df["PhraseAdjusted"])
    df["PhraseAdjusted"] = stemWords(df["PhraseAdjusted"])
    vectorizedList = vectorizeList(df["PhraseAdjusted"])

    # Get predictions and scoring data.
    # Target is the rating that we want to predict.
    X_test, y_test, y_predicted = modelAndPredict(vectorizedList, df["Sentiment"])

    from sklearn import metrics

    # Draw the confusion matrix.
    def showConfusionMatrix(y_test, y_predicted):
        # You can print a simple confusion matrix with no formatting – this is easiest.
        cm = metrics.confusion_matrix(y_test.values, y_predicted)
        print(cm)

    showConfusionMatrix(y_test, y_predicted)

    from collections import Counter
    from nltk.util import ngrams

    def generateWordList(wordDf, scoreStart, scoreEnd, n_gram_size):
        resultDf = wordDf[
            (wordDf["Sentiment"] >= scoreStart) & (wordDf["Sentiment"] <= scoreEnd)
        ]

        sentences = [sentence.split() for sentence in resultDf["PhraseAdjusted"]]
        wordArray = []
        for i in range(0, len(sentences)):
            wordArray += sentences[i]

        counterList = Counter(ngrams(wordArray, n_gram_size)).most_common(80)

        print("\n***N-Gram")
        for i in range(0, len(counterList)):
            print("Occurrences: ", str(counterList[i][1]), end=" ")
            delimiter = " "
            print("  N-Gram: ", delimiter.join(counterList[i][0]))

        return counterList

    # Create two column matrix.
    dfSub = df[["Sentiment", "PhraseAdjusted"]]
    SCORE_RANGE_START = 0
    SCORE_RANGE_END = 1
    SIZE = 2
    counterList = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)

    SIZE = 3
    counterList = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)


def ex11():
    import nltk
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from nltk.tokenize import RegexpTokenizer
    from sklearn.metrics import accuracy_score

    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    import math

    # To get stop words.
    nltk.download("stopwords")

    # -------------------------------------------------------------
    # Create lower case array of words with no punctuation.
    # -------------------------------------------------------------
    def createTokenizedArray(sentences):
        # Initialize tokenizer and empty array to store modified sentences.
        tokenizer = RegexpTokenizer(r"\w+")
        tokenizedArray = []
        for i in range(0, len(sentences)):
            # Convert sentence to lower case.
            sentence = sentences[i].lower()

            # Split sentence into array of words with no punctuation.
            words = tokenizer.tokenize(sentence)

            # Append word array to list.
            tokenizedArray.append(words)

        print(tokenizedArray)
        return tokenizedArray  # send modified contents back to calling function.

    # -------------------------------------------------------------
    # Create array of words with no punctuation or stop words.
    # -------------------------------------------------------------
    def removeStopWords(tokenList):
        stopWords = set(stopwords.words("english"))
        shorterSentences = []  # Declare empty array of sentences.

        for sentence in tokenList:
            shorterSentence = []  # Declare empty array of words in single sentence.
            for word in sentence:
                if word not in stopWords:
                    # Remove leading and trailing spaces.
                    word = word.strip()

                    # Ignore single character words and digits.
                    if len(word) > 1 and word.isdigit() == False:
                        # Add remaining words to list.
                        shorterSentence.append(word)
            shorterSentences.append(shorterSentence)
        return shorterSentences

    # -------------------------------------------------------------
    # Removes suffixes and rebuids the sentences.
    # -------------------------------------------------------------
    def stemWords(sentenceArrays):
        ps = PorterStemmer()
        stemmedSentences = []
        for sentenceArray in sentenceArrays:
            stemmedArray = []  # Declare empty array of words.
            for word in sentenceArray:
                stemmedArray.append(ps.stem(word))  # Add stemmed word.

            # Convert array back to sentence of stemmed words.
            delimeter = " "
            sentence = delimeter.join(stemmedArray)

            # Append stemmed sentence to list of sentences.
            stemmedSentences.append(sentence)
        return stemmedSentences

    # -------------------------------------------------------------
    # Creates a matrix of word vectors.
    # -------------------------------------------------------------
    def vectorizeList(stemmedList):
        # cv  = CountVectorizer(binary=True, ngram_range=(1, 4))
        cv = CountVectorizer(binary=True)

        cv.fit(stemmedList)
        X = cv.transform(stemmedList)
        print("\nNumber vector size: " + str(X.shape))
        return X

    import pandas as pd
    from sklearn import metrics
    from sklearn.metrics import classification_report

    # -------------------------------------------------------------
    # Build model and predict scores.
    #
    # Parameters:
    # X         - X contains the stemmed and vectorized sentences.
    # target    - The target is the known rating (0 to 4).

    # Returns X_test, y_test, and y_predicted values.
    # -------------------------------------------------------------
    def modelAndPredict(X, target):
        # Create training set with 75% of data and test set with 25% of data.
        X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.75)

        # Build the model with the training data.
        model = LogisticRegression(solver="newton-cg").fit(X_train, y_train)

        # Predict target values.
        y_prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_prediction)

        print("\n\n*** The accuracy score is: " + str(accuracy))

        print(classification_report(y_test, y_prediction))

        rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
        print("RMSE: " + str(rmse2))

        # Your Python functions can return multiple values.
        return X_test, y_test, y_prediction

    # Read in the file.
    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    CLEAN_DATA = "cleanedMovieReviews.tsv"
    df = pd.read_csv(
        PATH + CLEAN_DATA,
        skiprows=1,
        sep="\t",
        names=("PhraseId", "SentenceId", "Phrase", "Sentiment"),
    )

    # Prepare the data.
    df["PhraseAdjusted"] = createTokenizedArray(df["Phrase"])
    df["PhraseAdjusted"] = removeStopWords(df["PhraseAdjusted"])
    df["PhraseAdjusted"] = stemWords(df["PhraseAdjusted"])
    vectorizedList = vectorizeList(df["PhraseAdjusted"])

    # Get predictions and scoring data.
    # Target is the rating that we want to predict.
    X_test, y_test, y_predicted = modelAndPredict(vectorizedList, df["Sentiment"])

    from sklearn import metrics

    # Draw the confusion matrix.
    def showConfusionMatrix(y_test, y_predicted):
        # You can print a simple confusion matrix with no formatting – this is easiest.
        cm = metrics.confusion_matrix(y_test.values, y_predicted)
        print(cm)

    showConfusionMatrix(y_test, y_predicted)

    from collections import Counter
    from nltk.util import ngrams

    def generateWordList(wordDf, scoreStart, scoreEnd, n_gram_size):
        resultDf = wordDf[
            (wordDf["Sentiment"] >= scoreStart) & (wordDf["Sentiment"] <= scoreEnd)
        ]

        sentences = [sentence.split() for sentence in resultDf["PhraseAdjusted"]]
        wordArray = []
        for i in range(0, len(sentences)):
            wordArray += sentences[i]

        counterList = Counter(ngrams(wordArray, n_gram_size)).most_common(80)

        print("\n***N-Gram")
        for i in range(0, len(counterList)):
            print("Occurrences: ", str(counterList[i][1]), end=" ")
            delimiter = " "
            print("  N-Gram: ", delimiter.join(counterList[i][0]))

        return counterList

    # Create two column matrix.
    dfSub = df[["Sentiment", "PhraseAdjusted"]]
    SCORE_RANGE_START = 4
    SCORE_RANGE_END = 4
    SIZE = 1
    counterList = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)

    SIZE = 3
    counterList = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)

    # Create DataFrame.
    simpleDataSet = {"PhraseAdjusted": ["the sky is blue"], "Sentiment": [4]}
    dfSimple = pd.DataFrame(simpleDataSet, columns=["Sentiment", "PhraseAdjusted"])
    SIZE = 2
    newNGrams = generateWordList(dfSimple, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)
    print(newNGrams)


def ex12():
    import nltk
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from nltk.tokenize import RegexpTokenizer
    from sklearn.metrics import accuracy_score

    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    import math

    # To get stop words.
    nltk.download("stopwords")

    # -------------------------------------------------------------
    # Create lower case array of words with no punctuation.
    # -------------------------------------------------------------
    def createTokenizedArray(sentences):
        # Initialize tokenizer and empty array to store modified sentences.
        tokenizer = RegexpTokenizer(r"\w+")
        tokenizedArray = []
        for i in range(0, len(sentences)):
            # Convert sentence to lower case.
            sentence = sentences[i].lower()

            # Split sentence into array of words with no punctuation.
            words = tokenizer.tokenize(sentence)

            # Append word array to list.
            tokenizedArray.append(words)

        print(tokenizedArray)
        return tokenizedArray  # send modified contents back to calling function.

    # -------------------------------------------------------------
    # Create array of words with no punctuation or stop words.
    # -------------------------------------------------------------
    def removeStopWords(tokenList):
        stopWords = set(stopwords.words("english"))
        shorterSentences = []  # Declare empty array of sentences.

        for sentence in tokenList:
            shorterSentence = []  # Declare empty array of words in single sentence.
            for word in sentence:
                if word not in stopWords:
                    # Remove leading and trailing spaces.
                    word = word.strip()

                    # Ignore single character words and digits.
                    if len(word) > 1 and word.isdigit() == False:
                        # Add remaining words to list.
                        shorterSentence.append(word)
            shorterSentences.append(shorterSentence)
        return shorterSentences

    # -------------------------------------------------------------
    # Removes suffixes and rebuids the sentences.
    # -------------------------------------------------------------
    def stemWords(sentenceArrays):
        ps = PorterStemmer()
        stemmedSentences = []
        for sentenceArray in sentenceArrays:
            stemmedArray = []  # Declare empty array of words.
            for word in sentenceArray:
                stemmedArray.append(ps.stem(word))  # Add stemmed word.

            # Convert array back to sentence of stemmed words.
            delimeter = " "
            sentence = delimeter.join(stemmedArray)

            # Append stemmed sentence to list of sentences.
            stemmedSentences.append(sentence)
        return stemmedSentences

    # -------------------------------------------------------------
    # Creates a matrix of word vectors.
    # -------------------------------------------------------------
    def vectorizeList(stemmedList):
        # cv  = CountVectorizer(binary=True, ngram_range=(1, 4))
        cv = CountVectorizer(binary=True)

        cv.fit(stemmedList)
        X = cv.transform(stemmedList)
        print("\nNumber vector size: " + str(X.shape))
        return X

    import pandas as pd
    from sklearn import metrics
    from sklearn.metrics import classification_report

    # -------------------------------------------------------------
    # Build model and predict scores.
    #
    # Parameters:
    # X         - X contains the stemmed and vectorized sentences.
    # target    - The target is the known rating (0 to 4).

    # Returns X_test, y_test, and y_predicted values.
    # -------------------------------------------------------------
    def modelAndPredict(X, target):
        # Create training set with 75% of data and test set with 25% of data.
        X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.75)

        # Build the model with the training data.
        model = LogisticRegression(solver="newton-cg").fit(X_train, y_train)

        # Predict target values.
        y_prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_prediction)

        print("\n\n*** The accuracy score is: " + str(accuracy))

        print(classification_report(y_test, y_prediction))

        rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
        print("RMSE: " + str(rmse2))

        # Your Python functions can return multiple values.
        return X_test, y_test, y_prediction

    # Read in the file.
    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    CLEAN_DATA = "tripadvisor_hotel_reviews.csv"
    df = pd.read_csv(
        PATH + CLEAN_DATA,
        skiprows=1,
        names=("Review", "Rating"),
    )

    # Prepare the data.
    df["PhraseAdjusted"] = createTokenizedArray(df["Review"])
    df["PhraseAdjusted"] = removeStopWords(df["PhraseAdjusted"])
    df["PhraseAdjusted"] = stemWords(df["PhraseAdjusted"])
    vectorizedList = vectorizeList(df["PhraseAdjusted"])

    # Get predictions and scoring data.
    # Target is the rating that we want to predict.
    X_test, y_test, y_predicted = modelAndPredict(vectorizedList, df["Rating"])

    from sklearn import metrics

    # Draw the confusion matrix.
    def showConfusionMatrix(y_test, y_predicted):
        # You can print a simple confusion matrix with no formatting – this is easiest.
        cm = metrics.confusion_matrix(y_test.values, y_predicted)
        print(cm)

    showConfusionMatrix(y_test, y_predicted)

    from collections import Counter
    from nltk.util import ngrams

    def generateWordList(wordDf, scoreStart, scoreEnd, n_gram_size):
        resultDf = wordDf[
            (wordDf["Rating"] >= scoreStart) & (wordDf["Rating"] <= scoreEnd)
        ]

        sentences = [sentence.split() for sentence in resultDf["PhraseAdjusted"]]
        wordArray = []
        for i in range(0, len(sentences)):
            wordArray += sentences[i]

        counterList = Counter(ngrams(wordArray, n_gram_size)).most_common(15)

        print("\n***N-Gram")
        for i in range(0, len(counterList)):
            print("Occurrences: ", str(counterList[i][1]), end=" ")
            delimiter = " "
            print("  N-Gram: ", delimiter.join(counterList[i][0]))

        return counterList

    # Create two column matrix.
    dfSub = df[["Rating", "PhraseAdjusted"]]
    SCORE_RANGE_START = 1
    SCORE_RANGE_END = 1
    SIZE = 3
    counterList = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)

    SCORE_RANGE_START = 5
    SCORE_RANGE_END = 5
    SIZE = 4
    counterList = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)


def main():
    # ex1()
    # ex2()
    # ex4()
    # ex5()
    # ex6()
    # ex8()
    # ex9()
    # ex11()
    ex12()


if __name__ == "__main__":
    main()
