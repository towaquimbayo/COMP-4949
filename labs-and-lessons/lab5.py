def ex1():
    import numpy as np
    import pandas as pd
    import torch
    import transformers as ppb
    import warnings

    warnings.filterwarnings("ignore")

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    PATH = "/Users/elber/Documents/COMP 4949 - Datasets/"
    FILE = "movie_reviewsBERT.csv"
    batch_1 = pd.read_csv(PATH + FILE, delimiter=",", header=None)

    print(batch_1.shape)
    ROW = 1
    print("Review 1st column: " + batch_1.iloc[ROW][0])
    print("Rating 2nd column: " + str(batch_1.iloc[ROW][1]))

    # Show counts for review scores.
    print("** Showing review counts")
    print(batch_1[1].value_counts())

    # Load pretrained models.
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel,
        ppb.DistilBertTokenizer,
        "distilbert-base-uncased",
    )

    ## Want BERT instead of distilBERT? Uncomment the following line:
    # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Tokenize the sentences.
    tokenized = batch_1[0].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True))
    )
    print("\n****************** Tokenized reviews ")
    print(tokenized)
    print(tokenized.values)
    print("******************")

    # For processing we convert to 2D array.
    max_len = 0

    # Get maximum number of tokens (get biggest sentence).
    print("\nGetting maximum number of tokens in a sentence")
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    print("Most tokens in a review (max_len): " + str(max_len))

    # Add padding
    print("------------")
    print("Padded so review arrays as same size: ")
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    print("These are the padded reviews:")
    print(padded)
    print("This is the last padded sentence:")
    LAST_INDEX = len(batch_1) - 1
    print(padded[LAST_INDEX])
    print("\n------------")
    print("Attention mask tells BERT to ignore the padding.")

    # Sending padded data to BERT would slightly confuse it
    # so create a mask to tell it to ignore the padding.
    attention_mask = np.where(padded != 0, 1, 0)
    print(attention_mask.shape)
    print(attention_mask)
    print(attention_mask[LAST_INDEX])
    print("=============")

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    print("Input ids which are padded reviews in torch tensor format:")
    print(input_ids)
    print("Attention mask in torch tensor format:")
    print(attention_mask)
    print("++++++++++++++")


# Exercise 2, 3, 4, 5, 6. 7
def ex2():
    import numpy as np
    import pandas as pd
    import torch
    import transformers as ppb
    import warnings

    warnings.filterwarnings("ignore")

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    dfEx = pd.DataFrame(columns=[0, 1])
    dfEx = dfEx._append(
        {0: "This brilliant movie is jaw-dropping.", 1: 1}, ignore_index=True
    )
    dfEx = dfEx._append({0: "This movie is awful.", 1: 0}, ignore_index=True)

    print(dfEx.shape)
    ROW = 1
    print("Review 1st column: " + dfEx.iloc[ROW][0])
    print("Rating 2nd column: " + str(dfEx.iloc[ROW][1]))

    # Show counts for review scores.
    print("** Showing review counts")
    print(dfEx[1].value_counts())

    # Load pretrained models.
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel,
        ppb.DistilBertTokenizer,
        "distilbert-base-uncased",
    )

    ## Want BERT instead of distilBERT? Uncomment the following line:
    # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Tokenize the sentences.
    tokenized = dfEx[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    print("\n****************** Tokenized reviews ")
    print(tokenized)
    print(tokenized.values)
    print("******************")

    # For processing we convert to 2D array.
    max_len = 0

    # Get maximum number of tokens (get biggest sentence).
    print("\nGetting maximum number of tokens in a sentence")
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    print("Most tokens in a review (max_len): " + str(max_len))

    # Add padding
    print("------------")
    print("Padded so review arrays as same size: ")
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    print("These are the padded reviews:")
    print(padded)
    print("This is the last padded sentence:")
    LAST_INDEX = len(dfEx) - 1
    print(padded[LAST_INDEX])
    print("\n------------")
    print("Attention mask tells BERT to ignore the padding.")

    # Sending padded data to BERT would slightly confuse it
    # so create a mask to tell it to ignore the padding.
    attention_mask = np.where(padded != 0, 1, 0)
    print(attention_mask.shape)
    print(attention_mask)
    print(attention_mask[LAST_INDEX])
    print("=============")

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    print("Input ids which are padded reviews in torch tensor format:")
    print(input_ids)
    print("Attention mask in torch tensor format:")
    print(attention_mask)
    print("++++++++++++++")


def ex8():
    import numpy as np
    import pandas as pd
    import torch
    import transformers as ppb
    import warnings

    warnings.filterwarnings("ignore")

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    dfEx = pd.DataFrame(columns=[0, 1])
    dfEx = dfEx._append(
        {0: "This brilliant movie is jaw-dropping.", 1: 1}, ignore_index=True
    )
    dfEx = dfEx._append({0: "This movie is awful.", 1: 0}, ignore_index=True)

    print(dfEx.shape)
    ROW = 1
    print("Review 1st column: " + dfEx.iloc[ROW][0])
    print("Rating 2nd column: " + str(dfEx.iloc[ROW][1]))

    # Show counts for review scores.
    print("** Showing review counts")
    print(dfEx[1].value_counts())

    # Load pretrained models.
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel,
        ppb.DistilBertTokenizer,
        "distilbert-base-uncased",
    )

    ## Want BERT instead of distilBERT? Uncomment the following line:
    # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Tokenize the sentences.
    tokenized = dfEx[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    print("\n****************** Tokenized reviews ")
    print(tokenized)
    print(tokenized.values)
    print("******************")

    # For processing we convert to 2D array.
    max_len = 0

    # Get maximum number of tokens (get biggest sentence).
    print("\nGetting maximum number of tokens in a sentence")
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    print("Most tokens in a review (max_len): " + str(max_len))

    # Add padding
    print("------------")
    print("Padded so review arrays as same size: ")
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    print("These are the padded reviews:")
    print(padded)
    print("This is the last padded sentence:")
    LAST_INDEX = len(dfEx) - 1
    print(padded[LAST_INDEX])
    print("\n------------")
    print("Attention mask tells BERT to ignore the padding.")

    # Sending padded data to BERT would slightly confuse it
    # so create a mask to tell it to ignore the padding.
    attention_mask = np.where(padded != 0, 1, 0)
    print(attention_mask.shape)
    print(attention_mask)
    print(attention_mask[LAST_INDEX])
    print("=============")

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    print("Input ids which are padded reviews in torch tensor format:")
    print(input_ids)
    print("Attention mask in torch tensor format:")
    print(attention_mask)
    print("++++++++++++++")

    # The model() function runs our sentences through BERT. The results of the
    # processing will be returned into last_hidden_states.
    print("BERT model transforms tokens and attention mask tensors into features ")
    print("for logistic regression.")
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    # Save the features and these will become
    # features of the logistic regression model.
    features = last_hidden_states[0][:, 0, :].numpy()

    print("Let's see the features: ")
    print(features)
    print("Length of features: " + str(len(features)))
    print("-------------------------")


def ex9():
    import tensorflow as tf

    # pip install tensorflow_text is needed.
    import tensorflow_text as text

    reloaded_model = tf.saved_model.load(
        "/Users/elber/Documents/COMP 4949 - Datasets/imdb_bert"
    )

    examples = [
        "this is such an amazing movie!",  # this is the same sentence tried earlier
        "The movie was great!",
        "The movie was meh.",
        "The movie was okish.",
        "The movie was terrible...",
        "Towa Quimbayo was fantastic",
        "Towa Quimbayo is the worst actor ever.",
    ]
    reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
    print("Results from the saved model:")

    for i in range(0, len(reloaded_results)):
        print(examples[i] + " " + str(reloaded_results[i][0]))


def main():
    # ex1()
    # ex2()
    # ex8()
    ex9()


if __name__ == "__main__":
    main()
