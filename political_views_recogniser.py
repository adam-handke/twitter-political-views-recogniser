import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import twitter_scraper
import re
import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.base import clone
from stop_words import get_stop_words
from joblib import dump, load

# parameters
dpi = mpl.rcParams['figure.dpi']
plt.rcParams['figure.facecolor'] = "#eeeeee"
img = plt.imread("compass.png")
height, width, depth = img.shape
figsize = width / float(dpi), height / float(dpi)
horizontal_max = 1200
horizontal_min = 200
vertical_max = 150
vertical_min = 1150
third_max = 1150
third_min = 150
nlp = spacy.load('pl_core_news_sm')


# drawing 3-axis political compass, all values should be in range [-1, 1]
def draw_compass(username, horizontal, vertical, third):
    if horizontal < -1 or horizontal > 1:
        horizontal_value = horizontal_min + int(np.round(float(horizontal_max - horizontal_min) / 2.0, 0))
        print(f"ERROR - wrong horizontal value: {horizontal}")
    else:
        horizontal_value = horizontal_min + int(
            np.round((float(horizontal + 1) / 2.0) * float(horizontal_max - horizontal_min), 0))

    if vertical < -1 or vertical > 1:
        vertical_value = vertical_max + int(np.round(float(vertical_min - vertical_max) / 2.0, 0))
        print(f"ERROR - wrong vertical value: {vertical}")
    else:
        vertical_value = vertical_max + int(
            np.round((float(-vertical + 1) / 2.0) * float(vertical_min - vertical_max), 0))

    if third < -1 or third > 1:
        third_value = third_min + int(np.round(float(third_max - third_min) / 2.0, 0))
        print(f"ERROR - wrong third value: {third}")
    else:
        third_value = third_min + int(np.round((float(third + 1) / 2.0) * float(third_max - third_min), 0))

    plt.figure(figsize=figsize)
    plt.scatter(x=[horizontal_value], y=[vertical_value], c="red", s=750, linewidths=1, edgecolors="black")
    plt.scatter(x=[1574], y=[third_value], c="red", s=11750, marker="_", linewidths=7)
    plt.imshow(img)
    plt.title(f"Political views compass for Twitter user @{username}", fontsize=24)
    plt.axis("off")
    plt.tight_layout(pad=1)
    plt.show()
    plt.close()


def print_statistics(name, values):
    print(name.ljust(12), str(len(values)).ljust(6), str(np.round(np.mean(values), 2)).ljust(6),
          str(np.round(np.median(values), 2)).ljust(6), str(np.round(np.std(values), 2)).ljust(6),
          str(np.round(np.max(values), 2)).ljust(6), str(np.round(np.min(values), 2)).ljust(6), sep="\t")


def load_texts(names, base_path="tweets\\"):
    result = []
    for name in names:
        with open(base_path + name + ".csv", encoding="utf8") as data:
            result.append("".join(data.readlines()[1:]))
    return result


def clean_text(text):
    result = re.sub(r"\"", "", text)  # remove quotation marks (especially around tweets)
    result = re.sub(r"(^|[^@\w])@(\w{1,15})\b", "", result)  # remove twitter usernames
    result = re.sub(r"http\S+", "", result)  # remove twitter URLs
    result = re.sub(r"[＃#](\w+)", "", result)  # remove hashtags
    result = re.sub(r"\d+", "", result)  # remove digits
    # lemmatization & punctuation removal
    result = " ".join([token.lemma_ for token in nlp(result) if not token.is_punct])
    return result


def train_model():
    usernames, horizontals, verticals, thirds = twitter_scraper.load_mypolitics_data()
    print("Axis        \tCount \tMean  \tMedian\tStd   \tMax   \tMin   ")
    print_statistics("horizontal", horizontals)
    print_statistics("vertical", verticals)
    print_statistics("third", thirds)

    original_texts = load_texts(usernames)
    texts = np.array([clean_text(text) for text in original_texts])

    np.random.seed(1)
    train_indices = np.random.rand(len(texts)) < 0.85  # 85% for training, rest for testing

    train_texts = texts[train_indices]
    train_horizontals = np.array(horizontals)[train_indices]
    train_verticals = np.array(verticals)[train_indices]
    train_thirds = np.array(thirds)[train_indices]

    test_texts = texts[~train_indices]
    test_horizontals = np.array(horizontals)[~train_indices]
    test_verticals = np.array(verticals)[~train_indices]
    test_thirds = np.array(thirds)[~train_indices]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(encoding='utf-8', stop_words=get_stop_words('polish'),
                                  min_df=0.1, max_df=0.9)),
        ('clf', Ridge(alpha=0.013, solver='auto', random_state=1))
    ])

    model_horizontal = clone(pipeline).fit(train_texts, train_horizontals)
    model_vertical = clone(pipeline).fit(train_texts, train_verticals)
    model_third = clone(pipeline).fit(train_texts, train_thirds)

    print("Mean score:", np.round(np.mean([model_horizontal.score(test_texts, test_horizontals),
                                           model_vertical.score(test_texts, test_verticals),
                                           model_third.score(test_texts, test_thirds)]), 3))
    print("Horizontal score:", np.round(model_horizontal.score(test_texts, test_horizontals), 3))
    print("Vertical score: ", np.round(model_vertical.score(test_texts, test_verticals), 3))
    print("Third score: ", np.round(model_third.score(test_texts, test_thirds), 3))

    print("Prediction / Actual comparison in test subset:")
    print("Case\tHorizontal\t\tVertical\t\tThird")
    for i, text in enumerate(test_texts):
        print(
            str(i).ljust(5),
            str(np.round(*model_horizontal.predict([text]), 2)).ljust(5),
            str(test_horizontals[i]).ljust(5),
            str(np.round(*model_vertical.predict([text]), 2)).ljust(5),
            str(test_verticals[i]).ljust(5),
            str(np.round(*model_third.predict([text]), 2)).ljust(5),
            str(test_thirds[i]).ljust(5),
            sep="\t"
        )

    dump(model_horizontal, "model_horizontal.joblib")
    dump(model_vertical, "model_vertical.joblib")
    dump(model_third, "model_third.joblib")


def most_influential_words_of_model(model, n):
    model_names = model['tfidf'].get_feature_names()
    model_coefs = model['clf'].coef_.flatten()
    pairs = [(name, np.round(abs(coef), 3)) for name, coef in zip(model_names, model_coefs)]
    return sorted(pairs, key=lambda tup: tup[1])[-n:]


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-help":
        print("Wrong arguments!",
              f"To recognise political views use: python {sys.argv[0]} [twitter_username]",
              f"To train the models use: python {sys.argv[0]} -train",
              f"To print the most influential words in the model use: python {sys.argv[0]} -most",
              f"To get help use: python {sys.argv[0]} -help",
              sep="\n")
    elif sys.argv[1] == "-train":
        train_model()
    elif sys.argv[1] == "-most":
        prepared_model_horizontal = load("model_horizontal.joblib")
        prepared_model_vertical = load("model_vertical.joblib")
        prepared_model_third = load("model_third.joblib")

        print("Most influential words in each model (word, abs(coef))")
        print("\nHorizontal", *most_influential_words_of_model(prepared_model_horizontal, 10), sep="\n")
        print("\nVertical", *most_influential_words_of_model(prepared_model_vertical, 10), sep="\n")
        print("\nThird", *most_influential_words_of_model(prepared_model_third, 10), sep="\n")
    else:
        prepared_model_horizontal = load("model_horizontal.joblib")
        prepared_model_vertical = load("model_vertical.joblib")
        prepared_model_third = load("model_third.joblib")

        name = sys.argv[1]
        data = clean_text(twitter_scraper.get_tweets_to_str(name, 1000))
        draw_compass(name,
                     prepared_model_horizontal.predict([data]),
                     prepared_model_vertical.predict([data]),
                     prepared_model_third.predict([data]))
