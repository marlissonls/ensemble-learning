# BBC Financial News Topic Modeling

# version 2 - Hyperparameters Optimizing for Voting method

from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from src.download_bbc_dataset import download_bbc_news
from src.download_nltk_stopwords import download_nltk_stopwords

# Dowload BBC News Dataset and get its directory
bbc_news_dir = download_bbc_news()

# Load data
news = load_files(container_path = bbc_news_dir, encoding = 'utf-8', decode_error = 'replace')

# Spliting input and output variables
# Receive the files
X = news.data
# Gets the labels (names of folders containing the files)
y = news.target

# Downlaod nltk stopwords if not downloaded yet
download_nltk_stopwords()

# List dictionary for the results
result = {}
result['soft'] = []
result['hard'] = []

# Loop for searching for the best parameters
for x in range(1, 100):

    # Split training/test
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size = 0.30, random_state = x)

    # Stop words list
    my_stop_words = list(stopwords.words('english'))

    # Vectorization
    vectorizer = TfidfVectorizer(norm = None, stop_words = my_stop_words, max_features = 1000, decode_error = 'ignore')

    # Applying vectorization
    X_training_vectors = vectorizer.fit_transform(X_training)
    X_test_vectors = vectorizer.transform(X_test)

    # Create the base models
    model01 = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', random_state = 30, max_iter = 1000)
    model02 = RandomForestClassifier(n_estimators = 1000, max_depth = 100, random_state = 1)
    model03 = MultinomialNB()

    # Loop for soft and hard methods
    for i in ['soft', 'hard']:
        voting_model = VotingClassifier(estimators = [ ('lg', model01), ('rf', model02), ('nb', model03) ], voting = i)
        voting_model = voting_model.fit(X_training_vectors, y_training)
        predictions = voting_model.predict(X_test_vectors)
        print('-Random State:', x, '-Voting:', i, '-Accuracy:', accuracy_score(y_test, predictions))
        result[i].append((x, accuracy_score(y_test, predictions)))

print('\nBest Results:')

# Extract the best results
h = max(result['hard'], key = lambda x:x[1])
s = max(result['soft'], key = lambda x:x[1])

# Print
print('-Random State:', h[0], '-Voting:hard', '-Accuracy:', h[1])
print('-Random State:', s[0], '-Voting:soft', '-Accuracy:', s[1])