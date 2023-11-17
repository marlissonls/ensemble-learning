# BBC Financial News Topic Modeling

# version 4 - Hyperparameters Optimizing for Stacking method

from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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

# List for the results
result = []

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
    base_models = [('rf', RandomForestClassifier(n_estimators = 100, random_state = 42)), ('nb', MultinomialNB())]

    # Stacking Model
    stacking_model = StackingClassifier(estimators = base_models, final_estimator = LogisticRegression(multi_class = 'multinomial', random_state = 30, max_iter = 1000))

    # Training
    accuracy = stacking_model.fit(X_training_vectors, y_training).score(X_test_vectors, y_test)

    # Results
    result.append((x, accuracy))
    print('-Random State:', x, '-Model Accuracy:', accuracy, '\n')

print('\nBest Results:')
mx = max(result, key = lambda x:x[1])
print('-Random State:', mx[0], '-Model Accuracy:', mx[1], '\n')