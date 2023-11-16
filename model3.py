# Financial News Topic Modeling

# version 3 - StackingClassifier

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

# Stop words list
my_stop_words = list(stopwords.words('english'))

# Split into training and test (70/30)
X_training, X_test, y_training, y_test = train_test_split(X, y, test_size = 0.30, random_state = 93)

# Vectorization
vectorizer = TfidfVectorizer(norm = None, stop_words = my_stop_words, max_features = 1000, decode_error = 'ignore')

# Applying vectorization
X_training_vectors = vectorizer.fit_transform(X_training)
X_test_vectors = vectorizer.transform(X_test)

# Starting the base models
base_models = [('rf', RandomForestClassifier(n_estimators = 100, random_state = 42)), ('nb', MultinomialNB())]

# Stacking Classifier - Ensemble Method
stacking_model = StackingClassifier(estimators = base_models, final_estimator = LogisticRegression(multi_class = 'multinomial', random_state = 30, max_iter = 1000))
print('\nStacking Model:\n')
print(stacking_model)

# Training
accuracy = stacking_model.fit(X_training_vectors.todense(), y_training).score(X_training_vectors.todense(), y_test)

# List for the result
result = []
result.append(accuracy)

print('\nModel Accuracy:', accuracy, '\n')