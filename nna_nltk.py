# Importar as bibliotecas necessárias
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer


# Definir as funções de pré-processamento de texto
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remover caracteres indesejados
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remover menções de usuários
    text = re.sub(r'https?:\/\/\S+', '', text)  # remover URLs
    text = re.sub(r'[^\w\s]', '', text)  # remover pontuação
    text = re.sub(r'\d+', '', text)  # remover números
    # Normalizar texto
    text = text.lower()
    # Remover stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Carregar os dados de tweets
tweets_df = pd.read_excel('base_de_dados_bitcoin.xlsx')

# Pré-processar os dados de texto
tweets_df['Tweet'] = tweets_df['Tweet'].apply(preprocess_text)

# Normalizar as palavras
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

# Processar as palavras normalizadas
tweets_df['Tweet'] = tweets_df['Tweet'].apply(lambda w: stemming(w))

# tweets_df.to_excel('tweets_df.xlsx')
print(tweets_df.head(50))


sia = SentimentIntensityAnalyzer()
polarities = []
for tweet in tweets_df.Tweet:
   polarity = sia.polarity_scores(tweet)['compound']
   polarities.append(polarity)
   
tweets_df['Polarity'] = polarities

def classify_sentiment(polarity):
    if polarity > 0.2:
        return 'positive'
    elif polarity < -0.2:
        return 'negative'
    else:
        return 'neutral'
    
tweets_df['Sentiment'] = tweets_df['Polarity'].apply(classify_sentiment)


# Criar a matriz de features
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(tweets_df['Tweet']).toarray()


# Criar o vetor de target
y = np.array(tweets_df['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0))
# y = np.array(tweets_df['Polarity'])

# df_y = pd.DataFrame(y).T
# df_y.to_excel('df_y.xlsx')
# print(y)


# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Definir a arquitetura da rede neural
model = Sequential()
model.add(Embedding(input_dim=8000, output_dim=32, input_length=X.shape[1]))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# # Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Treinar o modelo
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=4)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, callbacks=[es])

# # Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy: {}'.format(accuracy))
