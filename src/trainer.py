import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from config import MODEL_CONFIG, NEURAL_NETWORKS
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.models = {}  # Инициализируем словарь моделей
        
    def prepare_data(self):
        """Подготовка данных для обучения"""
        texts = self.df['text'].values
        labels = self.df['group_id_encoded'].values

        # Разделение на train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels, 
            test_size=MODEL_CONFIG['test_size'], 
            random_state=MODEL_CONFIG['random_state']
        )

        print(f"Данные разделены: {len(self.X_train)} train, {len(self.X_test)} test")

        # TF-IDF векторизация для классических моделей
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)

        # Токенизация и паддинг для нейросетей
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MODEL_CONFIG['max_num_words'])
        self.tokenizer.fit_on_texts(self.X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)
        self.X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=MODEL_CONFIG['max_sequence_length'])
        self.X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=MODEL_CONFIG['max_sequence_length'])

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_random_forest(self):
        """Обучение Random Forest"""
        print("Обучение Random Forest...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(self.X_train_tfidf, self.y_train)
        
        y_pred = model.predict(self.X_test_tfidf)
        self._evaluate_model(model, y_pred, "Random Forest")
        self.models['random_forest'] = model  # Добавляем модель в словарь
        return model

    def train_gradient_boosting(self):
        """Обучение Gradient Boosting"""
        print("Обучение Gradient Boosting...")
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(self.X_train_tfidf, self.y_train)
        
        y_pred = model.predict(self.X_test_tfidf)
        self._evaluate_model(model, y_pred, "Gradient Boosting")
        self.models['gradient_boosting'] = model  # Добавляем модель в словарь
        return model

    def _evaluate_model(self, model, y_pred, model_name):
        """Оценка классической модели"""
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        print(f"\n{model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("-" * 50)

    def create_lstm_model(self):
        """Создание LSTM модели"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                input_dim=MODEL_CONFIG['max_num_words'], 
                output_dim=128, 
                input_length=MODEL_CONFIG['max_sequence_length']
            ),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(self.y_train)), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_cnn_lstm_model(self):
        """Создание CNN-LSTM модели"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                input_dim=MODEL_CONFIG['max_num_words'], 
                output_dim=128, 
                input_length=MODEL_CONFIG['max_sequence_length']
            ),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(self.y_train)), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_cnn_model(self):
        """Создание CNN модели"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                input_dim=MODEL_CONFIG['max_num_words'], 
                output_dim=128, 
                input_length=MODEL_CONFIG['max_sequence_length']
            ),
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=len(np.unique(self.y_train)), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_neural_network(self, model, model_name):
        """Обучение нейросетевой модели"""
        print(f"Обучение {model_name}...")
        history = model.fit(
            self.X_train_pad, self.y_train,
            epochs=NEURAL_NETWORKS['epochs'],
            batch_size=NEURAL_NETWORKS['batch_size'],
            validation_split=NEURAL_NETWORKS['validation_split'],
            verbose=1
        )
        
        # Оценка модели
        y_pred = np.argmax(model.predict(self.X_test_pad), axis=1)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        print(f"\n{model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Сохраняем модель в словарь с правильным ключом
        model_key = model_name.lower().replace(' ', '_')
        self.models[model_key] = model
        return model, history

    def train_all_models(self):
        """Обучение всех моделей"""
        print("Подготовка данных...")
        self.prepare_data()
        
        # Классические модели
        print("\nОбучение классических моделей...")
        self.train_random_forest()
        self.train_gradient_boosting()
        
        # Нейросетевые модели
        print("\nОбучение нейросетевых моделей...")
        
        # LSTM
        lstm_model = self.create_lstm_model()
        self.train_neural_network(lstm_model, "LSTM Neural Network")
        
        # CNN-LSTM
        cnn_lstm_model = self.create_cnn_lstm_model()
        self.train_neural_network(cnn_lstm_model, "CNN-LSTM Neural Network")
        
        # CNN
        cnn_model = self.create_cnn_model()
        self.train_neural_network(cnn_model, "CNN Neural Network")
        
        print(f"Обучено моделей: {len(self.models)}")
        return self.models

    def save_models(self):
        """Сохранение обученных моделей"""
        import pickle
        
        # Проверяем, есть ли модели для сохранения
        if not hasattr(self, 'models') or not self.models:
            print("Нет моделей для сохранения")
            return
        
        # Создание папки models если не существует
        os.makedirs('models', exist_ok=True)
        
        # Сохранение нейросетевых моделей с правильными именами
        model_names = {
            'lstm_neural_network': 'text_classification_model1.h5',
            'cnn_lstm_neural_network': 'text_classification_model2.h5', 
            'cnn_neural_network': 'text_classification_model3.h5'
        }
        
        for name, model in self.models.items():
            if hasattr(model, 'save'):
                # Сохраняем с понятными именами
                filename = model_names.get(name, f'{name}.h5')
                model.save(f'models/{filename}')
                print(f"Сохранена модель: models/{filename}")
        
        # Сохранение классических моделей
        for name, model in self.models.items():
            if not hasattr(model, 'save'):  # Это классические модели
                with open(f'models/{name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print(f"Сохранена модель: models/{name}.pkl")
        
        # Сохранение токенизатора и векторизатора
        if hasattr(self, 'tokenizer'):
            with open('models/tokenizer.pickle', 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Сохранен токенизатор")
        
        if hasattr(self, 'tfidf_vectorizer'):
            with open('models/tfidf_vectorizer.pickle', 'wb') as handle:
                pickle.dump(self.tfidf_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Сохранен TF-IDF векторизатор")
        
        if hasattr(self, 'label_encoder'):
            with open('models/label_encoder.pickle', 'wb') as handle:
                pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Сохранен Label Encoder")
        
        print("Все модели и компоненты сохранены!")