import re
import numpy as np
import pickle
import tensorflow as tf
from config import MODEL_CONFIG
import nltk
from nltk.corpus import stopwords

class Predictor:
    def __init__(self, model_path='models/cnn-lstm_neural_network.h5'):
        """Инициализация predictor с загрузкой модели и всех компонентов"""
        try:
            # Загрузка модели
            self.model = tf.keras.models.load_model(model_path)
            print(f"Модель загружена: {model_path}")
            
            # Загрузка токенизатора
            with open('models/tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            print("Токенизатор загружен")
            
            # Загрузка label encoder
            with open('models/label_encoder.pickle', 'rb') as handle:
                self.label_encoder = pickle.load(handle)
            print("Label Encoder загружен")
            
            # Инициализация компонентов предобработки
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('russian'))
            
            print("Predictor успешно инициализирован")
            
        except Exception as e:
            print(f"Ошибка загрузки компонентов: {e}")
            self.model = None
            self.tokenizer = None
            self.label_encoder = None
    
    def preprocess_text(self, text):
        """Предобработка текста (аналогично обучению)"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^а-яё\s]', '', text)

        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def predict(self, text, threshold=0.7):
        """Предсказание группы для текста"""
        if self.model is None or self.tokenizer is None or self.label_encoder is None:
            return "Ошибка: не все компоненты загружены"
        
        try:
            # Предобработка текста
            processed_text = self.preprocess_text(text)
            
            # Токенизация и паддинг
            seq = self.tokenizer.texts_to_sequences([processed_text])
            pad_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MODEL_CONFIG['max_sequence_length'])

            # Предсказание
            pred = self.model.predict(pad_seq, verbose=0)
            pred_prob = pred[0]
            
            print(f"Вероятности по классам: {[f'{p:.3f}' for p in pred_prob]}")
            
            max_prob_index = np.argmax(pred_prob)
            max_prob = pred_prob[max_prob_index]

            if max_prob < threshold:
                return f"Не относится не к одной группе (вероятность: {max_prob:.2f} < {threshold})"
            
            # Получаем оригинальное название группы
            predicted_group = self.label_encoder.inverse_transform([max_prob_index])[0]
            
            return f"Группа: {predicted_group} (вероятность: {max_prob:.2f})"
            
        except Exception as e:
            return f"Ошибка предсказания: {e}"