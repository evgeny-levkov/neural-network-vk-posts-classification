"""
Основной скрипт для запуска проекта классификации постов ВК
"""

import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.trainer import ModelTrainer
from src.predictor import Predictor
import pandas as pd
import os

def main():
    print("Запуск проекта классификации постов ВК")
    
    try:
        # Создаем необходимые папки
        folders = ['data/raw', 'data/processed', 'models', 'results/images']
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        
        # 1. Загрузка данных
        print("Загрузка данных...")
        data_loader = DataLoader()
        df = data_loader.load_from_csv('data/raw/vk_posts1.csv')
        print(f"Загружено {len(df)} записей")
        
        # 2. Предобработка
        print("Предобработка текста...")
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df)
        print(f"Данные обработаны. Осталось {len(df_processed)} записей")
        
        # 3. Обучение моделей
        print("Обучение моделей...")
        trainer = ModelTrainer(df_processed)
        
        # Передаем label_encoder в trainer для сохранения
        trainer.label_encoder = preprocessor.get_label_encoder()
        
        results = trainer.train_all_models()
        
        # Проверяем, что модели были обучены
        if not hasattr(trainer, 'models') or not trainer.models:
            print("Модели не были обучены. Пропускаем сохранение.")
        else:
            # 4. Сохранение моделей
            print("Сохранение моделей...")
            trainer.save_models()
        
        # 5. Демонстрация предсказаний (если есть модели)
        if hasattr(trainer, 'models') and trainer.models:
            print("\nТестирование предсказаний...")
            predictor = Predictor()
            
            test_texts = [
                "Лучшие пистолеты в CS2 Выбирая оружие, игрокам приходится учитывать сразу несколько факторов",
                "С Днем защиты детей, друзья! Забота о судьбах наших детей – верный залог",
                "Здание вокзала в Новом Петергофе напоминает сказочный замок!"
            ]
            
            for i, text in enumerate(test_texts, 1):
                print(f"\nПример {i}:")
                print(f"Текст: {text[:80]}...")
                prediction = predictor.predict(text)
                print(f"Результат: {prediction}")
        
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()