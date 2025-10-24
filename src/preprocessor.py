import re
import emoji
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

class TextPreprocessor:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('russian'))
        # Убираем pymorphy2 из-за несовместимости с Python 3.13
        # self.morph = pymorphy2.MorphAnalyzer()
    
    def clean_text(self, text):
        """Очистка текста от эмодзи"""
        if isinstance(text, str):
            return emoji.replace_emoji(text)
        return text
    
    def preprocess_text(self, text):
        """Полная предобработка текста (без лемматизации)"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        
        # Временно убираем лемматизацию из-за проблем с pymorphy2
        # words = [self.morph.parse(word)[0].normal_form for word in words]
        
        return ' '.join(words)
    
    def preprocess_dataframe(self, df):
        """Предобработка всего DataFrame"""
        df_clean = df.copy()
        df_clean['text'] = df_clean['text'].apply(self.clean_text)
        df_clean['text'] = df_clean['text'].apply(self.preprocess_text)
        df_clean = df_clean.drop_duplicates().dropna()
        
        # Кодирование меток групп
        if 'group_id' in df_clean.columns:
            self.label_encoder = LabelEncoder()
            df_clean['group_id_encoded'] = self.label_encoder.fit_transform(df_clean['group_id'])
        
        return df_clean
    
    def get_label_encoder(self):
        """Получение label encoder"""
        return self.label_encoder