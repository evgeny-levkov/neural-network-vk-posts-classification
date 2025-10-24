import requests
import time
import pandas as pd
import os
from config import VK_CONFIG

class DataLoader:
    def __init__(self):
        self.access_token = VK_CONFIG['access_token']
        self.version = VK_CONFIG['version']
        self.group_ids = VK_CONFIG['group_ids']
    
    def get_posts(self, group_id, count=1000):
        """Парсинг постов из VK API"""
        all_posts = []
        offset = 0
        while offset < count:
            response = requests.get('https://api.vk.com/method/wall.get', params={
                'owner_id': f'-{group_id}',
                'count': 100,
                'offset': offset,
                'access_token': self.access_token,
                'v': self.version
            }).json()

            if 'response' in response:
                posts = response['response']['items']
                all_posts.extend(posts)
                offset += 100
                if len(posts) < 100:
                    break
            else:
                print(f"Error fetching posts from group {group_id}: {response}")
                break
            time.sleep(0.5)
        return all_posts

    def load_from_vk(self):
        """Загрузка данных через VK API"""
        all_data = []
        for group_id in self.group_ids:
            print(f"Fetching posts from group {group_id}")
            posts = self.get_posts(group_id)
            for post in posts:
                all_data.append({
                    'text': post.get('text', ''),
                    'comments': post['comments']['count'] if 'comments' in post else 0,
                    'likes': post['likes']['count'] if 'likes' in post else 0,
                    'group_id': group_id
                })
            print(f"Fetched {len(posts)} posts from group {group_id}")

        df = pd.DataFrame(all_data, columns=['text', 'comments', 'likes', 'group_id'])
        return df

    def load_from_csv(self, filepath):
        """Загрузка данных из CSV файла"""
        # Создаем папку, если её нет
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"Файл {filepath} не найден. Загружаем данные из VK API...")
            df = self.load_from_vk()
            self.save_to_csv(df, filepath)
            return df

    def save_to_csv(self, df, filepath):
        """Сохранение данных в CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Данные сохранены в {filepath}")