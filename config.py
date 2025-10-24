"""
Конфигурационные параметры проекта
"""

# Конфигурационные параметры VK API
VK_CONFIG = {
    'access_token': 'a8574a42a8574a42a8574a4259ab4f14beaa857a8574a42ce1aa1231f96d9580997a7e1',
    'version': '5.131',
    'group_ids': ['766121', '205220394', '100490890', '38545497']
}

# Параметры моделей
MODEL_CONFIG = {
    'max_num_words': 40000,
    'max_sequence_length': 100,
    'test_size': 0.2,
    'random_state': 42
}

# Параметры нейронных сетей
NEURAL_NETWORKS = {
    'epochs': 10,
    'batch_size': 64,
    'validation_split': 0.2
}

# Старые переменные для обратной совместимости (можно удалить после обновления кода)
MAX_NUM_WORDS = 40000
MAX_SEQUENCE_LENGTH = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42