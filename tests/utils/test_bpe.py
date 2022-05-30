"""
Тестирование работы модуля ap.utils.bpe
"""


from ap.utils.bpe import load_bpe_models


def test_bpe():
    bpe_models = load_bpe_models('tests/data/bpe')

    assert isinstance(bpe_models, dict)
    assert len(bpe_models.keys()) == 2
    assert 'ru' in bpe_models
    assert 'en' in bpe_models
    assert isinstance(bpe_models['ru'].encode('Мороженое было вкусным.'), list)
