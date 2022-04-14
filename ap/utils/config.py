"""
Файл с путями до различных данных.
"""

data_path = '/data/datasets/Antiplagiat/'

# rubrics
path_grnti_mapping = data_path + 'grnti_to_number.json'
path_articles_rubrics_train_oecd = data_path + 'oecd_codes.json'
path_articles_rubrics_train_udk = data_path + 'udk_codes.json'
path_articles_rubrics_train_grnti = data_path + 'grnti_codes.json'
path_elib_train_rubrics_grnti = data_path + 'articles_train_val2/' + \
                                'data_train/elib_train_grnti_codes.json'
path_elib_val_rubrics_grnti = data_path + 'articles_train_val2/' + \
                                'data_val/elib_val_grnti_codes.json'
path_elib_train_val_rubrics_grnti = data_path + 'articles_train_val2/' + \
                                'elib_train_val_grnti_codes.json'
path_vak_rubrics = data_path + 'val_vak/val_vak.json'

# subsamples
path_articles_subsamples_udk = data_path + 'subsamples_15_udk/'
path_articles_subsamples_grnti = data_path + 'subsamples_15_grnti/'
path_val_1_subsamples_grnti = data_path + 'subsamples_15_val_1_grnti/'
path_val_2_subsamples_grnti = data_path + 'subsamples_15_val_2_grnti/'
path_val_1_2_subsamples_grnti = data_path + 'subsamples_15_val_1_2_grnti/'
path_val_combined_subsamples_grnti = data_path + 'subsamples_15_val_combined_grnti/'
path_vak_subsamples = data_path + 'val_vak/subsamples'

path_model_base = '/data/antiplagiat_models/' + \
                    'TM_for_100_languages_with_wiki_with_dictionary_reduction_BPE_120k_2k/'

# only wiki
path_wiki_train_bpe = data_path + 'wiki_100/wiki_100_train_bpe_120k.txt'
path_wiki_train_batches = data_path + 'wiki_100/batches_train'
path_wiki_test = data_path + 'wiki_100/test'
path_wiki_test_bpe = data_path + 'wiki_100/test_BPE/'

# only articles
path_articles_train_raw = data_path + 'data_raw/texts_raw'
path_articles_train = data_path + 'texts_vw/train.txt'
path_articles_train_dataset = data_path + 'texts_vw/train.csv'
path_articles_train_bpe = data_path + 'texts_vw/train_BPE_120k.txt'
path_articles_train_batches = data_path + 'texts_vw/batches_train'
path_articles_test_bpe = data_path + 'texts_vw/test_BPE/test_test/'
path_articles_val_1 = data_path + 'validation_1/validation_validation'
path_articles_val_1_bpe = data_path + 'validation_1_BPE/validation_validation'
path_articles_val_2 = data_path + 'validation_2/validation_validation'
path_articles_val_2_bpe = data_path + 'validation_2_BPE/validation_validation'
path_articles_val_1_2_bpe = data_path + 'validation_1_2_BPE/validation_validation'

path_wiki_with_articles_train_batches = data_path + \
                                        'wiki_with_articles/batches_train'
path_wiki_with_articles_train_topicnet_dataset = data_path + \
                                        'wiki_with_articles/wiki_with_articles.csv'

# elibrary
path_elib_train_raw = data_path + 'articles_train_val2/data_train/'
path_elib_train = data_path + 'articles_train_val2/train_elib.txt'
path_elib_train_bpe = data_path + 'articles_train_val2/train_elib_bpe.txt'
path_elib_train_batches = data_path + 'articles_train_val2/batches_train/'
path_elib_val_raw = data_path + 'articles_train_val2/data_val/'
path_elib_val = data_path + 'articles_train_val2/val/'
path_elib_val_bpe = data_path + 'articles_train_val2/val_bpe/'

# wiki + articles + elibrary
path_combined_train_batches = data_path + 'combined_train/batches'

# wiki + elibrary
path_wiki_with_elib_train_batches = data_path + 'wiki_with_elib_train/batches'

# articles + elibrary
path_combined_val_bpe = data_path + 'combined_val/'

# vak (dissertations)
path_vak_val_raw = data_path + 'val_vak/'
path_vak_val = data_path + 'val_vak/val_ru.txt'
path_vak_val_bpe = data_path + 'val_vak/val_ru_120k.txt'

# dictinaries
dict_path = data_path + 'dictionaties/'
path_dict_whole = dict_path + 'dictionary_train_wiki_100_BPE_120k_all.txt'
path_dict_reduced = dict_path + 'dictionary_train_BPE_wiki_100_120k_2k.txt'
path_dict_11k = dict_path + 'dictionary_train_BPE_wiki_100_120k_11k.txt'
path_dict_11k_without_sw = dict_path + 'dictionary_train_BPE_wiki_100_120k_11k_no_stopwords.txt'
path_stop_words = dict_path + 'stop_wodrs.joblib'

path_BPE_models = data_path + 'BPE_models/'

LANGUAGES_MAIN = [
    'ru',
    'en',
    'cs',
    'de',
    'es',
    'fr',
    'it',
    'ja',
    'kk',
    'ky',
    'nl',
    'pl',
    'pt',
    'tr',
    'zh'
]

LANGUAGES_ALL = [
    'af',
    'am',
    'ar',
    'av',
    'az',
    'ba',
    'be',
    'bg',
    'bn',
    'bs',
    'ca',
    'ce',
    'cs',
    'cv',
    'cy',
    'da',
    'de',
    'el',
    'en',
    'eo',
    'es',
    'et',
    'eu',
    'fa',
    'fi',
    'fr',
    'gd',
    'gl',
    'gu',
    'he',
    'hi',
    'hr',
    'hu',
    'hy',
    'ia',
    'id',
    'inh',
    'is',
    'it',
    'ja',
    'jv',
    'ka',
    'kaa',
    'kbd',
    'kk',
    'kl',
    'km',
    'ko',
    'krc',
    'ky',
    'la',
    'lez',
    'lo',
    'lt',
    'lv',
    'mg',
    'mhr',
    'mi',
    'mk',
    'ml',
    'mn',
    'mo',
    'ms',
    'my',
    'myv',
    'ne',
    'nl',
    'no',
    'oc',
    'os',
    'pl',
    'pt',
    'rm',
    'rn',
    'ro',
    'ru',
    'sah',
    'sh',
    'si',
    'sk',
    'sl',
    'sm',
    'so',
    'sq',
    'sr',
    'sv',
    'sw',
    'ta',
    'tg',
    'th',
    'tk',
    'tr',
    'tt',
    'udm',
    'uk',
    'ur',
    'uz',
    'vi',
    'yi',
    'zh'
]

LANGUAGES_ELIB = [
    'ar',
    'bg',
    'bs',
    'ca',
    'cs',
    'cy',
    'da',
    'de',
    'en',
    'eo',
    'es',
    'et',
    'eu',
    'fa',
    'fr',
    'gl',
    'he',
    'hr',
    'hu',
    'ia',
    'id',
    'is',
    'it',
    'ja',
    'kk',
    'ko',
    'ky',
    'ms',
    'nl',
    'no',
    'pl',
    'pt',
    'ro',
    'ru',
    'sh',
    'sk',
    'sl',
    'sq',
    'sv',
    'tr',
    'uk',
    'vi',
    'zh'
]
