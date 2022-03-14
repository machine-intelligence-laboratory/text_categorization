path_experiment = '/home/common/potapova.ps/experiments/best_model'
need_augmentation = False
aug_proportion = 0.2
num_collection_passes = 25
num_rubric = 69
NUM_TOPICS = 125
num_not_sp = 1
tau_DecorrelatorPhi = 0.05
tau_SmoothTheta = 0.025
tau_SparseTheta = -0.1
train_dict_path = (
    '/home/common/potapova.ps/data/train_rubrics_modalities/' +
    'train_with_combined_grnti_69_rubric_dict.joblib'
)
dictionary_path = ('/home/common/potapova.ps/data/' +
                   'dictionaries/dictionary_train_BPE_wiki_100_120k_11k.txt')
path_wiki_train_batches = '/home/common/potapova.ps/data/wiki_100/batches_train'
path_articles_rubrics_train_grnti = '/home/common/potapova.ps/data/grnti_codes.json'
path_elib_train_rubrics_grnti = ('/home/common/potapova.ps/data/elib_train_grnti_codes.json')
path_grnti_mapping = '/home/common/potapova.ps/data/grnti_to_number.json'
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
modalities_with_weights = {'@' + lang: 1 for lang in LANGUAGES_ALL + ['UDK', 'GRNTI']}