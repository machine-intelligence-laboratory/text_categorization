from pathlib import Path

import matplotlib.pyplot as plt

import config


def show_cos_distribution(cos_distribution, path_save_figs, rubrics, lang, n_bins):
    """Function to visualize cos distribution."""
    path_save_figs = Path(path_save_figs)
    for rubric in sorted(cos_distribution['in_class'].keys()):
        _, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].hist(cos_distribution['not_in_class'][rubric], label=f'rubric_{rubric}_not_in_class',
                    density=True, alpha=0.7, bins=n_bins)
        axs[0].hist(cos_distribution['in_class'][rubric], label=f'rubric_{rubric}_in_class',
                    density=True, alpha=0.7, bins=n_bins)
        axs[0].grid()
        axs[0].legend()
        axs[0].set_title(f'Распределение косинусной близости, {lang}')
        axs[0].set_xlabel('Косинусная близость')
        axs[0].set_ylabel('Относительное количество документов')

        axs[1].hist(cos_distribution['not_in_class'][rubric], label=f'rubric_{rubric}_not_in_class',
                    alpha=0.7, bins=n_bins)
        axs[1].hist(cos_distribution['in_class'][rubric], label=f'rubric_{rubric}_in_class',
                    alpha=0.7, bins=n_bins)
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title(f'Распределение косинусной близости, {lang}')
        axs[1].set_xlabel('Косинусная близость')
        axs[1].set_ylabel('Абсолютное количество документов')

        plt.savefig(path_save_figs.joinpath(f'cos_distribution_{lang}_{rubric}.png'))


def show_analogy_distribution(model_name, pair_analogy, path_save_figs, lang, n_bins):
    """Function to visualize analogy distribution."""
    path_save_figs = Path(path_save_figs)
    for current_lang in config.LANGUAGES_MAIN:
        if (current_lang == lang) or \
                (lang == 'ru' and current_lang == 'en') or (lang == 'en' and current_lang == 'ru'):
            continue
        _, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].hist(
            pair_analogy[f'{model_name}_{current_lang}_{lang}']['not_relevant'],
            label=f'{current_lang}_{lang}_not_relevant', density=True, alpha=0.7, bins=n_bins
        )
        axs[0].hist(
            pair_analogy[f'{model_name}_{current_lang}_{lang}']['relevant'],
            label=f'{current_lang}_{lang}_relevant', density=True, alpha=0.7, bins=n_bins
        )
        axs[0].grid()
        axs[0].legend()
        axs[0].set_title('Распределение аналогий')
        axs[0].set_xlabel('Косинусная близость')
        axs[0].set_ylabel('Относительное количество документов')

        axs[1].hist(
            pair_analogy[f'{model_name}_{current_lang}_{lang}']['not_relevant'],
            label=f'{current_lang}_{lang}_not_relevant', alpha=0.7, bins=n_bins
        )
        axs[1].hist(
            pair_analogy[f'{model_name}_{current_lang}_{lang}']['relevant'],
            label=f'{current_lang}_{lang}_relevant', alpha=0.7, bins=n_bins
        )
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title('Распределение аналогий')
        axs[1].set_xlabel('Косинусная близость')
        axs[1].set_ylabel('Абсолютное количество документов')

        plt.savefig(path_save_figs.joinpath(f'analogy_distribution_{current_lang}_{lang}.png'))
