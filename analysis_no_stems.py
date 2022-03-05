import os
import re
import shutil
import string
import pandas as pd
import matplotlib.pyplot as plt
import collections
import nltk
from nltk.corpus import stopwords


def output_all_graphs(df, graph_title, output_dir, max_index, interval, file_stem):
    current_index = 0
    while current_index <= max_index:
        upper_limit = min(current_index + interval - 1, max_index)

        df_subset = df.iloc[current_index:upper_limit]
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot horizontal bar graph
        df_subset.sort_values(by='count').plot.barh(x='words',
                                                    y='count',
                                                    ax=ax,
                                                    color="purple")

        ax.set_title('{0} (Items {1}-{2})'.format(graph_title, current_index + 1, upper_limit + 1))
        plt.savefig(
            os.path.join(output_dir, '{0}_{1}_to_{2}.png'.format(file_stem, current_index + 1, upper_limit + 1)))
        plt.close(fig)

        current_index += interval


if __name__ == "__main__":
    with open("Inferno.txt", "r", encoding="utf8") as inferno_file:
        inferno_text = inferno_file.read().replace('\n', ' ').lower()  # Replace line breaks with spaces + to lowercase
        # Remove punctuation, except apostrophes and hyphens
        inferno_text = inferno_text.translate(
            str.maketrans('', '', string.punctuation.replace('-', '“”')))
        # Replace hyphens and dashes with spaces to not form merged words
        inferno_text = inferno_text.replace('—', ' ').replace('-', ' ')
        # Removes single quotes but not apostrophes
        inferno_text = inferno_text.replace(' ’', ' ').replace('’ ', ' ')
        # Convert apostrophes to ' character
        inferno_text = inferno_text.replace('’', '\'')
        inferno_text = re.sub(' +', ' ', inferno_text)  # Remove any instances of multiple spaces and make lowercase
        inferno_file.close()

    # Text has been cleaned, we can split into words now
    inferno_words = inferno_text.split()
    word_frequencies = collections.Counter(inferno_words)

    # Remove stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    inferno_words_no_stopwords = [word for word in inferno_words if not word in stop_words]
    word_frequencies_no_stopwords = collections.Counter(inferno_words)

    df = pd.DataFrame(word_frequencies.most_common(len(word_frequencies)), columns=['words', 'count'])
    no_stopwords_df = pd.DataFrame(word_frequencies_no_stopwords.most_common(len(word_frequencies_no_stopwords)),
                                   columns=['words', 'count'])

    # Prepare to export many, many graphs
    if os.path.exists('out_no_stems'):
        shutil.rmtree('out_no_stems')

    os.mkdir('out_no_stems')

    output_all_graphs(df, 'Most Common Words in Mandelbaum Inferno', 'out_no_stems', 100, 15, 'word_count')
    output_all_graphs(no_stopwords_df, 'Most Common Words in Mandelbaum Inferno (No Stopwords)', 'out_no_stems', 100, 15,
                      'no_stopword_word_count')
