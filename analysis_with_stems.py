import re
import matplotlib.pyplot as plt
import os
import pandas as pd
import string
import shutil
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')

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

    stop_words = stopwords.words('english')

    # Clean stop words
    inferno_no_stop_words = [word for word in inferno_words if word not in stop_words]

    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(word) for word in inferno_words]
    stems_no_stop_words = [stemmer.stem(word) for word in inferno_no_stop_words]

    freq_dist = FreqDist(stems)
    freq_dist_no_stop_words = FreqDist(stems_no_stop_words)

    # Prepare to export many, many graphs
    if os.path.exists('out_stems'):
        shutil.rmtree('out_stems')

    os.mkdir('out_stems')

    text = nltk.Text(inferno_words)
    text_no_stop_words = nltk.Text(inferno_no_stop_words)

    df = pd.DataFrame(freq_dist_no_stop_words.most_common(30), columns=['words', 'count'])
    df['words'] = df['words'].replace('guid', 'guide')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot horizontal bar graph
    df.sort_values(by='count').plot.barh(x='words',
                                         y='count',
                                         ax=ax,
                                         color="purple")

    ax.set_title('Most Common Words in Mandelbaum Inferno (No Stopwords)')
    plt.savefig(os.path.join('out_stems', '30_Most_Common_No_Stopwords.png'))
    plt.close(fig)

    df_sw = pd.DataFrame(freq_dist.most_common(30), columns=['words', 'count'])
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot horizontal bar graph
    df_sw.sort_values(by='count').plot.barh(x='words',
                                         y='count',
                                         ax=ax,
                                         color="purple")

    ax.set_title('Most Common Words in Mandelbaum Inferno')
    plt.savefig(os.path.join('out_stems', '30_Most_Common.png'))
    plt.close(fig)

    top_50 = freq_dist_no_stop_words.most_common(50)
    top_61_sw = freq_dist.most_common(61)
    selected = []
    selected.append(top_61_sw[3])
    selected.append(top_61_sw[11])
    selected.append(top_61_sw[12])
    selected.append(top_61_sw[60])
    selected.append(top_50[1])
    selected.append(top_50[2])
    selected.append(top_50[3])
    selected.append(top_50[5])
    selected.append(top_50[8])
    selected.append(top_50[11])
    selected.append(top_50[13])
    selected.append(top_50[15])
    selected.append(top_50[18])
    selected.append(top_50[23])
    selected.append(top_50[27])
    selected.append(top_50[33])
    selected.append(top_50[40])
    selected.append(top_50[41])
    selected.append(top_50[49])

    df_selected = pd.DataFrame(selected, columns=['words', 'count'])
    df_selected['words'] = df_selected['words'].replace('guid', 'guide')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot horizontal bar graph
    df_selected.sort_values(by='count').plot.barh(x='words',
                                            y='count',
                                            ax=ax,
                                            color="purple")

    ax.set_title('Interesting Common Words in Mandelbaum Inferno')
    plt.savefig(os.path.join('out_stems', 'Selected_Most_Common.png'))
    plt.close(fig)

    # print(freq_dist.most_common(50))
    # print(freq_dist_no_stop_words.most_common(50))
    #
    # print(text_no_stop_words.common_contexts(['see']))
