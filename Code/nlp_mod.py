import pandas as pd
import numpy as np
from PIL import Image
import re
import os
import seaborn as sns
import string
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import lemmatizer
from spacy.tokenizer import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake
from wordcloud import WordCloud
import altair as alt
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from collections import Counter
from polyglot.text import Text, Word
from polyglot.downloader import downloader
from sklearn.manifold import TSNE
import gensim
from gensim.models import Word2Vec

sid_obj = SentimentIntensityAnalyzer()

w2v_cat_df = pd.read_csv('Final_DB.csv')

w2v_cat_df.head()

"""Removing Punctuations, Stop Words and Emojis"""

punctuations = string.punctuation
stop_words = list(STOP_WORDS)
stop_words.extend(
    ["five", "star", "stars", "good", "nice", "buy", "great", "buying", "value", "four", "purchase", "excellent",
     "classic", "love", "item", "awesome", "day", "review", "ultra", "fast", "delivery", "item", "year", "£", "price",
     'grand', "money",
     "recommend",
     "recommended", "company", 'hi', "lovely", "young", "old", "sure", "simple", "age", "do", "fine", "clear", 'come',
     'lol',
     'please',
     'maybe', 'someday', 'best'])
stop_words = set(stop_words)


def decontracted(phrase):
    phrase = phrase.lower()
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"n\ \'\ t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\ \'\ ve", " have", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\ \'\ s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\ \'\ t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\ \'\ ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\.", " . ", phrase)
    phrase = re.sub(r'\s+', ' ', phrase)
    phrase = re.sub("<|>", " ", phrase)
    phrase = re.sub(r"\d+\w+|\w+\d+|\d+", " ", phrase)
    return phrase


def stpw_rem(sentences):
    if type(sentences) is str:
        sentences = word_tokenize(sentences, "english")
    b = " ".join(word for word in sentences if word.strip() not in stop_words)  # Removing stopwords
    return b


def punc_rem(sentences):
    sentences = sentences.translate(str.maketrans('', '', punctuations))  # Removing Punctuations
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', sentences)  # Emoji removal


"""Lemmatization"""

lemmatizer = WordNetLemmatizer()


def lemma_id(sentence):
    doc = word_tokenize(sentence)
    lem = " ".join(lemmatizer.lemmatize(token) for token in doc)

    return lem


def linkrem(i):
    i = re.sub(r'(www.\S+|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', "", i)
    return i


'''Loading Model'''
model = gensim.models.KeyedVectors.load_word2vec_format('w2v_amazon.bin', binary=True)


def graph(keys):
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in model.most_similar(word):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=4500, random_state=32, learning_rate=150,
                            n_jobs=5)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
        plt.figure(figsize=(16, 9))
        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
        for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
            x = embeddings[:, 0]
            y = embeddings[:, 1]
            plt.scatter(x, y, c=color, alpha=a, label=label)
            for i, word in enumerate(words):
                plt.annotate(word, alpha=0.8, xy=(x[i], y[i]), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom', size=8)
        plt.legend(loc=4)
        plt.title(title)
        plt.grid(True)
        if filename:
            plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        plt.show()

    tsne_plot_similar_words('Similar words from Your Review', keys, embeddings_en_2d, word_clusters, 1, \
                            "static\sim_grph.png")


'''Main Function Starts'''
set_l = []
t = ['train', 'kite', 'toy', 'outdoor', 'set', 'flight', 'child', 'eraser', 'puzzle', 'game', 'track']
for i in w2v_cat_df.amazon_category_and_sub_category.unique():
    for k in t:
        if k in i.lower():
            set_l.append(i)
        try:
            a = dict(model.similar_by_word(k, topn=15)).keys()
        except:
            continue
        for j in a:
            if j in i.lower():
                set_l.append(i)

unl_cat_l = set(w2v_cat_df.amazon_category_and_sub_category.unique().flatten()).symmetric_difference(set_l)


def mainf(strg):
    c_dic = {}
    cat_arr = []
    cat_no = 0
    grph_tok = []

    def comp_ext(strg):  # Extracting Comapanies
        strg = punc_rem(strg)  # polyglot
        text = Text(strg)
        company_l = []
        for i in text.entities:
            company_l.extend(i)

        nlp = spacy.load("en_core_web_md")
        doc = nlp(strg)
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                company_l.append(ent.text)

        tokens = nltk.word_tokenize(strg)
        tagged = nltk.pos_tag(tokens)
        entities = nltk.chunk.ne_chunk(tagged)
        company = [" ".join(w for w, t in elt) for elt in entities if isinstance(elt, nltk.Tree)]
        company.extend(company_l)

        company_fin = [i for i in set(company) if len(word_tokenize(i)) < 4]
        return company_fin

    def cat_pred(cat, cat_no):  # Predict the Category
        for j in cat.groupby(['amazon_category_and_sub_category'])['Review'].unique():
            m = []
            m.extend(j)
            tok_rev = Counter(word_tokenize(punc_rem(str(word_tokenize(str(m))))))
            c = 0
            score = 0
            for ele in token_fin:
                if ele in tok_rev.keys():
                    c += 1
                    score += tok_rev[ele]
                    try:
                        sim_dic = dict(model.similar_by_word(ele, topn=7)).keys()
                    except:
                        continue
                    for sim_w in sim_dic:
                        grph_tok.append(ele)
                        if sim_w in tok_rev.keys():
                            c += 1
                            score += tok_rev[ele]
                        c_dic[cat_no] = [c, score]
            cat_no += 1
        return c_dic

    def _main(company):  # Main function
        flag = 0
        for i in company:
            if i in w2v_cat_df.manufacturer.unique():
                flag = 1
                cat = w2v_cat_df[w2v_cat_df.manufacturer.str.lower() == i.lower()][
                    ['amazon_category_and_sub_category', 'Review']]
                cat_arr.extend(cat.amazon_category_and_sub_category.unique())
                c_dict = cat_pred(cat, cat_no)

        if flag == 0:
            c_dict = unla_rev()

        return c_dict

    def unla_rev():
        # Finding the Most Relevant Word Category
        max = 0
        idx = -1
        for i, word in enumerate(t):
            try:
                a = list(dict(model.similar_by_word(word, topn=10)).keys())
            except:
                continue
            a.append(word)
            if max < len(list(set(token_fin).intersection(set(a)))):
                max = len(list(set(token_fin).intersection(set(a))))
                idx = i

        if idx == -1:  # if not found then proceed with unlabeled ones
            cat_list = unl_cat_l
        else:  # else find the all categories present of that word
            set_l = []
            for i in w2v_cat_df.amazon_category_and_sub_category.unique():
                if t[idx] in i.lower():
                    set_l.append(i)
                try:
                    a = dict(model.similar_by_word(t[idx], topn=15)).keys()
                except:
                    continue
                for j in a:
                    if j in i.lower():
                        set_l.append(i)
            cat_list = set(set_l)

        for i in cat_list:
            cat = w2v_cat_df[w2v_cat_df.amazon_category_and_sub_category == i][
                ['amazon_category_and_sub_category', 'Review']]
            leng = len(cat_arr)
            cat_arr.append(i)
            c_dict = cat_pred(cat, leng)

        return c_dict

    if strg:
        company_fin = comp_ext(strg)
        strng = lemma_id(punc_rem(stpw_rem(decontracted(strg))))
        # print(strng)
        token_fin = list(Counter(word_tokenize(strng)).keys())[:10]
        token_fin = [x for x in token_fin if x.capitalize() not in company_fin and x.upper() not in company_fin]
        if token_fin:
            # print(token_fin)
            c_dict = _main(company_fin)
            c_dic = sorted(c_dict.items(), key=lambda kv: (kv[1], kv[0]))
            if grph_tok:
                graph(set(grph_tok))
            print(c_dic[::-1])
            return c_dic[::-1], cat_arr, company_fin


def pred(strg):
    if strg:
        strg = linkrem(strg)
        c_dic, cat_arr, _ = mainf(strg)
        cat_arngd = []
        maxp = round(c_dic[0][1][0], -1)
        if maxp < c_dic[0][1][0]:
            maxp = maxp + 10
        for i in c_dic:
            a=[]
            a.append(cat_arr[i[0]])
            a.append((i[1][0]/maxp)*100)
            if (i[1][0]/maxp)*100 > 40:
                cat_arngd.append(a)
        return cat_arngd


'''Word Cloud of Negative and Positive Words'''

pos_w = []
neg_w = []


def sentimentAn(strg):
    if strg:
        strg = linkrem(strg)
        for i in word_tokenize(strg):
            if sid_obj.polarity_scores(i)['compound'] < -0.2 and len(i) > 2:
                print("Negative Words-", '\n', i)
                neg_w.append(i)
            elif sid_obj.polarity_scores(i)['compound'] > 0.2 and len(i) > 2:
                print("Positive Words-", '\n', i)
                pos_w.append(i)
        sen_l = []
        kar, _ = keyword(strg)
        for i in kar:
            lo_dic = sid_obj.polarity_scores(i)
            maxv = str(lo_dic['pos']) + ' 😊'
            if lo_dic['neu'] > 0.85:
                maxv = str(lo_dic['neu']) + ' 😐'
            elif lo_dic['compound'] < 0.0:
                maxv = str(lo_dic['neg'] * -1) + ' 😢'
            sen_l.append([i, maxv])
        if pos_w:
            wrdcld(pos_w, fn='static\pos_grph.png')
        if neg_w:
            wrdcld(neg_w, fn='static\grph_neg.png')
        return sid_obj.polarity_scores(strg)['compound'], sen_l


def wrdcld(wl, fn):
    wc_data = ' '.join(i for i in wl)
    mask = np.array(Image.open("cloud.png"))
    wc = WordCloud(mask=mask).generate(wc_data)
    plt.figure(figsize=[9, 9])
    plt.imshow(wc)
    plt.savefig(fn)


def keyword(strg):
    '''Keyword and Concept extraction'''
    if strg:
        strg = linkrem(strg)
        r = Rake()
        a = r.extract_keywords_from_text(strg)

        return r.get_ranked_phrases()[:10], r.get_ranked_phrases_with_scores()[:10]
pred("""The Lego Creative (Medium) Creative set comes to in a nice plastic storage box. I knew what was included, so I cannot blame Lego. There are simply not enough basic bricks to build with. I have uploaded several pictures for your review. When all is said and done, you are left with a lot of extra room in your storage box. For us, this was okay, but if your little builder wants to build a house, you may want to check out the Lego Education Basic Brick Set. That set has a lot of building bricks. This set has more specialty pieces, such as wheels. Overall, a great quality product. Lego never disappoints in that regard. This set may disappoint you IF you don't realize exactly what you are getting. I think my pictures will help you there! As you can see, there aren't many basic bricks.""")