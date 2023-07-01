import os
import re
import sys

import jieba
import jieba.posseg as psg
import numpy as np
import pandas as pd
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import config

dic_file = os.path.join(sys.path[0], "stop_dic/dict.txt")
stop_file = os.path.join(sys.path[0], "stop_dic/stopwords.txt")

# 页数
max_page = config.max_page
# 关键字
search_keyword = config.search_keyword


def preprocessing():
    # output_path = 'output/'
    # file_path = '微博清单_唐山打人_前10页.csv'
    # os.chdir(file_path)
    data = pd.read_csv(
        os.path.join(
            sys.path[0],
            "dataset/test/微博清单_{}_前{}页.csv".format(search_keyword, max_page),
        )
    )  # content type
    return data
    # os.chdir(output_path)


def chinese_word_cut(mytext):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding="utf-8")
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ["n", "nz", "vn"]
    for line in stopword_list:
        line = re.sub("\n|\\r", "", line)
        stop_list.append(line)

    word_list = []
    # jieba分词
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        word = re.sub("[^\u4e00-\u9fa5]", "", seg_word.word)
        # word = seg_word.word  #如果想要分析英语文本，注释这行代码，启动下行代码
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:  # this word is stopword
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)
    return (" ").join(word_list)


def analysis(data, n_topics=8):
    def print_top_words(model, feature_names, n_top_words):
        tword = []
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            topic_w = " ".join(
                [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
            )
            tword.append(topic_w)
            print(topic_w)
        return tword

    n_features = 1000  # 提取1000个特征词语
    tf_vectorizer = CountVectorizer(
        strip_accents="unicode",
        max_features=n_features,
        stop_words="english",
        max_df=0.5,
        min_df=10,
    )
    tf = tf_vectorizer.fit_transform(data.content_cutted)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method="batch",
        learning_offset=50,
        #                                 doc_topic_prior=0.1,
        #                                 topic_word_prior=0.01,
        random_state=0,
    )
    lda.fit(tf)

    n_top_words = 25
    tf_feature_names = tf_vectorizer.get_feature_names()
    topic_word = print_top_words(lda, tf_feature_names, n_top_words)

    topics = lda.transform(tf)
    topic = []
    for t in topics:
        topic.append("Topic #" + str(list(t).index(np.max(t))))
    data["概率最大的主题序号"] = topic
    data["每个主题对应概率"] = list(topics)

    path = "result/微博清单_{}_前{}页/LDA/".format(search_keyword, max_page)
    if not os.path.exists(path):
        os.makedirs(path)

    data.to_excel(path + "data_topic.xlsx", index=False)

    return lda, tf, tf_vectorizer, n_topics


def visualize(lda, tf, tf_vectorizer, n_topics):
    pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    # pyLDAvis.display(pic)

    path = "result/微博清单_{}_前{}页/LDA/".format(search_keyword, max_page)

    pyLDAvis.save_html(pic, path + "lda_pass" + str(n_topics) + ".html")
    # pyLDAvis.display(pic)


def LDA():
    n_topics = 8
    data = preprocessing()
    data["content_cutted"] = data.微博内容.apply(chinese_word_cut)
    lda, tf, tf_vectorizer, n_topics = analysis(data, n_topics)
    visualize(lda, tf, tf_vectorizer, n_topics)


if __name__ == "__main__":
    LDA()
