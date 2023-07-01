import re

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.utils import shuffle

"""
    数据集来源：
        1、weibo_senti_100k:
            数据概览： 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条
            https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb
            
        2、simplifyweibo_4_moods：(本案例并未使用36万条，只使用了一半)
            数据概览： 36 万多条，带情感标注 新浪微博，包含 4 种情感，其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条
            https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb
            
        3、online_shopping_10_cats：
            数据概览： 10个类别，共6万多条评论数据，正、负向评论各约3万条，包括书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店
            https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb
            
        4、CCF_NewsSenAna_Transformers2：
            数据概览： 对新闻情绪进行分类，0代表正面情绪、1代表中性情绪、2代表负面情绪。
            https://github.com/DefuLi/Emotional-Analysis-Transformers2.0-Bert/tree/master/CCF_NewsSenAna_Transformers2/data
            
        5、CCF_NewsSenAna
            数据概览： 对新闻情绪进行分类，0代表正面情绪、1代表中性情绪、2代表负面情绪。
            https://github.com/DefuLi/Emotional-Analysis-of-Internet-News/tree/master/CCF_NewsSenAna/trainfiles
          
            
        6、ChnSentiCorp_htl_all：
            数据概览： 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论
            https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb
            
        7、waimai_10k：
            数据概览： 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条
            https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb
            
        8、Training_data_for_Emotion_Classification：
            数据概览： 第二届自然语言处理与中文计算会议（NLP&CC 2013），大小：10 000 条微博语料，标注了7 emotions: like, disgust, happiness, sadness, anger, surprise, fear
            http://tcci.ccf.org.cn/conference/2014/dldoc/evtestdata1.zip
            
            
            
"""


def weibo_senti_100k():  # 数据集处理
    wei = pd.read_csv("dataset/raw/weibo_senti_100k.csv")
    return wei


def simplifyweibo_4_moods():  # 数据集处理
    sim = pd.read_csv(
        "dataset/raw/simplifyweibo_4_moods.csv"
    )  # 原数据集{0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}
    dic = {0: 1, 1: 0, 2: 0, 3: 0}
    sim["label"] = sim["label"].map(dic)
    return sim


def online_shopping_10_cats():  # 数据集处理
    onl = pd.read_csv("dataset/raw/online_shopping_10_cats.csv")
    onl.drop(["cat"], axis=1, inplace=True)
    return onl


def CCF_NewsSenAna_Transformers2():  # 数据集处理
    new = pd.read_csv(
        "dataset/raw/CCF_NewsSenAna_Transformers2.csv"
    )  # 原数据集0代表正面情绪、1代表中性情绪、2代表负面情绪
    dic = {0: 1, 1: 1, 2: 0}
    new["label"] = new["label"].map(dic)
    new["review"] = new["text"]
    new.drop(["id", "text"], axis=1, inplace=True)
    return new


def CCF_NewsSenAna():  # 数据集处理
    new2 = pd.read_csv("dataset/raw/CCF_NewsSenAna.csv")  # 原数据集0代表正面情绪、1代表中性情绪、2代表负面情绪
    dic = {0: 1, 1: 1, 2: 0}
    new2["label"] = new2["label"].map(dic)
    new2["review"] = new2["content"]
    new2.drop(["id", "content", "title"], axis=1, inplace=True)
    return new2


def ChnSentiCorp_htl_all():  # 数据集处理
    chn = pd.read_csv("dataset/raw/ChnSentiCorp_htl_all.csv")
    return chn


def waimai_10k():  # 数据集处理
    wai = pd.read_csv("dataset/raw/waimai_10k.csv")
    return wai


def Training_data_for_Emotion_Classification():  # 数据集处理
    def emotion_convert(string):
        dictionary = {
            "like": 1,
            "disgust": 0,
            "happiness": 1,
            "sadness": 0,
            "anger": 0,
            "surprise": 1,
            "fear": 0,
        }
        return dictionary.get(string, None)

    file = open(
        "dataset/raw/Training data for Emotion Classification.xml",
        "r",
        encoding="utf-8",
    )
    txt = file.read()
    file.close()

    soup = BeautifulSoup(txt, "html.parser")
    nlpcc = pd.DataFrame(columns=["label", "review"])
    for tag in soup.find_all("sentence"):
        if tag.attrs["opinionated"] == "N":
            label = 1
        elif tag.attrs["opinionated"] == "Y":
            label = emotion_convert(tag.attrs["emotion-1-type"])
        nlpcc.loc[len(nlpcc.index)] = [label, tag.string]  # 向nlpcc的末尾添加

    return nlpcc


def remove_chinese_characters(content):  # 只保留文本中的中文   ？？？？不确定是否要进行这一步
    content = re.sub("[^\u4e00-\u9fa5]+", "", content)
    return content


def clean_data(df):  # 对合并后的数据集进行数据清洗
    df = df.dropna(how="any")
    df.review = df.review.map(remove_chinese_characters)
    df = shuffle(df)
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(keep="last")
    df = df.dropna(how="any")
    return df


if __name__ == "__main__":
    wei = weibo_senti_100k()
    sim = simplifyweibo_4_moods()
    onl = online_shopping_10_cats()
    new = CCF_NewsSenAna_Transformers2()
    new2 = CCF_NewsSenAna()
    chn = ChnSentiCorp_htl_all()
    wai = waimai_10k()
    nlpcc = Training_data_for_Emotion_Classification()

    # 数据合并
    df = pd.concat([wei, sim, onl, new, new2, chn, wai, nlpcc], axis=0)

    # 数据清洗
    df = clean_data(df)

    # 数据保存
    df.to_csv("dataset/train/weibo_train_data.csv", index=False)
