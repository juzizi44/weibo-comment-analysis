import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd  # 存取csv文件
import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

import config
from bert_data_preprocesssing import remove_chinese_characters

# 页数
max_page = config.max_page
# 关键字
search_keyword = config.search_keyword
# 正向情感趋势图的时间单位
time_division_way = config.time_division_way


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertClassificationModel(nn.Module):
    def __init__(self, hidden_size=768):  # bert默认最后输出维度为768
        super(BertClassificationModel, self).__init__()
        model_name = "bert-base-chinese"
        # 读取分词器
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name
        )
        # 读取预训练模型
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)

        for p in self.bert.parameters():  # 冻结bert参数
            p.requires_grad = False

        num_class = 2
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, batch_sentences):  # [batch_size,1]
        # 编码
        sentences_tokenizer = self.tokenizer(
            batch_sentences,
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=True,
        )
        input_ids = torch.tensor(sentences_tokenizer["input_ids"]).to(device)  # 变量
        attention_mask = torch.tensor(sentences_tokenizer["attention_mask"]).to(
            device
        )  # 变量
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # 模型

        last_hidden_state = bert_out[0].to(
            device
        )  # [batch_size, sequence_length, hidden_size] # 变量
        bert_cls_hidden_state = last_hidden_state[:, 0, :].to(device)  # 变量
        fc_out = self.fc(bert_cls_hidden_state)  # 模型
        return fc_out


def crawl_data_preprocessing(df):
    """
    对爬取下来的微博数据进行清洗。
    :param df: 爬取的微博数据列表
    :return: df
    """
    weibo_list = df.dropna(how="any")
    weibo_list["微博内容"] = weibo_list["微博内容"].map(remove_chinese_characters)  # 去除非中文字符
    weibo_list = weibo_list.drop_duplicates(keep="last")  # 去重
    weibo_list = weibo_list.dropna(how="any")  # 删除空值

    weibo_list["发布时间"] = pd.to_datetime(weibo_list["发布时间"])  # 将数据类型转换为日期类型
    weibo_list.set_index("发布时间", inplace=True)  # 将发布时间设置为index
    weibo_list.sort_values(by="发布时间", ascending=True, inplace=True)  # 时间升序--其实没什么用吧

    return weibo_list


def positive_sentiment_prediction(df, model):
    """
    将df中的所有数据一次性传入模型中进行训练，进行情感预测。（没有进度条，不建议用这个）
    :param df: 已清洗的微博数据
    :param model: 训练好的模型
    :return: total_positive_rate,  # 正向情感比率
             pos_list,  # 正向情感评论列表
             nag_list   # 负向情感评论列表
    """
    # 将数据输入模型当中
    output = model(list(df["微博内容"]))
    # 正面负面评论列表
    pos_list = []
    nag_list = []
    # 统计正面评论个数
    total_positive = 0
    for i, label in enumerate(output.argmax(1)):
        if label == 1:
            total_positive += 1
            pos_list.append(df["微博内容"][i])
        else:
            nag_list.append(df["微博内容"][i])
    # 正面评论比率
    total_positive_rate = total_positive / len(output)
    pos_df = pd.Series(data=pos_list)
    nag_df = pd.Series(data=nag_list)

    return (total_positive_rate, pos_df, nag_df)


def sentiment_prediction_per_time(df, model, time_division_way):
    """
    将df按照时间划分，传入模型中进行训练，分别对不同时间点进行情感预测。
    :param df: 已清洗的微博数据
    :param model: 训练好的模型
    :param time_division_way: 时间划分方式
    :return: total_positive_rate,  # 总体正向情感比率
             positive_rate_per_unit_time,  # 不同时间正向情感比率
             pos_df,   # 正向情感评论表
             nag_df    # 负向情感评论表
    """
    df_period = df.to_period(time_division_way)  # 按月/时显示，但不统计
    times = df_period.index.drop_duplicates(keep="first")  # 时间点

    positive_rate_per_unit_time = []  # 存储不同时间的正向比率
    pos_list = dict([(k, []) for k in times])  # 正面评论汇总
    nag_list = dict([(k, []) for k in times])  # 负面评论汇总
    total_positive = 0  # 总体正向数量

    for time in tqdm(times):
        contents_per_unit_time = df_period[df_period.index == time][
            "微博内容"
        ]  # 每个时间区域的微博内容
        output = model(list(contents_per_unit_time))  # 将微博内容以列表形式输入模型
        pos_per_unit_time = 0  # 每个时间区域的正向微博内容数量

        for i, label in enumerate(output.argmax(1)):
            if label == 1:
                pos_per_unit_time += 1
                pos_list[time].append(contents_per_unit_time[i])
            else:
                nag_list[time].append(contents_per_unit_time[i])

        temp = pos_per_unit_time / len(output)
        positive_rate_per_unit_time.append(temp)
        total_positive += pos_per_unit_time  # 总体正向比率

    total_positive_rate = total_positive / len(df)
    # 将正向评论负向评论按照时间，存入dataframe中
    pos_df = pd.DataFrame(
        dict([(k, pd.Series(v, dtype="object")) for k, v in pos_list.items()])
    )  # 不等长字典转化为dataframe类型
    nag_df = pd.DataFrame(
        dict([(k, pd.Series(v, dtype="object")) for k, v in nag_list.items()])
    )

    return (total_positive_rate, positive_rate_per_unit_time, pos_df, nag_df)


def emotional_development_pic(
    df, time_division_way, positive_rate_per_unit_time, total_positive_rate
):
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
    matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号

    # t：时间坐标
    df_period = df.to_period(time_division_way)  # 按月/时显示，但不统计
    times = df_period.index.drop_duplicates(keep="first")
    t = []
    for time in times:
        t.append(str(time))

    # 画布
    plt.figure(figsize=(25, 15))
    plt.plot(
        t, positive_rate_per_unit_time, "r--", marker="d", label="正向比率", linewidth=1
    )
    plt.legend()
    plt.xticks(rotation=45)
    plt.title("正向情感比率走势图")
    plt.text(4, 1, "总体正向情感比率为{}".format(total_positive_rate))

    # 判断结果文件夹是否存在
    path = "result/微博清单_{}_前{}页/statistical_pictures".format(search_keyword, max_page)
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + "/emotional_development_pic.jpg")


def heat_development_pic(df, time_division_way):
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
    matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号

    dfn = df.resample(time_division_way).sum()
    # 画布
    plt.figure(figsize=(25, 15))

    plt.plot(
        dfn.index,
        dfn.点赞数,
        marker="d",
        label="点赞数",
    )  # o,x,d,v
    plt.plot(dfn.index, dfn.评论数, marker="o", label="评论数")  # o,x,d,v
    plt.plot(dfn.index, dfn.转发数, marker="v", label="转发数")  # o,x,d,v

    # 显示图例
    plt.legend()
    plt.xticks(rotation=45)
    plt.title("事件热度走势图")

    # 判断结果文件夹是否存在
    path = "result/微博清单_{}_前{}页/statistical_pictures".format(search_keyword, max_page)
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + "/heat_development_pic.jpg")


if __name__ == "__main__":

    # 读取数据
    df = pd.read_csv("dataset/test/微博清单_{}_前{}页.csv".format(search_keyword, max_page))

    # 读取模型--如果设备是cpu，选择第一个，如果是gpu，选择第二个（有gpu尽量用gpu）
    model = torch.load("model/model_20.pth", map_location="cpu")
    # model = torch.load('model/model_20.pth').to(device)

    # 清洗数据
    df = crawl_data_preprocessing(df)

    # 总体情感预测--不建议用这个
    # total_positive_rate, pos_df, nag_df = positive_sentiment_prediction(df, model)

    # 不同时间情感预测（这种方式也能输出总体正向情感比率）
    (
        total_positive_rate,
        positive_rate_per_unit_time,
        pos_df,
        nag_df,
    ) = sentiment_prediction_per_time(df, model, time_division_way)

    # 判断结果文件夹是否存在
    path = "result/微博清单_{}_前{}页".format(search_keyword, max_page)
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + " 创建成功")

    # 正面评论保存至csv中
    pos_df.to_csv("result/微博清单_{}_前{}页/预测正面评论.csv".format(search_keyword, max_page))
    # 负面评论保存至csv中
    nag_df.to_csv("result/微博清单_{}_前{}页/预测负面评论.csv".format(search_keyword, max_page))

    print("正面评论占比：{}".format(total_positive_rate))
    print("分时段正面评论占比:{}".format(positive_rate_per_unit_time))

    # 绘图
    emotional_development_pic(
        df, time_division_way, positive_rate_per_unit_time, total_positive_rate
    )

    heat_development_pic(df, time_division_way)
# %%
