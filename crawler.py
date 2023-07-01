import datetime  # 转换时间用
import multiprocessing as mp  # 多进程
import os
import re  # 正则表达式提取文本

import pandas as pd  # 存取csv文件
import requests  # 发送请求
from jsonpath import jsonpath  # 解析json数据

import config


def trans_time(v_str):
    """转换GMT时间为标准格式"""
    gmt_format = "%a %b %d %H:%M:%S +0800 %Y"
    time_array = datetime.datetime.strptime(v_str, gmt_format)
    ret_time = time_array.strftime("%Y-%m-%d %H:%M:%S")
    return ret_time


def get_weibo_list(v_keyword, page):
    """
    爬取微博内容列表
    :param v_keyword:搜索关键字
    :param v_max_page: 爬取前几页
    :return: None
    """
    # 请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36 Edg/103.0.1264.62",
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
    }
    df_all = pd.DataFrame(
        columns=["页码", "微博id", "微博bid", "微博作者", "发布时间", "微博内容", "转发数", "评论数", "点赞数"]
    )

    print("===开始爬取第{}页微博==".format(page))
    # 请求地址
    url = "https://m.weibo.cn/api/container/getIndex"
    # 请求参数
    params = {
        "containerid": "100103type=1&q={}".format(v_keyword),
        "page_type": "searchall",
        "page": page,
    }
    # 发送请求
    r = requests.get(url, headers=headers, params=params)
    # 解析json数据
    cards = r.json()["data"]["cards"]
    # 微博内容
    text_list = jsonpath(cards, "$..mblog.text")
    # 微博内容-正则表达式数据清洗
    dr = re.compile(r"<[^>]+>", re.S)
    text2_list = []
    if not text_list:  # 如果未获取到微博内容，进入下一轮循环
        print("{}页爬取失败".format(page))
        return pd.DataFrame(
            columns=["页码", "微博id", "微博bid", "微博作者", "发布时间", "微博内容", "转发数", "评论数", "点赞数"]
        )
    if type(text_list) == list and len(text_list) > 0:
        for text in text_list:
            text2 = dr.sub("", text)  # 正则表达式提取微博内容
            text2_list.append(text2)
    # 微博创建时间
    time_list = jsonpath(cards, "$..mblog.created_at")
    time_list = [trans_time(v_str=i) for i in time_list]
    # 微博作者
    author_list = jsonpath(cards, "$..mblog.user.screen_name")
    # 微博id
    id_list = jsonpath(cards, "$..mblog.id")
    # 微博bid
    bid_list = jsonpath(cards, "$..mblog.bid")
    # 转发数
    reposts_count_list = jsonpath(cards, "$..mblog.reposts_count")
    # 评论数
    comments_count_list = jsonpath(cards, "$..mblog.comments_count")
    # 点赞数
    attitudes_count_list = jsonpath(cards, "$..mblog.attitudes_count")
    # 把列表数据保存成DataFrame数据
    df = pd.DataFrame(
        {
            "页码": [page] * len(id_list),
            "微博id": id_list,
            "微博bid": bid_list,
            "微博作者": author_list,
            "发布时间": time_list,
            "微博内容": text2_list,
            "转发数": reposts_count_list,
            "评论数": comments_count_list,
            "点赞数": attitudes_count_list,
        }
    )

    return df


def crawler(max_page, search_keyword):
    v_weibo_file = "dataset/test/微博清单_{}_前{}页.csv".format(search_keyword, max_page)
    if os.path.exists(v_weibo_file):
        os.remove(v_weibo_file)
        print("文件已存在，已删除：{}".format(v_weibo_file))

    res_l = []  # 储存进程池中输出的结果
    po = mp.Pool(10)  # 引入进程池
    for page in range(max_page):
        res = po.apply_async(
            get_weibo_list, (search_keyword, page + 1)
        )  # getInfo为具体的爬取函数
        res_l.append(res)
    po.close()  # 关闭进程池，关闭后po不再接收新的请求
    po.join()  # 等待po中的所有子进程执行完成，必须放在close语句之后
    print("爬取完毕，开始整合数据...")

    df = pd.DataFrame(
        columns=["页码", "微博id", "微博bid", "微博作者", "发布时间", "微博内容", "转发数", "评论数", "点赞数"]
    )  # 储存进程池中输出的结果
    for res in res_l:
        res_ = pd.DataFrame(res.get())
        df = pd.concat([df, res_])  # 把进程池的结果合并到dataframe里面

    # 数据清洗-去重
    df.drop_duplicates(subset=["微博id"], inplace=True, keep="first")
    # 再次保存到csv文件
    df.to_csv(v_weibo_file, index=False, encoding="utf-8")
    print("数据保存完毕")
    return df


if __name__ == "__main__":
    # # 爬取前几页
    max_page = config.max_page
    # 搜索关键字
    search_keyword = config.search_keyword

    # 保存文件名
    df = crawler(max_page, search_keyword)
