# 微博评论情感分析案例

- 使用crawler.py爬取包含某一热搜（自定义关键词）的微博评论作为预测数据。
- 使用bert_data_preprocesssing.py合并几个现有的微博评论数据集，做训练数据。
- 使用bert_fine_tuning.py训练模型。
- 使用new_data_predict.py预测自己爬取的评论数据，情感分为正向和负向。
  

# 微博LDA主题分析案例

- 使用crawler.py爬取包含某一热搜（自定义关键词）的微博评论作为预测数据。

- 使用LDA.py对需要预测的评论数据做主题分析。

# docker（已训练好bert模型的情况）

``` bash
docker build -t weibo_analyse:v0.1 ./
docker run \
--env MAX_PAGE=10 \
--env SEARCH_KEYWORD=唐山打人 \
--env TIME_UNIT=H \
-v result_store_path:/weibo_analysis/result \
weibo_analyse -itd
```

docker run --env MAX_PAGE=10 --env SEARCH_KEYWORD=唐山打人 --env TIME_UNIT=H -v your\path\weibo_analyse\result:/weibo_analysis/result weibo_analyse:latest -itd

