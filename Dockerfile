from python:slim
COPY ./ weibo_analysis
WORKDIR /weibo_analysis
RUN pip install -r requirements.txt -i https://mirror.lzu.edu.cn/pypi/web/simple
ENTRYPOINT [ "bash", "run.sh" ]