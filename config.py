import os

# 爬取前几页
max_page = int(os.environ.get("MAX_PAGE")) if os.environ.get("MAX_PAGE") != None else 10
# 搜索关键字
search_keyword = (
    os.environ.get("SEARCH_KEYWORD")
    if os.environ.get("SEARCH_KEYWORD") != None
    else "唐山打人"
)
# 正向情感趋势图的时间单位
time_division_way = (
    os.environ.get("TIME_UNIT") if os.environ.get("TIME_UNIT") != None else "H"
)  # H、D、M
