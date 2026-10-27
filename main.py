from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests


# 项目目录
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# 设置中文图表字体，适用于 Windows
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

# 国家代码和图表中使用的中文名称
COUNTRY_NAMES = {
    "CHN": "中国",
    "USA": "美国",
    "JPN": "日本",
    "DEU": "德国",
    "IND": "印度",
    "BRA": "巴西",
}

# 世界银行指标：使用互联网的人口比例
INDICATOR = "IT.NET.USER.ZS"


def fetch_data():
    """从世界银行公开 API 获取数据，并保存原始 CSV。"""
    country_path = ";".join(COUNTRY_NAMES.keys())
    url = (
        f"https://api.worldbank.org/v2/country/"
        f"{country_path}/indicator/{INDICATOR}"
    )

    response = requests.get(
        url,
        params={
            "format": "json",
            "per_page": 20000,
        },
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()

    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or payload[1] is None
    ):
        raise ValueError(f"API 返回了无法识别的数据：{payload}")

    records = []

    for item in payload[1]:
        records.append(
            {
                "country": item["country"]["value"],
                "country_code": item["countryiso3code"],
                "year": item["date"],
                "internet_users_pct": item["value"],
            }
        )

    raw_data = pd.DataFrame(records)
    raw_path = DATA_DIR / "internet_usage_raw.csv"
    raw_data.to_csv(raw_path, index=False, encoding="utf-8-sig")

    print(f"第一步完成：获取了 {len(raw_data)} 行原始数据。")
    print(f"原始数据已保存到：{raw_path}")


def clean_data():
    """读取原始数据，处理类型、缺失值、重复值和异常范围。"""
    raw_path = DATA_DIR / "internet_usage_raw.csv"
    raw_data = pd.read_csv(raw_path)

    # 将年份和指标转换为数字；无法转换的内容会变成缺失值
    raw_data["year"] = pd.to_numeric(
        raw_data["year"],
        errors="coerce",
    )
    raw_data["internet_users_pct"] = pd.to_numeric(
        raw_data["internet_users_pct"],
        errors="coerce",
    )

    # 删除关键字段缺失的记录
    clean = raw_data.dropna(
        subset=["year", "internet_users_pct"]
    ).copy()

    clean["year"] = clean["year"].astype(int)

    # 本项目只研究 2000 年及以后的数据
    clean = clean[clean["year"] >= 2000]

    # 百分比应当位于 0 到 100 之间
    clean = clean[
        clean["internet_users_pct"].between(0, 100)
    ]

    # 同一个国家、同一个年份只保留一条记录
    clean = clean.drop_duplicates(
        subset=["country_code", "year"],
        keep="last",
    )

    clean["country_zh"] = clean["country_code"].map(
        COUNTRY_NAMES
    )

    clean = clean.sort_values(
        ["country_code", "year"]
    ).reset_index(drop=True)

    clean_path = DATA_DIR / "internet_usage_clean.csv"
    clean.to_csv(
        clean_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"第二步完成：清洗后剩余 {len(clean)} 行数据。")
    print(f"清洗数据已保存到：{clean_path}")

    return clean


def analyze_and_plot(clean):
    """统计数据，生成三张图表和三条数据结论。"""

    # 找出六个国家都有数据的年份，避免拿不同年份直接比较
    countries_per_year = clean.groupby("year")[
        "country_code"
    ].nunique()

    complete_years = countries_per_year[
        countries_per_year == len(COUNTRY_NAMES)
    ].index

    if len(complete_years) == 0:
        raise ValueError("没有找到六个国家都有数据的共同年份。")

    first_year = int(min(complete_years))
    latest_year = int(max(complete_years))

    # 图一：各国互联网普及率变化趋势
    plt.figure(figsize=(11, 6))

    for country_code, country_name in COUNTRY_NAMES.items():
        group = clean[
            clean["country_code"] == country_code
        ].sort_values("year")

        plt.plot(
            group["year"],
            group["internet_users_pct"],
            linewidth=2,
            label=country_name,
        )

    plt.title("六国互联网普及率变化趋势")
    plt.xlabel("年份")
    plt.ylabel("使用互联网的人口比例（%）")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "01_trend.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # 最新共同年份的国家对比
    latest = clean[clean["year"] == latest_year].copy()
    latest = latest.sort_values(
        "internet_users_pct",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))
    plt.bar(
        latest["country_zh"],
        latest["internet_users_pct"],
    )
    plt.title(f"{latest_year} 年六国互联网普及率对比")
    plt.xlabel("国家")
    plt.ylabel("使用互联网的人口比例（%）")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "02_latest_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # 计算第一个共同年份到最新共同年份的变化
    comparison = clean[
        clean["year"].isin([first_year, latest_year])
    ]

    comparison_wide = comparison.pivot(
        index="country_zh",
        columns="year",
        values="internet_users_pct",
    )

    comparison_wide["increase_pp"] = (
        comparison_wide[latest_year]
        - comparison_wide[first_year]
    )

    comparison_wide = comparison_wide.sort_values(
        "increase_pp"
    )

    comparison_wide.to_csv(
        DATA_DIR / "country_comparison.csv",
        encoding="utf-8-sig",
    )

    # 图三：各国互联网普及率增幅
    plt.figure(figsize=(10, 6))
    plt.barh(
        comparison_wide.index,
        comparison_wide["increase_pp"],
    )
    plt.title(
        f"{first_year}—{latest_year} 年六国互联网普及率增幅"
    )
    plt.xlabel("增幅（百分点）")
    plt.ylabel("国家")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "03_growth.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # 生成三条结论
    top = latest.iloc[0]
    bottom = latest.iloc[-1]

    gap = (
        top["internet_users_pct"]
        - bottom["internet_users_pct"]
    )

    fastest_country = comparison_wide[
        "increase_pp"
    ].idxmax()

    fastest_growth = comparison_wide.loc[
        fastest_country,
        "increase_pp",
    ]

    china_value = latest.loc[
        latest["country_code"] == "CHN",
        "internet_users_pct",
    ].iloc[0]

    six_country_average = latest[
        "internet_users_pct"
    ].mean()

    if china_value >= six_country_average:
        relation = "高于"
    else:
        relation = "低于"

    summary = f"""六国互联网普及率分析结论

1. 在六个国家都有完整数据的最新年份（{latest_year} 年），{top['country_zh']}的互联网普及率最高，为 {top['internet_users_pct']:.1f}%；{bottom['country_zh']}最低，为 {bottom['internet_users_pct']:.1f}%，两者相差 {gap:.1f} 个百分点。

2. 从 {first_year} 年到 {latest_year} 年，{fastest_country}的互联网普及率增幅最大，增加了 {fastest_growth:.1f} 个百分点。这说明其在该统计期间内增长较快，但不能仅凭这一指标推断增长原因。

3. {latest_year} 年，中国的互联网普及率为 {china_value:.1f}%，{relation}六国平均值 {six_country_average:.1f}%。这里的平均值仅代表本项目选取的六个国家，不代表全球平均水平。

分析限制：
本研究只比较国家层面的年度指标，没有分析城乡、年龄和收入群体差异，也不对变化原因作因果推断。
"""

    summary_path = BASE_DIR / "analysis_summary.txt"
    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print("第三步完成：三张图表已经生成。")
    print(summary)
    print(f"结论已保存到：{summary_path}")


def main():
    fetch_data()
    clean = clean_data()
    analyze_and_plot(clean)
    print("全部流程运行完成！")


if __name__ == "__main__":
    main()