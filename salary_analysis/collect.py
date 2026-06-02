"""
招聘数据采集器 - 在宿主机上运行，采集真实数据写入 Docker MySQL

用法:
  python collect.py --city 北京 --pages 3            # 单个城市
  python collect.py --all --pages 3                   # 全部城市
  python collect.py --cron                            # 显示crontab配置

数据来源: 前程无忧(51job)
写入目标: Docker MySQL (localhost:3306 -> salary_analysis.recruitment_info)
"""

import time
import random
import json
import re
import sys
import argparse
import requests
import pymysql
from datetime import datetime

# ── MySQL 连接（通过宿主机 127.0.0.1 连 Docker） ──
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '@Swb112988',
    'database': 'salary_analysis',
    'charset': 'utf8mb4'
}

CITIES = ['北京', '上海', '深圳', '杭州', '广州', '成都', '武汉', '南京', '西安', '重庆']

CITY_CODES = {
    '北京': '010000', '上海': '020000', '深圳': '040000', '杭州': '080000',
    '广州': '030000', '成都': '090000', '武汉': '170000', '南京': '070000',
    '西安': '250000', '重庆': '060000'
}

INSERT_SQL = """
    INSERT INTO recruitment_info
    (platform, position_name, company_name, salary_min, salary_max, salary_avg,
     city, experience, education, publish_time)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    salary_min=VALUES(salary_min), salary_max=VALUES(salary_max),
    salary_avg=VALUES(salary_avg), crawl_time=CURRENT_TIMESTAMP
"""

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS recruitment_info (
        id INT AUTO_INCREMENT PRIMARY KEY,
        platform VARCHAR(50),
        position_name VARCHAR(100),
        company_name VARCHAR(100),
        salary_min INT,
        salary_max INT,
        salary_avg DECIMAL(10,2),
        city VARCHAR(50),
        experience VARCHAR(20),
        education VARCHAR(20),
        publish_time VARCHAR(20),
        crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""


def ensure_table():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(CREATE_TABLE_SQL)
    conn.commit()
    cursor.close()
    conn.close()
    print('[OK] 表已就绪')


def fetch_page(session, city_code, page):
    """请求51job单页，返回原始JSON中的职位列表"""
    url = (
        'https://search.51job.com/jobsearch/searchResult.php'
        f'?keyword=Python&jobarea={city_code}'
        f'&curr_page={page}&pagesize=50'
        f'&lang=c&postchannel=0000&workyear=99&cotype=99'
        f'&degreefrom=99&jobterm=99&companysize=99&ord_field=0'
    )
    resp = session.get(url, timeout=20)
    resp.encoding = 'gbk'
    if resp.status_code != 200:
        print(f'    HTTP {resp.status_code}')
        return []
    html = resp.text
    if len(html) < 1000:
        print(f'    页面过短({len(html)}字符)')
        return []
    match = re.search(r'window\.__SEARCH_RESULT__\s*=\s*({.*?});', html, re.DOTALL)
    if not match:
        print('    未找到JSON数据')
        return []
    data = json.loads(match.group(1))
    return data.get('engine_search_result', []) or data.get('jobList', [])


def parse_jobs(jobs, city):
    """解析原始职位列表为入库记录"""
    records = []
    for job in jobs:
        try:
            pos = (job.get('job_name', '') or job.get('jobtitle', '') or '').strip()
            if not pos:
                continue
            company = (job.get('company_name', '') or job.get('companyname', '') or '').strip()
            salary_str = (job.get('providesalary', '') or job.get('salary', '') or '')
            sal_match = re.search(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*(?:[Kk万])', salary_str)
            if not sal_match:
                continue
            a, b = float(sal_match.group(1)), float(sal_match.group(2))
            if a <= 0 or b <= 0 or b - a > 80:
                continue
            if '万' in salary_str:
                min_sal, max_sal = int(a * 10000), int(b * 10000)
            else:
                min_sal, max_sal = int(a * 1000), int(b * 1000)
            if min_sal < 2000 or max_sal > 200000:
                continue
            avg_sal = (min_sal + max_sal) // 2
            records.append((
                '前程无忧', pos, company,
                min_sal, max_sal, avg_sal,
                city,
                str(job.get('workyear', '')),
                str(job.get('degree', '')),
                str(job.get('issuedate', ''))
            ))
        except Exception:
            continue
    return records


def save_batch(cursor, records):
    count = 0
    for rec in records:
        try:
            cursor.execute(INSERT_SQL, rec)
            count += 1
        except pymysql.Error:
            continue
    return count


def collect_one_city(city, pages=3):
    """采集一个城市的数据"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://www.51job.com/',
    })
    session.get('https://www.51job.com/', timeout=10)

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    total = 0
    city_code = CITY_CODES.get(city, '010000')
    now = datetime.now().strftime('%H:%M:%S')

    print(f'[{now}] {city} 开始采集 {pages} 页...')

    for page in range(1, pages + 1):
        try:
            jobs = fetch_page(session, city_code, page)
            if not jobs:
                print(f'  [第{page}页] 无数据')
                time.sleep(random.uniform(2, 4))
                continue
            records = parse_jobs(jobs, city)
            saved = save_batch(cursor, records)
            conn.commit()
            total += saved
            print(f'  [第{page}页] 获取{len(jobs)}个 -> 保存{saved}个')
        except Exception as e:
            print(f'  [第{page}页] 异常: {e}')
            conn.rollback()
        time.sleep(random.uniform(1.5, 3))

    cursor.close()
    conn.close()
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {city} 完成, 共保存 {total} 条')
    return total


def main():
    parser = argparse.ArgumentParser(description='前程无忧招聘数据采集器')
    parser.add_argument('--city', default='北京', help='城市')
    parser.add_argument('--industry', default='互联网', help='行业(仅用于展示)')
    parser.add_argument('--pages', type=int, default=3, help='每城市采集页数')
    parser.add_argument('--all', action='store_true', help='采集所有城市')
    parser.add_argument('--cron', action='store_true', help='显示crontab配置')
    args = parser.parse_args()

    print('=' * 45)
    print('前程无忧招聘数据采集器')
    print('=' * 45)

    ensure_table()

    if args.cron:
        script_path = sys.argv[0]
        full_path = script_path if script_path.startswith('/') else f'$(pwd)/{script_path}'
        print()
        print('将以下行加入 crontab (crontab -e):')
        print()
        print(f'0 */2 * * * cd {sys.path[0] or "."} && python3 {full_path} --all --pages 3 >> /tmp/collect.log 2>&1')
        print()
        print('或立即执行一轮: python3 collect.py --all --pages 3')
        return

    if args.all:
        total = 0
        for city in CITIES:
            count = collect_one_city(city, args.pages)
            total += count
        print(f'\n全部完成! 共 {total} 条数据')
    else:
        collect_one_city(args.city, args.pages)


if __name__ == '__main__':
    main()
