"""
IMSDb 電影劇本爬蟲
爬取電影元數據、用戶評論和劇本文本

注意：採用「全有全無」策略 - 如果劇本抓不到，整部電影的資料都不會被寫入
"""

import requests
import time
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
import logging
import argparse

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置
BASE_URL = "https://imsdb.com"
DATA_DIR = Path("data")
SCRIPTS_DIR = DATA_DIR / "scripts"
PROGRESS_FILE = DATA_DIR / "progress.json"
MOVIES_FILE = DATA_DIR / "movies.json"
DEFAULT_DELAY = 2.0  # 預設請求間隔（秒）

# HTTP Session
session = requests.Session()
session.headers.update({
    "User-Agent": "MovieResearchBot/1.0 (Academic Research Project)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
})


def slugify(title: str) -> str:
    """將標題轉為 URL-safe ID"""
    slug = re.sub(r'[^\w\s-]', '', title)
    slug = re.sub(r'[-\s]+', '-', slug).strip('-')
    return slug.lower()


def fetch(url: str, delay: float, retries: int = 3) -> str:
    """帶重試的 HTTP GET 請求"""
    full_url = urljoin(BASE_URL, url) if not url.startswith('http') else url

    for attempt in range(retries):
        try:
            resp = session.get(full_url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"請求失敗 (嘗試 {attempt + 1}/{retries}): {full_url} - {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    raise Exception(f"無法取得: {full_url}")


def get_movie_list(delay: float) -> list[dict]:
    """從 all-scripts.html 取得所有電影連結"""
    logger.info("正在取得電影列表...")
    html = fetch("/all-scripts.html", delay)
    soup = BeautifulSoup(html, "lxml")

    movies = []
    for link in soup.select('a[href*="/Movie Scripts/"]'):
        href = link.get("href", "")
        title = link.get_text(strip=True)

        if href and title and "Script.html" in href:
            movie_id = slugify(title)
            movies.append({
                "id": movie_id,
                "title": title,
                "detail_url": href,
            })

    logger.info(f"找到 {len(movies)} 部電影")
    return movies


def scrape_detail_page(url: str, delay: float) -> dict:
    """爬取電影詳情頁，提取元數據和用戶評論"""
    html = fetch(url, delay)
    soup = BeautifulSoup(html, "lxml")

    data = {
        "writers": [],
        "genres": [],
        "avg_rating": None,
        "script_url": None,
        "script_date": None,
        "reviews": [],
    }

    details_table = soup.select_one('table.script-details')

    if details_table:
        # 提取編劇
        for link in details_table.select('a[href*="writer.php"]'):
            writer = link.get_text(strip=True)
            if writer and writer not in data["writers"]:
                data["writers"].append(writer)

        # 提取類型
        for link in details_table.select('a[href*="/genre/"]'):
            genre = link.get_text(strip=True)
            if genre and genre not in data["genres"]:
                data["genres"].append(genre)

        # 提取劇本連結
        script_link = details_table.select_one('a[href*="/scripts/"]')
        if script_link:
            data["script_url"] = script_link.get("href")

        # 提取平均評分
        details_text = details_table.get_text()
        avg_match = re.search(r'\((\d+\.?\d*)\s*out of\s*10\)', details_text)
        if avg_match:
            try:
                data["avg_rating"] = float(avg_match.group(1))
            except ValueError:
                pass

        # 提取劇本日期
        date_match = re.search(r'Script Date\s*:\s*([^\n<]+)', str(details_table))
        if date_match:
            data["script_date"] = date_match.group(1).strip()

    # 提取用戶評論
    comments_table = soup.select_one('table.script-comments')
    if comments_table:
        reviews = extract_reviews_from_table(comments_table)
        data["reviews"] = reviews

    return data


def extract_reviews_from_table(comments_table) -> list[dict]:
    """從評論表格提取用戶評論"""
    reviews = []
    html_content = str(comments_table)

    review_pattern = re.compile(
        r'<p><b>([^<]+)</b>\s*'
        r'<img[^>]*src=["\']?[^"\']*?(\d+)-stars?[^"\']*["\']?[^>]*>\s*'
        r'\((\d+)\s*out\s*of\s*10\s*\)\s*<br>?'
        r'(.*?)</p>',
        re.DOTALL | re.IGNORECASE
    )

    for match in review_pattern.finditer(html_content):
        user = match.group(1).strip()
        rating = int(match.group(3))
        review_html = match.group(4)

        review_soup = BeautifulSoup(review_html, "lxml")
        review_text = review_soup.get_text(strip=True)
        review_text = re.sub(r'^/?>?\s*', '', review_text)

        if user and review_text and len(review_text) > 3:
            reviews.append({
                "user": user,
                "rating": rating,
                "text": review_text[:1000],
            })

    return reviews


def scrape_script(url: str, delay: float) -> str:
    """爬取劇本文本"""
    html = fetch(url, delay)
    soup = BeautifulSoup(html, "lxml")

    # 劇本通常在 <pre> 標籤中
    pre_tag = soup.find("pre")
    if pre_tag:
        return pre_tag.get_text()

    # 備選: 有些可能在 <td class="scrtext">
    script_td = soup.select_one("td.scrtext")
    if script_td:
        pre_in_td = script_td.find("pre")
        if pre_in_td:
            return pre_in_td.get_text()
        return script_td.get_text()

    return ""


def load_progress() -> set:
    """載入已完成的電影 ID"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()


def save_progress(completed: set):
    """保存進度"""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed": list(completed)}, f)


def load_movies_data() -> list[dict]:
    """載入已爬取的電影數據"""
    if MOVIES_FILE.exists():
        with open(MOVIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_movies_data(movies: list[dict]):
    """保存電影數據"""
    with open(MOVIES_FILE, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)


def main(limit: int = None, delay: float = DEFAULT_DELAY):
    """
    主程式

    採用「全有全無」策略：
    - 如果劇本 URL 不存在 → 跳過該電影
    - 如果劇本內容抓不到 → 跳過該電影，不保存任何資料
    - 只有成功取得劇本時，才保存元數據和劇本文件

    Args:
        limit: 限制爬取數量（用於測試）
        delay: 請求間隔（秒）
    """
    # 確保目錄存在
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # 取得電影列表
    movie_list = get_movie_list(delay)

    if limit:
        movie_list = movie_list[:limit]
        logger.info(f"限制爬取 {limit} 部電影（測試模式）")

    # 載入進度
    completed = load_progress()
    movies_data = load_movies_data()
    existing_ids = {m["id"] for m in movies_data}

    # 統計
    total = len(movie_list)
    skipped = 0
    success = 0
    failed = 0
    no_script = 0  # 新增：沒有劇本的計數

    logger.info(f"開始爬取，共 {total} 部電影，已完成 {len(completed)} 部")

    for i, movie in enumerate(movie_list):
        movie_id = movie["id"]

        # 跳過已完成的
        if movie_id in completed:
            skipped += 1
            continue

        try:
            logger.info(f"[{i+1}/{total}] 正在處理: {movie['title']}")

            # Step 1: 爬取詳情頁
            detail = scrape_detail_page(movie["detail_url"], delay)
            time.sleep(delay)

            # Step 2: 檢查是否有劇本 URL
            if not detail["script_url"]:
                logger.warning(f"跳過 {movie['title']}: 沒有劇本連結")
                no_script += 1
                continue

            # Step 3: 爬取劇本內容
            script_text = scrape_script(detail["script_url"], delay)
            time.sleep(delay)

            # Step 4: 全有全無檢查 - 如果劇本內容為空，跳過整部電影
            if not script_text.strip():
                logger.warning(f"跳過 {movie['title']}: 劇本內容為空（全有全無策略）")
                no_script += 1
                continue

            # Step 5: 劇本成功取得，保存劇本文件
            script_file = SCRIPTS_DIR / f"{movie_id}.txt"
            script_file.write_text(script_text, encoding="utf-8")

            # Step 6: 組合並保存電影數據
            movie_data = {
                "id": movie_id,
                "title": movie["title"],
                "detail_url": movie["detail_url"],
                "writers": detail["writers"],
                "genres": detail["genres"],
                "avg_rating": detail["avg_rating"],
                "script_url": detail["script_url"],
                "reviews": detail["reviews"],
            }

            # 更新或添加到列表
            if movie_id in existing_ids:
                for j, m in enumerate(movies_data):
                    if m["id"] == movie_id:
                        movies_data[j] = movie_data
                        break
            else:
                movies_data.append(movie_data)
                existing_ids.add(movie_id)

            # Step 7: 標記完成並保存進度
            completed.add(movie_id)
            save_progress(completed)

            # 每 10 部保存一次數據
            if (i + 1) % 10 == 0:
                save_movies_data(movies_data)
                logger.info(f"已保存進度: {len(completed)} 部完成")

            success += 1
            logger.info(f"成功: {movie['title']}")

        except Exception as e:
            logger.error(f"處理失敗: {movie['title']} - {e}")
            failed += 1
            continue

    # 最終保存
    save_movies_data(movies_data)

    # 統計報告
    logger.info(f"""
    ========== 爬取完成 ==========
    總數: {total}
    成功: {success}
    跳過（已完成）: {skipped}
    跳過（無劇本）: {no_script}
    失敗: {failed}
    評論總數: {sum(len(m.get('reviews', [])) for m in movies_data)}
    ==============================
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMSDb 電影劇本爬蟲")
    parser.add_argument("--limit", type=int, default=None, help="限制爬取數量（用於測試）")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="請求間隔（秒）")
    args = parser.parse_args()

    main(limit=args.limit, delay=args.delay)
