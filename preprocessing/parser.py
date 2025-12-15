"""
劇本解析器
將原始劇本文本解析為結構化 JSON，提取場景、角色和互動關係
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置
DATA_DIR = Path("data")
SCRIPTS_DIR = DATA_DIR / "scripts"
PARSED_DIR = DATA_DIR / "parsed"

# 確保目錄存在
PARSED_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 正則表達式模式 ====================

# 場景標題: INT. LOCATION - TIME 或 EXT. LOCATION - DAY
# 也處理製作劇本格式: 1     EXT. OUTER SPACE     1
SCENE_HEADER_PATTERN = re.compile(
    r'^\s*'
    r'(?:\d+\s+)?'  # 可選場景號（在開頭）
    r'(INT\.?|EXT\.?|INT\.?/EXT\.?|I/E\.?|INTERIOR|EXTERIOR)'
    r'[\s\.\-:]+(.+?)(?:\s*[-–—]\s*(.+?))?'
    r'(?:\s+\d+)?\s*$',  # 可選場景號（在結尾）
    re.IGNORECASE | re.MULTILINE
)

# 角色名: 全大寫，可能帶括號標記 (CONT'D), (V.O.), (O.S.)
CHARACTER_PATTERN = re.compile(
    r'^[ \t]{10,50}([A-Z][A-Z\s\.\'\-]{1,30}?)(?:\s*\(([^)]+)\))?\s*$'
)

# 簡化的角色識別（用於統計）
CHARACTER_SIMPLE = re.compile(r'^([A-Z][A-Z\s\.\'\-]{2,25})$')

# 對話擴展標記
DIALOGUE_EXTENSIONS = {'CONT\'D', 'CONTD', 'CONT', 'V.O.', 'VO', 'O.S.', 'OS', 'O.C.', 'OC', 'OVER', 'FILTER'}

# 場景轉換
TRANSITIONS = re.compile(
    r'^\s*(FADE\s+(IN|OUT|TO)|CUT\s+TO|DISSOLVE\s+TO|SMASH\s+CUT|'
    r'MATCH\s+CUT|JUMP\s+CUT|WIPE\s+TO|IRIS\s+(IN|OUT)|THE\s+END):?\s*$',
    re.IGNORECASE
)

# 排除的「角色」（常見的非角色全大寫文字）
EXCLUDED_NAMES = {
    'INT', 'EXT', 'INTERIOR', 'EXTERIOR', 'FADE', 'CUT', 'DISSOLVE',
    'CONTINUED', 'CONTINUOUS', 'MORE', 'CONT', 'THE END', 'END',
    'SCENE', 'ACT', 'TITLE', 'SUPER', 'SUPERIMPOSE', 'INSERT',
    'FLASHBACK', 'FLASH', 'BACK', 'MONTAGE', 'SERIES', 'ANGLE',
    'POV', 'CLOSE', 'WIDE', 'MEDIUM', 'SHOT', 'ON', 'OF',
    'LATER', 'MOMENTS', 'SAME', 'DAY', 'NIGHT', 'MORNING', 'EVENING',
    'DAWN', 'DUSK', 'NOON', 'AFTERNOON', 'SUNRISE', 'SUNSET',
}


def clean_character_name(name: str) -> str:
    """清理角色名，移除標記"""
    name = name.strip()
    # 移除常見後綴
    for ext in DIALOGUE_EXTENSIONS:
        name = re.sub(rf"\s*\(?{re.escape(ext)}\.?\)?", "", name, flags=re.IGNORECASE)
    return name.strip()


def is_valid_character(name: str) -> bool:
    """檢查是否為有效角色名"""
    name = clean_character_name(name)
    if not name or len(name) < 2:
        return False
    if name.upper() in EXCLUDED_NAMES:
        return False
    if name.isdigit():
        return False
    # 檢查是否全大寫（允許空格和常見標點）
    clean = re.sub(r'[\s\.\'\-]', '', name)
    if not clean.isupper():
        return False
    return True


def split_into_scenes(text: str) -> list[dict]:
    """將劇本切分為場景"""
    scenes = []
    lines = text.split('\n')

    current_scene = None
    current_lines = []
    scene_number = 0

    for i, line in enumerate(lines):
        # 檢查是否為場景標題
        match = SCENE_HEADER_PATTERN.match(line)
        if match:
            # 保存前一場景
            if current_scene is not None:
                current_scene['text'] = '\n'.join(current_lines).strip()
                current_scene['line_end'] = i - 1
                scenes.append(current_scene)

            scene_number += 1
            location_type = match.group(1).upper().replace('INTERIOR', 'INT').replace('EXTERIOR', 'EXT')
            location = match.group(2).strip() if match.group(2) else ""
            time_of_day = match.group(3).strip() if match.group(3) else None

            current_scene = {
                'number': scene_number,
                'header': line.strip(),
                'location_type': location_type,
                'location': location,
                'time': time_of_day,
                'line_start': i,
                'line_end': None,
            }
            current_lines = []
        elif current_scene is not None:
            current_lines.append(line)

    # 保存最後一場景
    if current_scene is not None:
        current_scene['text'] = '\n'.join(current_lines).strip()
        current_scene['line_end'] = len(lines) - 1
        scenes.append(current_scene)

    return scenes


def extract_dialogues_from_scene(scene_text: str) -> list[dict]:
    """從場景文本中提取對話"""
    dialogues = []
    lines = scene_text.split('\n')

    current_character = None
    current_dialogue = []

    for line in lines:
        stripped = line.strip()

        # 跳過空行
        if not stripped:
            if current_character and current_dialogue:
                dialogues.append({
                    'character': current_character,
                    'text': ' '.join(current_dialogue).strip(),
                })
                current_character = None
                current_dialogue = []
            continue

        # 檢查是否為角色名行
        char_match = CHARACTER_PATTERN.match(line)
        if char_match:
            # 保存前一段對話
            if current_character and current_dialogue:
                dialogues.append({
                    'character': current_character,
                    'text': ' '.join(current_dialogue).strip(),
                })

            name = char_match.group(1).strip()
            if is_valid_character(name):
                current_character = clean_character_name(name)
                current_dialogue = []
            else:
                current_character = None
                current_dialogue = []
            continue

        # 如果在收集對話
        if current_character:
            # 對話行通常有縮進
            if line.startswith('    ') or line.startswith('\t'):
                # 跳過括號動作指示 (beat), (pause) 等
                if stripped.startswith('(') and stripped.endswith(')'):
                    continue
                current_dialogue.append(stripped)
            else:
                # 遇到非縮進行，結束當前對話
                if current_dialogue:
                    dialogues.append({
                        'character': current_character,
                        'text': ' '.join(current_dialogue).strip(),
                    })
                current_character = None
                current_dialogue = []

    # 保存最後一段對話
    if current_character and current_dialogue:
        dialogues.append({
            'character': current_character,
            'text': ' '.join(current_dialogue).strip(),
        })

    return dialogues


def extract_all_characters(text: str) -> list[str]:
    """從全文提取所有角色名"""
    characters = set()

    for line in text.split('\n'):
        match = CHARACTER_PATTERN.match(line)
        if match:
            name = match.group(1).strip()
            if is_valid_character(name):
                characters.add(clean_character_name(name))

    return sorted(characters)


def build_character_interactions(scenes: list[dict]) -> list[dict]:
    """建立角色互動關係"""
    # 統計角色對在同一場景出現的次數
    interaction_counts = defaultdict(lambda: {'count': 0, 'scenes': []})

    for scene in scenes:
        # 提取場景中所有角色
        chars_in_scene = set()
        for dialogue in scene.get('dialogues', []):
            chars_in_scene.add(dialogue['character'])

        # 建立角色對
        chars_list = sorted(chars_in_scene)
        for i, char_a in enumerate(chars_list):
            for char_b in chars_list[i + 1:]:
                pair = (char_a, char_b)
                interaction_counts[pair]['count'] += 1
                interaction_counts[pair]['scenes'].append(scene['number'])

    # 轉換為列表格式
    interactions = []
    for (char_a, char_b), data in sorted(interaction_counts.items()):
        interactions.append({
            'a': char_a,
            'b': char_b,
            'count': data['count'],
            'scenes': data['scenes'],
        })

    return interactions


def compute_character_stats(scenes: list[dict]) -> dict[str, dict]:
    """計算角色統計資訊"""
    stats = defaultdict(lambda: {
        'dialogue_count': 0,
        'scene_appearances': set(),
        'first_appearance': None,
        'total_words': 0,
    })

    for scene in scenes:
        for dialogue in scene.get('dialogues', []):
            char = dialogue['character']
            stats[char]['dialogue_count'] += 1
            stats[char]['scene_appearances'].add(scene['number'])
            stats[char]['total_words'] += len(dialogue['text'].split())

            if stats[char]['first_appearance'] is None:
                stats[char]['first_appearance'] = scene['number']

    # 轉換 set 為 list
    result = {}
    for char, data in stats.items():
        result[char] = {
            'dialogue_count': data['dialogue_count'],
            'scene_count': len(data['scene_appearances']),
            'scenes': sorted(data['scene_appearances']),
            'first_appearance': data['first_appearance'],
            'total_words': data['total_words'],
        }

    return result


def parse_script(text: str, movie_id: str) -> dict:
    """解析劇本，返回結構化數據"""
    # 切分場景
    scenes = split_into_scenes(text)

    # 為每個場景提取對話
    for scene in scenes:
        scene['dialogues'] = extract_dialogues_from_scene(scene.get('text', ''))
        scene['characters'] = sorted(set(d['character'] for d in scene['dialogues']))

    # 提取所有角色
    all_characters = extract_all_characters(text)

    # 建立互動關係
    interactions = build_character_interactions(scenes)

    # 計算角色統計
    character_stats = compute_character_stats(scenes)

    return {
        'id': movie_id,
        'total_scenes': len(scenes),
        'total_characters': len(all_characters),
        'total_dialogues': sum(len(s.get('dialogues', [])) for s in scenes),
        'characters': all_characters,
        'character_stats': character_stats,
        'interactions': interactions,
        'scenes': scenes,
    }


def parse_single_file(script_file: Path) -> tuple[str, bool, str]:
    """解析單個劇本檔案，返回 (movie_id, success, message)"""
    movie_id = script_file.stem

    try:
        text = script_file.read_text(encoding='utf-8', errors='ignore')

        if not text.strip():
            return movie_id, False, "空白劇本"

        parsed = parse_script(text, movie_id)

        output_file = PARSED_DIR / f"{movie_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        msg = f"{parsed['total_scenes']} 場景, {parsed['total_characters']} 角色, {parsed['total_dialogues']} 對話"
        return movie_id, True, msg

    except Exception as e:
        return movie_id, False, str(e)


def main():
    """主程式：解析所有劇本"""
    parser = argparse.ArgumentParser(description='解析劇本')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='並行工作數 (預設: CPU 核心數)')
    args = parser.parse_args()

    script_files = list(SCRIPTS_DIR.glob("*.txt"))

    if not script_files:
        logger.warning(f"在 {SCRIPTS_DIR} 中沒有找到劇本檔案")
        return

    workers = args.workers or multiprocessing.cpu_count()
    logger.info(f"開始解析 {len(script_files)} 個劇本（{workers} 個並行工作）...")

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parse_single_file, f): f for f in script_files}

        for future in as_completed(futures):
            movie_id, ok, msg = future.result()
            if ok:
                logger.info(f"已解析: {movie_id} - {msg}")
                success += 1
            else:
                logger.error(f"解析失敗: {movie_id} - {msg}")
                failed += 1

    # 統計報告
    logger.info(f"""
    ========== 解析完成 ==========
    成功: {success}
    失敗: {failed}
    輸出目錄: {PARSED_DIR}
    ==============================
    """)


if __name__ == "__main__":
    main()
