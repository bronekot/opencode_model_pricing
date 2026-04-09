import urllib.request
from urllib.error import URLError, HTTPError
import re
import json
from datetime import datetime
import os


def parse_price(price_str):
    if price_str == 'Free':
        return 0.0
    elif price_str.startswith('$'):
        return float(price_str.replace('$', '').replace(',', ''))
    elif price_str == '-' or price_str.strip() == '':
        return float('inf')
    else:
        return float('inf')


def load_env():
    """Загрузка переменных из .ENV файла."""
    env_file = os.path.join(os.path.dirname(__file__), '.ENV')
    env_vars = {}
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env_vars


# Модели, для которых не нужно запрашивать бенчмарки
SKIP_BENCHMARK_MODELS = {'big pickle'}


def get_benchmark_cache_path():
    return os.path.join(os.path.dirname(__file__), 'benchmark_cache.json')


def load_benchmark_cache():
    """Загрузка кеша бенчмарков из файла."""
    cache_path = get_benchmark_cache_path()
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_benchmark_cache(cache):
    """Сохранение кеша бенчмарков в файл."""
    cache_path = get_benchmark_cache_path()
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def is_null_entry_expired(entry):
    """Проверяет, истёк ли срок хранения записи с пустыми данными (текущий день)."""
    null_date = entry.get('_null_date')
    if not null_date:
        return True
    today = datetime.now().strftime('%Y-%m-%d')
    return null_date != today


def fetch_benchmarks(api_key):
    """Получение всех бенчмарков через Artificial Analysis API."""
    url = 'https://artificialanalysis.ai/api/v2/data/llms/models'
    try:
        req = urllib.request.Request(url, headers={
            'x-api-key': api_key,
            'User-Agent': 'Mozilla/5.0'
        })
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))

        benchmarks = {}
        for model in data.get('data', []):
            evals = model.get('evaluations', {})
            name = model.get('name', '')
            creator = model.get('model_creator', {}).get('name', '')

            gpqa = evals.get('gpqa')
            coding = evals.get('artificial_analysis_coding_index')

            benchmarks[name] = {
                'creator': creator,
                'gpqa': gpqa,
                'coding': coding,
            }

        return benchmarks
    except (URLError, HTTPError, json.JSONDecodeError) as e:
        print(f"Не удалось получить бенчмарки: {e}")
        return None


def needs_api_fetch(model_names, cache):
    """Возвращает список моделей, требующих обновления кеша.

    Обновление нужно если:
    - в таблице цен есть модель, которой нет в кеше совсем
    - в таблице цен есть модель с просроченной null-записью в кеше
    """
    today = datetime.now().strftime('%Y-%m-%d')
    missing = []

    for name in model_names:
        if normalize_name(name) in SKIP_BENCHMARK_MODELS:
            continue
        entry = cache.get(name)
        if entry is None:
            missing.append(name)
        elif entry.get('gpqa') is None and entry.get('coding') is None:
            if entry.get('_null_date') != today:
                missing.append(name)

    return missing


def enrich_with_bridgebench(cache):
    """Обогащает кеш данными BridgeBench (статические данные)."""
    for bb_name, score in BRIDGEBENCH_DATA.items():
        norm_bb = normalize_name(bb_name)
        search_patterns = BRIDGEBENCH_TO_AA_MAP.get(norm_bb, [bb_name])

        # Ищем соответствующую модель в кеше
        cache_key = None
        for pattern in search_patterns:
            pattern_lower = pattern.lower()
            for key in cache:
                if pattern_lower in key.lower():
                    cache_key = key
                    break
            if cache_key:
                break

        if cache_key:
            cache[cache_key]['bridgebench'] = score
        else:
            # Если не нашли в кеше, создаём запись с помощью маппинга из opencode
            # Пробуем найти через model_map из match_model_to_benchmarks
            for key in cache:
                norm_key = normalize_name(key)
                # Проверяем по ключевым словам
                bb_words = set(norm_bb.split())
                key_words = set(norm_key.split())
                if bb_words & key_words:  # Есть общие слова
                    # Проверяем, что это действительно нужная модель
                    if ('opus' in norm_bb and 'opus' in norm_key) or \
                       ('minimax' in norm_bb and 'minimax' in norm_key) or \
                       ('gpt' in norm_bb and 'gpt' in norm_key and 'codex' in norm_key) or \
                       ('kimi' in norm_bb and 'kimi' in norm_key) or \
                       ('glm' in norm_bb and 'glm' in norm_key):
                        cache[key]['bridgebench'] = score
                        break

    return cache


def get_benchmarks_for_models(model_names, api_key):
    """Получение бенчмарков с учётом кеша. Вызывает API только если нужно."""
    cache = load_benchmark_cache()
    today = datetime.now().strftime('%Y-%m-%d')

    missing = needs_api_fetch(model_names, cache)

    if not missing:
        print(f"Все бенчмарки загружены из кеша ({len(cache)} записей)")
        return cache

    if not api_key:
        print("API ключ Artificial Analysis не найден в .ENV, бенчмарки не загружены")
        return cache if cache else None

    print(f"Нет актуального кеша для {len(missing)} моделей, запрос к Artificial Analysis API...")
    api_benchmarks = fetch_benchmarks(api_key)
    if not api_benchmarks:
        print(f"API недоступен, используется кеш ({len(cache)} записей)")
        return cache if cache else None

    print(f"Получены данные API для {len(api_benchmarks)} моделей")

    # Сохраняем ВСЕ модели из API в кеш (кроме тех у которых уже есть данные)
    new_total = 0
    for name, bench in api_benchmarks.items():
        existing = cache.get(name)
        # Не перезаписываем уже имеющиеся данные (с gpqa или coding)
        if existing and (existing.get('gpqa') is not None or existing.get('coding') is not None):
            continue
        cache[name] = {
            'creator': bench.get('creator'),
            'gpqa': bench.get('gpqa'),
            'coding': bench.get('coding'),
        }
        if bench.get('gpqa') is None and bench.get('coding') is None:
            cache[name]['_null_date'] = today
        new_total += 1

    # Для моделей из таблицы цен, которые не нашлись в API напрямую — пробуем маппинг
    for name in missing:
        if name in cache and (cache[name].get('gpqa') is not None or cache[name].get('coding') is not None):
            continue
        bench = match_model_to_benchmarks(name, api_benchmarks)
        if bench and (bench.get('gpqa') is not None or bench.get('coding') is not None):
            cache[name] = {'creator': bench.get('creator'), 'gpqa': bench.get('gpqa'), 'coding': bench.get('coding')}
        else:
            cache[name] = {'gpqa': None, 'coding': None, '_null_date': today}

    print(f"Кеш обновлён: {len(cache)} записей ({new_total} из API)")

    # Обогащаем данными BridgeBench
    cache = enrich_with_bridgebench(cache)
    save_benchmark_cache(cache)
    return cache


def normalize_name(name):
    """Нормализация имени модели для сопоставления."""
    name = name.lower().strip()
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'[<>]=?\s*\d+k', '', name)
    name = re.sub(r'\bfree\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.replace('-', ' ').replace('_', ' ')
    return name


def get_effort_rank(name):
    """Извлекает приоритет effort-варианта из названия модели."""
    match = re.search(r'\(([^()]*)\)\s*$', name)
    if not match:
        return 0

    suffix = match.group(1).lower()

    if 'xhigh' in suffix or 'max effort' in suffix or 'adaptive reasoning' in suffix:
        return 3
    if 'high' in suffix:
        return 2
    if 'medium' in suffix:
        return 1
    if 'low' in suffix:
        return 0
    if 'minimal' in suffix:
        return -1
    return 0


def benchmark_similarity(model_name, benchmark_name):
    """Оценивает похожесть имени модели и имени из бенчмарка."""
    model_norm = normalize_name(model_name)
    benchmark_norm = normalize_name(benchmark_name)

    model_words = set(model_norm.split())
    benchmark_words = set(benchmark_norm.split())
    if not model_words or not benchmark_words:
        return 0.0

    intersection = len(model_words & benchmark_words)
    union = len(model_words | benchmark_words)
    return intersection / union if union else 0.0


def has_useful_benchmark(benchmark):
    return bool(benchmark) and (benchmark.get('gpqa') is not None or benchmark.get('coding') is not None)


def match_model_to_benchmarks(model_name, benchmarks):
    """Сопоставление модели из таблицы цен с данными бенчмарков."""
    if not benchmarks:
        return None

    best_match = None
    best_score = (-1, -1, -1, -1)
    for index, api_name in enumerate(benchmarks):
        benchmark = benchmarks[api_name]
        similarity = benchmark_similarity(model_name, api_name)
        effort_rank = get_effort_rank(api_name)
        useful_rank = 1 if has_useful_benchmark(benchmark) else 0
        score = (
            useful_rank,
            similarity,
            effort_rank,
            -index,
        )
        if score > best_score:
            best_score = score
            best_match = api_name

    if best_match:
        return benchmarks[best_match]
    return None


def fetch_html_from_website():
    url = 'https://opencode.ai/docs/zen'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode('utf-8')
    except (URLError, HTTPError) as e:
        print(f"Сайт недоступен: {e}")
        return None


def parse_html_to_models(html_content):
    models = []
    
    table_pattern = re.compile(r'<table>(.*?)</table>', re.DOTALL)
    tables = table_pattern.findall(html_content)
    
    if len(tables) < 2:
        print("Не удалось найти таблицу с ценами на сайте")
        return None
    
    table_html = tables[1]
    
    row_pattern = re.compile(r'<tr>(.*?)</tr>', re.DOTALL)
    rows = row_pattern.findall(table_html)
    
    for row in rows:
        if '<th>' in row or '<thead' in row:
            continue
        
        cell_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
        cells = cell_pattern.findall(row)
        
        if len(cells) < 3:
            continue
        
        clean_cells = []
        for cell in cells:
            clean = re.sub(r'<[^>]+>', '', cell)
            clean = clean.strip()
            clean_cells.append(clean)
        
        if clean_cells:
            models.append(clean_cells)
    
    return models


def save_models_to_cache(models):
    cache_file = os.path.join(os.path.dirname(__file__), 'model_pricing_cache')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(f"# Cache timestamp: {timestamp}\n")
        f.write("Model\tInput\tOutput\tCached Read\tCached Write\n")
        for model in models:
            f.write('\t'.join(model) + '\n')


def load_models_from_cache():
    cache_file = os.path.join(os.path.dirname(__file__), 'model_pricing_cache')
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        timestamp = None
        models = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('# Cache timestamp:'):
                timestamp = line.replace('# Cache timestamp:', '').strip()
            elif line and not line.startswith('#') and not line.startswith('Model'):
                parts = line.split('\t')
                if len(parts) >= 3:
                    models.append(parts)
        
        return models, timestamp
    except FileNotFoundError:
        return None, None


# Якоря для CodIndex: Haiku 3.5 → 10, Opus 4.6 → 100
# Значения Coding (AA Coding Index) для якорных моделей
COD_INDEX_MIN = 10.7   # Claude Haiku 3.5
COD_INDEX_MAX = 48.1   # Claude Opus 4.6
COD_INDEX_MIN_VAL = 10  # Целевое значение для Haiku 3.5
COD_INDEX_MAX_VAL = 100  # Целевое значение для Opus 4.6

# BridgeBench бенчмарк (источник: bridgemind.ai/bridgebench)
# Overall score на реальных задачах программирования
BRIDGEBENCH_DATA = {
    'Claude Sonnet 4.6': 94.9,
    'Claude Opus 4.6': 94.8,
    'GPT-5.3 Codex': 94.6,
    'Qwen3.5 Plus 02-15': 93.6,
    'GPT-5.2 Codex': 92.8,
    'MiniMax M2.5': 92.3,
    'Qwen3.5 397B A17B': 92.1,
    'GLM-5': 89.5,
    'Kimi K2.5': 89.1,
    'Gemini 3.1 Pro Preview': 75.6,
    'Aurora Alpha': 57.6,
}

# Маппинг имён BridgeBench на имена в AA API для поиска в кеше
BRIDGEBENCH_TO_AA_MAP = {
    'claude sonnet 4.6': ['Claude Sonnet 4.6', 'Sonnet 4.6'],
    'claude opus 4.6': ['Claude Opus 4.6', 'Opus 4.6'],
    'gpt 5.3 codex': ['GPT-5.3 Codex', 'GPT 5.3 Codex'],
    'qwen3.5 plus 02-15': ['Qwen3.5 Plus', 'Qwen 3.5 Plus'],
    'gpt 5.2 codex': ['GPT-5.2 Codex', 'GPT 5.2 Codex'],
    'minimax m2.5': ['MiniMax-M2.5', 'MiniMax M2.5'],
    'qwen3.5 397b a17b': ['Qwen3.5 397B', 'Qwen 3.5 397B'],
    'glm 5': ['GLM-5', 'GLM 5'],
    'kimi k2.5': ['Kimi K2.5'],
    'gemini 3.1 pro preview': ['Gemini 3.1 Pro', 'Gemini 3.1 Pro Preview'],
    'aurora alpha': ['Aurora Alpha'],
}


def compute_cod_index(coding):
    """Вычисляет CodIndex — линейный индекс качества кодинга.

    Формула: CodIndex = 10 + 90 * (x - min) / (max - min)
    Якоря: Claude Haiku 3.5 (Coding=10.7) → 10, Claude Opus 4.6 (Coding=48.1) → 100.
    Модели ниже якоря → 10, выше → >100.
    """
    if coding is None:
        return None
    t = (coding - COD_INDEX_MIN) / (COD_INDEX_MAX - COD_INDEX_MIN)
    t = max(t, 0.0)
    return round(COD_INDEX_MIN_VAL + (COD_INDEX_MAX_VAL - COD_INDEX_MIN_VAL) * t, 1)


def format_benchmark(value):
    """Форматирование значения бенчмарка для отображения."""
    if value is None:
        return '-'
    if isinstance(value, float):
        if value < 1:
            return f"{value * 100:.1f}%"
        return f"{value:.1f}"
    return str(value)


def format_cod_index(value):
    """Форматирование CodIndex."""
    if value is None:
        return '-'
    return f"{value:.1f}"


def main():
    html_content = fetch_html_from_website()
    
    if html_content:
        models_data = parse_html_to_models(html_content)
        if models_data:
            save_models_to_cache(models_data)
            print("Загружено с сайта")
            source_data = models_data
        else:
            print("Не удалось распарсить данные с сайта")
            models_data, timestamp = load_models_from_cache()
            if models_data:
                print(f"Сайт недоступен, используется кэш от {timestamp}")
                source_data = models_data
            else:
                print("Кэш не найден. Пожалуйста, проверьте соединение с интернетом.")
                return
    else:
        models_data, timestamp = load_models_from_cache()
        if models_data:
            print(f"Сайт недоступен, используется кэш от {timestamp}")
            source_data = models_data
        else:
            print("Сайт недоступен и кэш не найден. Пожалуйста, проверьте соединение с интернетом.")
            return
    
    # Загрузка бенчмарков
    env_vars = load_env()
    api_key = env_vars.get('ARTIFICICAL_ANALYSIS_API') or env_vars.get('ARTIFICIAL_ANALYSIS_API')
    model_names = [row[0] for row in source_data]
    benchmarks = get_benchmarks_for_models(model_names, api_key)
    
    models = []
    for row in source_data:
        name = row[0]
        input_price = row[1]
        output_price = row[2]
        
        input_val = parse_price(input_price)
        output_val = parse_price(output_price)
        
        weighted_price = 0.9784 * input_val + 0.0216 * output_val if input_val != float('inf') and output_val != float('inf') else float('inf')

        bench = benchmarks.get(name) if benchmarks else None

        # Ищем BridgeBench через маппинг (по нормализованным именам)
        bridgebench = None
        if benchmarks:
            norm_name = normalize_name(name)
            for bb_name, score in BRIDGEBENCH_DATA.items():
                norm_bb = normalize_name(bb_name)
                # Точное совпадение
                if norm_name == norm_bb:
                    bridgebench = score
                    break

                # Проверяем по словам
                bb_words = set(norm_bb.split())
                name_words = set(norm_name.split())

                # Если в BB есть "codex", то и в имени должно быть "codex"
                bb_has_codex = 'codex' in bb_words
                name_has_codex = 'codex' in name_words
                if bb_has_codex != name_has_codex:
                    continue

                # Исключаем "codex" из проверки (уже проверили выше)
                exclude_words = {'codex', 'pro', 'preview', 'plus'}
                bb_key_words = bb_words - exclude_words
                if bb_key_words and bb_key_words.issubset(name_words):
                    # Проверяем версии
                    bb_nums = set(w for w in bb_words if any(c.isdigit() for c in w))
                    name_nums = set(w for w in name_words if any(c.isdigit() for c in w))
                    if bb_nums == name_nums or not bb_nums:
                        bridgebench = score
                        break

        coding = bench.get('coding') if bench else None
        models.append({
            'name': name,
            'input_price': input_price,
            'output_price': output_price,
            'weighted_price': weighted_price,
            'gpqa': bench.get('gpqa') if bench else None,
            'coding': coding,
            'cod_index': compute_cod_index(coding),
            'bridgebench': bridgebench,
        })
    
    models_sorted = sorted(
        models,
        key=lambda x: (
            x['weighted_price'],
            float('inf') if x['cod_index'] is None else -x['cod_index'],
        ),
    )
    
    has_benchmarks = benchmarks is not None
    has_bridgebench = any(m.get('bridgebench') is not None for m in models)

    if has_benchmarks:
        bb_col = 7 if has_bridgebench else 0
        bb_header = f" {'Bridge':<{bb_col}}" if has_bridgebench else ""
        header = f"{'Model':<40} {'Input':<10} {'Output':<10} {'Weighted':<10} {'CodIdx':<8} {'Coding':<8} {'GPQA':<8}" + bb_header
        separator = "-" * (94 + bb_col)
    else:
        header = f"{'Model':<40} {'Input':<12} {'Output':<12} {'Weighted':<12}"
        separator = "-" * 76

    print()
    print(header)
    print(separator)
    for model in models_sorted:
        if model['weighted_price'] != float('inf'):
            weighted_str = f"${model['weighted_price']:.4f}"
        else:
            weighted_str = "-"

        if has_benchmarks:
            cod_idx_str = format_cod_index(model['cod_index'])
            coding_str = format_benchmark(model['coding'])
            gpqa_str = format_benchmark(model['gpqa'])
            bb_str = format_benchmark(model['bridgebench']) if has_bridgebench else ""
            if has_bridgebench:
                print(f"{model['name']:<40} {model['input_price']:<10} {model['output_price']:<10} {weighted_str:<10} {cod_idx_str:<8} {coding_str:<8} {gpqa_str:<8} {bb_str:<{bb_col}}")
            else:
                print(f"{model['name']:<40} {model['input_price']:<10} {model['output_price']:<10} {weighted_str:<10} {cod_idx_str:<8} {coding_str:<8} {gpqa_str:<8}")
        else:
            print(f"{model['name']:<40} {model['input_price']:<12} {model['output_price']:<12} {weighted_str:<12}")

    if has_benchmarks:
        print()
        print(f"CodIdx: линейный индекс кодинга (10=Haiku 3.5, 100=Opus 4.6). Coding = AA Coding Index, GPQA = GPQA Diamond")
        if has_bridgebench:
            print(f"Bridge = BridgeBench Overall (источник: bridgemind.ai)")
        print("Источник бенчмарков: artificialanalysis.ai")


if __name__ == '__main__':
    main()
