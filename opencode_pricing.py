"""Цены OpenCode Zen и бенчмарки Artificial Analysis, без внешних зависимостей."""

import argparse
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import urllib.request
from urllib.error import URLError

BASE_DIR = Path(__file__).resolve().parent
PRICING_URL = 'https://opencode.ai/docs/zen/'
BENCHMARK_URL = 'https://artificialanalysis.ai/api/v2/data/llms/models'
BENCHMARK_TTL = 24 * 60 * 60
INPUT_WEIGHT = 0.9784
OUTPUT_WEIGHT = 0.0216
COD_INDEX_MIN = 10.7
COD_INDEX_MAX = 48.1

with (BASE_DIR / 'data/bridgebench.json').open(encoding='utf-8') as f:
    BRIDGEBENCH = json.load(f)
with (BASE_DIR / 'data/model_aliases.json').open(encoding='utf-8') as f:
    MODEL_ALIASES = json.load(f)


def parse_price(price_str):
    price_str = unescape(price_str).strip()
    if price_str.lower() == 'free':
        return 0.0
    if re.fullmatch(r'\$\s*\d[\d,]*(?:\.\d+)?', price_str):
        return float(price_str[1:].replace(',', '').strip())
    return float('inf')


def load_env():
    """Переменные процесса имеют приоритет над .ENV и .env."""
    env_vars = {}
    for filename in ('.env', '.ENV'):
        try:
            lines = (BASE_DIR / filename).read_text(encoding='utf-8').splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            line = line.strip()
            if line.startswith('export '):
                line = line[7:]
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '\"\'':
                value = value[1:-1]
            else:
                value = re.split(r'\s+#', value, maxsplit=1)[0].rstrip()
            env_vars[key.strip()] = value
    env_vars.update(os.environ)
    return env_vars


def get_api_key():
    env = load_env()
    return next((env[key] for key in (
        'NUXT_ARTIFICIAL_ANALYSIS_API', 'ARTIFICIAL_ANALYSIS_API',
        'ARTIFICICAL_ANALYSIS_API',
    ) if env.get(key)), None)


def get_benchmark_cache_path():
    return BASE_DIR / 'benchmark_cache.json'


def valid_benchmarks(models):
    return isinstance(models, dict) and bool(models) and all(
        isinstance(name, str) and isinstance(entry, dict) and all(
            entry.get(field) is None or (
                type(entry[field]) in (int, float) and math.isfinite(entry[field])
            ) for field in ('gpqa', 'coding')
        ) for name, entry in models.items()
    )


def load_benchmark_cache():
    try:
        with open(get_benchmark_cache_path(), encoding='utf-8') as f:
            cache = json.load(f)
        # Старый формат содержит недостоверные результаты нечёткого сопоставления.
        if (isinstance(cache, dict) and cache.get('version') == 2
                and isinstance(cache.get('fetched_at'), (int, float))
                and valid_benchmarks(cache.get('models'))):
            return cache
    except (OSError, ValueError):
        pass
    return {}


def save_benchmark_cache(cache):
    path = Path(get_benchmark_cache_path())
    try:
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(path)
    except OSError as error:
        print(f'Не удалось сохранить кэш бенчмарков: {error}')


def fetch_benchmarks(api_key):
    try:
        request = urllib.request.Request(BENCHMARK_URL, headers={
            'x-api-key': api_key, 'User-Agent': 'OpenCode-Model-Pricing',
        })
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode('utf-8'))
        if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
            raise ValueError('неожиданный формат ответа API')
        benchmarks = {}
        for model in payload['data']:
            if not isinstance(model, dict) or not isinstance(model.get('name'), str):
                continue
            evaluations = model.get('evaluations') or {}
            creator = model.get('model_creator') or {}
            benchmarks[model['name']] = {
                'creator': creator.get('name'),
                'gpqa': evaluations.get('gpqa'),
                'coding': evaluations.get('artificial_analysis_coding_index'),
            }
        if not valid_benchmarks(benchmarks):
            raise ValueError('API не вернул корректные бенчмарки')
        return benchmarks
    except (URLError, OSError, ValueError, AttributeError) as error:
        print(f'Не удалось получить бенчмарки: {error}')
        return None


def get_benchmarks_for_models(model_names, api_key, *, offline=False, refresh=False):
    """Кэшируем исходный ответ API на сутки; сопоставляем имена заново при запуске."""
    cache = load_benchmark_cache()
    now = datetime.now(timezone.utc).timestamp()
    age = now - cache.get('fetched_at', 0)
    fresh = bool(cache) and 0 <= age < BENCHMARK_TTL
    if not offline and api_key and (refresh or not fresh):
        benchmarks = fetch_benchmarks(api_key)
        if benchmarks:
            cache = {'version': 2, 'fetched_at': now, 'models': benchmarks}
            save_benchmark_cache(cache)
            fresh = True
            print(f'Бенчмарки обновлены: {len(benchmarks)} записей API')
    elif not offline and not api_key and not fresh:
        print('Ключ Artificial Analysis не настроен; используются доступные данные кэша')
    if not cache:
        return None
    timestamp = datetime.fromtimestamp(cache['fetched_at'], timezone.utc).isoformat(timespec='seconds')
    print(f'Бенчмарки от {timestamp}' + ('' if fresh else ' (устаревший кэш)'))
    source = cache['models']
    return {name: match_model_to_benchmarks(name, source) for name in model_names}


def normalize_name(name):
    """Сохраняем версии и варианты; убираем только контекст, Free и режим reasoning."""
    name = unescape(name).lower().strip()

    def strip_metadata(match):
        value = match.group(1)
        if re.search(r'\b(?:effort|reasoning|tokens)\b|^(?:max|xhigh|high|medium|low|minimal)$', value):
            return ' '
        if re.fullmatch(r'[<>≤≥]=?\s*\d+k(?:\s+tokens)?', value):
            return ' '
        return match.group(0)

    name = re.sub(r'\(([^()]*)\)', strip_metadata, name)
    name = re.sub(r'\bfree\s*$', '', name)
    name = re.sub(r'[-_]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def model_identity(name, *, aliases=True):
    normalized = normalize_name(name)
    if aliases:
        normalized = MODEL_ALIASES.get(normalized, normalized)
    # Claude 4.5 Sonnet и Claude Sonnet 4.5 — одна модель.
    return tuple(sorted(normalized.split()))


def get_effort_rank(name):
    suffix = ' '.join(re.findall(r'\(([^()]*)\)', name.lower()))
    for rank, effort in ((5, 'max'), (4, 'xhigh'), (3, 'high'), (2, 'medium'), (1, 'low'), (0, 'minimal')):
        if re.search(r'\b' + effort + r'\b', suffix):
            return rank
    if 'non-reasoning' in suffix:
        return -1
    return 2 if 'reasoning' in suffix else 0


def has_useful_benchmark(benchmark):
    return bool(benchmark) and any(benchmark.get(field) is not None for field in ('gpqa', 'coding'))


def match_model_to_benchmarks(model_name, benchmarks):
    if not benchmarks or normalize_name(model_name) == 'big pickle':
        return None
    identity = model_identity(model_name)
    candidates = [(name, bench) for name, bench in benchmarks.items()
                  if model_identity(name) == identity]
    if not candidates:
        return None
    name, benchmark = max(candidates, key=lambda item: (
        has_useful_benchmark(item[1]), get_effort_rank(item[0]),
    ))
    return {**benchmark, 'source_model': name}


def find_bridgebench(name):
    identity = model_identity(name, aliases=False)
    return next((score for model, score in BRIDGEBENCH['scores'].items()
                 if model_identity(model, aliases=False) == identity), None)


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table = []
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = []
        elif tag == 'br' and self.cell is not None:
            self.cell.append(' ')

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def fetch_html_from_website():
    try:
        request = urllib.request.Request(PRICING_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode('utf-8')
    except (URLError, OSError, UnicodeError) as error:
        print(f'Сайт недоступен: {error}')
        return None


def parse_html_to_models(html_content):
    parser = TableParser()
    parser.feed(html_content)
    for table in parser.tables:
        for index, row in enumerate(table):
            headers = [cell.lower() for cell in row]
            if not {'model', 'input', 'output'}.issubset(headers):
                continue
            columns = [headers.index(field) if field in headers else None
                       for field in ('model', 'input', 'output', 'cached read', 'cached write')]
            models = []
            for cells in table[index + 1:]:
                if any(column is not None and column >= len(cells) for column in columns[:3]):
                    continue
                model = [cells[column] if column is not None and column < len(cells) else '-'
                         for column in columns]
                if model[0] and any(math.isfinite(parse_price(price)) for price in model[1:3]):
                    models.append(model)
            if models:
                return models
    return None


def save_models_to_cache(models):
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    path = BASE_DIR / 'model_pricing_cache'
    try:
        temp = path.with_suffix('.tmp')
        temp.write_text(f'# Cache timestamp: {timestamp}\n'
                        + 'Model\tInput\tOutput\tCached Read\tCached Write\n'
                        + ''.join('\t'.join(model) + '\n' for model in models), encoding='utf-8')
        temp.replace(path)
    except OSError as error:
        print(f'Не удалось сохранить кэш цен: {error}')


def load_models_from_cache():
    try:
        lines = (BASE_DIR / 'model_pricing_cache').read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError):
        return None, None
    timestamp = None
    models = []
    for line in lines:
        if line.startswith('# Cache timestamp:'):
            timestamp = line.partition(':')[2].strip()
        elif line and not line.startswith('#') and not line.startswith('Model\t'):
            parts = line.split('\t')
            if len(parts) >= 3 and any(math.isfinite(parse_price(price)) for price in parts[1:3]):
                models.append(parts)
    return models or None, timestamp


def compute_cod_index(coding):
    """Историческая шкала: Coding 10.7 → 10, Coding 48.1 → 100; верхнего предела нет."""
    if coding is None:
        return None
    return round(10 + 90 * max((coding - COD_INDEX_MIN) / (COD_INDEX_MAX - COD_INDEX_MIN), 0), 1)


def format_benchmark(value, *, percentage=False):
    if value is None:
        return '-'
    return f'{value * 100:.1f}%' if percentage else f'{value:.1f}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='использовать только локальный кэш')
    parser.add_argument('--refresh-benchmarks', action='store_true', help='обновить бенчмарки до истечения суток')
    parser.add_argument('--no-benchmarks', action='store_true', help='не загружать бенчмарки Artificial Analysis')
    args = parser.parse_args(argv)
    if args.offline and args.refresh_benchmarks:
        parser.error('--offline несовместим с --refresh-benchmarks')
    html = None if args.offline else fetch_html_from_website()
    source = parse_html_to_models(html) if html else None
    if source:
        save_models_to_cache(source)
        print(f'Загружено с сайта: {len(source)} строк тарифов')
    else:
        source, timestamp = load_models_from_cache()
        if not source:
            print('Не удалось получить цены; локальный кэш отсутствует или повреждён.')
            return 1
        print(f'Используется кэш цен от {timestamp or "неизвестной даты"}')
    benchmarks = None if args.no_benchmarks else get_benchmarks_for_models(
        [row[0] for row in source], None if args.offline else get_api_key(),
        offline=args.offline, refresh=args.refresh_benchmarks,
    )
    models = []
    for row in source:
        name, input_price, output_price = row[:3]
        bench = (benchmarks or {}).get(name) or {}
        coding = bench.get('coding')
        models.append({
            'name': name, 'input_price': input_price, 'output_price': output_price,
            'weighted_price': INPUT_WEIGHT * parse_price(input_price) + OUTPUT_WEIGHT * parse_price(output_price),
            'coding': coding, 'cod_index': compute_cod_index(coding), 'gpqa': bench.get('gpqa'),
            'bridgebench': find_bridgebench(name),
        })
    models.sort(key=lambda model: (model['weighted_price'], -(model['cod_index'] or 0)))
    has_benchmarks = any(has_useful_benchmark(model) for model in models)
    has_bridgebench = any(model['bridgebench'] is not None for model in models)
    name_width = max(40, max(len(model['name']) for model in models))
    header = f'{"Model":<{name_width}} {"Input":<10} {"Output":<10} {"Weighted":<10}'
    if has_benchmarks:
        header += f' {"CodIdx":<8} {"Coding":<8} {"GPQA":<8}'
    if has_bridgebench:
        header += f' {"Bridge":<8}'
    print('\n' + header + '\n' + '-' * len(header))
    for model in models:
        price = f"${model['weighted_price']:.4f}" if math.isfinite(model['weighted_price']) else '-'
        line = f'{model["name"]:<{name_width}} {model["input_price"]:<10} {model["output_price"]:<10} {price:<10}'
        if has_benchmarks:
            line += f' {format_benchmark(model["cod_index"]):<8} {format_benchmark(model["coding"]):<8} {format_benchmark(model["gpqa"], percentage=True):<8}'
        if has_bridgebench:
            line += f' {format_benchmark(model["bridgebench"]):<8}'
        print(line)
    print('\nUSD за 1 млн токенов. Weighted = 97.84% Input + 2.16% Output; кэширование не учтено.')
    if has_benchmarks:
        print('Coding = AA Coding Index; GPQA = GPQA Diamond. Источник: https://artificialanalysis.ai/')
        print('CodIdx: историческая шкала 10.7 → 10, 48.1 → 100; может превышать 100.')
        print('Для одной модели выбирается доступный вариант с наибольшим reasoning effort.')
    if has_bridgebench:
        print(f'Bridge = {BRIDGEBENCH["metric"]}, снимок от {BRIDGEBENCH["checked_at"]}: {BRIDGEBENCH["source"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
