import urllib.request
from urllib.error import URLError, HTTPError
import re
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
    
    models = []
    for row in source_data:
        name = row[0]
        input_price = row[1]
        output_price = row[2]
        
        input_val = parse_price(input_price)
        output_val = parse_price(output_price)
        
        weighted_price = 0.9784 * input_val + 0.0216 * output_val if input_val != float('inf') and output_val != float('inf') else float('inf')
        
        models.append({
            'name': name,
            'input_price': input_price,
            'output_price': output_price,
            'weighted_price': weighted_price
        })
    
    models_sorted = sorted(models, key=lambda x: x['weighted_price'])
    
    print(f"{'Model':<40} {'Input':<12} {'Output':<12} {'Weighted':<12}")
    print("-" * 76)
    for model in models_sorted:
        if model['weighted_price'] != float('inf'):
            weighted_str = f"${model['weighted_price']:.4f}"
        else:
            weighted_str = "-"
        print(f"{model['name']:<40} {model['input_price']:<12} {model['output_price']:<12} {weighted_str:<12}")


if __name__ == '__main__':
    main()
