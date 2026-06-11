# OpenCode Zen Nuxt

Nuxt приложение для сравнения цен и бенчмарков AI моделей с OpenCode Zen.

## Установка

```bash
npm install
```

## Настройка

Скопируйте `.ENV.example` в `.ENV` и добавьте ваш API ключ:

```bash
cp .ENV.example .ENV
```

Отредактируйте `.ENV`:
```
ARTIFICIAL_ANALYSIS_API=your_api_key_here
```

## Запуск

```bash
npm run dev
```

Приложение будет доступно по адресу http://localhost:3000

## Сборка для продакшена

```bash
npm run build
npm run preview
```

### Генерация статического сайта

```bash
npm run generate
```

## Функционал

- Автоматическая загрузка цен с https://opencode.ai/docs/zen
- Получение бенчмарков из Artificial Analysis API
- Расчёт взвешенной цены (97.84% вход + 2.16% выход)
- CodIndex — линейный индекс качества кодинга
- BridgeBench бенчмарки для актуальных моделей
- Адаптивный дизайн

## Python скрипт (оригинал)

Оригинальный Python скрипт доступен в `opencode_pricing.py` для локального использования:

```bash
python opencode_pricing.py
```
