# Trading Signal Bot

Bot de señales de trading para criptomonedas estilo swing, solo LONG/compras. Analiza datos públicos de Binance Spot, guarda señales en SQLite, envía alertas por Telegram y muestra un dashboard web con Flask.

Este proyecto no compra, no vende y no conecta con ninguna cuenta de exchange. Solo genera señales informativas.

## Estrategia

La estrategia `swing_long_v1` genera una señal cuando:

- En 1D, EMA 50 > EMA 200.
- En 4H, el precio actual está cerca de la EMA 50.
- RSI 14 está entre 40 y 60.
- RSI actual es mayor que el RSI anterior.
- Volumen actual es mayor que el promedio de las últimas 20 velas.

La entrada sugerida usa el precio actual. El stop loss se ubica debajo del mínimo reciente de las últimas 10 velas de 4H. TP1 usa 1:2 y TP2 usa 1:3.

## Instalación local

```bash
git clone <URL_DE_TU_REPOSITORIO>
cd trading-signal-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuración

Copia el archivo de ejemplo:

```bash
cp .env.example .env
```

En Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Luego completa:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

## Crear un bot de Telegram

1. Abre Telegram y busca `@BotFather`.
2. Ejecuta `/newbot`.
3. Elige nombre y usuario para tu bot.
4. Copia el token entregado por BotFather en `TELEGRAM_BOT_TOKEN`.
5. Envía un mensaje a tu bot.
6. Obtén tu `chat_id` usando `https://api.telegram.org/bot<TOKEN>/getUpdates`.
7. Copia el valor del chat en `TELEGRAM_CHAT_ID`.

## Ejecutar el scanner

```bash
python scanner.py
```

El scanner analiza `BTCUSDT`, `ETHUSDT` y `SOLUSDT` usando 1D para tendencia y 4H para entrada. Si ya existe una señal activa reciente del mismo par en las últimas 24 horas, no crea otra.

## Ejecutar el dashboard

```bash
python app.py
```

Abre `http://localhost:5000` para ver las últimas señales guardadas.

## Despliegue en Render

1. Sube el proyecto a GitHub.
2. En Render, crea un nuevo Blueprint y conecta el repositorio.
3. Render leerá `render.yaml`.
4. Configura las variables de entorno `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`.
5. El Web Service ejecutará `gunicorn app:app`.
6. El Cron Job ejecutará `python scanner.py` cada 4 horas.

## Estructura

```text
trading-signal-bot/
├── app.py
├── scanner.py
├── requirements.txt
├── render.yaml
├── .env.example
├── README.md
├── config/
│   └── settings.py
├── services/
│   ├── binance_service.py
│   ├── telegram_service.py
│   └── signal_service.py
├── strategies/
│   └── swing_long_v1.py
├── database/
│   ├── db.py
│   └── signals.db
└── templates/
    └── dashboard.html
```

## Riesgo financiero

Este software es educativo e informativo. Las señales pueden fallar y el mercado de criptomonedas es volátil. No es asesoría financiera. Haz tu propio análisis, usa gestión de riesgo y nunca arriesgues dinero que no puedas permitirte perder.

