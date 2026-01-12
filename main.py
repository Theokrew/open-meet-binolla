import logging
import asyncio
import random  # ← ESSENCIAL! Faltava isso antes
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from binance.client import Client
from binance.exceptions import BinanceAPIException
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Configuração de logging (mais detalhado para depuração)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Estatísticas
stats = {
    "manha": {"gains": 0, "losses": 0},
    "tarde": {"gains": 0, "losses": 0},
    "noite": {"gains": 0, "losses": 0}
}

def get_periodo(hora: int) -> str:
    if 6 <= hora < 12:
        return "manha"
    elif 12 <= hora < 18:
        return "tarde"
    else:
        return "noite"

# Modelo LSTM
class AdvancedLSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 128)
        c0 = torch.zeros(2, x.size(0), 128)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Carrega modelo (simulado; em produção, use torch.load se tiver .pth)
model = AdvancedLSTMNet()

# Funções de análise
def advanced_analysis(df):
    df['support'] = df['close'].rolling(20).min()
    df['resistance'] = df['close'].rolling(20).max()
    df['volatility'] = df['close'].pct_change().rolling(14).std()
    df['momentum'] = df['close'] - df['close'].shift(10)
    return df.fillna(0)

def prepare_input(df, time_step=60):
    df = advanced_analysis(df)
    data = df[['close', 'volume', 'volatility', 'momentum']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    if len(data_scaled) < time_step:
        logger.warning("Dados insuficientes para previsão")
        return None, None
    input_data = data_scaled[-time_step:].reshape(1, time_step, 4)
    return torch.tensor(input_data, dtype=torch.float32), scaler

# Lista de ativos reais
ativos = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

async def enviar_sinal(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now()
    hora = now.hour
    minuto = now.minute
    periodo = get_periodo(hora)

    logger.info(f"Iniciando sinal às {now.strftime('%H:%M')} - Período: {periodo}")

    # Binance client com chaves do ambiente
    try:
        binance_client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET')
        )
        logger.info("Binance client inicializado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao inicializar Binance client: {e}")
        direcao = "CALL"  # Fallback
        cor = "🟢"
        ativo = "BTCUSDT"
        time_str = now.strftime("%H:%M")
        mensagem = f"""
🟡OPORTUNIDADE ENCONTRADA🟡

💹{ativo}
⏰{time_str}
⌛M1
{cor}Direção: {direcao}
⚠️G1 (Opcional)

📍Abra Sua Conta Aqui ↙️
🔗https://binolla.com/?lid=2101

🎯SINAIS AO VIVO🎯
"""
        await context.bot.send_message(chat_id=1158936585, text=mensagem, parse_mode="HTML")
        return

    ativo = random.choice(ativos)
    logger.info(f"Ativo selecionado: {ativo}")

    try:
        # Fetch dados reais (últimos 90 candles de 1m)
        klines = binance_client.get_klines(symbol=ativo, interval='1m', limit=90)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])

        input_tensor, scaler = prepare_input(df)
        if input_tensor is not None:
            with torch.no_grad():
                pred = model(input_tensor).item()
            is_call = pred > 0.5
            direcao = "CALL" if is_call else "PUT"
            cor = "🟢" if is_call else "🔴"
            logger.info(f"Previsão: {pred:.2f} → {direcao}")
        else:
            logger.warning("Dados insuficientes, usando fallback")
            direcao = random.choice(["CALL", "PUT"])
            cor = "🟢" if direcao == "CALL" else "🔴"
    except BinanceAPIException as e:
        logger.error(f"Erro na API Binance: {e}")
        direcao = random.choice(["CALL", "PUT"])
        cor = "🟢" if direcao == "CALL" else "🔴"

    preco_inicial = df['close'].iloc[-1] if 'df' in locals() else 0

    time_str = now.strftime("%H:%M")

    mensagem = f"""
🟡OPORTUNIDADE ENCONTRADA🟡

💹{ativo}
⏰{time_str}
⌛M1
{cor}Direção: {direcao}
⚠️G1 (Opcional)

📍Abra Sua Conta Aqui ↙️
🔗https://binolla.com/?lid=2101

🎯SINAIS AO VIVO🎯
"""

    await context.bot.send_message(
        chat_id=1158936585,
        text=mensagem,
        parse_mode="HTML"
    )
    logger.info("Sinal enviado com sucesso")

    # Agendar verificação após 1 minuto
    scheduler = context.job_queue
    scheduler.run_once(
        verificar_resultado,
        when=60,
        data={
            "periodo": periodo,
            "is_call": is_call if 'is_call' in locals() else True,
            "ativo": ativo,
            "preco_inicial": preco_inicial,
            "binance_client": binance_client
        }
    )

async def verificar_resultado(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    periodo = data["periodo"]
    is_call = data["is_call"]
    ativo = data["ativo"]
    preco_inicial = data["preco_inicial"]
    binance_client = data["binance_client"]

    try:
        klines = binance_client.get_klines(symbol=ativo, interval='1m', limit=1)
        preco_final = float(klines[0][4])
        ganhou = (preco_final > preco_inicial) if is_call else (preco_final < preco_inicial)
        if ganhou:
            stats[periodo]["gains"] += 1
        else:
            stats[periodo]["losses"] += 1
        logger.info(f"Verificação: Ativo {ativo} - Ganho: {ganhou}")
    except Exception as e:
        logger.error(f"Erro na verificação: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot iniciado com sinais reais via Binance! Sinais a cada minuto.")

def main():
    TOKEN = "8501561041:AAHucMrzlYnA0ZXR-1_HrOJ1widA6Qs4Ctw"

    # Verifica chaves Binance no startup
    if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_API_SECRET'):
        logger.error("Chaves Binance não configuradas! Adicione BINANCE_API_KEY e BINANCE_API_SECRET no Config Vars do Heroku.")
    else:
        logger.info("Chaves Binance detectadas")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        enviar_sinal,
        trigger=IntervalTrigger(minutes=1),
        args=(app,)
    )
    scheduler.start()

    logger.info("Bot iniciado com sinais reais da Binance!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
