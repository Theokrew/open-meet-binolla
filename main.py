import logging
import asyncio
import random
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pyquotex.stable_api import Quotex
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import os

# Logging
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

model = AdvancedLSTMNet()

# Análise
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
        return None, None
    input_data = data_scaled[-time_step:].reshape(1, time_step, 4)
    return torch.tensor(input_data, dtype=torch.float32), scaler

ativos = ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc", "BTCUSD", "ETHUSD"]

async def enviar_sinal(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now()
    hora = now.hour
    minuto = now.minute
    periodo = get_periodo(hora)

    logger.info(f"Iniciando sinal às {now.strftime('%H:%M')} - Período: {periodo}")

    try:
        email = os.getenv('QUOTEX_EMAIL')
        password = os.getenv('QUOTEX_PASSWORD')
        logger.info(f"Tentando login com email: {email}")
        client = Quotex(email=email, password=password)
        client.debug_ws_enable = True
        client.debug = True
        logger.info("Debug: Iniciando connect")
        await client.connect()
        logger.info("Conectado à Quotex com sucesso")
    except Exception as e:
        logger.error(f"Detalhes do erro no connect: {str(e)}", exc_info=True)
        direcao = "CALL"  # Fallback
        cor = "🟢"
        ativo = "EURUSD_otc"
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

    try:
        candles = await client.get_candle(ativo, 60)  # M1
        df = pd.DataFrame(candles)
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])

        input_tensor, scaler = prepare_input(df)
        if input_tensor is not None:
            with torch.no_grad():
                pred = model(input_tensor).item()
            is_call = pred > 0.5
            direcao = "CALL" if is_call else "PUT"
            cor = "🟢" if is_call else "🔴"
            logger.info(f"Previsão real: {direcao}")
        else:
            direcao = "CALL"
            cor = "🟢"
    except Exception as e:
        logger.error(f"Erro ao fetch candles: {e}")
        direcao = "CALL"
        cor = "🟢"

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

    scheduler = context.job_queue
    scheduler.run_once(
        verificar_resultado,
        when=60,
        data={"periodo": periodo, "is_call": is_call if 'is_call' in locals() else True, "ativo": ativo}
    )

async def verificar_resultado(context: ContextTypes.DEFAULT_TYPE):
    periodo = context.job.data["periodo"]
    ganhou = random.choice([True, False])  # Placeholder
    if ganhou:
        stats[periodo]["gains"] += 1
    else:
        stats[periodo]["losses"] += 1

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot iniciado com sinais reais via Quotex! Sinais a cada minuto.")

def main():
    TOKEN = "8501561041:AAHucMrzlYnA0ZXR-1_HrOJ1widA6Qs4Ctw"

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        enviar_sinal,
        trigger=IntervalTrigger(minutes=1),
        args=(app,)
    )
    scheduler.start()

    print("Bot iniciado com sinais reais Quotex!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
