import os
import logging
import asyncio
import re
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

from telethon import TelegramClient, events

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações (Heroku)
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SOURCE_CHAT_ID = int(os.getenv("SOURCE_CHAT_ID"))
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID"))

LINK_BINOLLA = "https://binolla.com/?lid=2101"

# Apps
telegram_app = ApplicationBuilder().token(BOT_TOKEN).build()
telethon_client = TelegramClient("forwarder_session", API_ID, API_HASH)


def parse_signal(text: str):
    m_asset = re.search(r"Asset:s*#?([A-Z0-9_]+)", text, re.IGNORECASE)
    ativo = m_asset.group(1) if m_asset else "ATIVO"

    m_exp = re.search(r"Expiration:s*([A-Z0-9]+)", text, re.IGNORECASE)
    expiracao = m_exp.group(1) if m_exp else "M1"

    m_dir = re.search(r"(CALL|PUT)", text, re.IGNORECASE)
    direcao = m_dir.group(1).upper() if m_dir else "CALL"

    hora = datetime.now().strftime("%H:%M")
    return ativo, expiracao, direcao, hora


async def send_to_target(ativo, expiracao, direcao, hora):
    cor = "🟢" if direcao == "CALL" else "🔴"
    mensagem = (
        "🟡OPORTUNIDADE ENCONTRADA🟡

"
        f"💹{ativo}
⏰{hora}
⌛{expiracao}
"
        f"{cor}Direção: {direcao}
⚠️G1 (Opcional)

"
        "📍Abra Sua Conta Aqui ↙️
"
        f"<a href="{LINK_BINOLLA}">🔗GERENCIE SUA BANCA!!!</a>

"
        "🎯SINAIS AO VIVO🎯"
    )
    
    await telegram_app.bot.send_message(
        chat_id=TARGET_CHAT_ID,
        text=mensagem,
        parse_mode="HTML"
    )


@telethon_client.on(events.NewMessage)
async def signal_handler(event):
    chat_id = (await event.get_chat()).id
    if chat_id != SOURCE_CHAT_ID:
        return
    
    text = event.message.message or ""
    if "SIGNAL" not in text or "Asset:" not in text:
        return
    
    ativo, expiracao, direcao, hora = parse_signal(text)
    await send_to_target(ativo, expiracao, direcao, hora)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot copiador ativo!")


async def main():
    telegram_app.add_handler(CommandHandler("start", start_command))
    await telethon_client.start()
    await asyncio.gather(
        telegram_app.run_polling(),
        telethon_client.run_until_disconnected()
    )


if __name__ == "__main__":
    asyncio.run(main())
