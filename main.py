import os
import logging
import asyncio
import re
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

from telethon import TelegramClient, events

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Configurações (Heroku Environment Variables)
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SOURCE_CHAT_ID = int(os.getenv("SOURCE_CHAT_ID"))  # ID do canal Pocket Signal
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID"))  # ID do seu canal/grupo/chat

LINK_BINOLLA = "https://binolla.com/?lid=2101"

# Telegram Bot App (para enviar)
telegram_app = ApplicationBuilder().token(BOT_TOKEN).build()

# Telethon Client (conta de usuário para escutar)
telethon_client = TelegramClient("forwarder_session", API_ID, API_HASH)


def parse_signal(text: str):
    """Extrai ativo, expiração e direção do texto do sinal."""
    # Asset
    m_asset = re.search(r"Asset:s*#?([A-Z0-9_]+)", text, re.IGNORECASE)
    ativo = m_asset.group(1) if m_asset else "ATIVO_DESCONHECIDO"
    
    # Expiration
    m_exp = re.search(r"Expiration:s*([A-Z0-9]+)", text, re.IGNORECASE)
    expiracao = m_exp.group(1) if m_exp else "M1"
    
    # Direção (CALL/PUT) - ajustado para seu formato
    m_dir = re.search(r"(CALL|PUT)", text, re.IGNORECASE)
    direcao = m_dir.group(1).upper() if m_dir else "CALL"
    
    # Hora atual
    hora = datetime.now().strftime("%H:%M")
    
    return ativo, expiracao, direcao, hora


async def send_to_target(ativo: str, expiracao: str, direcao: str, hora: str):
    """Monta e envia mensagem no modelo exato que você pediu."""
    cor = "🟢" if direcao == "CALL" else "🔴"
    
    mensagem = (
        "🟡OPORTUNIDADE ENCONTRADA🟡

"
        f"💹{ativo}
"
        f"⏰{hora}
"
        f"⌛{expiracao}
"
        f"{cor}Direção: {direcao}
"
        "⚠️G1 (Opcional)

"
        "📍Abra Sua Conta Aqui ↙️
"
        f"<a href="{LINK_BINOLLA}">🔗GERENCIE SUA BANCA!!!</a>

"
        "🎯SINAIS AO VIVO🎯"
    )
    
    try:
        await telegram_app.bot.send_message(
            chat_id=TARGET_CHAT_ID,
            text=mensagem,
            parse_mode="HTML"
        )
        logger.info(f"✅ Sinal enviado: {ativo} {direcao}")
    except Exception as e:
        logger.error(f"❌ Erro ao enviar: {e}")


@telethon_client.on(events.NewMessage)
async def signal_handler(event):
    """Escuta sinais no canal de origem."""
    chat = await event.get_chat()
    chat_id = getattr(chat, "id", None)
    
    if chat_id != SOURCE_CHAT_ID:
        return
    
    text = event.message.message or ""
    
    # Só processa mensagens que parecem ser sinais
    if "SIGNAL" not in text or "Asset:" not in text:
        return
    
    logger.info(f"🔍 Sinal detectado no canal {SOURCE_CHAT_ID}")
    
    ativo, expiracao, direcao, hora = parse_signal(text)
    await send_to_target(ativo, expiracao, direcao, hora)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start básico."""
    await update.message.reply_text(
        f"🤖 Bot copiador de sinais ativo!

"
        f"📡 Escutando: {SOURCE_CHAT_ID}
"
        f"📤 Enviando para: {TARGET_CHAT_ID}

"
        "Quando chegar sinal no canal de origem, replico aqui no formato solicitado."
    )


async def main():
    """Inicia bot e client juntos."""
    telegram_app.add_handler(CommandHandler("start", start_command))
    
    # Inicia Telethon
    await telethon_client.start()
    logger.info("✅ Telethon iniciado (user account)")
    
    logger.info("🚀 Bot copiador de sinais rodando!")
    
    # Roda tudo junto
    await asyncio.gather(
        telegram_app.run_polling(allowed_updates=Update.ALL_TYPES),
        telethon_client.run_until_disconnected()
    )


if __name__ == "__main__":
    asyncio.run(main())
