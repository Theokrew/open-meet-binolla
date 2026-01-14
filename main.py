import os
import logging
import asyncio
import re
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telethon import TelegramClient, events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SOURCE_CHAT_ID = int(os.getenv("SOURCE_CHAT_ID"))
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID"))

LINK_BINOLLA = "https://binolla.com/?lid=2101"

telegram_app = ApplicationBuilder().token(BOT_TOKEN).build()
telethon_client = TelegramClient("forwarder_session", API_ID, API_HASH)

def parse_signal(text):
    m_asset = re.search(r"Asset:s*#?([A-Z0-9_]+)", text, re.IGNORECASE)
    ativo = m_asset.group(1) if m_asset else "ATIVO"
    
    m_exp = re.search(r"Expiration:s*([A-Z0-9]+)", text, re.IGNORECASE)
    expiracao = m_exp.group(1) if m_exp else "M1"
    
    m_dir = re.search(r"(CALL|PUT)", text, re.IGNORECASE)
    dir
