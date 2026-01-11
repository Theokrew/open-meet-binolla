import logging
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import random  # Temporário - depois substitua pela sua IA real

# Configuração de logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Estatísticas (em memória - para produção use SQLite ou Redis)
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

async def enviar_sinal(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now()
    hora = now.hour
    minuto = now.minute
    periodo = get_periodo(hora)

    # Simulação de sinal (substitua pela sua lógica de previsão real)
    direcao = random.choice(["CALL", "PUT"])
    cor = "🟢" if direcao == "CALL" else "🔴"
    
    # Simulação de resultado (depois verifique o preço real após 1 minuto)
    ganhou = random.choice([True, False])
    if ganhou:
        stats[periodo]["gains"] += 1
    else:
        stats[periodo]["losses"] += 1

    time_str = now.strftime("%H:%M")
    
    mensagem = f"""
📊𝗘𝗡𝗧𝗥𝗔𝗗𝗔 𝗖𝗢𝗡𝗙𝗜𝗥𝗠𝗔𝗗𝗔

💹EURGBP_otc
⏰{time_str}
⏳M1
{cor}Direção: {direcao}
⚠️G1 (Opcional)

📌Abra Sua Conta Aqui ↙️ 
🔗GERENCIE SUA BANCA!!!

🎁DUVIDAS CHAME SUPORTE!!!

🎯SINAIS AO VIVO🎯
"""

    # Enviar o sinal para o seu chat privado
    await context.bot.send_message(
        chat_id=1158936585,  # ← Seu chat_id aqui!
        text=mensagem,
        parse_mode="HTML"
    )

    # Enviar relatório a cada hora cheia (opcional - pode remover se quiser)
    if minuto == 0:
        relatorio = f"""
📊 Relatório do dia até agora ({now.strftime("%d/%m/%Y %H:%M")})

Manhã:   {stats['manha']['gains']} gains  •  {stats['manha']['losses']} losses
Tarde:   {stats['tarde']['gains']} gains  •  {stats['tarde']['losses']} losses
Noite:   {stats['noite']['gains']} gains  •  {stats['noite']['losses']} losses

Total:   {sum(s['gains'] for s in stats.values())} gains  •  {sum(s['losses'] for s in stats.values())} losses
"""
        await context.bot.send_message(
            chat_id=1158936585,
            text=relatorio
        )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot iniciado!\n\nSinais serão enviados **a cada minuto** aqui no seu chat privado.\n"
        "Para parar, use /stop (ainda não implementado)."
    )

def main():
    # Seu token real
    TOKEN = "8501561041:AAHucMrzlYnA0ZXR-1_HrOJ1widA6Qs4Ctw"

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    # Agendador: envia sinal a cada 1 minuto
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        enviar_sinal,
        trigger=IntervalTrigger(minutes=1),
        args=(app,)
    )
    scheduler.start()

    print("Bot iniciado! Enviando sinais a cada minuto para chat_id 1158936585")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
