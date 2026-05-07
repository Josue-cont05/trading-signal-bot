import logging

import requests

from config.settings import REQUEST_TIMEOUT_SECONDS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


logger = logging.getLogger(__name__)


class TelegramService:
    def __init__(
        self,
        bot_token: str = TELEGRAM_BOT_TOKEN,
        chat_id: str = TELEGRAM_CHAT_ID,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(self, message: str) -> bool:
        if not self.is_configured():
            logger.warning("Telegram is not configured. Signal saved but alert was not sent.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Telegram alert failed: %s", exc)
            return False

        logger.info("Telegram alert sent successfully.")
        return True

if __name__ == "__main__":
    telegram = TelegramService()
    telegram.send_message("🚀 Trading Signal Bot conectado correctamente.")