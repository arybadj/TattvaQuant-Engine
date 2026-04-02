from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from app.config import get_settings
from app.utils.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class EmailService:
    async def send_email(self, to_email: str, subject: str, body: str) -> bool:
        if not settings.sendgrid_api_key:
            logger.info("email_mock_send", to_email=to_email, subject=subject)
            return True
        try:
            message = Mail(from_email=settings.email_from, to_emails=to_email, subject=subject, plain_text_content=body)
            SendGridAPIClient(settings.sendgrid_api_key).send(message)
            logger.info("email_sent", to_email=to_email, subject=subject)
            return True
        except Exception as exc:
            logger.exception("email_send_failed", to_email=to_email, error=str(exc))
            return False

