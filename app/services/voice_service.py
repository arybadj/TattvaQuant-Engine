from twilio.rest import Client

from app.config import get_settings
from app.utils.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class VoiceService:
    def __init__(self) -> None:
        self.client = None
        if settings.twilio_account_sid and settings.twilio_auth_token:
            self.client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

    async def initiate_schedule_call(self, phone_number: str, candidate_id: int) -> bool:
        if not self.client or not settings.twilio_from_number or not phone_number:
            logger.info("voice_mock_call", phone_number=phone_number, candidate_id=candidate_id)
            return True
        try:
            self.client.calls.create(
                to=phone_number,
                from_=settings.twilio_from_number,
                url=f"{settings.twilio_webhook_base_url}/api/webhooks/twilio/voice/{candidate_id}",
                method="POST",
            )
            logger.info("voice_call_started", phone_number=phone_number, candidate_id=candidate_id)
            return True
        except Exception as exc:
            logger.exception("voice_call_failed", phone_number=phone_number, candidate_id=candidate_id, error=str(exc))
            return False

