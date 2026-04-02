from pydantic import BaseModel

from app.agents.client import AIClient
from app.agents.prompts import VOICE_SCHEDULING_PROMPT


class VoiceConversationPlan(BaseModel):
    greeting: str
    qualification_acknowledgement: str
    slot_offer_strategy: str
    fallback_if_no_answer: str
    confirmation_script: str


class VoiceSchedulingAgent:
    def __init__(self) -> None:
        self.client = AIClient()

    async def plan(self, timezone: str, slots: list[str]) -> VoiceConversationPlan:
        if self.client.enabled:
            prompt = VOICE_SCHEDULING_PROMPT.format(timezone=timezone, slots=", ".join(slots))
            return await self.client.invoke_structured(prompt, VoiceConversationPlan)
        return VoiceConversationPlan(
            greeting="Hello, this is the recruiting team calling to schedule your interview.",
            qualification_acknowledgement="We were impressed with your background and would like to coordinate a time.",
            slot_offer_strategy="Offer the first two business-hour slots, then ask for another preference.",
            fallback_if_no_answer="Leave a voicemail and send a follow-up email with a scheduling link.",
            confirmation_script="Great, I have confirmed that slot and will send the details by email.",
        )

