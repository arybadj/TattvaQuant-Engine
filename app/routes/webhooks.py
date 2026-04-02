from fastapi import APIRouter, Response
from twilio.twiml.voice_response import Gather, VoiceResponse

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/twilio/voice/{candidate_id}")
async def twilio_voice_webhook(candidate_id: int) -> Response:
    response = VoiceResponse()
    gather = Gather(input="speech", action=f"/api/webhooks/twilio/voice/{candidate_id}/handle", method="POST")
    gather.say(
        "Hello, this is the recruiting team calling to schedule your interview. "
        "Please tell us your preferred availability after the tone."
    )
    response.append(gather)
    response.say("We will follow up with an email if we miss you. Goodbye.")
    return Response(content=str(response), media_type="application/xml")


@router.post("/twilio/voice/{candidate_id}/handle")
async def twilio_voice_handle(candidate_id: int) -> Response:
    response = VoiceResponse()
    response.say("Thank you. We captured your availability and will confirm the slot by email shortly.")
    return Response(content=str(response), media_type="application/xml")

