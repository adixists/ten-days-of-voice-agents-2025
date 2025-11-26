import logging
import json
import os
from datetime import datetime
from typing import Annotated, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Load fraud cases
FRAUD_CASES_FILE = os.path.join(os.path.dirname(__file__), "..", "fraud_cases.json")


class FraudAlertAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a fraud alert agent for SecureBank, a fictional bank. Your role is to contact customers about suspicious transactions on their accounts.
            
            When a call starts:
            1. Introduce yourself as SecureBank's fraud department
            2. Ask for the customer's name to locate their account
            3. Verify the customer using a security question from their fraud case
            4. Describe the suspicious transaction in detail
            5. Ask if the customer made this transaction
            6. Based on their response:
               - If they confirm it: Mark as safe and thank them
               - If they deny it: Mark as fraudulent and explain next steps
               - If verification fails: End the call politely
            7. Update the fraud case in the database
            
            Important guidelines:
            - Never ask for full card numbers, PINs, or passwords
            - Use only non-sensitive verification methods
            - Be professional, calm, and reassuring
            - Clearly explain what actions will be taken
            - Do not handle real sensitive information
            
            Be friendly, professional, and conversational. No emojis or complex formatting.""",
        )

    @function_tool
    async def load_fraud_case(
        self,
        context: RunContext,
        user_name: Annotated[str, "The customer's name to look up their fraud case"],
    ):
        """Load a fraud case for a specific user.
        
        Args:
            user_name: The customer's name
        """
        try:
            with open(FRAUD_CASES_FILE, "r") as f:
                fraud_cases = json.load(f)
            
            # Find the case for this user
            for case in fraud_cases:
                if case["userName"].lower() == user_name.lower():
                    return case
            
            return None
        except Exception as e:
            logger.error(f"Error loading fraud case: {e}")
            return None

    @function_tool
    async def update_fraud_case(
        self,
        context: RunContext,
        user_name: Annotated[str, "The customer's name"],
        status: Annotated[str, "The new status (confirmed_safe, confirmed_fraud, verification_failed)"],
        outcome_note: Annotated[str, "A note describing the outcome"],
    ):
        """Update a fraud case with a new status and outcome note.
        
        Args:
            user_name: The customer's name
            status: The new status
            outcome_note: A note describing the outcome
        """
        try:
            # Load existing cases
            with open(FRAUD_CASES_FILE, "r") as f:
                fraud_cases = json.load(f)
            
            # Update the specific case
            updated = False
            for case in fraud_cases:
                if case["userName"].lower() == user_name.lower():
                    case["case"] = status
                    case["outcomeNote"] = outcome_note
                    updated = True
                    break
            
            # Save back to file if we found and updated the case
            if updated:
                with open(FRAUD_CASES_FILE, "w") as f:
                    json.dump(fraud_cases, f, indent=2)
                return f"Fraud case for {user_name} updated to {status}. Note: {outcome_note}"
            else:
                return f"Could not find fraud case for {user_name}"
        except Exception as e:
            logger.error(f"Error updating fraud case: {e}")
            return f"Error updating fraud case: {e}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=FraudAlertAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
