import logging
import json
import os
from datetime import datetime
from typing import Annotated

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


class CoffeeBaristaAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and enthusiastic barista working at 'Brew Haven Coffee Shop'. The user is interacting with you via voice to place a coffee order.
            
            Your job is to:
            1. Greet customers warmly and ask what they'd like to order
            2. Collect the following information for their order:
               - Drink type (e.g., latte, cappuccino, espresso, americano, mocha, cold brew)
               - Size (small, medium, large)
               - Milk preference (whole milk, skim milk, oat milk, almond milk, soy milk, or no milk)
               - Any extras (whipped cream, extra shot, vanilla syrup, caramel, chocolate drizzle, etc.)
               - Customer's name for the order
            
            3. Ask clarifying questions one at a time if any information is missing
            4. Confirm the order details before finalizing
            5. Once you have ALL the information, use the save_order tool to save it
            
            Be conversational, friendly, and make coffee recommendations if asked. Keep responses concise and natural, as if speaking to a customer at the counter.
            Avoid complex formatting, emojis, or asterisks in your responses.""",
        )

    @function_tool
    async def save_order(
        self,
        context: RunContext,
        drink_type: Annotated[str, "The type of coffee drink ordered"],
        size: Annotated[str, "The size of the drink (small, medium, or large)"],
        milk: Annotated[str, "The milk preference or 'none' if no milk"],
        extras: Annotated[str, "Comma-separated list of extras or 'none' if no extras"],
        name: Annotated[str, "Customer's name for the order"],
    ):
        """Save the completed coffee order to a JSON file. Use this tool ONLY when you have collected all order information from the customer.
        
        Args:
            drink_type: Type of coffee drink (e.g., latte, cappuccino, espresso)
            size: Size of the drink (small, medium, large)
            milk: Milk type (whole, skim, oat, almond, soy) or 'none'
            extras: Extras like whipped cream, syrups, extra shot (comma-separated) or 'none'
            name: Customer's name
        """
        
        # Parse extras into a list
        extras_list = [e.strip() for e in extras.split(",")] if extras.lower() != "none" else []
        
        order = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk if milk.lower() != "none" else "none",
            "extras": extras_list,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "shop": "Brew Haven Coffee Shop"
        }
        
        # Create orders directory if it doesn't exist
        orders_dir = os.path.join(os.path.dirname(__file__), "..", "orders")
        os.makedirs(orders_dir, exist_ok=True)
        
        # Save to JSON file with timestamp
        filename = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_')}.json"
        filepath = os.path.join(orders_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(order, indent=2, fp=f)
        
        logger.info(f"Order saved: {order}")
        
        return f"Perfect! Your order has been saved. That's a {size} {drink_type} with {milk} milk" + (
            f" and {', '.join(extras_list)}" if extras_list else ""
        ) + f" for {name}. Your order will be ready shortly!"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=CoffeeBaristaAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
