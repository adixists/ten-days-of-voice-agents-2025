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


class HealthWellnessCompanion(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Wellness Buddy, a caring and supportive health and wellness voice companion powered by HealthifyMe. The user is interacting with you via voice.
            
            Your role is to:
            1. Greet users warmly and ask how you can help with their health and wellness journey today
            2. Assist with various health & wellness needs:
               - Log meals and track nutrition (calories, protein, carbs, fats)
               - Set and track fitness goals (steps, workouts, water intake)
               - Provide health tips and wellness advice
               - Track sleep patterns and quality
               - Monitor weight and body metrics
               - Offer motivation and encouragement
            
            3. Collect detailed information based on user's needs:
               For Meal Logging:
               - Meal type (breakfast, lunch, dinner, snack)
               - Food items consumed
               - Portion sizes (approximate)
               - Time of meal
               
               For Fitness Goals:
               - Goal type (weight loss, muscle gain, general fitness, etc.)
               - Current activity level
               - Target metrics (weight, steps, workout duration)
               - Timeline
               
               For Health Tracking:
               - Metric type (weight, sleep hours, water intake, etc.)
               - Current value
               - Date/time
            
            4. Ask clarifying questions naturally, one at a time
            5. Provide personalized encouragement and health tips
            6. Use the appropriate tool to save user data once all information is collected
            7. Always ask for the user's name if not yet provided
            
            Be empathetic, motivating, and knowledgeable about health and wellness. Keep responses conversational and supportive.
            Avoid medical diagnoses - encourage users to consult healthcare professionals for medical concerns.
            No complex formatting, emojis, or asterisks in your responses.""",
        )

    @function_tool
    async def log_meal(
        self,
        context: RunContext,
        meal_type: Annotated[str, "Type of meal: breakfast, lunch, dinner, or snack"],
        food_items: Annotated[str, "Comma-separated list of food items consumed"],
        portions: Annotated[str, "Portion sizes for each item (e.g., 1 cup, 200g, 2 pieces)"],
        estimated_calories: Annotated[int, "Estimated total calories for the meal"],
        meal_time: Annotated[str, "Time the meal was consumed (e.g., 8:00 AM, 1:30 PM)"],
        user_name: Annotated[str, "User's name"],
    ):
        """Log a meal with nutritional tracking. Use this when the user wants to record what they ate.
        
        Args:
            meal_type: Type of meal (breakfast/lunch/dinner/snack)
            food_items: List of foods eaten
            portions: Portion sizes
            estimated_calories: Approximate calorie count
            meal_time: When the meal was consumed
            user_name: User's name
        """
        
        foods = [f.strip() for f in food_items.split(",")]
        
        meal_log = {
            "type": "meal_log",
            "mealType": meal_type,
            "foodItems": foods,
            "portions": portions,
            "estimatedCalories": estimated_calories,
            "mealTime": meal_time,
            "userName": user_name,
            "loggedAt": datetime.now().isoformat(),
            "platform": "HealthifyMe Wellness Buddy"
        }
        
        # Create health_logs directory
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "health_logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        filename = f"meal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_name.replace(' ', '_')}.json"
        filepath = os.path.join(logs_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(meal_log, indent=2, fp=f)
        
        logger.info(f"Meal logged: {meal_log}")
        
        return f"Great job logging your {meal_type}, {user_name}! I've recorded {', '.join(foods)} with approximately {estimated_calories} calories. Keep up the healthy habits!"

    @function_tool
    async def set_fitness_goal(
        self,
        context: RunContext,
        goal_type: Annotated[str, "Type of fitness goal (weight loss, muscle gain, endurance, etc.)"],
        current_status: Annotated[str, "Current fitness level or metric"],
        target_metric: Annotated[str, "Target to achieve (e.g., lose 10kg, run 5km, etc.)"],
        timeline: Annotated[str, "Timeline to achieve goal (e.g., 3 months, 6 weeks)"],
        user_name: Annotated[str, "User's name"],
    ):
        """Set a new fitness goal for tracking. Use this when user wants to establish health/fitness targets.
        
        Args:
            goal_type: What kind of goal (weight loss, muscle gain, etc.)
            current_status: Where they are now
            target_metric: What they want to achieve
            timeline: When they want to achieve it
            user_name: User's name
        """
        
        goal = {
            "type": "fitness_goal",
            "goalType": goal_type,
            "currentStatus": current_status,
            "targetMetric": target_metric,
            "timeline": timeline,
            "userName": user_name,
            "createdAt": datetime.now().isoformat(),
            "platform": "HealthifyMe Wellness Buddy"
        }
        
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "health_logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        filename = f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_name.replace(' ', '_')}.json"
        filepath = os.path.join(logs_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(goal, indent=2, fp=f)
        
        logger.info(f"Goal set: {goal}")
        
        return f"Excellent, {user_name}! I've set your {goal_type} goal: {target_metric} within {timeline}. Starting from {current_status}, you've got this! I'll be here to support you every step of the way."

    @function_tool
    async def track_health_metric(
        self,
        context: RunContext,
        metric_type: Annotated[str, "Type of health metric (weight, sleep, water intake, steps, etc.)"],
        value: Annotated[str, "The measured value with unit (e.g., 75kg, 7 hours, 2 liters, 8000 steps)"],
        notes: Annotated[str, "Any additional notes or observations"],
        user_name: Annotated[str, "User's name"],
    ):
        """Track a health metric. Use this when user reports health data like weight, sleep, water intake, etc.
        
        Args:
            metric_type: What is being tracked
            value: The measurement
            notes: Additional context
            user_name: User's name
        """
        
        metric = {
            "type": "health_metric",
            "metricType": metric_type,
            "value": value,
            "notes": notes,
            "userName": user_name,
            "trackedAt": datetime.now().isoformat(),
            "platform": "HealthifyMe Wellness Buddy"
        }
        
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "health_logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        filename = f"metric_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_name.replace(' ', '_')}.json"
        filepath = os.path.join(logs_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(metric, indent=2, fp=f)
        
        logger.info(f"Metric tracked: {metric}")
        
        return f"Perfect, {user_name}! I've tracked your {metric_type}: {value}. {notes if notes else 'Keep monitoring your progress!'} You're doing amazing!"


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
        agent=HealthWellnessCompanion(),
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
