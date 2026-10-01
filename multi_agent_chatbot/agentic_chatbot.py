import base64
import os

import chainlit as cl
import dotenv
from agents import InputGuardrailTripwireTriggered, Runner, SQLiteSession
from nutrition_agent import exa_search_mcp, nutrition_agent
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

dotenv.load_dotenv()
openai_client = AsyncOpenAI()


@cl.on_chat_start
async def on_chat_start():
    session = SQLiteSession("conversation_history")
    cl.user_session.set("agent_session", session)
    
    # Connect Exa MCP server safely
    try:
        await exa_search_mcp.connect()
    except Exception as e:
        print(f"MCP connection warning: {e}")


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")
    msg = cl.Message(content="", author="Ojas")

    # Multimodal Vision Analysis: detect uploaded food images
    query_content = message.content or ""
    images = [
        el for el in (message.elements or [])
        if isinstance(el, cl.Image) or getattr(el, "mime", "").startswith("image/")
    ]

    if images:
        image = images[0]
        try:
            with cl.Step(name="Analyzing food image with Vision AI...", type="tool") as vision_step:
                with open(image.path, "rb") as f:
                    encoded_img = base64.b64encode(f.read()).decode("utf-8")
                mime_type = getattr(image, "mime", None) or "image/jpeg"

                vision_resp = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional nutrition vision assistant. Analyze the food photo accurately and concisely. "
                                "Identify each dish, visible ingredients, estimated portion sizes, and preparation style "
                                "so our calorie and nutrition agent can calculate precise calories."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"User question: {query_content or 'Analyze the food and estimate calories.'}\nIdentify all food items, ingredients, and portion sizes in this picture.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{encoded_img}"},
                                },
                            ],
                        },
                    ],
                    max_tokens=600,
                )
                food_description = vision_resp.choices[0].message.content or ""
                vision_step.output = food_description[:300] + ("..." if len(food_description) > 300 else "")

                query_content = (
                    f"The user shared a photo of a meal. Here is the visual breakdown of the food items and estimated portions:\n"
                    f"{food_description}\n\n"
                    f"User request: {message.content or 'Calculate the calories and breakdown the meal nutrition.'}"
                )
        except Exception as img_err:
            print(f"Vision analysis error: {img_err}")

    try:
        result = Runner.run_streamed(
            nutrition_agent,
            query_content,
            session=session,
        )

        async for event in result.stream_events():
            # Stream final message text to screen
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                await msg.stream_token(token=event.data.delta)
                print(event.data.delta, end="", flush=True)

            elif (
                event.type == "raw_response_event"
                and hasattr(event.data, "item")
                and hasattr(event.data.item, "type")
                and event.data.item.type == "function_call"
                and len(event.data.item.arguments) > 0
            ):
                with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                    step.input = event.data.item.arguments
                    print(
                        f"\nTool call: {event.data.item.name} with args: {event.data.item.arguments}"
                    )

        await msg.update()

    except InputGuardrailTripwireTriggered:
        fallback = (
            "Namaste! I'm Ojas, your everyday nutrition companion. In Ayurveda, Ojas is the vitality "
            "that comes from eating well, and that's what I'm here to help you build. "
            "I focus exclusively on food, nutrition, recipes, and meal planning. "
            "Tell me what you ate today or what you'd like to plan, and I'll help you out!"
        )
        await msg.stream_token(fallback)
        await msg.update()
    except Exception as e:
        print(f"Error during message processing: {e}")
        error_msg = "I encountered an unexpected issue while processing your request. Please try again!"
        await msg.stream_token(error_msg)
        await msg.update()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == (
        os.getenv("CHAINLIT_USERNAME"),
        os.getenv("CHAINLIT_PASSWORD"),
    ):
        return cl.User(
            identifier="Student",
            metadata={"role": "student", "provider": "credentials"},
        )
    else:
        return None
