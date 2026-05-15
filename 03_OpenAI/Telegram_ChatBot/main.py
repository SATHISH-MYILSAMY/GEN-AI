from dotenv import load_dotenv
import os
from aiogram import Bot, Dispatcher, types, executor
import openai
import sys

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

class Reference:
    '''
    A class to store the reference from the Openai API
    '''

    def __init__(self) -> None:
        self.reference = ""
        
reference = Reference()
model_name = "gpt-3.5-turbo"

# Initialize bot and dispatcher
bot = Bot(token = TELEGRAM_BOT_TOKEN)
dispatcher = Dispatcher(bot)

@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    This handler receives messages with the commands /start or /help command
    """
    await message.reply("Hi\nI am a Tele Bot!\nCreated by Sathish.\nHow can I assist you?")


@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    """
    This handler receives messages with the commands /start or /help command
    """

    help_command = """
    Hi There, I'm chatGPT Telegram bot created by Sathish! Please follow these commands -
    /start - to start the conversation
    /clear - to clear the past conversation and context.
    /help - to get this help menu.
    I hope this helps. :)
    """
    await message.reply(help_command)

def clear_past():
    '''
    This function will clear the past conversation and context.
    '''
    reference.reference = ""

@dispatcher.message_handler(commands=['clear'])
async def clear(message: types.Message):
    """
    A handler to clear the past conversation and context.
    """
    clear_past()
    await message.reply("Cleared the past conversation and context. How can I assist you now?")

@dispatcher.message_handler()
async def chatgpt(message: types.Message):
    """
    A handler to process the user's input and generate a response using the ChatGPT API
    """
    print(f">>> USER: \n\t{message.text}")
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "assistant", "content": reference.reference},
            {"role": "user", "content": message.text}
        ]
    )
    reference.reference = response["choices"][0]["message"]["content"]
    print(f">>> ChatGPT: \n\t{reference.reference}") 
    await bot.send_message(chat_id=message.chat.id, text=reference.reference)



if __name__ == "__main__":
    executor.start_polling(dispatcher, skip_updates=True)