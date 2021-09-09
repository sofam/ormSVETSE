from dotenv import load_dotenv
import discord
from megahal import *
import os
import re


load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

megahal = MegaHAL(brainfile='./hej55.brn',max_length=500,timeout=5)
client = discord.Client()

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    # don't respond to ourselves
    if message.author == client.user:
        return
    if message.content[0] == '>':
        return
    else:
        print("Got this message: " + message.content)
        msg = re.sub('<.*> ','',message.content)
        megahal.learn(msg)
        megahal.sync()
        if len(message.mentions) > 0:
            for a in message.mentions:
                if a.id == client.user.id:
                    reply = megahal.get_reply_nolearn(msg)
                    print("Replying " + reply.text[:1999])
                    await message.channel.send(reply.text)
                else:
                    return

client.run(TOKEN)

megahal.sync()  # flush any changes to disc
megahal.close()  # flush changes and close
