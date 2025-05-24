import openai
import json
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
from utils.tools import ItemPrice, GenerateCards, Speaker

openai.api_key = OPENAI_API_KEY

def call_openai(messages):
    ''' messages including system_prompt + history(user & assistant messages) + new message '''
    image = None

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools= ItemPrice.tools + GenerateCards.tools,
        temperature=OPENAI_TEMPERATURE
    )

    if response.choices[0].finish_reason=="tool_calls": # default finish_reason = 'stop'
        # assistant call tool
        message = response.choices[0].message 
        response, image = handle_tool_call(message)
        messages.append(message)
        messages.append(response)

        # assistant wrap retrieved item price and write answer text back to user
        response = openai.chat.completions.create(
            model=OPENAI_MODEL, 
            messages=messages
        ) 

    response_text = response.choices[0].message.content

    # call speaker
    # Speaker.speak(response_text)
    
    return response_text, image


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    tool_name = tool_call.function.name
    image = None

    # tool reponse with price (or unknown)
    if tool_name == "get_item_price":
        item = arguments.get('item')
        price = ItemPrice.get_item_price(item)
        response = {
            "role": "tool",
            "content": json.dumps({"item": item, "price": price}),
            "tool_call_id": tool_call.id
        } 
    elif tool_name == "generate_gift_card":
        name = arguments.get('name')
        response = {
            "role": "tool",
            "content": name,
            "tool_call_id": tool_call.id
        } 
        image = GenerateCards.genImage(name)
    else:
        response = {
            "role": "tool",
            "content": "unknown",
            "tool_call_id": tool_call.id
        } 

    return response, image
