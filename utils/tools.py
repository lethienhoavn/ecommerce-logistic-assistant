import openai
import base64
from io import BytesIO
from PIL import Image

import os
from playsound import playsound
from datetime import datetime

class ItemPrice:

    def __init__(self):
        self.item_prices = {"earbuds": "49.99 USD", "backpack": "79.50 USD", "usb-c": "12.00 SGD", "smartwatch x pro": "199.00 EUR", "laptop sleeve": "19.99 USD"}

        price_function = {
            "name": "get_item_price",
            "description": "Get the price of an item. Call this whenever you need to know the item price, for example when a user asks 'How much does this item cost ?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The item that the user asks",
                    },
                },
                "required": ["item"], # LLM will perform intent detection, then entity extraction (with semantic search & correcting typos) for us
                "additionalProperties": False
            }
        }
        self.tools = [{"type": "function", "function": price_function}]


    def get_item_price(self, item):
        print(f"Tool get_item_price called for {item}")
        item = item.lower()
        return self.item_prices.get(item, "Unknown")


class GenerateCards:

    def __init__(self):
        gen_card_function = {
            "name": "generate_gift_card",
            "description": "Generate Image Gift Card. Call this whenever a user asks 'Generate an image gift card to someone with the name ...'",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person that the card is sent to",
                    },
                },
                "required": ["name"], # LLM will perform intent detection, then entity extraction (with semantic search & correcting typos) for us
                "additionalProperties": False
            }
        }
        
        self.tools = [{"type": "function", "function": gen_card_function}]

    def genImage(self, name):

        print(f"Tool genImage called for {name}")

        image_response = openai.images.generate(
                model="dall-e-3",
                prompt=f"An image gift card with some nice words on that card to {name}",
                size="1024x1024",
                n=1,
                response_format="b64_json",
            )
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)

        # Load image with PIL
        image = Image.open(BytesIO(image_data))

        # Prepare filename with timestamp: yy-mm-dd-hh-mm-ss
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"generated_card_{timestamp}.png"

        # Optional: make sure output directory exists
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)

        # Save image
        image_path = os.path.join(output_dir, filename)
        image.save(image_path)

        return image_path

        # return Image.open("resources/sample.jpg")



class Speaker:

    def speak(message):
        response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=message)

        audio_stream = BytesIO(response.content)

        speaker_temp_dir = "speaker_temp"
        if not os.path.exists(speaker_temp_dir):
            os.makedirs(speaker_temp_dir)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = os.path.join(speaker_temp_dir, f"output_audio_{timestamp}.mp3")
        with open(output_filename, "wb") as f:
            f.write(audio_stream.read())
            f.flush()               # Đẩy từ buffer Python → OS buffer
            os.fsync(f.fileno())    # Đẩy từ OS buffer → đĩa thật

        playsound(output_filename)