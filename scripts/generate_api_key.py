"""generate a single Husika TTS API key and print it for use in .env"""

import secrets

key = "hsk_" + secrets.token_hex(16)
print(f"\ngenerated API key: {key}")
print(f"\nadd this line to your .env file:\n  API_KEY={key}\n")
