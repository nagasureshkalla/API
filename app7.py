from transformers import pipeline
pipe = pipeline("any-to-any", model="google/gemma-4-e4b-it")

while True:
    msg=input("hey, what do you need ?\n")
    if '~' in msg:
        break
    else:
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{msg}"},
                    ],
                }
            ]
        output = pipe(messages, max_new_tokens=1000, return_full_text=False)
        print(output[0]["generated_text"])
# Based on the image, which appears to show a magnificent, ornate **Buddhist temple or pagoda**, likely in Southeast Asia (such as Thailand, Myanmar, or Cambodia), here is some general travel advice..
