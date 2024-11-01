import time
import openai 
import re 

def convert_to_seconds(time_str): 
    match = re.search(r'(\d+)m(\d+)s', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds
    else:
        match = re.search(r'(\d+\.\d+)s', time_str)
        return float(match.group(1)) if match else None

def ask_chatgpt(params, client, tries=0):
    try:
        if tries >= 1: 
            return "0, rate limit error"
        result = client.chat.completions.create(**params)
        response = result.choices[0].message.content
        return response

    except openai.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 60
        print(f"{e.message}")
        
        if "Request too large" in e.message:
            return "0, request too large error"
                
        retry_time = convert_to_seconds(e.message)
        if retry_time is None:
            retry_time = 60        
        print(f"retry_time = {retry_time} seconds...")
        time.sleep(retry_time+2)
        return ask_chatgpt(params, client, tries=tries+1)

    except openai.UnprocessableEntityError as e:
        retry_time = 10  # Adjust the retry time as needed
        print(f"{e.message}")
        print(f"Service is unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return ask_chatgpt(params, client)

    except openai.APIError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"{e.message}")
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return ask_chatgpt(params, client)

    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(f"{e.message}")
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")      
        time.sleep(retry_time)
        return ask_chatgpt(params, client)
    
    except openai.APITimeoutError as e:
        retry_time = 10
        print(f"Service is unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return ask_chatgpt(params, client)
