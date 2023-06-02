import os
import openai
import time
from model.utils import get_first_line_not_comment

SLEEP_SECOND =2.8
MAX_SLEEP_SECOND = 120



os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

def query(prompt, temperature=0.2, max_tokens=128, top_p=1, frequency_penalty=0, presence_penalty=0):
    """
    This function queries the OpenAI Codex API to generate code based on the given prompt.

    Args:
    prompt: str, the prompt to generate code from
    temperature: float, the value used to module the next token probabilities
    max_tokens: int, the maximum number of tokens to generate
    top_p: float, the cumulative probability for top-p filtering
    frequency_penalty: float, value to penalize new tokens based on their existing frequency in the text so far
    presence_penalty: float, value to penalize new tokens based on whether they appear in the text so far

    Returns:
    OpenAI Completion object, the response from the OpenAI Codex API
    """

    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    return response

# handle the rate limit error
def query_with_retry(prompt, temperature=0.2, max_tokens=128, top_p=1, frequency_penalty=0, presence_penalty=0):
    """
    This function queries the OpenAI Codex API to generate code based on the given prompt.

    Args:
    prompt: str, the prompt to generate code from
    sleep_second: int, the number of seconds to sleep when the rate limit error is raised
    temperature: float, the value used to module the next token probabilities
    max_tokens: int, the maximum number of tokens to generate
    top_p: float, the cumulative probability for top-p filtering
    frequency_penalty: float, value to penalize new tokens based on their existing frequency in the text so far
    presence_penalty: float, value to penalize new tokens based on whether they appear in the text so far

    Returns:
    OpenAI Completion object, the response from the OpenAI Codex API
    """
    sleep_second = 10
    retry = True
    while retry:
        try:
            response = query(prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)

            time.sleep(SLEEP_SECOND)

            return response
        except:
            # sleep 30 seconds if the rate limit error is raised
            time.sleep(sleep_second)

            # double the sleep time if it is less than 60 seconds
            if sleep_second < MAX_SLEEP_SECOND:
                sleep_second *= 2
            
            # if the sleep time is greater than 60 seconds, then sleep 60 seconds
            else:
                sleep_second = MAX_SLEEP_SECOND

# get the next line of code from the OpenAI Codex API response
def get_frist_line(
        response, 
        language="python",
        remove_comment=True):
    """
    This function gets the code from the OpenAI Codex API response.

    Args:
    response: OpenAI Completion object, the response from the OpenAI Codex API
    remove_comment: bool, whether to remove the comments from the code

    Returns:
    Str, the code from the OpenAI Codex API response
    """

    code = response.choices[0].text

    if not remove_comment:
        # get the first line of code, except the first line is \n
        first_line = code.lstrip('\n').split('\n')[0]
    else:
        first_line = get_first_line_not_comment(code, language)
    
    return first_line