from typing import (
    List, 
    Dict, 
    Optional, 
    Callable, 
    Any, 
) 
from collections import defaultdict 
import time 
import warnings
import tiktoken
from openai import OpenAI 

def compute_openai_cost(
    model: str, 
    messages: List[Dict[str, str]], 
    response: str = '' 
) -> Dict[str, int | float]:
    """
    Compute the cost of openai api. 
    
    Each token will cost money and the cost is different for different models. This function helps 
    us know the total cost of experiments and do budget control in advance. The list of models is 
    available at https://platform.openai.com/docs/models. The cost of each model is available 
    at https://openai.com/pricing. 

    This is just an estimate; the calculated results may not be accurate.

    Parameters
    ----------
    model: str
        The model from OpenAI you use. Currently, 'gpt-4o', 'gpt-4o-2024-05-13', 'gpt-3.5-turbo-0125', 
        'gpt-3.5-turbo-instruct', and 'gpt-4-turbo' are supported. 
    messages: list of dict
        A list of messages comprising the conversation so far. For more details, please see 
        https://platform.openai.com/docs/guides/chat-completions. 
    response: str, default '' 
        The corresponding response obtained by feeding the messages to the model. By default, it is set to
        an empty string, indicating that when calculating the cost, the expense from the model's output is 
        not considered.

    Returns
    -------
    stat: dict 
        A dict containing the following keys: 
        - 'num_input_tokens': The number of input tokens. 
        - 'num_output_tokens': The number of output tokens. 
        - 'cost': The total cost (in USD) spent by an input-output pair. 

    Notes
    -----
    - The available models and their prices may change at any time. The function is likely to be 
      deprecated. Please stay updated with the latest news from OpenAI.

    Examples
    --------
    >>> messages = [
    ...     {"role": "system", "content": "You are a helpful assistant."}, 
    ...     {"role": "user", "content": "How are you?"}
    ... ]
    >>> response = "I'm doing well, thanks for asking! How about you?"
    >>> compute_openai_cost("gpt-4o", messages, response=response)
    {'num_input_tokens': 21, 'num_output_tokens': 12, 'cost': 0.000285}
    """
    supported_version = {
        'gpt-4o', 
        'gpt-4o-2024-05-13', 
        'gpt-3.5-turbo-0125', 
        'gpt-3.5-turbo-instruct', 
        'gpt-4-turbo', 
    } 
    if model not in supported_version:
        warnings.warn(
            f"{model} is not supported currently. It should be one of {supported_version}. The function just returns 0.0.", 
            UserWarning
        )
        return {'cost': 0.0, 'num_input_tokens': 0, 'num_output_tokens': 0}
    
    encoding = tiktoken.encoding_for_model(model)
    num_input_tokens = sum(len(encoding.encode(message[key])) for message in messages for key in message)
    num_output_tokens = len(encoding.encode(response))
    # every message will be prepended with some tokens
    num_input_tokens += len(messages) * 3
    # every response will be prepended with "<|start|>assistant<|message|>"
    num_input_tokens += 3
    # see https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    num_input_tokens += sum(1 for message in messages for key in message if key == 'name')
    stat = {'num_input_tokens': num_input_tokens, 'num_output_tokens': num_output_tokens}
    
    num_input_tokens, num_output_tokens = num_input_tokens / 1000, num_output_tokens / 1000

    if model == 'gpt-4o':
        cost = num_input_tokens * 0.0050 + num_output_tokens * 0.0150
    elif model == 'gpt-4o-2024-05-13':
        cost = num_input_tokens * 0.0050 + num_output_tokens * 0.0150
    elif model == 'gpt-3.5-turbo-0125':
        cost = num_input_tokens * 0.0005 + num_output_tokens * 0.0015
    elif model == 'gpt-3.5-turbo-instruct':
        cost = num_input_tokens * 0.0015 + num_output_tokens * 0.0020
    else:
        cost = num_input_tokens * 0.0100 + num_output_tokens * 0.0300
    
    stat['cost'] = cost
    return stat 


class OpenAIClient(OpenAI): 
    """
    A subclass of ``openai.OpenAI`` that wraps around the chat completions API to provide a simple interface 
    for generating text with the OpenAI language models, along with cost estimation.

    This class provides functionality for interacting with OpenAI's chat models, allowing for 
    customized generation with options like temperature, top_p, streaming, and token cost estimation. 
    It also supports post-processing of the generated content and retrying the request up to a 
    specified tolerance level if any errors occur during the API call.
    """

    def get_text_generation_output(
        self, 
        messages: List[Dict[str, str]], 
        model: str = 'gpt-4-turbo', 
        post_processor: Optional[Callable[[str], Any]] = None,
        max_tolerance: int = 3,
        temperature: float = 0.75, 
        top_p: float = 0.95,
        stream: bool = True, 
        return_stat: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the OpenAI chat completions API and return the response content, 
        with optional post-processing and cost estimation.

        Parameters
        ----------
        messages: list of dict
            The list of messages in the conversation, where each message is a dictionary with keys 
            such as 'role' and 'content'.
        model: str, default 'gpt-4-turbo'
            The model to use for text generation. This can be one of the supported models like 
            'gpt-4-turbo', 'gpt-4o', etc.
        post_processor: callable, optional
            A function that processes the generated content before returning it. If not provided, the 
            generated content is returned as-is.
        max_tolerance: int, default 3
            The maximum number of retry attempts if an error occurs during the API call.
        temperature: float, default 0.75
            Control the randomness of the response. Higher values (e.g., 1.0) make the output more 
            random, while lower values (e.g., 0.2) make it more deterministic.
        top_p: float, default 0.95
            Control the nucleus sampling. When set to 0.95, it considers only the top 95% of the 
            probability mass when generating responses.
        stream: bool, default True
            If True, enables streaming of the response in chunks. If False, waits until the entire 
            response is complete.
        return_stat: bool, default True
            If True, returns additional statistics, including the cost of the API call, the number of 
            tokens in messages, the number of tokens of response, and the time taken.
        **kwargs
            Additional arguments to pass to the OpenAI chat completions API. For more details, please 
            see https://platform.openai.com/docs/api-reference/chat/create. 

        Returns
        -------
        outputs: dict
            A dict containing the following keys:
            - 'content': The raw generated content.
            - 'processed_content': The post-processed content (if a post-processor is provided).
            - 'cost': The estimated cost (in USD) of the API call (if return_stat is True).
            - 'num_input_tokens': The number of tokens in the input  (if return_stat is True).
            - 'num_output_tokens': The number of tokens in the generated response (if return_stat is True).
            - 'time': The time taken to generate the response (if return_stat is True).
        
        Examples
        --------
        Suppose we have a valid ``api_key`` and the corresponding ``base_url``. 

        >>> client = OpenAIClient(
        ...     api_key=api_key, 
        ...     base_url=base_url, 
        ...     timeout=60,
        ...     max_retries=3,  
        ... )
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."}, 
        ...     {"role": "user", "content": "How are you?"}
        ... ]
        >>> client.get_text_generation_output(
        ...     messages,
        ...     max_tolerance=3,
        ...     model="gpt-4o", 
        ...     stream=True, 
        ...     temperature=0.75, 
        ...     top_p=0.95, 
        ... )
        {'content': "I'm just a computer program, so I don't have feelings, but I'm here and ready to help you! How can I assist you today?",
         'processed_content': "I'm just a computer program, so I don't have feelings, but I'm here and ready to help you! How can I assist you today?",
         'num_input_tokens': 21.0,
         'num_output_tokens': 28.0,
         'cost': 0.0005250000000000001,
         'time': 3.6033990383148193}
        
        Add a post-processor to split the response into a list of words. 

        >>> client.get_text_generation_output(
        ...     messages,
        ...     post_processor=lambda response: response.split(),
        ...     model="gpt-4o", 
        ...     stream=False, 
        ...     temperature=0.75, 
        ...     top_p=0.95, 
        ...     max_tokens=4, 
        ... )
        {'content': "I'm just a computer",
         'processed_content': ["I'm", 'just', 'a', 'computer'],
         'num_input_tokens': 21.0,
         'num_output_tokens': 4.0,
         'cost': 0.000165,
         'time': 2.5210800170898438}

        By default, it will return related statistics of API call. Disable it by setting ``return_stat`` to *False*. 

        >>> client.get_text_generation_output(
        ...     messages,
        ...     model="gpt-4o", 
        ...     return_stat=False, 
        ... )
        {'content': "I'm just a computer program, so I don't have feelings, but thank you for asking! How can I assist you today?",
         'processed_content': "I'm just a computer program, so I don't have feelings, but thank you for asking! How can I assist you today?"}
        """
        response_content = None 
        report = defaultdict(float)
        counter = 0 
        start_time = time.time() 
        content = ''

        while response_content is None and counter <= max_tolerance:
            try:
                response = self.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream,
                    **kwargs
                )
                if stream:
                    chunks = []
                    for chunk in response:
                        chunks.append(chunk.choices[0].delta.content or '')
                    content = ''.join(chunks)
                else:
                    content = response.choices[0].message.content
                if return_stat:
                    for key, value in compute_openai_cost(model, messages, content).items():
                        report[key] += value
            except Exception as e:
                print(e)
            finally: 
                response_content = content if post_processor is None else post_processor(content)
                counter += 1
        
        end_time = time.time() 
        outputs = {
            "content": content, 
            "processed_content": response_content,
            **report 
        }
        if return_stat: 
            outputs['time'] = end_time - start_time  

        return outputs
