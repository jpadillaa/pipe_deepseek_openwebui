"""
title: Azure DeepSeek Pipe
author: jpadillaa
author_url: https://github.com/jpadillaa
version: 0.0.1
license: MIT
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import os


class Pipe:
    class Valves(BaseModel):
        # Configuration for Azure DeepSeek Model
        AZURE_DEEPSEEK_API_KEY: str = os.getenv("AZURE_DEEPSEEK_API_KEY")
        AZURE_DEEPSEEK_ENDPOINT: str = os.getenv("AZURE_DEEPSEEK_ENDPOINT")
        AZURE_DEEPSEEK_API_VERSION: str = os.getenv("AZURE_DEEPSEEK_API_VERSION")

    def __init__(self):
        self.name = "Azure DeepSeek Pipe"
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"on_startup:{__name__} - Azure DeepSeek Pipe")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__} - Azure DeepSeek Pipe")
        pass

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__} - Azure DeepSeek Pipe")

        headers = {
            "Authorization": f"Bearer {self.valves.AZURE_DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        # Constructing URL for Azure AI Studio model endpoints (OpenAI compatible)
        url = f"{self.valves.AZURE_DEEPSEEK_ENDPOINT.rstrip('/')}/v1/chat/completions?api-version={self.valves.AZURE_DEEPSEEK_API_VERSION}"

        # Standard OpenAI API parameters. DeepSeek should be compatible.
        allowed_params = {
            "messages",
            "temperature",
            "top_p",
            "n",  # Number of chat completion choices to generate for each input message.
            "stream",
            "stop",  # Up to 4 sequences where the API will stop generating further tokens.
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",  # Modify the likelihood of specified tokens appearing in the completion.
            "user",  # A unique identifier representing your end-user.
            "function_call",  # (Deprecated)
            "functions",  # (Deprecated)
            "tools",
            "tool_choice",
            # "log_probs", # Whether to return log probabilities of the output tokens or not.
            # "top_logprobs", # An integer between 0 and 5 specifying the number of tokens to return log probabilities for.
            "response_format",
            "seed",
            # Azure specific or less common params from original that might not apply or need verification for DeepSeek:
            # "role", # Already part of messages
            # "content", # Already part of messages
            # "contentPart",
            # "contentPartImage",
            # "enhancements",
            # "dataSources",
        }

        # Remap user field if necessary (standard practice from original)
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = (
                body["user"]["id"] if "id" in body["user"] else str(body["user"])
            )

        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        # Log fields that were filtered out
        dropped_params = set(body.keys()) - set(filtered_body.keys())
        if dropped_params:
            print(f"Dropped params: {', '.join(dropped_params)}")

        try:
            r = requests.post(
                url=url,
                json=filtered_body,
                headers=headers,
                stream=body.get(
                    "stream", False
                ),  # Ensure stream parameter is correctly obtained
            )

            r.raise_for_status()

            if body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()
        except requests.exceptions.HTTPError as http_err:
            error_content = r.text if r else "No response content"
            print(
                f"HTTP error in Azure DeepSeek pipeline: {http_err} - Response: {error_content}"
            )
            return f"Error: {http_err} ({error_content})"
        except Exception as e:
            print(f"Requests error in Azure DeepSeek pipeline: {e}")
            # If 'r' was defined before exception (e.g. connection error vs. HTTPError)
            error_text = ""
            if "r" in locals() and r is not None:
                error_text = r.text
            return f"Error: {e} ({error_text if error_text else 'N/A'})"
