import openai
from abc import ABC, abstractmethod


class LLMAgent(ABC):
    """
    Base LLM Agent Abstract Class
    """
    @abstractmethod
    def call_no_history(self, content):

        """
        Makes a single call to the agent without using history

        Args:
            content: The prompt to send to the LLM

        Returns:
            response: The agent response
        """
        pass


class OpenAIResearchAgent(LLMAgent):
    def __init__(self,
                 model_name: str,
                 api_key: str):
        """
        Sets up a basic OpenAI research agent with a system message and client

        Args:
            model_name: The model name to use
            api_key: The OpenAI API key
        """
        self.model_name = model_name
        self.client = openai.Client(api_key=api_key)
        self.sys_message = \
            {"role": "system", "content": "You are a scientific research helper. You provide concise and accurate "
                                          "summaries that highlight the most important data."}

    @staticmethod
    def _format_user_message(content: str) -> dict[str, str]:
        return {"role": "user", "content": content}

    @staticmethod
    def _format_agent_message(content: str) -> dict[str, str]:
        return {"role": "assistant", "content": content}

    def call_no_history(self, content: str, **kwargs) -> str:
        """
        Makes a single call to the agent without using history.

        Args:
            content: The prompt to send to OpenAI LLM

        Returns:
            response: The response from the OpenAI LLM
        """
        msg = self._format_user_message(content)
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[self.sys_message, msg],
            **kwargs
        ).choices[0].message.content
