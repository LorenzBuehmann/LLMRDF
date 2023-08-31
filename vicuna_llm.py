from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)


class VicunaLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "vicuna"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")

        import requests

        headers = {
            # Already added when you pass json=
            # 'Content-Type': 'application/json',
        }

        json_data = {
            'model': 'vicuna-13b-v1.5',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            'temperature': 0,
            'max_tokens': 2048,
            'key': 'M7ZQL9ELMSDXXE86',
        }

        response = requests.post('https://turbo.skynet.coypu.org/', headers=headers, json=json_data)

        text = response.json()[0]['choices'][0]['message']['content']

        text = text.replace("\\_", "_") # needed to avoid invalid escaped JSON keys for reuqested JSON response in LangChain

        return text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}


if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What are 5 good names for a company that makes {product}?",
    )
    llm = VicunaLLM()

    chain = LLMChain(llm=llm, prompt=prompt)

    print(chain.run("colorful socks"))

    prompt = PromptTemplate(
        input_variables=["country"],
        template="Create a SPARQL query against Wikidata to answer the following questions."
                 ""
                 "Question: What are 5 sectors affected by a state act announced by {country}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run("Germany"))
