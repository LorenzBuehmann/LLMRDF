import os
from typing import List

import requests

DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)


class Instructor:
    def __init__(self, instructor_url, **kwargs):
        self.instructor_url = instructor_url

        if os.getenv("INSTRUCTOR_API_KEY") is not None:
            self.api_key = os.getenv("INSTRUCTOR_API_KEY")
            print("INSTRUCTOR_API_KEY is ready")
        else:
            print("INSTRUCTOR_API_KEY environment variable not found")
            raise ValueError(
                "Please provide the INSTRUCTOR_API_KEY as environment variable"
            )

        super().__init__(**kwargs)

    def compute_embedding(self, instruction: str, text: str):
        return self.compute_embeddings(instruction, [text])[0]

    def compute_embeddings(self,
                           instruction: str | List[str],
                           text: str | List[str]):

        headers = {
            # Already added when you pass json=
            # 'Content-type': 'application/json',
        }

        json_data = {
            'instruction': instruction,
            'sentence': text,
            'key': self.api_key,
        }

        response = requests.post(self.instructor_url, headers=headers, json=json_data)

        data = response.json()

        embeddings = data['embeddings']

        return embeddings
