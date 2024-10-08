from typing import List, Optional, Union

from pydantic import Field

from mistral_common.base import MistralBase


class EmbeddingRequest(MistralBase):
    input: Union[str, List[str]] = Field(description="Text to embed.")
    model: str = Field(description="ID of the model to use.")
    encoding_format: Optional[str] = Field(default="float", description="The format to return the embeddings in.")
