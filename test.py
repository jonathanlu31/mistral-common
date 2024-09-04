# Import needed packages:
from mistral_common.protocol.instruct.messages import (
    UserMessage,
    ToolMessage,
    FinetuningAssistantMessage,
    SystemMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    Tool,
    ToolCall,
    FunctionCall
)
from mistral_common.protocol.instruct.validator import (
    ValidationMode,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# Load Mistral tokenizer
is_tekken = False

if not is_tekken:
    tokenizer_path = "/home/jonathan_lu/research/project/mistral-common/src/tokenizer_new.model.v3"
else:
    tokenizer_path = str(MistralTokenizer._data_path() / "tekken_240718.json")
tokenizer = MistralTokenizer.from_file(tokenizer_path, mode=ValidationMode.finetuning) # type: ignore

# Tokenize a list of messages
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        tools=[
            Tool(
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the users location.",
                            },
                        },
                        "required": ["location", "format"],
                    },
                )
            )
        ],
        messages=[
            SystemMessage(content="this is a system message"),
            UserMessage(content="What's the weather like today in Paris"),
            FinetuningAssistantMessage(content="Let me search that up for you", tool_calls=[ToolCall(id="3XhQnxLsT", function=FunctionCall(name="get_current_weather", arguments='{"location": "Paris, FR", "format": "celsius"}'))]),
            ToolMessage(tool_call_id="3XhQnxLsT", content="20"),
            FinetuningAssistantMessage(content="The weather is 20 degrees Celsius"),
            UserMessage(content="Describe what that temperature feels like"),
            FinetuningAssistantMessage(content="It feels warm"),
        ],
        model="DOESNT_MATTER",
    )
)
tokens, text = tokenized.tokens, tokenized.text

# Count the number of tokens
print(len(tokens))
print(text.replace("▁", " "))
