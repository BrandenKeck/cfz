from langchain.llms import GPT4All
from langchain.schema import HumanMessage
llm = GPT4All(model="./gpt4all-13b-snoozy-q4_0.gguf")
messages = [
    HumanMessage(content="""
        What would be a good company name for a company that makes colorful socks? Please respond with JSON formatted data.
    """)
    ]
llm.invoke(messages)
