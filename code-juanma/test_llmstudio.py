from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="llama-3.2-3b-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key="not_required",
)

messages = [
    (
        "system",
        "You are a helpful assistant that only translates English to Spanish.",
    ),
    ("human", "Pepe is passionate about his master's in Artificial Intelligence."),
]

aiMsg = model.invoke(messages)

print(aiMsg.content)