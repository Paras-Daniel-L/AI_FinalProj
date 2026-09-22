from groq import Groq

client = Groq(
    api_key="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"
)

models = client.models.list()
for model in models.data:
    print(model.id)