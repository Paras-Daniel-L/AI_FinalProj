from groq import Groq

client = Groq(
    api_key=""
)

models = client.models.list()
for model in models.data:
    print(model.id)