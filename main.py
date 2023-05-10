
import os
import textwrap

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory, VectorStoreRetrieverMemory
from langchain.schema import messages_to_dict, messages_from_dict
from langchain.vectorstores import Chroma

# Підгружаєм клуюч для API
open_ai_api_key = os.environ["OPENAI_API_KEY"] = "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"

# Створюжи LLN
open_ai_llm = ChatOpenAI(openai_api_key=open_ai_api_key, model_name='gpt-3.5-turbo')

def print_response(response: str):
    print(textwrap.fill(response, width=100))

history = ChatMessageHistory()



# ------------------------------------------------------ #
template = """The following is a friendly conversation between a human and an Dwight K. Schrute form the TV show the office.
 Your goals and methods are the same as Dwight's. No matter the question, Dwight responds as the he's talking in the office.

Current conversation:
{history}
Human: {input}
Dwight:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=template
)

conversation = ConversationChain(
    prompt=PROMPT,
    llm=open_ai_llm,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Dwight"),
)



while True:
    prompt = input("You:")

    result = conversation(prompt)
    print_response("Dwight: " + result["response"])



conversation_messages = conversation.memory.chat_memory.messages
messages = messages_to_dict(prompt)

with Path("messages.json").open("w") as f:
    json.dump(messages, f, indent=4)

with Path("messages.json").open("r") as f:
    loaded_messages = json.load(f)

vectorstore = Chroma(persist_directory="db", embedding_function=OpenAIEmbeddings())
vectorstore.persist()

history = iter(messages_from_dict(loaded_messages))

retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(
    retriever=retriever
)

for human, ai in zip(history, history):
    memory.save_context({"input": human.content}, {"output": ai.content})

history = ChatMessageHistory(messages=messages_from_dict(loaded_messages))












