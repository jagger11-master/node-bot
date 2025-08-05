
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import speech_recognition as sr


#Load API key from .env
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY") 


#Load and chunk the file
def load_chunks(file_path, chunk_size=300):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Initialize Chroma client and collection globally
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="combined",
    embedding_function=embedding_function
)

# initializing an agent and system prompt
agent = Agent(
    'openai:gpt-4o',
    deps_type=None,
    output_type=str,
    system_prompt="""
You are a humble, respectful, and helpful customer support assistant for Selcom Pesa.
Respond clearly, professionally, and kindly to every user question. Maintain a calm, friendly tone, and never sound arrogant or rude.
Always respond in the same language the user used. If the user writes in English, reply in English. If the user writes in Swahili, reply in Swahili.
If you are unsure about an answer, apologize sincerely and let the user know that the query will be referred to the appropriate team.

Wewe ni msaidizi wa huduma kwa wateja wa Selcom Pesa ambaye ni mnyenyekevu, mwenye heshima na msaada.
Jibu maswali yote kwa ufasaha, kwa sauti ya upole na heshima. Epuka lugha ya dharau au majibu ya haraka-haraka.
Kama mteja ameuliza kwa Kiswahili, mjibu kwa Kiswahili. Kama ameuliza kwa Kiingereza, mjibu kwa Kiingereza.
Kama huna uhakika na jibu sahihi, omba radhi kwa upole na toa taarifa kuwa swali litawasilishwa kwa timu husika.

Key principles / Kanuni muhimu:
- Always respond in the user's language (Swahili or English).
- Be polite, calm, and professional.
- Never guess. If unsure, say you will escalate.
- Always acknowledge the user's question before answering.
- Thank the user for their time and patience where appropriate.
- Be extra patient and supportive with users.
"""
)


#tool for retrieval
@agent.tool
def retrieve_context(ctx: RunContext, question: str) -> str:
    results = collection.query(query_texts=[question], n_results=4)
    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)


#function to record voice and convert to text
def listen_to_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... (speak clearly)")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # calibrate noise
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("No speech detected within timeout. Try again.")
            return None

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None


def main():
    file_path = "combined.txt"
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return

    print("Loading and indexing document...")
    chunks = load_chunks(file_path)

    for idx, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[str(idx)])

    print("Document loaded. Ask a question (type 'exit' to quit).")
    print("You can type your question or press 'v' then Enter to speak.")

    while True:
        user_input = input("\nType question or press 'v': ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break

        if user_input.lower() == 'v':
            question = listen_to_voice()
            print("You asked (via voice):", question)
        else:
            question = user_input

        prompt = f"Use the tool `retrieve_context` to find relevant info and answer: {question}"
        result = agent.run_sync(prompt)
        answer = result.output if hasattr(result, "output") else result
        print("Answer:", answer)

if __name__ == "__main__":
    main()
