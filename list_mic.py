import speech_recognition as sr

# List all available microphone names
microphones = sr.Microphone.list_microphone_names()
print("Available microphone devices:")
for i, mic in enumerate(microphones):
    print(f"{i}: {mic}")
