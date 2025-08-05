import speech_recognition as sr

def mic_test():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Say something:")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("Timeout, no speech detected.")
            return

    try:
        text = r.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print(f"Request error: {e}")

if __name__ == "__main__":
    mic_test()
