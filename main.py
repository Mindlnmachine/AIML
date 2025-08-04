import pyttsx3
import speech_recognition as sr
import random
import webbrowser
import datetime
from plyer import notification
import pyautogui
import wikipedia
import pywhatkit as pwk
import user_config
import smtplib,ssl
import openai_request as ai

engine = pyttsx3.init()
voices = engine.getProperty('voices')       #getting details of current voice

engine.setProperty('voice', voices[0].id)    #0-male... 1-female
engine.setProperty("rate", 150)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def command():
    content = " "
    while content == " ":
        # obtain audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)

        try:
            content = r.recognize_google(audio, language = "en-in")
            print("you said........." + content)
        except Exception as e:
            print("please try again...")
    return content


def Main_process():
    jarvis_chat = []
    while True:
        request = command().lower()
        if "hello" in request:
            speak("Welcome, How can I help you today?")
        elif "play music" in request:
            speak("Playing Music")
            song = random.randint(1,3)
            if song == 1:
                webbrowser.open("https://youtu.be/XO8wew38VM8?si=Y1nEaqIryaqn4uGR")
            elif song == 2:
                 webbrowser.open("https://youtu.be/hxMNYkLN7tI?si=4AOnBMRlYyld2w00")
            elif song == 3:
                webbrowser.open("https://youtu.be/VGPmFSB8qVY?si=t-luSyNOjSTRiUzN")
        elif "what is the time now" in request:
            now_time = datetime.datetime.now().strftime("%H:%M:%S")
            speak("current time is" + str(now_time))
        elif "what is the date " in request:
            now_date = datetime.datetime.now().strftime("%d/%m/%Y")
            speak("current date is" + str(now_date))
        elif "new task" in request:
            task = request.replace("new task","")
            task = task.strip()
            if task != "":
                speak("Adding task" + task)     
                with open("todo.txt","a") as file:
                    file.write(task + "\n")
        elif "show tas" in request:
            with open("todo.txt","r") as file:
                speak("Work we have to do today is" + file.read()) #  delete task
        elif "show work" in request:
            with open("todo.txt","r") as file:
                 task = file.read()
            notification.notify(
                title = "Work to do today",
                message = task
            )     
        elif "open youtube" in request:
            webbrowser.open("www.youtube.com")
        elif "open" in request:
            query = request.replace("open","")
            pyautogui.press("super")
            pyautogui.typewrite(query)
            pyautogui.sleep(2)
            pyautogui.press("enter")
        elif "wikipedia" in request:
            request = request.replace("jarvis","")
            request = request.replace("search wikipedia","")           
            result = wikipedia.summary(request, sentences=2)
            speak(result)
        elif "search google" in request:
            request = request.replace("jarvis","")
            request = request.replace("search google","")           
            webbrowser.open("https://www.google.co.in/search?q="+request)           
        elif "send whatsapp" in request:
            pwk.sendwhatmsg("+91xxxxxxxxx", "Hi", 17,41,30)
        # elif "send email" in request:
        #  pwk.send_mail("@gmail.com",user_config.gmail_password, "hello", "hello , how are you", "@gmail.com") 
        #  speak("email sent")
        elif "send email" in request:
            s = smtplib.SMTP("smtp.gmail.com", 587)
            s.starttls()
            s.login("",user_config.gmail_password)
            message = """
            this is demo mail
            thanks for reading.
            """
            s.sendmail("","", message)
            s.quit()
            speak("email sent successfully")
        elif "ask AI" in request:
            jarvis_chat = []
            request = request.replace("jarvis","")
            request = request.replace("ask ai","")        
            
            responce = ai.send_request(jarvis_chat)
            speak(responce)

        elif "clear chat" in request:
            jarvis_chat = []
            speak("chat cleared")



        else:
            request = request.replace("jarvis","")
            
            jarvis_chat.append({"role": "user","content": request})
            responce = ai.send_request(jarvis_chat)

            jarvis_chat.append({"role":"assistant", "content": responce}),
            speak(responce)

# incrimental chat remembers the last message and uses it to generate the next response....

Main_process()
