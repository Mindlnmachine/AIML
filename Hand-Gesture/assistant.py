import speech_recognition as sr
import os
import webbrowser
import openai
import datetime
import random
import win32com.client  # For Windows text-to-speech
import requests
import pyautogui
import pygetwindow as gw
import ctypes
import subprocess

WEATHER_API_KEY = ''
NEWS_API_KEY = ''
CHAT_HISTORY_PATH = "C:/Users/kb290/Documents/Chat_history.txt"
OPENAI_API_KEY = '' 
script_path1 = 'C:/Users/kb290/Desktop/Python/Minor_Project/mouse phase/PHASE2.py'


chatStr = ""
standby_mode = False  

def chat(query):
    global chatStr
    print(chatStr)
    openai.api_key = OPENAI_API_KEY
    chatStr += f"Mr.Bhardwaj: {query}\n God: "
try:
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response)
except Exception as e:
    print(f"Error during OpenAI API call: {e}")


def ai(prompt):
    openai.api_key = OPENAI_API_KEY
    text = f"OpenAI response for Prompt: {prompt} \n *************************\n\n"

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",  
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text += response["choices"][0]["text"]
    if not os.path.exists("Openai"):
        os.mkdir("Openai")

    with open(f"Openai/{''.join(prompt.split('intelligence')[1:]).strip()}.txt", "w") as f:
        f.write(text)

def say(text):
    speaker = win32com.client.Dispatch("SAPI.SpVoice")  # Windows text-to-speech
    speaker.Speak(text)

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            return "Some Error Occurred. Sorry from Jarvis"

def wishTime():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        say("Good morning, sir!")
    elif 12 <= hour < 18:
        say("Good afternoon, sir!")
    else:
        say("Good evening, sir!")


def get_weather(city):
    base_url = 'http://api.openweathermap.org/data/2.5/weather'
    params = {'q': city, 'appid': WEATHER_API_KEY}

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            weather_description = data['weather'][0]['description']
            temperature = data['main']['temp']
            return f"The weather in {city} is {weather_description}. The temperature is {temperature} degrees Celsius."
        else:
            return "Sorry, unable to fetch weather information at the moment."

    except Exception as e:
        print(f"Error fetching weather: {e}")
        return "Sorry, an error occurred while fetching weather information."

def get_news():
    base_url = 'https://newsapi.org/v2/top-headlines'
    params = {'country': 'us', 'apiKey': NEWS_API_KEY}

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            articles = data.get('articles', [])
            if articles:
                news_headlines = "\n".join([f"{article['title']}" for article in articles])
                return f"Here are the top headlines:\n{news_headlines}"
            else:
                return "No news articles available at the moment."
        else:
            return "Sorry, unable to fetch news information at the moment."

    except Exception as e:
        print(f"Error fetching news: {e}")
        return "Sorry, an error occurred while fetching news information."

def open_application(app_name):
    try:
        os.startfile(app_name)
        say(f"Opening {app_name} sir...")
    except Exception as e:
        print(e)
        say(f"Sorry, there was an issue opening {app_name}.")

def close_application(app_name):
    try:
        os.system(f'taskkill /f /im {app_name}.exe')
        say(f"Closing {app_name} sir...")
    except Exception as e:
        print(e)
        say(f"Sorry, there was an issue closing {app_name}.")
    
def switch_to_application(title):
    try:
        window = gw.getWindowsWithTitle(title)
        if window:
            window[0].activate()
            say(f"Switching to {title} sir...")
        else:
            say(f"No window found with the title {title}.")
    except Exception as e:
        print(e)
        say(f"Sorry, there was an issue switching to {title}.")

def search_google(query):
    try:
        search_url = f"https://www.google.com/search?q={query}"
        webbrowser.open(search_url)
        say(f"Searching Google for {query}, sir.")
    except Exception as e:
        print(e)
        say("Sorry, there was an issue performing the Google search.")

def put_to_sleep():
    try:
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        say("Putting the system to sleep, sir.")
    except Exception as e:
        print(e)
        say("Sorry, there was an issue putting the system to sleep.")

def sign_out():
    try:
        ctypes.windll.user32.ExitWindowsEx(0, 0x1)
        say("Signing out, sir.")
    except Exception as e:
        print(e)
        say("Sorry, there was an issue signing out.")

def shut_down():
    try:
        os.system("shutdown /s /t 1")
        say("Shutting down the system, sir.")
    except Exception as e:
        print(e)
        say("Sorry, there was an issue shutting down the system.")

def restart():
    try:
        os.system("shutdown /r /t 1")
        say("Restarting the system, sir.")
    except Exception as e:
        print(e)
        say("Sorry, there was an issue restarting the system.")

def play_youtube_video(video_query):
    try:
        youtube_search_url = f"https://www.youtube.com/results?search_query={video_query}"

        webbrowser.open(youtube_search_url)

        say(f"Playing the video {video_query} on YouTube, sir.")
    except Exception as e:
        print(e)
        say("Sorry, there was an issue playing the YouTube video.")

def run_external_script(script_name):
    script_path = os.path.join(os.getcwd(), script_name)
    try:
        subprocess.run(['python', script_path], check=True)
        say(f"Running the external script, sir.")
    except subprocess.CalledProcessError as e:
        print(e)
        say("Sorry, there was an issue running the external script.") 

def take_notes(note_text):
    notes_path = "C:/Users/kb290/Documents/Notes.txt"
    with open(notes_path, "a", encoding="utf-8") as notes_file:
        notes_file.write(f"{datetime.datetime.now()}: {note_text}\n")
    say("Note added successfully.")

if __name__ == '__main__':
    print('Welcome to GestroAI')
    wishTime()  
    say("If there is anything you require assistance with during this period, I am at your service.")
    while True:
        print("Listening...")
        query = takeCommand().lower()

        if 'standby' in query:
            standby_mode = not standby_mode
            if standby_mode:
                say("Entering standby mode. I'll be here when you need me.")
            else:
                say("Exiting standby mode. How may I assist you?")

        if not standby_mode:
            sites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"], ["google", "https://www.google.com"],]
            for site in sites:
                if f"Open {site[0]}".lower() in query:
                    say(f"Opening {site[0]} sir...")
                    webbrowser.open(site[1])

            if 'play music' in query:
                music_dir = 'C:/Users/kb290/Music/songs'
                songs = os.listdir(music_dir)
                print(songs)
                os.startfile(os.path.join(music_dir, songs[0]))

            elif 'time' in query:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                say(f"Sir, the time is {strTime}")

            elif 'virtual mouse' in query:
                codePath = '"C:/Users/kb290/Desktop/Python/Minor_Project/mouse phase/PHASE2.py"'
                try:
                    say(f"Activating Virtual Mouse sir...")
                    os.system(f'python {codePath}')
                except Exception as e:
                    print(e)
                    say("Sorry, there was an issue running the other code.")
            elif 'how are you' in query:
                say(f"I am functioning excellently sir. My algorithms are finely tuned... And What about you sire..")
            
            elif 'fine' in query:
                say(f"Good to hear. Is there a specific task or inquiry you wish for me to attend to?")
            elif 'thank' in query:
                say(f"your welcome Sire...")
            elif 'weather' in query:
                say("Sure, please specify the city.")
                city = takeCommand().lower()
                weather_response = get_weather(city)
                say(weather_response)
            elif 'news' in query:
                news_response = get_news()
                say(news_response)
            elif 'open application' in query:
            
                app_name = query.split('open application')[1].strip()
                open_application(app_name)
            elif 'close application' in query:
            
                app_name = query.split('close application')[1].strip()
                close_application(app_name)
            elif 'switch to' in query:
            
                title = query.split('switch to')[1].strip()
                switch_to_application(title)
            elif 'switch to whatsapp' in query:
                switch_to_application("WhatsApp - [Problem]")
            elif 'search on google' in query:
            
                search_query = query.split('search on google')[1].strip()
                search_google(search_query)
            elif 'put to sleep' in query:
                put_to_sleep()

            elif 'shut down' in query:
                shut_down()

            elif 'restart' in query:
                restart()
            elif 'sign out' in query:
                sign_out()
            elif 'play on youtube' in query:
            
                video_query = query.split('play on youtube')[1].strip()
                play_youtube_video(video_query)
            elif 'run script' in query:
                say("Sure, please specify the script name.")
                script_name = takeCommand().lower()
                run_external_script(script_name)
            elif 'take note' in query:
                say("Sure, please dictate the note.")
                note_text = takeCommand().lower()
                take_notes(note_text)