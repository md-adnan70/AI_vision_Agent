import os
import elevenlabs
from elevenlabs.client import ElevenLabs
import subprocess
import platform
from pydub import AudioSegment



def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client=ElevenLabs(api_key="sk_3ab5fa4fa08d002c7c6f5707a92c99f48d8e223f7f719fd5")
    audio=client.text_to_speech.convert(
        text= input_text,
        voice_id="ZF6FPAbjXT4488VcRRnw", #"JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format= "mp3_22050_32",
    )
    elevenlabs.save(audio, output_filepath)
    sound = AudioSegment.from_mp3(output_filepath)
    sound.export("output.wav", format="wav")
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":  # Windows
            subprocess.run(['ffplay', '-nodisp', '-autoexit', output_filepath])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")


# text_to_speech_with_elevenlabs("Hello, how are you? My name is Adnan", "output.wav")


from gtts import gTTS

def text_to_speech_with_gtts(input_text, output_filepath):
    language="en"

    audioobj= gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":  # Windows
            subprocess.run(['ffplay', '-nodisp', '-autoexit', output_filepath])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")