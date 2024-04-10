from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch #used for machine learning and deep learning tasks, in this code tensors is used using pytorch 
import librosa # audio and music signal analysis. used for extraction of audio files, features and processing
import pydub #used for chunk divison and split on silence 
from pydub import AudioSegment
import os 
import glob 
from moviepy.editor import * #imports all classes , functions from the moviepy library
from pydub.effects import split_on_silence
from moviepy.editor import VideoFileClip #VideoFileClip is used for loading, spliting, extracting etc the audio file.

def transcribe_audio_wav2vec2(audio_file_path):
    processor, model = load_audio_model('facebook/wav2vec2-base-100h')
    speech_array, _ = librosa.load(audio_file_path, sr=16000)
    inputs = processor(speech_array, return_tensors="pt", padding=True).input_values
    logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

audio_file_path = 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\data\\intermin'

#loading the WAV2VEC2 Model
def load_audio_model(model_name):
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    return processor, model

# Function to extract audio from MP4 and save as MP3
def extract_audio(input_file, output_file):
    video = VideoFileClip(input_file) 
    audio = video.audio
    audio.write_audiofile(output_file)
    audio.close()
    video.close()

# Function to extract MP3 from MP4 files in the input directory
def extract_mp3_from_videos(input_directory, output_directory):
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Ensure input directory exists and contains video files
    if not os.path.exists(input_directory):
        print(f"Input directory '{input_directory}' doesn't exist.")
        return

    # Extract MP3 from each MP4 file in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".mp4"):
            input_file_path = os.path.join(input_directory, file_name) #mp4 path extraction using join functions
            output_file_name = os.path.splitext(file_name)[0] + ".mp3"
            output_file_path = os.path.join(output_directory, output_file_name)
            extract_audio(input_file_path, output_file_path)
            print(f"Extracted audio from {input_file_path} and saved as {output_file_path}")


# Function to decode the transcription from Wav2Vec2
def decode_transcription(logits, processor):
    predictions = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predictions[0])
    return transcription

# Directory paths for Windows OS
input_directory = 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\data\\videos'
output_directory = 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\data\\raw'
path_text = 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\data\\processed'

# Call the function to extract MP3 from videos
extract_mp3_from_videos(input_directory, output_directory)

# Get a list of MP3 files
filelist = glob.glob(os.path.join(output_directory, '*.mp3'))

# Create the output directory for text files if it doesn't exist
os.makedirs(path_text, exist_ok=True)

pydub.AudioSegment.ffmpeg = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\gyan"

# Process each MP3 file
for file in filelist:
    audio_mp3 = AudioSegment.from_mp3(file) #extraction of audio from mp3 file
    chunks = split_on_silence(audio_mp3, min_silence_len=300, silence_thresh=audio_mp3.dBFS - 14, keep_silence=300) 
    full_text = ""

    # Process every chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        temp_wav_file = f"temp_chunk{i}.wav" #f is working for i basically excuting i serial wise like temp_chunk 1, temp_chunk 2 , etc
        audio_chunk.export(temp_wav_file, format="wav") # it is putting data in temp wav file that it is stored in audio_chunk
        # Ensure the WAV file exists before attempting transcription
        if os.path.exists(temp_wav_file):
            transcription = transcribe_audio_wav2vec2(temp_wav_file)
            full_text += transcription + " "

    # Write the full_text to a text file
    output_text_file = os.path.join(path_text, os.path.splitext(os.path.basename(file))[0] + ".txt")
    with open(output_text_file, 'w') as f:
        f.write(full_text)
        