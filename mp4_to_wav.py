import subprocess
import os

def convert_mp4_to_wav(mp4_file, output_wav):
    command = f"<path_to_ffmpeg_executable> -i {mp4_file} -acodec pcm_s16le -ar 16000 -ac 1 {output_wav}"

mp4_file_name = 'C:\\Users\\Lenovo\\Downloads\\C in 100 Seconds.mp4'
output_wav_file = 'output_audio.wav'

# create a directory if it does not exist
if not os.path.exists(os.path.dirname(output_wav_file)):
    os.makedirs(os.path.dirname(output_wav_file))

convert_mp4_to_wav(mp4_file_name, output_wav_file)