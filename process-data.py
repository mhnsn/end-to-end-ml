import os
import whisper
from pydub import AudioSegment
from tqdm import tqdm
from datetime import timedelta

# Load the Whisper model. Consider a larger model for better quality, or smaller for speed if quality is still good.
model = whisper.load_model("base")

def process_folder(folder_path):
    """
    Process all .m4a files in the given folder and generate annotation files.
    """
    # Get all .m4a files in the folder
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]
    print(f"Found {len(audio_files)} audio files in {folder_path}.")

    for audio_file in tqdm(audio_files):
        file_path = os.path.join(folder_path, audio_file)
        annotation_file = file_path.replace(".mp3", ".txt")

        # Skip if annotation already exists
        if os.path.exists(annotation_file):
            tqdm.write(f"Skipping {audio_file}, annotation already exists.")
            continue
        else:
            tqdm.write(f"Processing867ret0df/d44++ {audio_file}")


        # Convert .m4a to temporary .wav
        temp_wav_path = os.path.join(folder_path, "temp_audio.wav")
        try:
            audio = AudioSegment.from_file(file_path, format="mp3")
            audio.export(temp_wav_path, format="wav")

            # Transcribe the .wav file
            result = model.transcribe(temp_wav_path)

            # Save annotations
            save_annotation(annotation_file, result["segments"])
        finally:
            # Clean up temporary .wav file
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

def save_annotation(annotation_file, segments):
    """
    Save transcription segments to an annotation file with timestamps.
    """
    with open(annotation_file, "w", encoding="utf-8") as f:
        for segment in segments:
            start_time = str(timedelta(seconds=segment["start"]))
            end_time = str(timedelta(seconds=segment["end"]))
            text = segment["text"]
            f.write(f"Start: {start_time} - End: {end_time}\n{text}\n\n")

if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        print(f"Running using your GPU {torch.cuda.get_device_name(0)}")
        print() 
    else:
        print()
        print("WARNING: RUNNING EXCLUSIVELY ON YOUR CPU. EXIT THE PROGRAM NOW AND FIX IF YOU WOULD RATHER USE YOUR GPU.")
        input()


    folder_name = input("Enter the folder name containing .m4a files: ").strip()
    
    if not os.path.isdir(folder_name):
        print("Invalid folder path.")
    else:
        process_folder(folder_name)
        print("Annotation process completed.")
