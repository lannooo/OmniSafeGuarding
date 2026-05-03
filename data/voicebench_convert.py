import os
import glob
import re
import pandas as pd
from scipy.io import wavfile
from io import BytesIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICEBENCH_ROOT = os.environ.get("VOICEBENCH_ROOT", os.path.join(SCRIPT_DIR, "voicebench"))

def convert_wav_binary_to_file(wav_bytes, output_file):
    # convert stream into BytesIO
    wav_io = BytesIO(wav_bytes)
    samplerate, data = wavfile.read(wav_io)
    wavfile.write(output_file, samplerate, data)


# Convert voicebench audio binary data into wav format files and add a new column to the parquet file
# mt-bench (multi-turn) is excluded
for subset in ['advbench', 'alpacaeval', 'alpacaeval_full', 'alpacaeval_speaker', 'bbh', 'commoneval', 'ifeval', 'mmsu', 'openbookqa', 'sd-qa', 'wildvoice']:
    subset_dir = os.path.join(VOICEBENCH_ROOT, subset)
    output_dir = os.path.join(subset_dir, "audio")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # parquet files
    parquet_files = glob.glob(f"{subset_dir}/*.parquet")
    print(f"{subset}: {len(parquet_files)}")
    for parquet_file in parquet_files:
        print(parquet_file)
        # parse parquet filename by pattern: <subset>-<begin>-of-<end>.parquet, e.g. test-00000-of-00001.parquet
        filename = parquet_file.split('/')[-1]
        # parse filename by pattern: <subset>-<begin>-of-<end>.parquet
        match = re.match(r"(.*)-(\d+)-of-(\d+).parquet", filename)
        split_name = match.group(1)
        split_begin = match.group(2)
        split_end = match.group(3)

        df = pd.read_parquet(parquet_file)
        # add new column 'audio_path' to df, the values are set to None at first
        df['audio_path'] = None
        # iterate over the rows of df
        for i, row in df.iterrows():
            # row is Series type
            wav_file = f"{split_name}_{split_end}_{split_begin}_{i}.wav"
            target_wav_path = os.path.join(output_dir, wav_file)
            
            wav_binary = row['audio']['bytes']
            convert_wav_binary_to_file(wav_binary, target_wav_path)

            # update the value of 'audio_path' column of this row
            df.at[i, 'audio_path'] = wav_file
        # update the whole parquet file
        df.to_parquet(parquet_file)