import os
import sys
from pathlib import Path
import re

new_wav_scp_path = sys.argv[1]
utt2spk_path = sys.argv[2]
new_wav_scp = {}

with open(new_wav_scp_path, "r") as f:
  for line in f:
    data = line.split()
    new_wav_scp[data[0]] = data[1]

pattern = re.compile(u'/[^/]+')
utts = list(new_wav_scp.items())
utt2spk = {}
for utt in utts:
  spk = pattern.findall(utt[1])[-2][1:]
  utt2spk[utt[0]] = spk

with open(utt2spk_path, 'w') as f:
  for i in sorted(utt2spk):
    f.write(i+'\t'+utt2spk[i]+'\n')