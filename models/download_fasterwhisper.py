# 第一步：下载 `whisper-medium` 目录到本地
# https://pan.baidu.com/s/1befgw3FQjl3orQQVPHxP8Q?pwd=63zj

# 第二步，验证模型
from faster_whisper import WhisperModel
import torch
import time
import os
import platform
from utils import torch_gc

os_name = platform.system()
project_path = os.path.dirname(os.path.dirname(__file__))

# Whisper模型的保存路径
model_path = os.path.join(os.path.join(project_path, "models"), "faster-whisper")
print("开始加载FasterWhisper模型:", model_path)
whisper_model = WhisperModel(model_path, device="cuda" if torch.cuda.is_available() else "cpu", local_files_only=True)
print("FasterWhisper模型加载完毕")

if "Windows" == os_name:
    ffmpeg_path = f'{project_path}\\ffmpeg_release\\ffmpeg_windows\\bin\\ffmpeg.exe'
else:
    ffmpeg_path = f'{project_path}/ffmpeg_release/ffmpeg_linux/ffmpeg'


# 音频或者视频转写为文本
def speech_to_text(video_file_path):
    torch_gc()
    print("开始转写音频或者视频")
    filename, file_ending = os.path.splitext(f'{video_file_path}')
    new_video_file_path = filename + "_" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + file_ending
    # 避免上传重名文件
    os.rename(video_file_path, new_video_file_path)

    audio_file_path = new_video_file_path.replace(file_ending, ".wav")
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    os.system(f'{ffmpeg_path} -i "{new_video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file_path}"')

    options = dict(language="zh", beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    segments_raw, info = whisper_model.transcribe(audio_file_path, **transcribe_options)

    segments = []
    for segment_chunk in segments_raw:
        print(segment_chunk.text)
        segments.append(segment_chunk.text)

        # todo: delete
        if len(segment_chunk) > 10:
            break

    # 返回文件的前缀和转写后的文本
    return os.path.basename(new_video_file_path).split(".")[0], " ".join(segments)


if __name__ == "__main__":
    video_file_path = os.path.join(os.path.join(project_path, "sources"), "2.mp4")
    speech_to_text(video_file_path)

