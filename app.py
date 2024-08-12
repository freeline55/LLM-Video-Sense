import sys
import importlib
importlib.reload(sys)
import datetime
import gradio as gr
import os
import torch
import time
import nltk
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
from utils import torch_gc
import imageio
from wordcloud import WordCloud
from datetime import datetime
import ffmpeg
import uuid
import numpy as np
from collections import defaultdict
from models.download_fasterwhisper import speech_to_text, whisper_model
from models.use_zhipu import get_qa

SENTENCE_SIZE = 512
SAMPLE_RATE = 16000
os.makedirs("output", exist_ok=True)
# 设置环境变量
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


# 抽取摘要的提示
prompt_template = """为下面的内容生成一份精简的摘要:


{text}


返回中文摘要内容
"""

# 使用refine模式抽取摘要的提示
refine_template = (
    "你的工作是生成一份全文摘要.\n"
    "我已经为某个文本片段生成了一份摘要: {existing_answer}\n"
    "请在给定新的上下文信息的情况下继续完善这份摘要。\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    ""
    "如果这段新的上下文信息不能提供额外的信息,请返回原始的摘要"
)


# 获取分割后的文本
def get_split_docs(output_txt_path):
    # 加载并分割转写文本
    loader = TextLoader(output_txt_path, encoding="utf-8")
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=SENTENCE_SIZE, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    return docs


# 生成文本摘要
def get_text_summary(output_txt_path):
    print("开始文本摘要")

    docs = get_split_docs(output_txt_path)
    for i, line in enumerate(tqdm(docs)):
        if i == 0:
            summary = get_qa(prompt_template.replace("{text}", line.page_content))
        else:
            summary = get_qa(refine_template.replace("{existing_answer}", summary).replace("{text}", line.page_content))

    return summary


# 生成关键词词云图
def get_wordcloud_pic(words_freq, **kwargs):
    bg_img = imageio.imread('./sources/{}.png'.format(kwargs['bg_name']))
    font_path = './sources/{}.ttf'.format(kwargs['font_type'])
    word_cloud = WordCloud(font_path=font_path, background_color=kwargs['color'], max_words=kwargs['top_k'], max_font_size=50, mask=bg_img)
    word_cloud.generate_from_frequencies(words_freq)
    word_cloud.to_file('./output/result.png')
    return imageio.imread('./output/result.png')


# 抽取关键词
def extract_keyword(output_txt_path):
    print("开始抽取关键词")

    docs = get_split_docs(output_txt_path)
    with open(output_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = {}
    for i, line in enumerate(tqdm(docs)):
        keyword_extracation_prompt = f"请从输入的文本中抽取出十个最重要的关键词,结果使用逗号分隔: \n{line.page_content}"
        keyword_extracation_res = get_qa(keyword_extracation_prompt).replace("，", ",").replace("：", ":").replace(":", "").strip("关键词").strip("。").strip()
        print("关键词抽取结果：", keyword_extracation_res)
        if "." in keyword_extracation_res:
            for r in keyword_extracation_res.split("\n"):
                if len(r) > 0:
                    count = text.count(r[r.index(".") + 1:].strip())
                    if count > 0:
                        words[r[r.index(".") + 1:].strip()] = count
        elif "," in keyword_extracation_res:
            for r in keyword_extracation_res.split(","):
                if len(r) > 0:
                    count = text.count(r.strip())
                    if count > 0:
                        words[r.strip()] = count
        elif "、" in keyword_extracation_res:
            for r in keyword_extracation_res.split("、"):
                if len(r) > 0:
                    count = text.count(r.strip())
                    if count > 0:
                        words[r.strip()] = count

    print("关键词词频统计结果:", words)
    if len(words) > 0:
        return get_wordcloud_pic(words, color='white', top_k=51, bg_name='bg', font_type='wryh')


# 离线视频分析
def offline_video_analyse(video_file_path):
    torch_gc()
    print("开始分析离线视频:", video_file_path)
    # 视频转文本
    file_prefix, transcribe_text = speech_to_text(video_file_path)

    # 转写文本保存
    output_txt_path = os.path.join("output",  file_prefix + ".txt")
    with open(output_txt_path, "w", encoding="utf-8") as wf:
        wf.write(transcribe_text)

    # 获取转写文本，文本摘要和关键词
    return transcribe_text, get_text_summary(output_txt_path), extract_keyword(output_txt_path)


class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.data = []
        self.full = False
        self.cur = 0

    def append(self, x):
        if self.size <= 0:
            return
        if self.full:
            self.data[self.cur] = x
            self.cur = (self.cur + 1) % self.size
        else:
            self.data.append(x)
            if len(self.data) == self.size:
                self.full = True

    def get_all(self):
        """ Get all elements in chronological order from oldest to newest. """
        all_data = []
        for i in range(len(self.data)):
            idx = (i + self.cur) % self.size
            all_data.append(self.data[idx])
        return all_data

    def has_repetition(self):
        prev = None
        for elem in self.data:
            if elem == prev:
                return True
            prev = elem
        return False

    def clear(self):
        self.data = []
        self.full = False
        self.cur = 0


def open_stream(stream, direct_url, preferred_quality):
    if direct_url:
        try:
            process = (
                ffmpeg.input(stream, loglevel="panic")
                .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
                .run_async(pipe_stdout=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return process, None

    import streamlink
    import subprocess
    import threading
    stream_options = streamlink.streams(stream)
    if not stream_options:
        print("No playable streams found on this URL:", stream)
        sys.exit(0)

    option = None
    for quality in [preferred_quality, 'audio_only', 'audio_mp4a', 'audio_opus', 'best']:
        if quality in stream_options:
            option = quality
            break
    if option is None:
        # Fallback
        option = next(iter(stream_options.values()))

    def writer(streamlink_proc, ffmpeg_proc):
        while (not streamlink_proc.poll()) and (not ffmpeg_proc.poll()):
            try:
                chunk = streamlink_proc.stdout.read(1024)
                ffmpeg_proc.stdin.write(chunk)
            except (BrokenPipeError, OSError):
                pass

    cmd = ['streamlink', stream, option, "-O"]
    streamlink_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        ffmpeg_process = (
            ffmpeg.input("pipe:", loglevel="panic")
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    thread = threading.Thread(target=writer, args=(streamlink_process, ffmpeg_process))
    thread.start()
    return ffmpeg_process, streamlink_process


stream_status = True


def update_stream_status():
    global stream_status
    stream_status = False


def stream_video_translate(url, max_len=10, language=None, interval=5, history_buffer_size=0, preferred_quality="audio_only", use_vad=True, direct_url=False, faster_whisper_args=True, **decode_options):
    global stream_status

    stream_status = True
    line_count = 0
    stream_video_file = f"output/stream_video_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt"
    res_list = []
    this_str = ""
    n_bytes = interval * SAMPLE_RATE * 2  # Factor 2 comes from reading the int16 stream as bytes
    audio_buffer = RingBuffer((history_buffer_size // interval) + 1)
    previous_text = RingBuffer(history_buffer_size // interval)

    if use_vad:
        from utils.vad import VAD
        vad = VAD()

    print("Opening stream...")
    ffmpeg_process, streamlink_process = open_stream(url, direct_url, preferred_quality)

    try:
        stream_summary, stream_keyword = None, None
        while ffmpeg_process.poll() is None and stream_status:
            # Read audio from ffmpeg stream
            in_bytes = ffmpeg_process.stdout.read(n_bytes)
            if not in_bytes:
                break

            torch_gc()
            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
            if use_vad and vad.no_speech(audio):
                print(f'{datetime.now().strftime("%H:%M:%S")}')
                continue
            audio_buffer.append(audio)

            # Decode the audio
            clear_buffers = False
            segments, info = whisper_model.transcribe(audio, language=language, **decode_options)

            decoded_language = "" if language else "(" + info.language + ")"
            decoded_text = ""
            previous_segment = ""
            for segment in segments:
                if segment.text != previous_segment:
                    decoded_text += segment.text
                    previous_segment = segment.text

            new_prefix = decoded_text
            previous_text.append(new_prefix)

            if clear_buffers or previous_text.has_repetition():
                audio_buffer.clear()
                previous_text.clear()

            # 把转写的结果写入文件
            with open(stream_video_file, "a+", encoding="utf-8") as f:
                context = f.read().strip() + " "
                context += decoded_text
                f.write(context)
                line_count += 1

            # 不要频繁的摘要生成关键词,太浪费时间,这里只是为了尽快展示效果
            if line_count % (max_len * 1) == 0:
                stream_summary = get_text_summary(stream_video_file)
                stream_keyword = extract_keyword(stream_video_file)

            tmp = f'{datetime.now().strftime("%H:%M:%S")} {decoded_language} {decoded_text}'
            print(tmp)

            length = len(res_list)
            if length >= max_len:
                res_list = res_list[length - max_len + 1:length]
            res_list.append(tmp)
            this_str = "\n".join(res_list)
            yield this_str, stream_summary, stream_keyword

        this_str += "\nStream ended"
        yield this_str, stream_summary, stream_keyword
    finally:
        ffmpeg_process.kill()
        if streamlink_process:
            streamlink_process.kill()


def reformat_freq(sr, y):
    """
    sample_rate不支持48000，转换为16000
    """
    if sr not in (
        48000,
        16000,
    ):  # Deepspeech only supports 16k, (we convert 48k -> 16k)
        raise ValueError("Unsupported rate", sr)
    if sr == 48000:
        y = (
            y
            .reshape((-1, 3))
            .mean(axis=1)
            .astype("int16")
        )
        sr = 16000
    return sr, y


mic_dicts = defaultdict(dict)


def get_summary_keyword(key):
    if key not in mic_dicts:
        return None, None

    return get_text_summary(mic_dicts[key]["filename"]), extract_keyword(mic_dicts[key]["filename"])


def microphone_translate(audio, key, language=None, interval_sec=5, **decode_options):
    if key is None or len(key) <= 0:
        key = ''.join(str(uuid.uuid4()).split('-'))
        filename = f"output/microphone_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt"
        mic_dicts[key] = {"line_count": 0,  "res_list": [], "filename": filename}

    torch_gc()
    """实时转录麦克风输入语音"""
    # 引用全局变量，也可以引用state存储状态信息比如stream_summary，因为流式输入函数内都是临时变量，不能做状态延续
    sample_rate, audio_stream = reformat_freq(*audio)
    # 数据转换，模型只支持16000采样率
    audio_stream = audio_stream.flatten().astype(np.float32) / 32768.0
    segments, info = whisper_model.transcribe(audio_stream, language=language, **decode_options)
    # 本次处理的转录文字
    decoded_text = ""
    previous_segment = ""
    for segment in segments:
        if segment.text != previous_segment:
            decoded_text += segment.text
            previous_segment = segment.text

    decoded_language = "" if language else "(" + info.language + ")"
    tmp = f'{datetime.now().strftime("%H:%M:%S")} {decoded_language} {decoded_text}'

    # 多次处理的转录文字
    mic_dicts[key]["res_list"].append(tmp)

    # 把转写的结果写入文件
    with open(mic_dicts[key]["filename"], "a+", encoding="utf-8") as f:
        context = f.read().strip() + " "
        context += decoded_text
        f.write(context)
        mic_dicts[key]["line_count"] += 1

    # 使用sleep控制单次处理的时长来提升识别效果，完全实时的情况，模型不能联系上下文效果很差
    time.sleep(interval_sec)

    # 返回状态
    return "\n".join(mic_dicts[key]["res_list"]), key


webui_title = """
# 🎉 视频内容智能感知 🎉

项目旨在将直播视频、视频文件和实时音频转写成文本，在文本摘要以及关键词抽取两大功能的加持下，辅助用户快速获取音频和视频的核心内容，提高学习和工作效率

"""


with gr.Blocks() as demo:
    gr.Markdown(webui_title)

    with gr.Tab("直播视频在线分析"):
        with gr.Row():
            with gr.Column():
                # 交互界面吊起
                url_input = gr.Textbox(label="输入url地址")
                with gr.Row():
                    btn_stream = gr.Button("直播转写")
                    btn_stop = gr.Button("停止转写")
                res_output = gr.Textbox(label="转写结果", lines=10, max_lines=15)

        with gr.Row():
            stream_text_summary = gr.Textbox(label="摘要结果", lines=10, max_lines=20)
            stream_text_image = gr.Image(label="关键词词云图")

        btn_stream.click(stream_video_translate, inputs=url_input, outputs=[res_output, stream_text_summary, stream_text_image], queue=True)
        btn_stop.click(update_stream_status)
    with gr.Tab("视频文件在线分析"):
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="音/视频文件", mirror_webcam=False)
                btn_analyse = gr.Button("视频分析")
        with gr.Row():
            text_translate = gr.Textbox(label="转写结果", lines=20, max_lines=50)
            text_summary = gr.Textbox(label="摘要结果", lines=20, max_lines=50)
            text_image = gr.Image(label="关键词词云图")

        btn_analyse.click(
            offline_video_analyse,
            inputs=[video_in],
            outputs=[text_translate, text_summary, text_image],
            queue=False
        )
    with gr.Tab("实时音频在线分析"):
        with gr.Row():
            with gr.Column():
                # 交互界面吊起
                mic_stream = gr.Audio(label="点击麦克风", source="microphone", type="numpy", streaming=True)
                btn_summary_keyword = gr.Button("生成摘要和关键词")
                key = gr.Textbox(label="key", lines=1, max_lines=1, interactive=False, visible=False)
                res_output = gr.Textbox(label="转写结果", lines=10, max_lines=15)

        with gr.Row():
            stream_text_summary = gr.Textbox(label="摘要结果", lines=10, max_lines=20)
            stream_text_image = gr.Image(label="关键词词云图")

        btn_summary_keyword.click(get_summary_keyword, inputs=key, outputs=[stream_text_summary, stream_text_image])
        mic_stream.stream(microphone_translate, inputs=[mic_stream, key], outputs=[res_output, key])

# 可能有遗留gr进程，关闭所有gr进程
gr.close_all()
time.sleep(3)
demo.queue().launch(server_name='0.0.0.0', server_port=7860, share=False, inbrowser=False)





