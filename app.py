import sys
import importlib
importlib.reload(sys)
import threading

from faster_whisper import WhisperModel
import datetime
import gradio as gr
import os
import torch
import wave
import contextlib
import time
from configs.model_config import *
import nltk
from models.chatglm_llm import ChatGLM
# from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import TextLoader

from textsplitter import ChineseTextSplitter
from tqdm import tqdm
from utils import torch_gc
import imageio
from wordcloud import WordCloud
import sys
from datetime import datetime
import ffmpeg
import numpy as np

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

whisper_models = ["tiny", "base", "small", "medium", "medium.en", "large-v1", "large-v2"]

source_languages = {
    "英文": "en",
    "中文": "zh"
}

source_language_list = [key[0] for key in source_languages.items()]

# 抽取摘要的提示
prompt_template = """为下面的内容生成一份精简的摘要:


{text}


返回中文摘要内容:"""

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

def get_text_summary(txt_path):
    print("starting summarizing")
    # loader = UnstructuredFileLoader(txt_path, mode="elements")
    loader = TextLoader(txt_path, encoding="utf-8")
    textsplitter = ChineseTextSplitter(pdf=False, sentence_size=SENTENCE_SIZE)
    docs = loader.load_and_split(text_splitter=textsplitter)

    for i, line in enumerate(tqdm(docs)):
        torch_gc()
        if i == 0:
            summary = next(llm._call(prompt=prompt_template.replace("{text}", line.page_content), history=[], streaming=False))[0]
        else:
            summary = next(llm._call(prompt=refine_template.replace("{existing_answer}", summary).replace("{text}", line.page_content), history=[], streaming=False))[0]

    return summary

# 加载ChatGLM模型
def load_chatglm():
    model_name = "THUDM/chatglm-6b-int8"
    print("正在加载模型:" + model_name)
    llm = ChatGLM()
    llm.load_model(model_name_or_path=model_name, llm_device="cuda:0", use_ptuning_v2=False, use_lora=False)
    llm.temperature = 1e-3
    print(model_name + "模型加载完毕")
    return llm


for i in range(5):
    try:
        llm = load_chatglm()
        selected_source_lang = "中文"
        break
    except:
        print("加载失败,正在尝试第 " + str(i + 1) + "次")
        time.sleep(5)


whisper_model = "medium"
for i in range(5):
    try:
        print("正在加载模型:" + whisper_model)
        model = WhisperModel(whisper_model, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8_float16")
        print(whisper_model + "模型加载完毕")
        break
    except Exception as e:
        print(whisper_model + "模型加载失败,正在尝试第 " + str(i + 1) + "次", e)
        time.sleep(5)


# 生成关键词词云图
def get_wordcloud_pic(words_freq, **kwargs):
    bg_img = imageio.imread('./sources/{}.png'.format(kwargs['bg_name']))
    font_path = './sources/{}.ttf'.format(kwargs['font_type'])
    word_cloud = WordCloud(font_path=font_path, background_color=kwargs['color'], max_words=kwargs['top_k'], max_font_size=50, mask=bg_img)
    word_cloud.generate_from_frequencies(words_freq)
    word_cloud.to_file('./output/result.png')
    return imageio.imread('./output/result.png')

# 抽取关键词
def extract_keyword(text):
    print("starting extracting keyword")
    keyword_extracation_prompt = f"你扮演的角色是关键词抽取工具,请从输入的文本中抽取出10个最重要的关键词,多个关键词之间用单个逗号分割: \n\n" + text
    print("抽取内容为:", keyword_extracation_prompt)
    keyword_extracation_res = next(llm._call(prompt=keyword_extracation_prompt, history=[], streaming=False))[0]
    keyword_extracation_res = keyword_extracation_res.strip().replace("，", ",").replace("：", ":").strip("关键词").strip(":").strip("。")
    print("抽取的关键词为:", keyword_extracation_res)
    words = {}
    torch_gc()

    if "." in keyword_extracation_res:
        for r in keyword_extracation_res.split("\n"):
            if len(r) > 0:
                words[r[r.index(".") + 1:].strip()] = text.count(r[r.index(".") + 1:].strip())
    elif "," in keyword_extracation_res:
        for r in keyword_extracation_res.split(","):
            if len(r) > 0:
                words[r.strip()] = text.count(r.strip())
    elif "、" in keyword_extracation_res:
        for r in keyword_extracation_res.split("、"):
            if len(r) > 0:
                words[r.strip()] = text.count(r.strip())

    print("关键词词频统计结果:", words)
    return get_wordcloud_pic(words, color='white', top_k=51, bg_name='bg', font_type='wryh')

def extract_keyword_from_file(file_name):
    print("starting extracting keyword")
    f = open(file_name, 'r', encoding='utf-8')
    text = f.read().strip()
    keyword_extracation_prompt = f"你扮演的角色是关键词抽取工具,请从输入的文本中抽取出10个最重要的关键词,多个关键词之间用单个逗号分割: \n\n" + text
    f.close()
    print("抽取内容为:", keyword_extracation_prompt)
    keyword_extracation_res = next(llm._call(prompt=keyword_extracation_prompt, history=[], streaming=False))[0]
    keyword_extracation_res = keyword_extracation_res.strip().replace("，", ",").replace("：", ":").strip("关键词").strip(":").strip("。")
    print("抽取的关键词为:", keyword_extracation_res)
    words = {}
    torch_gc()

    if "." in keyword_extracation_res:
        for r in keyword_extracation_res.split("\n"):
            if len(r) > 0:
                words[r[r.index(".") + 1:].strip()] = text.count(r[r.index(".") + 1:].strip())
    elif "," in keyword_extracation_res:
        for r in keyword_extracation_res.split(","):
            if len(r) > 0:
                words[r.strip()] = text.count(r.strip())
    elif "、" in keyword_extracation_res:
        for r in keyword_extracation_res.split("、"):
            if len(r) > 0:
                words[r.strip()] = text.count(r.strip())

    print("关键词词频统计结果:", words)
    return get_wordcloud_pic(words, color='white', top_k=51, bg_name='bg', font_type='wryh')


def speech_to_text(video_file_path):  # selected_source_lang, whisper_model):
    # for i in range(5):
    #     try:
    #         print("正在加载模型:" + whisper_model)
    #         model = WhisperModel(whisper_model, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8_float16")
    #         print(whisper_model + "模型加载完毕")
    #         break
    #     except Exception as e:
    #         print(whisper_model + "模型加载失败,正在尝试第 " + str(i + 1) + "次", e)
    #         time.sleep(5)

    if(video_file_path == None):
        raise ValueError("Error no video input")

    print("原始路径:", video_file_path)

    try:
        filename, file_ending = os.path.splitext(f'{video_file_path}')
        new_video_file_path = filename + "_" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + file_ending
        os.rename(video_file_path, new_video_file_path)
        print("新的路径:", new_video_file_path)

        print(f'file enging is {file_ending}')
        audio_file = new_video_file_path.replace(file_ending, ".wav")
        print("starting conversion to wav")
        os.system(f'ffmpeg -i "{new_video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')

        # Get duration
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")

        options = dict(language=source_languages[selected_source_lang], beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(audio_file, **transcribe_options)

        segments = []
        for segment_chunk in segments_raw:
            segments.append(segment_chunk.text)
        transcribe_text = " ".join(segments)
        print("transcribe audio done with fast whisper")

        output_txt_path = os.path.join("output", os.path.basename(new_video_file_path).split(".")[0] + ".txt")
        with open(output_txt_path, "w", encoding="utf-8") as wf:
            wf.write(transcribe_text)
            torch_gc()
            print("transcribe text writen into txt file")

        return transcribe_text, get_text_summary(output_txt_path), extract_keyword(transcribe_text)
    except Exception as e:
        raise RuntimeError(e)



SAMPLE_RATE = 16000

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


def stream_video_translate(url, max_len=10, language=None, interval=5, history_buffer_size=0, preferred_quality="audio_only",
         use_vad=True, direct_url=False, faster_whisper_args=True, **decode_options):

    line_count = 0
    stream_video_file = f"output/stream_video_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt"
    res_list = []
    this_str = ""
    n_bytes = interval * SAMPLE_RATE * 2  # Factor 2 comes from reading the int16 stream as bytes
    audio_buffer = RingBuffer((history_buffer_size // interval) + 1)
    previous_text = RingBuffer(history_buffer_size // interval)
    # 声明加载好的模型
    global model

    if use_vad:
        from utils.vad import VAD
        vad = VAD()

    print("Opening stream...")
    ffmpeg_process, streamlink_process = open_stream(url, direct_url, preferred_quality)

    try:
        stream_summary, stream_keyword = None, None
        while ffmpeg_process.poll() is None:
            # Read audio from ffmpeg stream
            in_bytes = ffmpeg_process.stdout.read(n_bytes)
            if not in_bytes:
                break

            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
            if use_vad and vad.no_speech(audio):
                print(f'{datetime.now().strftime("%H:%M:%S")}')
                continue
            audio_buffer.append(audio)

            # Decode the audio
            clear_buffers = False
            if faster_whisper_args:
                segments, info = model.transcribe(audio, language=language, **decode_options)

                decoded_language = "" if language else "(" + info.language + ")"
                decoded_text = ""
                previous_segment = ""
                for segment in segments:
                    if segment.text != previous_segment:
                        decoded_text += segment.text
                        previous_segment = segment.text

                new_prefix = decoded_text

            else:
                result = model.transcribe(np.concatenate(audio_buffer.get_all()),
                                          prefix="".join(previous_text.get_all()),
                                          language=language,
                                          without_timestamps=True,
                                          **decode_options)

                decoded_language = "" if language else "(" + result.get("language") + ")"
                decoded_text = result.get("text")
                new_prefix = ""
                for segment in result["segments"]:
                    if segment["temperature"] < 0.5 and segment["no_speech_prob"] < 0.6:
                        new_prefix += segment["text"]
                    else:
                        # Clear history if the translation is unreliable, otherwise prompting on this leads to
                        # repetition and getting stuck.
                        clear_buffers = True

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
                stream_keyword = extract_keyword_from_file(stream_video_file)

            tmp = f'{datetime.now().strftime("%H:%M:%S")} {decoded_language} {decoded_text}'
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

res_list = []
microphone_file = f"output/microphone_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.txt"

def microphone_translate(audio, stream_summary=None, stream_keyword=None, line_count=0, max_len=10, language=None, interval_sec=5, **decode_options):
    """实时转录麦克风输入语音"""
    # 引用全局变量，也可以引用state存储状态信息比如stream_summary，因为流式输入函数内都是临时变量，不能做状态延续
    global model, res_list, microphone_file
    sample_rate, audio_stream = reformat_freq(*audio)
    # 数据转换，模型只支持16000采样率
    audio_stream = audio_stream.flatten().astype(np.float32) / 32768.0
    segments, info = model.transcribe(audio_stream, language=language, **decode_options)
    # 本次处理的转录文字
    decoded_text = ""
    previous_segment = ""
    for segment in segments:
        if segment.text != previous_segment:
            decoded_text += segment.text
            previous_segment = segment.text

    decoded_language = "" if language else "(" + info.language + ")"
    tmp = f'{datetime.now().strftime("%H:%M:%S")} {decoded_language} {decoded_text}'
    length = len(res_list)
    if length >= max_len:
        res_list = res_list[length - max_len + 1:length]
    # 多次处理的转录文字
    res_list.append(tmp)
    stream = "\n".join(res_list)

    # 把转写的结果写入文件
    with open(microphone_file, "a+", encoding="utf-8") as f:
        context = f.read().strip() + " "
        #context += stream
        context += decoded_text
        f.write(context)
        line_count += 1

    # 不要频繁的摘要生成关键词,太浪费时间,这里只是为了尽快展示效果
    if line_count % (max_len * 1) == 0:
        stream_summary = get_text_summary(microphone_file)
        stream_keyword = extract_keyword_from_file(microphone_file)

    # 使用sleep控制单次处理的时长来提升识别效果，完全实时的情况，模型不能联系上下文效果很差
    time.sleep(interval_sec)
    # 返回状态
    return stream, stream_summary, stream_keyword, line_count

webui_title = """
# 🎉 ChatGLM-Video-Sense+ 🎉

项目旨在将直播视频和视频文件转写成文本,在文本摘要以及关键词抽取两大功能的加持下,辅助用户实现视频内容智能感知

项目地址为: [https://github.com/freeline55/ChatGLM-Video-Sense](https://github.com/freeline55/ChatGLM-Video-Sense)
"""


with gr.Blocks() as demo:
    gr.Markdown(webui_title)

    with gr.Tab("直播视频实时转写"):
        with gr.Row():
            with gr.Column():
                # 交互界面吊起
                url_input = gr.Textbox(label="输入url地址")
                btn_stream = gr.Button("直播转写")
                res_output = gr.Textbox(label="转写结果", lines=10, max_lines=15)

        with gr.Row():
            stream_text_summary = gr.Textbox(label="摘要结果", lines=10, max_lines=20)
            stream_text_image = gr.Image(label="关键词词云图")

        btn_stream.click(stream_video_translate, inputs=url_input, outputs=[res_output, stream_text_summary, stream_text_image], queue=True)
    with gr.Tab("视频文件智能分析"):
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="音/视频文件", mirror_webcam=False, )
                # selected_source_lang = gr.Dropdown(choices=source_language_list, type="value", value="中文", label="视频语种", interactive=True)
                # selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value", value="medium", label="选择模型", interactive=True)
                btn_analyse = gr.Button("视频分析")
        with gr.Row():
            text_translate = gr.Textbox(label="转写结果", lines=20, max_lines=50)
            text_summary = gr.Textbox(label="摘要结果", lines=20, max_lines=50)
            text_image = gr.Image(label="关键词词云图")

        btn_analyse.click(
            speech_to_text,
            inputs=[video_in],
            # inputs=[video_in, selected_source_lang, selected_whisper_model],
            outputs=[text_translate, text_summary, text_image],
            queue=False
        )
    with gr.Tab("麦克风实时转写"):
        with gr.Row():
            with gr.Column():
                # 交互界面吊起
                mic_stream = gr.Audio(label="点击麦克风", source="microphone", type="numpy", streaming=True)
                line_count = gr.Number(label="累计行数", value=0)
                res_output = gr.Textbox(label="转写结果", lines=10, max_lines=15)

        with gr.Row():
            stream_text_summary = gr.Textbox(label="摘要结果", lines=10, max_lines=20)
            stream_text_image = gr.Image(label="关键词词云图")
        # 实时更新stream_text_summary, stream_text_image
        mic_stream.stream(microphone_translate, inputs=[mic_stream, stream_text_summary, stream_text_image, line_count], outputs=[res_output, stream_text_summary, stream_text_image, line_count])

# 可能有遗留gr进程，关闭所有gr进程
gr.close_all()
time.sleep(3)
demo.queue().launch(server_name='0.0.0.0', share=False, inbrowser=False)
