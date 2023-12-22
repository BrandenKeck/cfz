import os, json, requests, librosa
from moviepy.editor import *
from selenium import webdriver
from pexelsapi.pexels import Pexels
from langchain.llms import GPT4All, LlamaCpp
from langchain import PromptTemplate
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from langchain.chains.summarize import load_summarize_chain
from webdriver_manager.core.driver_cache import DriverCacheManager
from langchain.text_splitter import RecursiveCharacterTextSplitter


class CFZVideoDriver():
    def get_bg(self, file, duration):
        img = ImageClip(file).set_start(0).set_duration(duration)
        img.fadein(1)
        img.fadeout(1)
        img.set_pos((0,0))
        img.resize(newsize=(1920,1080))
        img.fps = 60
        return img
    def get_bot(self, file, duration):
        img = ImageClip(file).set_start(0).set_duration(duration)
        img.fadein(1)
        img.fadeout(1)
        img.set_pos((1380,540))
        img.resize(newsize=(540,540))
        img.fps = 60
        return img
    def process(self, 
                components,
                audio_clip,
                output):
        duration=librosa.get_duration(path=audio_clip)
        botimg = self.get_image('img/avatar-neutral.png', duration)
        bgimg = self.get_image(components[0], duration)
        compvideo = CompositeVideoClip([bgimg, botimg])
        compaudio = AudioFileClip(audio_clip)
        compaudio = CompositeAudioClip([compaudio])
        compvideo.audio = compaudio
        compvideo.write_videofile(output)
    def resize_func(self, t):
        if t < 2: return 1 + 0.2*t
        elif 2 <= t <= 4: return 1 + 0.2*2
        else: return 1 + 0.2*(self.duration-t)
    def position_func(self, t):
        if t < 2: return (-20*t, -20*t)
        elif 2 <= t <= 4: return (-20*2, -20*2)
        else: return (-20*2 + 20*(t-4), -20*2 + 20*(t-4))

class CFZLanguageModel():
    def __init__(self, 
                 model="./models/gpt4all-mistral-7b.gguf",
                 chunk_size=4000, 
                 chunk_overlap=160,
                 gpu=True):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        if gpu:
            self.llm = LlamaCpp(
                model_path="./models/llamacpp-mistral-7b.gguf",
                n_gpu_layers=33,
                n_batch=512,
                n_ctx=8192,
                f16_kv=True
            )
        else: self.llm = GPT4All(model=model)
    def process(self, input):
        map_prompt_template = """
            Write a summary of this chunk of text that includes the main points and any important details.
            {text}
            """
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt_template = """
            Please summarize the following text in essay form with at least six paragraphs of content.
            The essay should include a proper introduction, body, and conclusion:
            ```{text}```
            """
        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        dox = splitter.create_documents([input])
        print(len(dox))
        map_reduce_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt
        )
        summary = map_reduce_chain(dox)
        return summary

class CFZImageScraper():
    def __init__(self):
        with open('auth.json', 'r') as f:
            self.api_key = json.load(f)["images"]
    def scrape_images(self, topic,
                tint="red",
                results=1):
        path = f'./img/{topic}'
        os.makedirs(path, exist_ok=True)
        pexel = Pexels(self.api_key)
        photos_dict = pexel.search_photos(
            query=topic, 
            orientation='landscape', 
            color=tint,
            page=1, 
            per_page=results)
        for idx, photo in enumerate(photos_dict['photos']):
            img_url = photo['src']['original']
            image_path = os.path.join(path, f"{topic}-{idx}.jpeg")
            response = requests.get(img_url, stream=True)
            with open(image_path, 'wb') as outfile:
                outfile.write(response.content)

class CFZWebScraper():
    def __init__(self):
        self.driver = self.get_driver()
        self.driver.implicitly_wait(50)
    def get_driver(self):
        # Driver options
        cache_manager=DriverCacheManager("./driver")
        driver = ChromeDriverManager(cache_manager=cache_manager).install()
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
        return driver
    def scrape_wiki(self, topic):
        self.driver.implicitly_wait(50)
        self.driver.get(f"https://wikipedia.org")
        search = self.driver.find_element("xpath", 
            f"//div[@id='search-input']/input")
        search.send_keys(topic)
        search.send_keys(Keys.ENTER)
        wiki = self.driver.find_element("xpath", 
            f"//main[@id='content']")
        return wiki.text

class CFZYoutubeAPI():

    def __init__(self):
        pass # TODO