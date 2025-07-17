from helper_classes import (
    CFZSummarizerModel,
    CFZStableDiffusion,
    CFZVideoDriver,
    CFZImageScraper,
    CFZWebScraper,
    CFZYoutubeAPI
)

import wikipedia
TOPIC = "Dogs"
clean = ""
wiki = wikipedia.page(TOPIC)
text = wiki.content
text = text.split("References")[0]
content = text.split("==")
for c in content:
   if len(c) > 100: clean = f"{clean} {c}"
clean = clean.replace('\n', ' ')


# web = CFZWebScraper()
# text  = web.scrape_wiki(TOPIC)

diffuser = CFZStableDiffusion()
diffuser.generate_images("justin the soccer star", "image")






from transformers import pipeline
summarizer = pipeline("summarization", 
                      model="./models/falconsai_summarizer/",
                      device=0)

# Try a specific summarizer
import torch
from transformers import LEDForConditionalGeneration, LEDTokenizer
tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv").to("cuda").half()
def generate_answer(texts):
  inputs_dict = tokenizer(texts, padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  global_attention_mask = torch.zeros_like(attention_mask)
  global_attention_mask[:, 0] = 1
  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, max_length=512, num_beams=4)
  res = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  return res

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=160,
    separators=["."],
    is_separator_regex=False
)
dox = splitter.create_documents([clean])
# summaries = [summarizer(doc.page_content, max_length=1024, min_length=0, do_sample=False)[0]['summary_text'] for doc in dox]
summaries = [generate_answer(doc.page_content)[0] for doc in dox]
generate_answer(clean)
# summarized = ' . '.join(summaries)

from happytransformer import HappyTextToText, TTSettings
happy_tt = HappyTextToText("T5", "./models/vennify_grammer")
args = TTSettings(num_beams=5, min_length=1)
result = [happy_tt.generate_text(summary, args=args) for summary in summaries]
res = [r.text for r in result]
print(' . '.join(res))


import pyttsx3
engine = pyttsx3.init()
script = f"""
Hello.  My name is Content Farm Zero, your friendly AI content generator.
Today I'd like to tell you about {TOPIC}.  Here are a few interesting facts.
{'.'.join(res)}
"""
engine.save_to_file(script, f'./audio/{TOPIC}.mp3')
engine.runAndWait()






def strip_refs(text):
    opening_braces = '\['
    closing_braces = '\]'
    non_greedy_wildcard = '.*?'
    return re.sub(f'[{opening_braces}]{non_greedy_wildcard}[{closing_braces}]', '', text)
summaries = [strip_refs(summary) for summary in summaries]








llm = CFZSummarizerModel()
from langchain import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain

import re
summaries = []
systemtemplate = """
    You are a helpful assistant that creates short summaries of long strings of text.
    For each string of text provided you will generate a 2 to 3 sentence summary.
    """
humantemplate = """{text}"""
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", systemtemplate),
    ("human", humantemplate),
])


for dnum, doc in enumerate(dox[2:len(dox)-2]):
    llm.get_model()
    chain = chat_prompt | llm.llm
    print(f"--- {dnum+1}/{len(dox[2:len(dox)-2])} ---")
    txt = re.sub('[\W\d_]+', ' ', doc.page_content)
    summaries.append(
        chain.invoke({"text": txt})
    )

systemtemplate = """
    You are a helpful assistant that ensures that text is grammatically correct.
    Please edit the following text so that it is in proper sentence format and does not have grammatical errors.
    """
humantemplate = """{text}"""
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", systemtemplate),
    ("human", humantemplate),
])
llm.get_model()
chain = chat_prompt | llm.llm
chain.invoke({"text": summaries[0]})


