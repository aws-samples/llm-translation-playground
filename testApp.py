
from ast import Str
from calendar import prmonth
from math import exp
from sqlalchemy import true
import streamlit as st
import json
import boto3
import logging
import pandas as pd
from lxml import etree
import clipboard
from botocore.exceptions import ClientError
from sacrebleu.metrics import BLEU

from bedrock_apis import (
    invokeLLM,
    getPromptXml,
    getPromptXml2,
)

from tmxFileProcess import (
    processTMXFile,
    loadEmbeddings,
    getExamples,
    populateRuleLanguageLookup,
)

logger = logging.getLogger(__name__)

#Language Choices
LAN_CHOICES = {"EN": "English", "FR": "French", "ES": "Spanish", "DE": "German", "MLM":"Malayalam", "TML":"Tamil", "JP":"Japanese"}

#Translator Model Choices
MODEL_CHOICES = {"anthropic.claude-3-sonnet-20240229-v1:0": "Claude 3 Sonnet", "anthropic.claude-3-haiku-20240307-v1:0": "Claude 3 Haiku", "mistral.mistral-large-2402-v1:0": "Mistral", "cohere.embed-multilingual-v3": "Cohere"}

bleu = BLEU()

def on_copy_click():
    # st.session_state.copied.append(text)
    if 'translated_text' in st.session_state:
      text = st.session_state['translated_text']
      clipboard.copy(text)

def populateExamplesXml(examplesRootElement): 
  if 'examples' in st.session_state :
    examples=st.session_state.examples
    
    for example in examples:
        exampleElement = etree.SubElement(examplesRootElement, 'example')
        source = etree.SubElement(exampleElement, 'source')
        source.text = example[sl]
        target = etree.SubElement(exampleElement, 'target')
        target.text = example[tl]
  

def loadRules(sl,tl):
  print(sl, tl)
  tmx_db=st.session_state.tmx_db
  matching_rules = tmx_db.similarity_search(text2translate, filter={"lang": sl})
  st.session_state.text2translate=text2translate
  st.session_state.sl=sl
  st.session_state.tl=tl
  st.session_state.matching_rules=matching_rules
  examples = getExamples(sl,tl,st.session_state.rule_language_lookup ,matching_rules)
  st.session_state.examples=examples

def displayExamples():
  examples=st.session_state.examples
  exampleText=""
  for example in examples:
    exampleText+= example[sl] + " : "
    exampleText+= example[tl]+ "\n"
  return exampleText

def getExamplesDF(text2translate, sl, tl):
  if st.session_state.sl!=sl or st.session_state.tl!=tl or st.session_state.text2translate!=text2translate:
    loadRules(sl,tl)
  
  examples=st.session_state.examples
  columns = [LAN_CHOICES[sl],LAN_CHOICES[tl]]
  rows=[None]*len(examples)
  data=[None]*len(examples)
  for index, example in enumerate(examples):
    rows[index]=index+1
    data[index]=[example[sl],example[tl]]
  
  exampleDF = pd.DataFrame(data=data, index=rows, columns=columns)
  return exampleDF




def getExampleText(text2translate, sl, tl):
  exampleText=""
  if st.session_state.sl==sl and st.session_state.tl==tl and st.session_state.text2translate==text2translate:
    exampleText=displayExamples()
  else:
    loadRules(sl,tl)
    exampleText=displayExamples()

  return exampleText  


def dict_to_xml(examples):
    xml_out = ''
    # Split each line on a colon and print the result
    for line in examples:
        if ":" in line:
            parts = line.split(":", 1)
            xml_out += '\n<example> <source>'+parts[0]+'</source> <target>'+parts[1]+'</target> </example>'
    return xml_out




def getCustomExampleXmlElement(examplesRootElement,examples):

    for line in examples:
        if ":" in line:
            parts = line.split(":", 1)
            example = etree.SubElement(examplesRootElement, 'example')
            source = etree.SubElement(example, 'source')
            source.text = parts[0].strip()
            target = etree.SubElement(example, 'target')
            target.text = parts[1].strip()

    return examplesRootElement

def populateCustomExampleXml(examplesRootElement):
  if custom_examples.strip() != ""  :
    # Split the string on newlines
    lines = custom_examples.split("\n")
    getCustomExampleXmlElement(examplesRootElement,lines) 
 


def generateExamplesXML():
  examplesRootElement = etree.Element('examples')
  populateCustomExampleXml(examplesRootElement)
  populateExamplesXml(examplesRootElement)
  return examplesRootElement


def on_evaluate_click():
  print("Running Evaluation")
  st.session_state['translated_text']
  if 'translated_text' in st.session_state and 'reference_text' in st.session_state:
    sys = st.session_state['translated_text'].split(".")
    refs = [st.session_state['reference_text'].split(".")]
    result = bleu.corpus_score(sys, refs)
    print(result)

st.title("Language Translator with LLMs")
text2translate=st.text_area("Text To translate")

col1, col2 = st.columns(2)

#Language Choices
with st.expander("Translation Choices",True):
  #st.header("Translation Choices")
  def format_func(option):
      return LAN_CHOICES[option]

  lcol1, lcol2 = st.columns(2)
  with lcol1:
    sl=st.selectbox("Select Source Language",options=list(LAN_CHOICES.keys()), format_func=format_func)
  with lcol2:
    tl=st.selectbox("Select Target Language",options=list(LAN_CHOICES.keys()), format_func=format_func)

  def format_func(option):
      return MODEL_CHOICES[option]
  model_id=st.selectbox("Select LLM models from Amazon Bedrock",options=list(MODEL_CHOICES.keys()), format_func=format_func)

#with st.expander("Explore the model"):
#  llm_q=st.text_area("What you want to know?")
#  st.session_state.llm_q=llm_q
# if st.button(MODEL_CHOICES[model_id]+" at Your Service"):
#      response = invokeLLM(llm_q,model_id)
#      result = json.loads(response.get("body").read())
#      output_list = result.get("content", [])
#      st.write(output_list[0]["text"])



with st.expander("Translation customization"):
  egcol1, egcol2 = st.columns(2)
  with egcol1:
  #TMX Examples Files
    filename = st.file_uploader("Upload a TMX file", type=["tmx"])
    st.write('You selected `%s`' % filename)

    #Embedding Model Choices
    EMBED_CHOICES = {"amazon.titan-embed-text-v2:0": "Titan Embedding Text v2", "cohere.embed-multilingual-v3": "Cohere Multilingual", "cohere.embed-english-v3": "Cohere English"}

    def format_func(option):
        return EMBED_CHOICES[option]

    embedding_id=st.selectbox("Select embedding models from Amazon Bedrock",options=list(EMBED_CHOICES.keys()), format_func=format_func)

    session_state = st.session_state
    examples= []
    if st.button("Process TMX File"):
      documents=processTMXFile(filename)
      tmx_db = loadEmbeddings(documents)
      st.session_state.tmx_db = tmx_db
      rule_language_lookup=populateRuleLanguageLookup(documents)
      st.session_state.rule_language_lookup = rule_language_lookup
      loadRules(sl,tl)

with egcol2:
  #if st.button("Collect Matching Rules"):
  #    loadRules(sl,tl)
  custom_examples=st.text_area("Custom Examples: "+ LAN_CHOICES[sl] + " : " +LAN_CHOICES[tl] +"\n")
  st.write("One example pair per line seperated by colon (:). Example: Hello, how are you? : Hola, ¿cómo estás?")

df=None
if 'tmx_db' in st.session_state :
  df=getExamplesDF(text2translate, sl, tl)

with st.expander("Exmples for RAG",expanded=True):
    #st.table(df)
    if df is not None :
      st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
      st.write(" ")

if st.button("Translate"):
 
  examplesXml=generateExamplesXML()
  #prompt = getPromptXml(sl,tl,text2translate,custom_example_xml,example_xml)
  prompt = getPromptXml2(LAN_CHOICES[sl],LAN_CHOICES[tl],text2translate,examplesXml)
  with st.expander("Generated Prompt"):
     st.text_area("Prompt",prompt)
  response=invokeLLM(prompt,model_id)

  
  # Process and print the response
  result = json.loads(response.get("body").read())
  input_tokens = result["usage"]["input_tokens"]
  output_tokens = result["usage"]["output_tokens"]
  output_list = result.get("content", [])

  print("Invocation details:")
  print(f"- The input length is {input_tokens} tokens.")
  print(f"- The output length is {output_tokens} tokens.")

  print(f"- The model returned {len(output_list)} response(s):")
  for output in result.get("usage",[]):
      print(output)

  for output in output_list:
      print(output["text"])

  translated2Text = {
              output_list[0]["text"]
          }
  st.session_state['translated_text'] = output_list[0]["text"]


with st.expander("Evaluation" ,expanded=True) :
  egcol1, egcol2 = st.columns(2)
  with egcol1:
     st.write("Reference- Paste your reference " +LAN_CHOICES[tl] +" translation  below")
     st.session_state['reference_text']=st.text_area()
     #st.button("Evaluate", on_click=on_evaluate_click, args=())
  with egcol2:
    st.write("Translation")
    if 'translated_text' in st.session_state:
      st.write(st.session_state['translated_text'])
      st.button("📋", on_click=on_copy_click, args=())
  
  #with st.expander("Metrics",expanded=True):
  #  st.write(f"- The input length is {input_tokens} tokens.")
  #  st.write(f"- The output length is {output_tokens} tokens.")
  st.button("Evaluate", on_click=on_evaluate_click, args=())
  