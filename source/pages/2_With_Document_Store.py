from calendar import prmonth
from math import exp
from sqlalchemy import true
import streamlit as st
import json
import boto3
import logging
import pandas as pd
import clipboard
from botocore.exceptions import ClientError
from sacrebleu.metrics import BLEU
from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
import nltk

nltk.download('wordnet')

from utils.bedrock_apis import (
    invokeLLM,
    converse,
    getFormattedPrompt,
    generateCustomTerminologyXml,
    generateExamplesXML,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
)

from processors.tmx_processor_oss import (
    processTMXFile,
    queryIndex,
    listIndices,
    populateRuleLanguageLookup,
    loadExamples,
)

from utils.ui_utils import (
    MODEL_CHOICES,
    loadLanguageChoices,
)

logger = logging.getLogger(__name__)

#Language Choices
#LAN_CHOICES = getLanguageChoices()

if "lang_list" not in st.session_state:
  st.session_state["lang_list"] = loadLanguageChoices()

bleu = BLEU()
bert_scorer = None

def on_copy_click():
    # st.session_state.copied.append(text)
    if 'translated_text' in st.session_state:
      text = st.session_state['translated_text']
      clipboard.copy(text)

def getLanguageChoices():
  if "lang_list" not in st.session_state:
     current_lang_mask = None
     if "lang_mask" in st.session_state:
        current_lang_mask = st.session_state["lang_mask"]
        st.session_state["lang_list"] = loadLanguageChoices(lang_mask=current_lang_mask)
  return st.session_state["lang_list"]

def loadRules(sl,tl):
  st.session_state.text2translate=text2translate
  st.session_state.sl=sl
  st.session_state.tl=tl
  examples = loadExamples(sl,tl,st.session_state.rule_language_lookup)
  st.session_state.examples=examples

def displayExamples(sl, tl):
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
  columns = [getLanguageChoices()[sl],getLanguageChoices()[tl]]
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

def refresh_metrics():
   with st.sidebar:
    st.subheader("Metrics")
    if 'latency' in st.session_state:
      latency=st.session_state['latency']
      st.metric(label="Latency(ms)", value=f'{latency:,}')
    if 'input_tokens' in st.session_state:
      input_tokens=st.session_state['input_tokens']
      st.metric(label="Input Tokens", value=f'{input_tokens:,}')
    if 'output_tokens' in st.session_state:
      output_tokens=st.session_state['output_tokens']
      st.metric(label="Output Tokens", value=f'{output_tokens:,}')
    if 'bleu' in st.session_state:
      bleu = st.session_state['bleu']
      if 'bleu_delta' in st.session_state['bleu']: st.metric(label="BLEU", value=str(round(bleu['score'], 2)), delta=str(round(bleu['bleu_delta'], 2)))
      else: st.metric(label="BLEU", value=str(round(bleu['score'], 2)))
    if 'bert_score' in st.session_state:
      bert_score = st.session_state['bert_score']
      if 'bert_delta' in st.session_state['bert_score']: 
         st.metric(label="BERTScore F1", value=str(round(bert_score['f1'] * 100, 2)), delta=str(round(bert_score['bert_delta'] * 100, 2)))
      else: 
         st.metric(label="BERTScore F1", value=str(round(bert_score['f1'] * 100, 2)))
    if 'meteor' in st.session_state:
      meteor = st.session_state['meteor']
      if 'meteor_delta' in meteor:
        st.metric(label="METEOR", value=str(round(meteor['score'] * 100, 2)), delta=str(round(meteor['meteor_delta'] * 100, 2)))
      else:
        st.metric(label="METEOR", value=str(round(meteor['score'] * 100, 2)))
    if 'chrf' in st.session_state:
        chrf = st.session_state['chrf']
        if 'chrf_delta' in chrf:
            st.metric(label="ChrF", value=str(round(chrf['score'] * 100, 2)), delta=str(round(chrf['chrf_delta'] * 100, 2)))
        else:
            st.metric(label="ChrF", value=str(round(chrf['score'] * 100, 2)))

def evaluate():
  print("Running Evaluation")
  if 'translated_text' in st.session_state and 'reference_text' in st.session_state:
    
    # BLEU score
    sys = st.session_state['translated_text'].split(".")
    refs = [st.session_state['reference_text'].split(".")]
    result = bleu.corpus_score(sys, refs)
         
    bleu_delta = None
    if 'bleu' in st.session_state:
       previous = st.session_state['bleu']['score']
       bleu_delta = result.score - previous
       st.session_state['bleu']['bleu_delta']=bleu_delta
    else:
       st.session_state['bleu'] = {}
    st.session_state['bleu']['score']=result.score

    # BERTScore
    global bert_scorer
    if bert_scorer is None or bert_scorer.lang != tl.lower():
      bert_scorer = BERTScorer(model_type='bert-base-uncased', lang=tl.lower())
    P, R, F1 = bert_scorer.score([st.session_state['translated_text']], [st.session_state['reference_text']])
    
    bert_delta = 0
    if 'bert_score' in st.session_state:
       previous = st.session_state['bert_score']['f1']
       bert_delta = F1.mean().item() - previous
       #st.session_state['bert_score']['bert_delta']=bert_delta
    else:
       st.session_state['bert_score'] = {}

    st.session_state['bert_score'] = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item(),
        'bert_delta': bert_delta
    }

    # METEOR score
    translated = st.session_state['translated_text'].split()
    reference = [st.session_state['reference_text'].split()]
    meteor_result = meteor_score(reference, translated)
    meteor_delta = None
    if 'meteor' in st.session_state:
        previous = st.session_state['meteor']['score']
        meteor_delta = meteor_result - previous
        st.session_state['meteor']['meteor_delta'] = meteor_delta
    else:
        st.session_state['meteor'] = {}
    st.session_state['meteor']['score'] = meteor_result

    # ChrF score
    chrf_result = sentence_chrf(
        reference=st.session_state['reference_text'],
        hypothesis=st.session_state['translated_text'],
        min_len=1,  # minimum n-gram length
        max_len=6,  # maximum n-gram length
        beta=3.0    # importance of recall over precision
    )

    chrf_delta = None
    if 'chrf' in st.session_state:
        previous = st.session_state['chrf']['score']
        chrf_delta = chrf_result - previous
        st.session_state['chrf']['chrf_delta'] = chrf_delta
    else:
        st.session_state['chrf'] = {}
    st.session_state['chrf']['score'] = chrf_result

def translate():
  examplesXml=generateExamplesXML(st.session_state['custom_examples'],sl,tl, st.session_state)
  customTermsXml=generateCustomTerminologyXml(st.session_state['custom_terms'])
  prompt = getFormattedPrompt(getLanguageChoices()[sl],getLanguageChoices()[tl],text2translate,examplesXml, userPrompt, systemPrompt, customTermsXml)
  st.session_state['prompt'] = prompt
  #response=invokeLLM(prompt,model_id)
  response=converse(systemPrompt,prompt,model_id, max_seq_len, temperature,top_p)

  # Process and print the response
  #result = json.loads(response.get("body").read())
  st.session_state['input_tokens'] = response["usage"]["inputTokens"]
  st.session_state['output_tokens'] = response["usage"]["outputTokens"]
  st.session_state['latency']=response["metrics"]["latencyMs"]
  output_list = response["output"]["message"]["content"]

  print(f"{len(output_list)} translation response(s) received")
  translated2Text = {
              output_list[0]["text"]
          }
  st.session_state['translated_text'] = output_list[0]["text"]
  
  if 'bleu' in st.session_state:
    st.session_state.pop("bleu")
  evaluate()

def on_index_change():
    current_selection = st.session_state.index_name
    print(f"Current selection: {current_selection}")
    if current_selection != "No Index Selected":
        documents = queryIndex(current_selection)
        rule_language_lookup = populateRuleLanguageLookup(documents)
        st.session_state.rule_language_lookup = rule_language_lookup
        st.session_state.tmx_loaded = True
        loadRules(sl, tl)



st.title("Language Translator with LLMs")
text2translate=st.text_area("Source Text")

col1, col2 = st.columns(2)

#Language Choices
with st.expander("Translation Configuration",True):
  #st.header("Translation Choices")
  def format_func(option):
      return getLanguageChoices()[option]

  lcol1, lcol2 = st.columns(2)
  with lcol1:
    sl=st.selectbox("Select Source Language",options=list(getLanguageChoices().keys()), format_func=format_func)
  with lcol2:
    tl=st.selectbox("Select Target Language",options=list(getLanguageChoices().keys()), format_func=format_func)

  def format_func(option):
      return MODEL_CHOICES[option]
  model_id=st.selectbox("Select an LLM:",options=list(MODEL_CHOICES.keys()), format_func=format_func)
  
  st.text("Tune Model Parameters")
  tmcol1,tmcol2,tmcol3 = st.columns(3)
  with  tmcol1:
    max_seq_len = st.number_input('Max Tokens', value=2000)
  with  tmcol2:
    temperature = st.slider('Temperature', value=0.5, min_value=0.0, max_value=1.0)
  with  tmcol3:
     top_p = st.slider('top_p', value=0.95, min_value=0.0, max_value=1.0)
  translate_button=st.button("Translate", on_click=translate,args=())

with st.expander("Prompt Configuration",False):
  systemPrompt=st.text_area("System Prompt", DEFAULT_SYSTEM_PROMPT)
  userPrompt =st.text_area("User Prompt", DEFAULT_USER_PROMPT)


with st.expander("Translation Customization"):
  egcol1, egcol2 = st.columns(2)
  with egcol1:
      list = ['No Index Selected']
      list.extend(listIndices())
      if len(list) > 0:
          st.session_state.index_list = list
          # Use a key for the selectbox and set a default value
          st.session_state.index_name = st.selectbox(
              "Select a translation memory index",
              options=tuple(st.session_state.index_list),
              key="index_selector",
              on_change=on_index_change
          )
      st.divider()
      tmx_file = st.file_uploader("Upload a new TMX file", type=["tmx"])
      if tmx_file is not None:
          file_name = tmx_file.name
          st.write('You selected `%s`' % file_name)
          examples = []
          if st.button("Process TMX File"):
              tmx_data = tmx_file.getvalue()
              loaded_index_name = processTMXFile(tmx_data, file_name)
              if st.session_state.index_list is not None:
                  st.session_state.index_list.append(loaded_index_name)
              # Update the selectbox to the newly added item
              # st.session_state.index_selector = loaded_index_name

  with egcol2:
    custom_examples=st.text_area("Provide translation memory manually: "+ getLanguageChoices()[sl] + " : " +getLanguageChoices()[tl] +"\n")
    custom_terms=st.text_area("Provide custom terminology manually: "+ getLanguageChoices()[sl] + " : " +getLanguageChoices()[tl] +"\n")
    st.session_state['custom_examples']=custom_examples
    st.session_state['custom_terms']=custom_terms
    st.write("One language sample pair per line seperated by colon (:). Example: Hello, how are you? : Hola, ¿cómo estás?")

df=None
if 'tmx_loaded' in st.session_state  and st.session_state.tmx_loaded == True:
  df=getExamplesDF(text2translate, sl, tl)

with st.expander("Translation pairs loaded from knowledge base",expanded=True):
    #st.table(df)
    if df is not None :
      st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
      st.write(" ")

with st.expander("Generated Prompt"):
     if 'prompt' in st.session_state:
      st.text_area("Prompt",st.session_state['prompt'])

if 'translated_text' in st.session_state:
  with st.expander("Translation", expanded=True):
    egcol1, egcol2 = st.columns(2)
    with egcol1:
      if 'translated_text' in st.session_state:
        st.write(st.session_state['translated_text'])
        bcol1, bcol2 = st.columns(2)
        with bcol1:
          st.button("✅ Evaluate", on_click=evaluate, args=())
        with bcol2:
          st.button("📋 Copy", on_click=on_copy_click, args=())
    with egcol2:
      st.text_area("Paste your reference " +getLanguageChoices()[tl] +" translation  below", key="reference_text")

refresh_metrics()