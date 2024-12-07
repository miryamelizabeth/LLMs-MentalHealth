import pandas as pd
import numpy as np
import time
import random
import math
import re
import torch
import tensorflow as tf

from datetime import datetime
from os import path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, set_seed
from transformers import BitsAndBytesConfig
from huggingface_hub import login

from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings('ignore')


# Establezca el valor seed inicial por todas partes para que sea reproducible.
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
tf.random.set_seed(SEED)
set_seed(SEED) # Transformers


DISORDER_TAG = 'SELF-HARM' # ANOREXIA DEPRESSION SELF-HARM PTSD

## =======================================================
login(token='XXXXX')


## ----------------------------------------------
## seq2seq
## ----------------------------------------------
MODEL_NAME = 'FlanT5-xxl'
checkpoint = 'google/flan-t5-xxl'

# MODEL_NAME = 'Mental-FlanT5'
# checkpoint = 'NEU-HAI/mental-flan-t5-xxl'

## ----------------------------------------------
## Causal
## ----------------------------------------------
# MODEL_NAME = 'LLaMA-3.1-70B-instruct'
# checkpoint = 'meta-llama/Meta-Llama-3.1-70B-Instruct'

# MODEL_NAME = 'LLaMA-2-70B-chat'
# checkpoint = 'meta-llama/Llama-2-70b-chat-hf'

# MODEL_NAME = 'Mental-LLaMA-chat-13B'
# checkpoint = 'klyang/MentaLLaMA-chat-13B'



### ----------------------------------------------
### Load tokenizer and model
### ----------------------------------------------
# bnb_config = BitsAndBytesConfig(
# 	load_in_4bit=True,
# 	bnb_4bit_quant_type='nf4',
# 	bnb_4bit_use_double_quant=True
# )

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to('cuda:0') # quantization_config=bnb_config
tokenizer.pad_token = tokenizer.eos_token


# tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to('cuda:0')
# tokenizer.pad_token = tokenizer.eos_token




## Maximum response
MAX_NEW_TOKENS = 5

## Get the maximum token length allowed by the model
MAX_LENGTH = tokenizer.model_max_length
MAX_LENGTH = MAX_LENGTH - MAX_NEW_TOKENS

MODE_PROMPTING = 'zero'

## Batch size
BATCH_SIZE = 4

print(f'------------>>>> Total length: {MAX_LENGTH}')


def read_dataset(controlFile, diagnosedFile):

	print(f'* Reading files...')
	start_time = time.time()

	testControl = pd.read_csv(controlFile)
	testDiagnosed = pd.read_csv(diagnosedFile)

	test = pd.concat([testDiagnosed, testControl])
	
	print(test.shape)
	print(test['class'].value_counts())

	# Guardar el user_id
	users_test = test['user_id'].values.tolist()

	# Timeline
	mainCol = [c for c in test.columns if c.startswith('all_')][0] # all_clean_text
	X_test = test[mainCol].values

	# Class tag
	y_test = test['class'].values


	print(f'{(time.time() - start_time) / 60.0}')

	return X_test, y_test, users_test



def make_prompts_zero_shot(tweets, choice=1):

	print(f'* Making ZERO prompts...')

	cleanTweets = []

	for timeline in tweets:

		timeline = timeline.replace('\\', '-').replace('\t', ' ').replace('nbsp', '')
		timeline = re.sub(' +', ' ', timeline).strip()

		## quitar palabras grandes
		allTimeline = [t for t in timeline.split() if len(t) < 12]
		allTimeline = ' '.join(allTimeline)
		allTimeline = re.sub(' +', ' ', allTimeline).strip()

		# ## START
		# ## --------------------------------------------------------
		# truncated_tweet = allTimeline[:MAX_TOKENS_INPUT].strip()
		# ## --------------------------------------------------------
		cleanTweets.append(allTimeline)


	if choice == 1:
		promptsLst = [f"Answer with 'yes' or 'no' if the person who wrote this posts on Reddit suffer from {DISORDER_TAG.lower()}: \"\"{tweet}\"\"" for tweet in cleanTweets]

	elif choice == 2:
		promptsLst = [f"Consider this series of posts on Reddit to answer the question: Is the poster likely to suffer from {DISORDER_TAG.lower()}? Only return 'yes' or 'no'. \"\"{tweet}\"\"" for tweet in cleanTweets]

	elif choice == 3:
		promptsLst = [f"Read this series of posts on Reddit and determine if the person suffer from {DISORDER_TAG.lower()}. Answer with 'yes' or 'no'. \"\"{tweet}\"\"?" for tweet in cleanTweets]

	else:
		raise Exception('Invalid choice')
	

	# chats = [{'role': 'user', 'content': prompt} for prompt in promptsLst]

	return promptsLst



def make_prompts_few_shot(tweets, choice=1):

	print(f'* Making FEW prompts...')

	### FEW SHOTS EXAMPLES
	if DISORDER_TAG == 'ANOREXIA':
		
		timeline1 = 'Timeline text'
		class1 = 'Tag'
		
		timeline2 = 'Timeline text'
		class2 = 'Tag'
		
		fewShot = f"Consider the following post timelines and their labels before answer the question. Posts: \\\"\\\"{timeline1}\\\"\\\"\\nLabel: {class1}\\nPosts: \\\"\\\"{timeline2}\\\"\\\"\\nLabel: {class2}"
	
	elif DISORDER_TAG == 'SELF-HARM':
		
		timeline1 = 'Timeline text'
		class1 = 'Tag'
		
		timeline2 = 'Timeline text'
		class2 = 'Tag'
		
		fewShot = f"Consider the following post timelines and their labels before answer the question. Posts: \\\"\\\"{timeline1}\\\"\\\"\\nLabel: {class1}\\nPosts: \\\"\\\"{timeline2}\\\"\\\"\\nLabel: {class2}"
	
	elif DISORDER_TAG == 'DEPRESSION':
		
		timeline1 = 'Timeline text'
		class1 = 'Tag'
		
		timeline2 = 'Timeline text'
		class2 = 'Tag'
		
		fewShot = f"Consider the following post timelines and their labels before answer the question. Posts: \\\"\\\"{timeline1}\\\"\\\"\\nLabel: {class1}\\nPosts: \\\"\\\"{timeline2}\\\"\\\"\\nLabel: {class2}"
	
	else:
		pass


	cleanTweets = []

	for timeline in tweets:

		timeline = timeline.replace('\\', '-').replace('\t', ' ').replace('nbsp', '')
		timeline = re.sub(' +', ' ', timeline).strip()

		## quitar palabras grandes
		allTimeline = [t for t in timeline.split() if len(t) < 12]
		allTimeline = ' '.join(allTimeline)
		allTimeline = re.sub(' +', ' ', allTimeline).strip()

		# ## START
		# ## --------------------------------------------------------
		# truncated_tweet = allTimeline[:MAX_TOKENS_INPUT].strip()
		# ## --------------------------------------------------------
		cleanTweets.append(allTimeline)



	if choice == 1:
		promptsLst = [fewShot + f"\\nAnswer with 'yes' or 'no' if the person who wrote this posts on Reddit suffer from {DISORDER_TAG.lower()}: \"\"{tweet}\"\"" for tweet in cleanTweets]

	elif choice == 2:
		promptsLst = [fewShot + f"\\nConsider this series of posts on Reddit to answer the question: Is the poster likely to suffer from {DISORDER_TAG.lower()}? Only return 'yes' or 'no'. \"\"{tweet}\"\"" for tweet in cleanTweets]

	elif choice == 3:
		promptsLst = [fewShot + f"\\nRead this series of posts on Reddit and determine if the person suffer from {DISORDER_TAG.lower()}. Answer with 'yes' or 'no'. \"\"{tweet}\"\"?" for tweet in cleanTweets]

	else:
		raise Exception('Invalid choice')


	# chats = [{'role': 'user', 'content': prompt} for prompt in promptsLst]

	return promptsLst



def make_inference_batch(prompts):

	print(f'* {MODE_PROMPTING.capitalize()} shot detection...')
	start_time = time.time()

	# chats = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

	# Batch tokenize the prompts
	inputs = tokenizer(prompts, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding=True, return_token_type_ids=False).to('cuda:0')
	
	# Generate model output
	outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)


	# Decode the responses
	responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

	print(f'{(time.time() - start_time) / 60.0}')

	return responses



def evaluate(tweets, choice, batch_size=32):

	print(f'* Running batches... {datetime.now().strftime("%H:%M:%S")}')
	start_time = time.time()

	responses = []

	totalRuns = math.ceil(len(tweets) / batch_size)
	contar = 0

	# Process tweets in batches
	for i in range(0, len(tweets), batch_size):

		contar += 1
		print(f'>> {datetime.now().strftime("%H:%M:%S")} - {DISORDER_TAG} - prompt {choice} -> running {contar}/{totalRuns}')

		batch_tweets = tweets[i:i + batch_size]

		if MODE_PROMPTING == 'zero':
			prompts = make_prompts_zero_shot(batch_tweets, choice)
		
		elif MODE_PROMPTING == 'few':
			prompts = make_prompts_few_shot(batch_tweets, choice)
		
		else:
			pass


		outputs = make_inference_batch(prompts)
		responses.extend(outputs)


		# Clear session
		K.clear_session()
		torch.cuda.empty_cache()

		# Volver a agregar las semillas
		random.seed(SEED)
		np.random.seed(SEED)
		tf.random.set_seed(SEED)
		set_seed(SEED)

	print(f'\n{datetime.now().strftime("%H:%M:%S")}')
	print(f'{(time.time() - start_time) / 60.0}')

	return responses



def save_answers(y_test, responses, users_test, rawPredictionsFilename):

	print(f'* Saving results...')

	df = pd.DataFrame(data={
						'user_id': users_test,
						'y_test': y_test,
						'responses': responses
					})
	df.to_csv(rawPredictionsFilename, index=False)

	print(f'Shape resultante: {df.shape}')
	print(f'Value counts:')
	print(df['y_test'].value_counts())



def run_model(dataDirectory, saveDirectory, language='eng'):

	print('\n---------------------')
	print(f'\n\n\t{DISORDER_TAG} - {MODE_PROMPTING}\n')
	print('\n---------------------')


	testControlFilename = path.join(dataDirectory, f'text_control.csv')
	testDiagnosedFilename = path.join(dataDirectory, f'text_diagnosed.csv')


	X_test, y_test, users_test = read_dataset(testControlFilename, testDiagnosedFilename)


	print(f'\n*** {DISORDER_TAG} - {MODEL_NAME} - {MODE_PROMPTING} ***')
	print(saveDirectory)

	for choice in [1, 2, 3]:

		print(f'\n> Running prompt {choice}')

		abbvrShot = 'zs' if MODE_PROMPTING == 'zero' else 'fs'
		fileRawPredictions = path.join(saveDirectory, f'{MODEL_NAME}_{abbvrShot}_prompt{choice}_raw-predictions_{language}.csv') # GPT-4o-mini_zs_prompt1_raw-predictions_eng

		responses = evaluate(X_test, choice, batch_size=BATCH_SIZE)

		save_answers(y_test, responses, users_test, fileRawPredictions)


		# Clear session
		K.clear_session()
		torch.cuda.empty_cache()

		# Volver a agregar las semillas
		random.seed(SEED)
		np.random.seed(SEED)
		tf.random.set_seed(SEED)
		set_seed(SEED)

	print('\nEND :)')


def main(dataDirectory, directoryResults, language='eng'):

	if MODE_PROMPTING == 'zero':
		run_model(dataDirectory, path.join(directoryResults, f'LLMS_ZERO-SHOT_{language.upper()}'))
	
	elif MODE_PROMPTING == 'few':
		run_model(dataDirectory, path.join(directoryResults, f'LLMS_FEW-SHOT_{language.upper()}'))
	
	else:
		pass



# --------------------------------
if __name__ == '__main__':

	language = 'eng'

	# ============================================
	DISORDER_TAG = 'ANOREXIA'

	dataDirectory = '/Experimentos/eRisk_anorexia_2018'
	resultsDirectory = '/Experimentos/Results'

	main(dataDirectory, resultsDirectory, language)
