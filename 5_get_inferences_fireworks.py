import pandas as pd
import numpy as np
import time
import re

from datetime import datetime
from os import path
from fireworks.client import Fireworks

import warnings
warnings.filterwarnings('ignore')


# Establezca el valor seed inicial por todas partes para que sea reproducible.
SEED = 42

MAX_TOKENS_INPUT = 1000 # Cambiar al input deseado
MAX_TOKES_OUTPUT = 3

DISORDER_TAG = 'SELF-HARM' # ANOREXIA DEPRESSION SELF-HARM PTSD
GLOBAL_DIR = 'D:\Users\Downloads'
MODE_PROMPTING = 'zero'

MODEL_NAME = 'LlaMA-3.1-405b-instruct'
checkpoint = 'accounts/fireworks/models/llama-v3p1-405b-instruct'



### Set the client from FireworksAI
API_KEY = 'secret_api_key'
client = Fireworks(api_key=API_KEY)



def read_dataset(controlFile, diagnosedFile):

	print(f'* Reading files...')
	start_time = time.time()


	testControl = pd.read_csv(controlFile)
	testDiagnosed = pd.read_csv(diagnosedFile)

	test = pd.concat([testDiagnosed, testControl])
	
	print(test.shape)

	# Guardar el user_id
	users_test = test['user_id'].values.tolist()

	# Timeline
	mainCol = [c for c in test.columns if c.startswith('all_')][0] # all_clean_text
	X_test = test[mainCol].values

	# Class tag
	y_test = test['class'].values


	print(f'{(time.time() - start_time) / 60.0}')

	return X_test, y_test, users_test



def make_prompt_zero_shot(timeline, choice=1):

	print(f'* Making ZERO prompts...')

	timeline = timeline.replace('\\', '-').replace('\t', ' ').replace('nbsp', '')
	timeline = re.sub(' +', ' ', timeline).strip()

	## quitar palabras grandes
	allTimeline = [t for t in timeline.split() if len(t) < 12]
	allTimeline = ' '.join(allTimeline)
	allTimeline = re.sub(' +', ' ', allTimeline).strip()

	## START
	## --------------------------------------------------------
	truncated_tweet = allTimeline[:MAX_TOKENS_INPUT].strip()
	## --------------------------------------------------------


	if choice == 1:
		prompt = f"Answer with 'yes' or 'no' if the person who wrote this posts on Reddit suffer from {DISORDER_TAG.lower()}: \"\"{truncated_tweet}\"\""

	elif choice == 2:
		prompt = f"Consider this series of posts on Reddit to answer the question: Is the poster likely to suffer from {DISORDER_TAG.lower()}? Only return 'yes' or 'no'. \"\"{truncated_tweet}\"\""

	elif choice == 3:
		prompt = f"Read this series of posts on Reddit and determine if the person suffer from {DISORDER_TAG.lower()}. Answer with 'yes' or 'no'. \"\"{truncated_tweet}\"\"?"

	else:
		raise Exception('Invalid choice')
	
	return prompt



def make_prompt_few_shot(timeline, choice=1):

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



	timeline = timeline.replace('\\', '-').replace('\t', ' ').replace('nbsp', '')
	timeline = re.sub(' +', ' ', timeline).strip()

	## quitar palabras grandes
	allTimeline = [t for t in timeline.split() if len(t) < 12]
	allTimeline = ' '.join(allTimeline)
	allTimeline = re.sub(' +', ' ', allTimeline).strip()

	## START
	max_input_api = MAX_TOKENS_INPUT - len(fewShot)
	## --------------------------------------------------------
	truncated_tweet = allTimeline[:max_input_api].strip()
	## --------------------------------------------------------



	if choice == 1:
		prompt = fewShot + f"\\nAnswer with 'yes' or 'no' if the person who wrote this posts on Reddit suffer from {DISORDER_TAG.lower()}: \"\"{truncated_tweet}\"\""

	elif choice == 2:
		prompt = fewShot + f"\\nConsider this series of posts on Reddit to answer the question: Is the poster likely to suffer from {DISORDER_TAG.lower()}? Only return 'yes' or 'no'. \"\"{truncated_tweet}\"\""

	elif choice == 3:
		prompt = fewShot + f"\\nRead this series of posts on Reddit and determine if the person suffer from {DISORDER_TAG.lower()}. Answer with 'yes' or 'no'. \"\"{truncated_tweet}\"\"?"

	else:
		raise Exception('Invalid choice')
	
	return prompt



def make_inference(prompt):

	print(f'* {MODE_PROMPTING.capitalize()} shot detection...')
	start_time = time.time()

	# Generate predictions for the batch
	answer = client.chat.completions.create(
									model=checkpoint,
									messages=[{
											'role': 'user',
											'content': prompt,
											}],
									max_tokens=MAX_TOKES_OUTPUT,
									context_length_exceeded_behavior='truncate'
								)

	try:
		# Decode the responses
		response = answer.choices[0].message.content

	except Exception as e:
		print(f'Error: {e}')
		response = 'Error'


	print(f'{(time.time() - start_time) / 60.0}')

	return response



def evaluate(X_test, choice):

	print(f'* Running batches... {datetime.now().strftime("%H:%M:%S")}')
	start_time = time.time()

	responses = []

	totalTweets = len(X_test)

	# Process tweets in batches
	for i, timeline in enumerate(X_test, start=1):

			if MODE_PROMPTING == 'zero':
				prompt = make_prompt_zero_shot(timeline, choice)
			
			elif MODE_PROMPTING == 'few':
				prompt = make_prompt_few_shot(timeline, choice)
			
			else:
				pass

			output = make_inference(prompt)
			responses.append(output)


			time.sleep(10) # Sleep for 10 seconds

			if i % 10 == 0:
				print(f'{datetime.now().strftime("%H:%M:%S")} -> {DISORDER_TAG} - choice{choice} - {i}/{totalTweets}...')
				# GUARDAR por si acaso la vida...
				pd.DataFrame(data=responses, columns=['response']).to_csv(path.join(GLOBAL_DIR, f'{MODE_PROMPTING}_{DISORDER_TAG}_choice{choice}_part{i}.csv'), index=False)

	print(f'\n{datetime.now().strftime("%H:%M:%S")}')
	print(f'{(time.time() - start_time) / 60.0}')

	return np.array(responses)



def save_answers(y_test, responses, users_test, rawPredictionsFilename):

	print(f'* Saving results...')

	df = pd.DataFrame(data={
			 			'user_id': users_test,
			 			'y_test': y_test,
						'response': responses,
					})
	df.to_csv(rawPredictionsFilename, index=False)

	print(f'Shape resultante: {df.shape}')
	print(f'Value counts:')
	print(df['y_test'].value_counts())



def run_model(dataDirectory, saveDirectory, language='eng'):

	print('\n---------------------')
	print(f'\n\n\t{DISORDER_TAG} - {MODE_PROMPTING}\n')
	print('\n---------------------')


	testControlFilename = path.join(dataDirectory, f'concat_rows_cleanText_control.csv')
	testDiagnosedFilename = path.join(dataDirectory, f'concat_rows_cleanText_diagnosed.csv')


	X_test, y_test, users_test = read_dataset(testControlFilename, testDiagnosedFilename)


	print(f'\n*** {DISORDER_TAG} - {MODEL_NAME} - {MODE_PROMPTING} ***')
	print(saveDirectory)

	for choice in [1, 2, 3]:

		print(f'\n> Running prompt {choice}') 

		abbvrShot = 'zs' if MODE_PROMPTING == 'zero' else 'fs'
		fileRawPredictions = path.join(saveDirectory, f'{MODEL_NAME}_{abbvrShot}_prompt{choice}_raw-predictions_{language}.csv')

		responses = evaluate(X_test, choice)

		save_answers(y_test, responses, users_test, fileRawPredictions)

		time.sleep(60*3) # Sleep for 3 minutes

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

	# =========================================================
	DISORDER_TAG = 'ANOREXIA'
	sourceDirectory = r'eRisk_anorexia_2018\Dataset\Test'
	destinyDirectory = r'eRisk_anorexia_2018\Results'
	GLOBAL_DIR = destinyDirectory
	main(sourceDirectory, destinyDirectory)
