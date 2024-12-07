import pandas as pd
import os
import re
import math


MAX_TOKENS_INPUT = 1000 # cambiar al input deseado
MAX_TOKES_OUTPUT = 3



def makeZeroShotRequests(sourceDirectory, sourceFilename, destinyDirectory, keyname, choice=1, model='gpt-3.5-turbo'):

	print(f'* Reading files...')

	testControl = pd.read_csv(os.path.join(sourceDirectory, f'{sourceFilename}_control.csv'))
	testDiagnosed = pd.read_csv(os.path.join(sourceDirectory, f'{sourceFilename}_diagnosed.csv'))

	test = pd.concat([testDiagnosed, testControl])


	mainCol = [c for c in test.columns if c.startswith('all_')][0] # all_clean_text


	requestsLst = []
	usersLst = test['user_id'].values
	timelinesLst = test[mainCol].values
	labelsLst = test['class'].values

	for user, label, timeline in zip(usersLst, labelsLst, timelinesLst):

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
			prompt = f"Answer with 'yes' or 'no' if the person who wrote this posts on Reddit suffer from {DISORDER_TAG.lower()}: \\\"\\\"{truncated_tweet}\\\"\\\""

		elif choice == 2:
			prompt = f"Consider this series of posts from Reddit to answer the question: Is the poster likely to suffer from {DISORDER_TAG.lower()}? Only return 'yes' or 'no'. \\\"\\\"{truncated_tweet}\\\"\\\""

		elif choice == 3:
			prompt = f"Read this series of posts from Reddit and determine if the person suffer from {DISORDER_TAG.lower()}. Answer with 'yes' or 'no'. \\\"\\\"{truncated_tweet}\\\"\\\""

		else:
			raise Exception('Invalid choice')
		

		tag = 'neg' if label == 'CONTROL' else 'pos'
		custom_id = f"{keyname}-{user.replace('subject', 's')}-{tag}"

		message = f"{{\"custom_id\": \"{custom_id}\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": \"{prompt}\"}}], \"max_tokens\": {MAX_TOKES_OUTPUT}}}}}"

		requestsLst.append(message)


	sizeParts = math.ceil(len(requestsLst) / 3)
	contar = 0
	for i in range(0, len(requestsLst), sizeParts):

		newLst = requestsLst[i:i+sizeParts]
		contar += 1

		modelKey = '-'.join(model.split('-')[:2])
		finalFilename = os.path.join(destinyDirectory, f'Full_{modelKey}_{DISORDER_TAG.lower()}_zs{choice}_test_eng_part{contar}.jsonl')
		with open(finalFilename, 'w') as file:  
			for line in newLst:
				file.write(line)  # Adding the line to the text.txt  
				file.write('\n')  # Adding a new line character  

	print('FIN!!! :)')



def makeFewShotRequests(sourceDirectory, sourceFilename, destinyDirectory, keyname, choice=1, model='gpt-3.5-turbo'):

	print(f'* Reading files...')

	testControl = pd.read_csv(os.path.join(sourceDirectory, f'{sourceFilename}_control.csv'))
	testDiagnosed = pd.read_csv(os.path.join(sourceDirectory, f'{sourceFilename}_diagnosed.csv'))

	test = pd.concat([testDiagnosed, testControl])


	mainCol = [c for c in test.columns if c.startswith('all_')][0] # all_clean_text

	
	### FEW SHOTS EXAMPLES
	if DISORDER_TAG == 'ANOREXIA':
		
		timeline1 = 'Timeline text'
		class1 = 'Tag'
		
		timeline2 = 'Timeline text'
		class2 = 'Tag'
		
		fewShot = f"Consider the following posts and their labels before answer the question. Posts: \\\"\\\"{timeline1}\\\"\\\"\\nLabel: {class1}\\nPosts: \\\"\\\"{timeline2}\\\"\\\"\\nLabel: {class2}"
	
	elif DISORDER_TAG == 'SELF-HARM':
		
		timeline1 = 'Timeline text'
		class1 = 'Tag'
		
		timeline2 = 'Timeline text'
		class2 = 'Tag'
		
		fewShot = f"Consider the following posts and their labels before answer the question. Posts: \\\"\\\"{timeline1}\\\"\\\"\\nLabel: {class1}\\nPosts: \\\"\\\"{timeline2}\\\"\\\"\\nLabel: {class2}"
	
	elif DISORDER_TAG == 'DEPRESSION':
		
		timeline1 = 'Timeline text'
		class1 = 'Tag'
		
		timeline2 = 'Timeline text'
		class2 = 'Tag'
		
		fewShot = f"Consider the following posts and their labels before answer the question. Posts: \\\"\\\"{timeline1}\\\"\\\"\\nLabel: {class1}\\nPosts: \\\"\\\"{timeline2}\\\"\\\"\\nLabel: {class2}"
	
	else:
		pass



	requestsLst = []
	usersLst = test['user_id'].values
	timelinesLst = test[mainCol].values
	labelsLst = test['class'].values

	for user, label, timeline in zip(usersLst, labelsLst, timelinesLst):

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
			prompt = fewShot + f"\\nAnswer with 'yes' or 'no' if the person who wrote this posts on Reddit suffer from {DISORDER_TAG.lower()}: \\\"\\\"{truncated_tweet}\\\"\\\""

		elif choice == 2:
			prompt = fewShot + f"\\nConsider this series of posts from Reddit to answer the question: Is the poster likely to suffer from {DISORDER_TAG.lower()}? Only return 'yes' or 'no'. \\\"\\\"{truncated_tweet}\\\"\\\""

		elif choice == 3:
			prompt = fewShot + f"\\nRead this series of posts from Reddit and determine if the person suffer from {DISORDER_TAG.lower()}. Answer with 'yes' or 'no'. \\\"\\\"{truncated_tweet}\\\"\\\""

		else:
			raise Exception('Invalid choice')
		

		tag = 'neg' if label == 'CONTROL' else 'pos'
		custom_id = f"{keyname}-{user.replace('subject', 's')}-{tag}"

		message = f"{{\"custom_id\": \"{custom_id}\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": \"{prompt}\"}}], \"max_tokens\": {MAX_TOKES_OUTPUT}}}}}"

		requestsLst.append(message)


	sizeParts = math.ceil(len(requestsLst) / 4)
	contar = 0
	for i in range(0, len(requestsLst), sizeParts):

		newLst = requestsLst[i:i+sizeParts]
		contar += 1

		modelKey = '-'.join(model.split('-')[:2])
		finalFilename = os.path.join(destinyDirectory, f'Full_{modelKey}_{DISORDER_TAG.lower()}_fs{choice}_test_eng_part{contar}.jsonl')
		with open(finalFilename, 'w') as file:  
			for line in newLst:
				file.write(line)  # Adding the line to the text.txt  
				file.write('\n')  # Adding a new line character  

	print('FIN!!! :)')



def main(sourceDirectory, sourceFilename, destinyDirectory, keyname, mode='zero'):

	print('\n---------------------')
	print(f'\n\n\t{DISORDER_TAG}\n')
	print('\n---------------------')

	for model in ['gpt-4o-mini']:

		print(f'\n*** {model} ***')
		
		for choice in [1, 2, 3]:
			
			if mode == 'zero':
				makeZeroShotRequests(sourceDirectory, sourceFilename, destinyDirectory, keyname, choice=choice, model=model)
			
			elif mode == 'few':
				makeFewShotRequests(sourceDirectory, sourceFilename, destinyDirectory, keyname, choice=choice, model=model)
			
			else:
				pass

	print('..........')




if __name__ == '__main__':

	sourceFilename = 'concat_rows_cleanText'
	# =========================================================
	DISORDER_TAG = 'ANOREXIA'
	keyname = 'eat'
	sourceDirectory = r'eRisk_anorexia_2018\Dataset\Test'
	destinyDirectory = r'eRisk_anorexia_2018\Prompts-few-shot-gpt'
	main(sourceDirectory, sourceFilename, destinyDirectory, keyname, mode='few')
