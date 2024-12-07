import pandas as pd
import os



def makePromptsEmbeddings(sourceDirectory, destinyDirectory, keyname, model='text-embedding-3-large'):

	print(f'\nStarting: {keyname.upper()}...')

	todo = pd.DataFrame()

	for partition in ['Train', 'Test']:

		print(f'> {partition}')

		control = pd.read_csv(os.path.join(sourceDirectory, partition, 'text_control.csv'))
		diagnosed = pd.read_csv(os.path.join(sourceDirectory, partition, 'text_diagnosed.csv'))

		final = pd.concat([diagnosed, control])
		final['id2'] = f'{keyname}_{partition.lower()}_'
		final['identifier'] = final['id2'] + final['text_id']

		todo = pd.concat([todo, final])
	
	print(f'Shape: {todo.shape[0]:,}')


	print(f'\nMaking prompts...')
	print(todo.isnull().sum())

	requestsLst = []
	for user, tweet in zip(todo['identifier'].values, todo['text'].values):

		message = f"{{\"custom_id\": \"{user}\", \"method\": \"POST\", \"url\": \"/v1/embeddings\", \"body\": {{\"model\": \"{model}\", \"input\": \"{tweet}\", \"encoding_format\": \"float\"}}}}"
		requestsLst.append(message)


	divisiones = len(requestsLst) / 50000
	print(f'\nTotal de archivos: {divisiones:.1f}')


	contar = 0
	for i in range(0, len(requestsLst), 50000):
		
		particionLst = requestsLst[i:i+50000]
		contar += 1

		print(f'* Particion {contar} - de {i:,} a {i+50000:,}')

		finalFilename = os.path.join(destinyDirectory, f'EMB_{keyname}_n{contar}.jsonl')
		with open(finalFilename, 'w') as file:
			for line in particionLst:
				file.write(line)  # Adding the line to the text.txt  
				file.write('\n')  # Adding a new line character  

	print('\nFIN!!! :)')





sourceDirectory = r'dataset\Full'
destinyDirectory = r'dataset\Prompts-embeddings-gpt'

makePromptsEmbeddings(sourceDirectory, destinyDirectory, keyname='eat')
