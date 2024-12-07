## -------------------------------------
# ANTES DE EJECUTAR...
# Especificar --> idioma ingles/español
# Especificar --> n_partitions
# Cambiar las rutas de los directorios en el main
## -------------------------------------

import dask.dataframe as ddf
import pandas as pd
import numpy as np
import spacy as sp
import time, re, os

from datetime import datetime


# ---------------------------------------
# CAMBIAR por lo que queramos...
n_partitions = 8
# ---------------------------------------

nlp = sp.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
nlp.add_pipe('sentencizer')



# -------------------------------------
def removePunctuation(text):
	""" Removes all puntuctuation symbols including ¿! (used in spanish) """

	punctuationStr = r'[:\.,\?!/\\_━🇧▪"\[!"\$%&\(\)\*\+;<=>\^`{\|}~¿¡¬‘’¶£¥€¢₩°«»“”—´¨™©º¸•¤‹›×–…·ا\]]'

	text = re.sub(punctuationStr, ' ', text)

	return re.sub(' +', ' ', text).strip()


def removeUrls(text):

	urlStr = r'https?:\/\/(www\.)?[\w.-]+(\/[\w\-?=&$@%.#]*)+'
	text = re.sub(urlStr, ' ', text)

	urlStr = r':\/\/(www\.)?[\w.-]+(\/[\w\-?=&$@%.#]*)+'
	text = re.sub(urlStr, ' ', text)

	urlStr = r'\w+\=\w+'
	text = re.sub(urlStr, ' ', text)

	return re.sub(' +', ' ', text).strip()


def helper_preprocess(text):

	user_regex = re.compile(r"@[a-zA-Z0-9_]{0,15}")
	text = user_regex.sub('@USER', text)


	shorten = 3
	repeated_regex = re.compile(r"(.)" + r"\1" * (shorten-1) + "+")
	text = repeated_regex.sub(r"\1"*shorten, text)


	letter_regex = re.compile(r'[Aa]{3,}')
	text = letter_regex.sub('a', text)

	letter_regex = re.compile(r'[Bb]{3,}')
	text = letter_regex.sub('b', text)

	letter_regex = re.compile(r'[Cc]{3,}')
	text = letter_regex.sub('c', text)

	letter_regex = re.compile(r'[Dd]{3,}')
	text = letter_regex.sub('d', text)

	letter_regex = re.compile(r'[Ee]{3,}')
	text = letter_regex.sub('e', text)

	letter_regex = re.compile(r'[Ff]{3,}')
	text = letter_regex.sub('f', text)

	letter_regex = re.compile(r'[Gg]{3,}')
	text = letter_regex.sub('g', text)

	letter_regex = re.compile(r'[Hh]{3,}')
	text = letter_regex.sub('h', text)

	letter_regex = re.compile(r'[Ii]{3,}')
	text = letter_regex.sub('i', text)

	letter_regex = re.compile(r'[Jj]{3,}')
	text = letter_regex.sub('j', text)

	letter_regex = re.compile(r'[Kk]{3,}')
	text = letter_regex.sub('k', text)

	letter_regex = re.compile(r'[Ll]{3,}')
	text = letter_regex.sub('l', text)

	letter_regex = re.compile(r'[Mm]{3,}')
	text = letter_regex.sub('m', text)

	letter_regex = re.compile(r'[Nn]{3,}')
	text = letter_regex.sub('n', text)

	letter_regex = re.compile(r'[Oo]{3,}')
	text = letter_regex.sub('o', text)

	letter_regex = re.compile(r'[Pp]{3,}')
	text = letter_regex.sub('p', text)

	letter_regex = re.compile(r'[Qq]{3,}')
	text = letter_regex.sub('q', text)

	letter_regex = re.compile(r'[Rr]{3,}')
	text = letter_regex.sub('r', text)

	letter_regex = re.compile(r'[Ss]{3,}')
	text = letter_regex.sub('s', text)

	letter_regex = re.compile(r'[Tt]{3,}')
	text = letter_regex.sub('t', text)

	letter_regex = re.compile(r'[Uu]{3,}')
	text = letter_regex.sub('u', text)

	letter_regex = re.compile(r'[Vv]{3,}')
	text = letter_regex.sub('v', text)

	letter_regex = re.compile(r'[Ww]{3,}')
	text = letter_regex.sub('w', text)

	letter_regex = re.compile(r'[Xx]{3,}')
	text = letter_regex.sub('x', text)

	letter_regex = re.compile(r'[Yy]{3,}')
	text = letter_regex.sub('y', text)

	letter_regex = re.compile(r'[Zz]{3,}')
	text = letter_regex.sub('z', text)


	laughter_regex = re.compile('[ha][ha]+ah[ha]+')
	replacement = 'haha'

	text = laughter_regex.sub(replacement, text)


	hour_regex = re.compile(r'([01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]')
	text = hour_regex.sub(' ', text)

	hour_regex = re.compile(r'([01]?[0-9]|2[0-3]):[0-5][0-9]')
	text = hour_regex.sub(' ', text)


	numbers_regex = re.compile(r'\d+\.?\d+')
	text = numbers_regex.sub(' ', text)


	text = text.replace(" ' ", " ").replace("' ", " ").replace(" '", " ").replace('hashtag', '').replace('http', '').replace('url', '').replace('nbsp', '').replace('\\', '-')


	return re.sub(' +', ' ', text).strip()


def removeOneWord(text):

	final = ''
	for t in text.split():
		if (t == 'I' or t == 'i') or len(t) > 1:
			final += t + ' '
	
	return re.sub(' +', ' ', final).strip()


def cleanFinal(text):
	final = text.replace(" '", " ").replace("' ", " ").replace(" '' ", " ").replace(" ' ", " ").replace("#", "")
	return re.sub(' +', ' ', final).strip()


# -------------------------------------
def processText(allText):
	""" Cleans all the garbage in the text and returns the cleaned text """

	# Remove extra newlines
	allText = [re.sub(r'[\r|\n|\r\n]+', ' ', t) for t in allText]

	# Remove extra whitespace
	allText = [re.sub(' +', ' ', t).strip() for t in allText]

	# Remove extra tabs
	allText = [t.replace('\t', ' ').replace('&nbsp', ' ').replace('nbsp', ' ') for t in allText]


	# Replace symbols (eg. I’m --> I'm   that´s --> that's)
	allText = [re.sub('’', '\'', t) for t in allText]
	allText = [re.sub('”', '\'', t) for t in allText]
	allText = [re.sub('´', '\'', t) for t in allText]
	allText = [re.sub('"', '\'', t) for t in allText]

	allText = [re.sub('‑', '-', t) for t in allText]
	allText = [re.sub('—', '-', t) for t in allText]

	allText = [removeUrls(t) for t in allText]

	allText = [helper_preprocess(t) for t in allText]

	allText = [removePunctuation(t) for t in allText]

	allText = [removeOneWord(t) for t in allText]

	allText = [cleanFinal(t) for t in allText]

	allText = [t.lower() for t in allText]

	return allText


# -------------------------------------
# Lemmatization
# -------------------------------------
def lemmatizeText(text):

	doc = nlp(text)

	lista_lemmatized = ' '.join([token.lemma_ for token in doc]).strip()

	filtered_tokens = ' '.join([token.text for token in doc if not token.is_stop]).strip()

	return lista_lemmatized, filtered_tokens


def cleanProcessDataframe(df):
	""" Process text and apply all the techniques programmed before, replace the x in id, split the date... """

	df[['day','time']] = df['text_created_at'].str.split(' ', expand=True,)

	df['clean_text'] = processText(df['text'].values)

	
	lst1 = []
	lst2 = []
	for t in df['clean_text'].values:
		t1, t2 = lemmatizeText(t)
		lst1.append(t1)
		lst2.append(t2)
	
	df['clean_text_lemma'] = lst1
	df['clean_text_nostop_lemma'] = lst2




	df['clean_title'] = processText(df['title'].values)

	lst1 = []
	lst2 = []
	for t in df['clean_title'].values:
		t1, t2 = lemmatizeText(t)
		lst1.append(t1)
		lst2.append(t2)
	
	df['clean_title_lemma'] = lst1
	df['clean_title_nostop_lemma'] = lst2

	return df


# -------------------------------------
def convertNum(value):
	return 0 if (value == np.nan) else value

def convertText(value):
	return '' if (value == np.nan) else value


def procesar_timelines(timelineDirectory, cleanUsersDirectory):
	
	print(f'*** {timelineDirectory} ***')

	usuarios = os.listdir(timelineDirectory)
	print(f'Tamaño lista: {len(usuarios)}')


	# iterar sobre c/usuario
	for user in usuarios:

		print(f'\n*** {user}***\n')
		print(f'{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
		
		print('Start...')
		count = 0

		start_time = time.time()

		df = pd.read_csv(os.path.join(timelineDirectory, user), low_memory=True,
						usecols=['SUBJECT_ID', 'TITLE', 'DATE', 'TEXT', 'CLASS'],
						converters={'TITLE': convertText, 'DATE': convertText, 'TEXT': convertText},
						dtype={'SUBJECT_ID': str}
						)

		df = df.rename(columns={'SUBJECT_ID': 'user_id',
								'DATE': 'text_created_at',
								'TITLE': 'title',
								'TEXT': 'text',
								'CLASS': 'class'})


		df['user_id'] = df['user_id'].apply(lambda x: re.sub(' +', ' ', x.replace('\n', '')).strip())
		df['text_created_at'] = df['text_created_at'].apply(lambda x: re.sub(' +', ' ', x.replace('\n', '')).strip())
		df['title'] = df['title'].apply(lambda x: re.sub(' +', ' ', x.replace('\n', '')).strip())
		df['text'] = df['text'].apply(lambda x: re.sub(' +', ' ', x.replace('\n', '')).strip())
		df['class'] = df['class'].apply(lambda x: re.sub(' +', ' ', x.replace('\n', '')).strip())


		# The empty DataFrame that you provide doesn't have the correct column names.
		# You don't provide any columns in your metadata, but your output does have them.
		# This is the source of your error.
		# The meta value should match the column names and dtypes of your expected output.
		# ... es por eso que se crean columnas vacías

	
		df[['day','time']] = ''
		
		df['clean_text'] = ''
		df['clean_text_lemma'] = ''
		df['clean_text_nostop_lemma'] = ''

		df['clean_title'] = ''
		df['clean_title_lemma'] = ''
		df['clean_title_nostop_lemma'] = ''



		dask_dataframe = ddf.from_pandas(df, npartitions=n_partitions)

		print(df.shape)
		result = dask_dataframe.map_partitions(cleanProcessDataframe, meta=df)
		df = result.compute()

		# Organizar columnas
		cleanData = df[['day','time',
						'text',
						'clean_text', 'clean_text_lemma', 'clean_text_nostop_lemma',
						'user_id',
						'title',
						'clean_title', 'clean_title_lemma', 'clean_title_nostop_lemma',
						'class']]



		cleanData = cleanData[(cleanData['clean_title'] != '') | (cleanData['clean_text'] != '')] # más rapido que remove

		print(cleanData.shape)

		# guardar en la nueva ubicacion
		cleanData.to_csv(os.path.join(cleanUsersDirectory, f'user_{user}'), index=False)

		end_time = time.time()
		print(f'{(end_time - start_time) / 60.0}')

		count += 1
		if count % 20 == 0:
			print(f'Processing {count}, {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')




# -----------------------------------------------------------------
if __name__ == '__main__':

	timelineDirectory = r'eRisk_anorexia_2018\Timelines'
	cleanUsersDirectory = r'eRisk_anorexia_2018\Clean_dataset'

	procesar_timelines(os.path.join(timelineDirectory, 'Train', 'Positive'), os.path.join(cleanUsersDirectory, 'Train', 'Positive'))
	procesar_timelines(os.path.join(timelineDirectory, 'Train', 'Negative'), os.path.join(cleanUsersDirectory, 'Train', 'Negative'))

	procesar_timelines(os.path.join(timelineDirectory, 'Test', 'Positive'), os.path.join(cleanUsersDirectory, 'Test', 'Positive'))
	procesar_timelines(os.path.join(timelineDirectory, 'Test', 'Negative'), os.path.join(cleanUsersDirectory, 'Test', 'Negative'))
