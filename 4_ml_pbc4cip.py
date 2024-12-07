### enviroment: tesis_elizabeth

import pandas as pd
import numpy as np
import time

from os import path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from PBC4cip.PBC4cip import PBC4cip

from pycm import ConfusionMatrix
# https://www.pycm.ir/doc/index.html#Full


N_JOBS = 8
SEED = 42

DISORDER_TAG = 'SELF-HARM' # ANOREXIA DEPRESSION SELF-HARM PTSD



def load_dataset(trainFile, testFile, method):

	print(f'Reading files...')
	start_time = time.time()

	train = pd.read_csv(trainFile)
	test = pd.read_csv(testFile)
	
	print(f'{(time.time() - start_time) / 60.0}')

	
	train['class'].replace({'CONTROL': 'negative', DISORDER_TAG: 'positive'}, inplace=True)
	test['class'].replace({'CONTROL': 'negative', DISORDER_TAG: 'positive'}, inplace=True)


	# Guardar el user_id
	users_train = train['user_id'].values.tolist()
	users_test = test['user_id'].values.tolist()


	if method == 'NGRAM' or method == 'NMF' or method == 'LDA':
		X_train = train['tweets_user'].values
		X_test = test['tweets_user'].values

	elif method == 'NUM':
		X_train = train.drop(['user_id', 'class'], axis=1)
		X_test = test.drop(['user_id', 'class'], axis=1)

		X_train.fillna(0, inplace=True)
		X_test.fillna(0, inplace=True)

	else:
		pass


	y_train = train[['class']]
	y_test = test[['class']]


	return X_train, X_test, y_train, y_test, users_train, users_test



def train_test_model(X_train, X_test, y_train, model, filePatterns):

	print(f'Training...')
	start_time = time.time()

	patterns = model.fit(X_train, y_train)

	print(f'\nPatterns Found:')
	# Writing to file
	with open(filePatterns, 'w') as file:
		for pattern in patterns:
			file.write(f'{pattern}\n\n')


	print(f'Testing...')	
	y_pred = model.predict(X_test) # lista de 1 y 0s --> 0: negative y 1:positive
	y_scores = model.score_samples(X_test)


	print(f'{(time.time() - start_time) / 60.0}')

	return y_pred



def tag(lstVal1, lstVal2):
	
	final = []
	
	for val1, val2 in zip(lstVal1, lstVal2):
		if val1 > val2:
			final.append('CONTROL')
		elif val2 > val1:
			final.append(DISORDER_TAG)
		else:
			final.append('CONTROL')
	
	return final

def get_group_predictions(test_users, y_test, y_pred, rawPredictionsFilename, predictionsFilename):

	print(f'Grouping predictions...')

	# convert one-hot-encoder classes to number value
	actual_vector = y_test['class'].values.tolist()
	predict_vector = ['negative' if y == 0 else 'positive' for y in y_pred]

	actual_vector = ['CONTROL' if y == 'negative' else DISORDER_TAG for y in actual_vector]
	predict_vector = ['CONTROL' if y == 'negative' else DISORDER_TAG for y in predict_vector]


	df = pd.DataFrame(data={
							'user_id': test_users,
							'y_test': actual_vector,
							'y_pred': predict_vector,
							# f'y_proba_{DISORDER_TAG.lower()}': y_proba[:, 1],
							# 'y_proba_control': y_proba[:, 0]
							})

	# GUARDAR!!!!
	print(f'Saving raw predictions...')
	df.to_csv(rawPredictionsFilename, index=False)

	# validación si no hay predicciones de todos los desórdenes
	etiquetasExistentes = [d for d in [DISORDER_TAG] if d not in df['y_pred'].unique()]

	group_predictions = df.groupby(['user_id'], sort=False)['y_pred'].value_counts().to_frame(name='total').reset_index()
	total_preds = group_predictions.pivot_table(index='user_id', columns='y_pred', values='total', sort=False).reset_index().fillna(value=0)
	# validación si no hay predicciones de todos los desórdenes
	for e in etiquetasExistentes:
		total_preds[e] = 0

	total_preds['y_pred'] = tag(total_preds['CONTROL'].values, total_preds[DISORDER_TAG].values)

	pred = total_preds[['user_id', 'y_pred']]
	#        user_id  CONTROL   DIAGNOSED      y_pred
	# 0            A      1.0         2.0   DIAGNOSED
	# 1            B      2.0         0.0     CONTROL
	# 2            C      3.0         1.0     CONTROL
	# 3            D      1.0         1.0     CONTROL

	# Obtener y_test
	test = df.groupby(['user_id'], sort=False)['y_test'].first().reset_index()

	# Merge con las predicciones
	merge = pd.merge(left=test, right=pred, on='user_id', how='left')

	# # Obtener las probabilidades agrupadas
	# probas = df.groupby(['user_id'], sort=False)[['y_proba_control', f'y_proba_{DISORDER_TAG.lower()}']].mean().reset_index()

	# Usuarios
	usersLst = df.groupby(['user_id'], sort=False).first().reset_index()['user_id']


	print(f'Saving final preds...')
	# Dataframe para guardar --> agrupadas las predicciones por mayoría de votos
	finalFile = pd.DataFrame(data={
								'user_id': usersLst,
								'y_test': merge['y_test'].values.tolist(),
								'y_pred': merge['y_pred'].values.tolist(),
								# f'y_proba_{DISORDER_TAG.lower()}': probas[f'y_proba_{DISORDER_TAG.lower()}'],
								# 'y_proba_control': probas['y_proba_control']
								})
	# Guardandoo!!
	finalFile.to_csv(predictionsFilename, index=False)


	return merge['y_pred'].values.tolist(), merge['y_test'].values.tolist()



def save_results(y_test, y_pred, metricsFilename, confusionFilename, normalizedConfusionFilename):

	print(f'Getting metrics...')

	cm = ConfusionMatrix(actual_vector=y_test,
						predict_vector=y_pred,
						digit=5)

	# Métricas de c/clase
	# GM - Geometric mean of specificity and sensitivity
	# MCC - Matthews correlation coefficient
	# PPV - Positive predictive value: precision
	# TPR - True positive rate: sensitivity/recall (e.g. the percentage of sick people who are correctly identified as having the condition)
	summary = pd.DataFrame(cm.class_stat).T
	summary.replace('None', 0, inplace=True)
	summary = summary.reset_index()
	summary = summary[summary['index'].isin(['AUC', 'F1', 'GM', 'MCC', 'PPV', 'TPR', 'ACC'])]
	summary['Macro_Average'] = (summary['CONTROL'] + summary[DISORDER_TAG]) / 2.0
	
	summary.to_csv(metricsFilename, index=False)


	# Matriz de confusion
	# En la documentación parece ser que columns = predict, rows = actual
	# por eso en el dataframe trasponemos la matriz
	cmDf = pd.DataFrame(cm.matrix).T.reset_index()
	cmDf.to_csv(confusionFilename, index=False)


	cmDf = pd.DataFrame(cm.normalized_matrix).T.reset_index()
	cmDf.to_csv(normalizedConfusionFilename, index=False)


	# Imprimir en pantalla métricas
	print(f'Report metrics...')

	f1_macro = summary[summary['index'] == 'F1'][DISORDER_TAG].values[0]
	auc_macro = summary[summary['index'] == 'AUC'][DISORDER_TAG].values[0]

	print(f'F1 = {f1_macro:.3f}\tAUC = {auc_macro:.3f}\n')



def get_objects(method):

	if method == 'LDA':
		vectorizer = CountVectorizer()
	
	elif method == 'NGRAM' or method == 'NMF':
	
		if DISORDER_TAG == 'ANOREXIA':
			vectorizer = TfidfVectorizer(min_df=15, max_features=10000, max_df=0.9)
		
		elif DISORDER_TAG == 'DEPRESSION':
			vectorizer = TfidfVectorizer(min_df=10, max_features=10000, max_df=0.95)
	
		elif DISORDER_TAG == 'SELF-HARM':
			vectorizer = TfidfVectorizer(min_df=5, max_features=10000, max_df=1.0)
		
		else:
			vectorizer = TfidfVectorizer()
		
		print(vectorizer)

	else:
		vectorizer = None
	

	if method == 'LDA':
		topic = LatentDirichletAllocation(learning_method='online', random_state=SEED, n_jobs=N_JOBS)
	elif method == 'NMF':
		topic = NMF(random_state=SEED) # alpha=1
	else:
		topic = None


	return vectorizer, topic



def run_model(dataDirectory, saveDirectory, nameResults, language, method):

	print('\n---------------------')
	print(f'\n\n\t{nameResults.upper()} - {DISORDER_TAG}\n')
	print('\n---------------------')

	classifiers_list = {
						'PBC4cip': PBC4cip()
					}

	trainFilename = path.join(dataDirectory, f'train_{language}.csv')
	testFilename = path.join(dataDirectory, f'test_{language}.csv')

	X_train, X_test, y_train_all, y_test_all, users_train, users_test = load_dataset(trainFilename, testFilename, method)


	vectorizer, topic = get_objects(method)


	for nameClf, clf in classifiers_list.items():

		print(f'\n*** {nameClf} ***')

		fileResults = path.join(saveDirectory, f'{nameClf}_{nameResults}_metrics_{language}.csv')
		fileMatrix = path.join(saveDirectory, f'{nameClf}_{nameResults}_matrix_{language}.txt')
		fileMatrixNorm = path.join(saveDirectory, f'{nameClf}_{nameResults}_normalized_matrix_{language}.txt')
		fileRawPredictions = path.join(saveDirectory, f'{nameClf}_{nameResults}_raw-predictions_{language}.csv')
		filePredictions = path.join(saveDirectory, f'{nameClf}_{nameResults}_predictions_{language}.csv')
		filePatterns = path.join(saveDirectory, f'{nameClf}_{nameResults}_patterns_{language}.txt')


		if method == 'NGRAM':
			
			vectorizer.fit(X_train)
			
			X_train_tfdif = vectorizer.transform(X_train).toarray()
			X_test_tfidf = vectorizer.transform(X_test).toarray()

			colNames = [f'att_{c}' for c in vectorizer.get_feature_names_out()]
			X_train = pd.DataFrame(data=X_train_tfdif, columns=colNames)
			X_test = pd.DataFrame(data=X_test_tfidf, columns=colNames)

		elif method == 'NMF' or method == 'LDA':
			pass


		elif method == 'NUM':
			pass


		else:
			pass


		try:
			print(saveDirectory)

			## The features (X_train, X_test) and the class (y_train, y_test)
			## should be DATAFRAMES
			## The classifier return a list of 0s and 1s
		
			y_pred_all = train_test_model(X_train, X_test, y_train_all, clf, filePatterns)
			# y_pred, y_test = get_group_predictions(users_test, y_test_all, y_pred_all, fileRawPredictions, filePredictions)
			# save_results(y_test, y_pred, fileResults, fileMatrix, fileMatrixNorm)
		
		except Exception as e:
			print(f'An exception occurred: {e}')


def main(partitionDirNormal, partitionDirLiwc, partitionDirOtros, directoryResults, language):

	# --------------------------
	# TEXT-STYLE
	# --------------------------
	run_model(partitionDirLiwc, path.join(directoryResults, 'STYLE_LIWC'), 'style_liwc', language, 'NUM')


	# --------------------------
	# TF-IDF Tokenization - WORDS
	# --------------------------	
	run_model(partitionDirNormal, path.join(directoryResults, 'WORD_UNIGRAM'), 'word_unigram', language, 'NGRAM')


	# --------------------------
	# EMBEDDINGS
	# --------------------------
	run_model(partitionDirOtros[0], path.join(directoryResults, 'EMBD_BERT'), 'embd_bert', language, 'NUM')
	run_model(partitionDirOtros[1], path.join(directoryResults, 'EMBD_MENTALBERT'), 'embd_mental-bert', language, 'NUM')
	run_model(partitionDirOtros[2], path.join(directoryResults, 'EMBD_ROBERTA'), 'embd_roberta', language, 'NUM')
	run_model(partitionDirOtros[3], path.join(directoryResults, 'EMBD_MENTALROBERTA'), 'embd_mental-roberta', language, 'NUM')
	run_model(partitionDirOtros[4], path.join(directoryResults, 'EMBD_FASTTEXT'), 'embd_fasttext', language, 'NUM')
	run_model(partitionDirOtros[5], path.join(directoryResults, 'EMBD_GLOVE'), 'embd_glove', language, 'NUM')
	run_model(partitionDirOtros[6], path.join(directoryResults, 'EMBD_GPT'), 'embd_gpt', language, 'NUM')



# --------------------------------
if __name__ == '__main__':

	language = 'eng'

	# ============================================
	DISORDER_TAG = 'SELF-HARM'

	partitionDirNormal = r'dataset\Txt'
	partitionDirLiwc = r'dataset\Liwc'
	partitionDirOtros = [r'dataset\Embeddings_Bert',
					  r'dataset\Embeddings_MentalBert',
					  r'dataset\Embeddings_RoBERTa',
					  r'dataset\Embeddings_MentalRoBERTa',
					  r'dataset\Embeddings_fastText',
					  r'dataset\Embeddings_GloVe',
					  r'dataset\Embeddings_GPT']

	directoryResults = r'dataset\Results'
	main(partitionDirNormal, partitionDirLiwc, partitionDirOtros, directoryResults, language)
