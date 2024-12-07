import pandas as pd
import numpy as np
import os
import random
import torch
import tensorflow as tf

from transformers import AutoTokenizer, AutoModel, set_seed

from tensorflow.keras import backend as K

from huggingface_hub import login


# Establezca el valor seed inicial por todas partes para que sea reproducible.
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
tf.random.set_seed(SEED)
set_seed(SEED) # Transformers


login(token='XXXXXX')



def load_data(file_path):
	"""Carga los datos desde un archivo CSV."""

	data = pd.read_csv(file_path)

	return data['text'], data['class'], data['user_id']


def get_model_and_tokenizer(model_name, device='cuda:0'):
	"""Carga el modelo y el tokenizador correspondiente."""

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name).to(device)

	return tokenizer, model


def get_embeddings(texts, tokenizer, model, device='cuda:0'):
	"""Obtiene las representaciones de los textos usando el modelo especificado."""

	model = model.to(device)
	embeddings = []

	with torch.no_grad():
	
		count = 0
		totalTexts = len(texts)

		for text in texts:
			# Prepare the texts for BERT
			inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
			# Feed the texts to the BERT model
			outputs = model(**inputs)
			# Obtain the representation vectors
			cls_embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
			embeddings.append(cls_embedding)
	
			count += 1
			if count % 25 == 0:
				print(f'{count}/{totalTexts}...')


	return embeddings


def save_embeddings(embeddings, labels, users, output_file):
	"""Guarda las representaciones de las incrustaciones en un archivo de texto."""


	totalColumns = embeddings[0].shape[1]
	totalRows = len(embeddings)

	data = np.reshape(embeddings, (totalRows, totalColumns))
	cols = [f'c{i+1}' for i in range(totalColumns)]

	final = pd.DataFrame(data, columns=cols)
	final['class'] = labels
	final['user_id'] = users

	orderCols = ['class', 'user_id'] + cols
	final[orderCols].to_csv(output_file, index=False)




def main(train_file, test_file, output_dir, name, tokenizer, model):
	"""Proceso principal para obtener y guardar las incrustaciones de los tweets."""

	print(f'\n\n\n==========================================')
	
	# Cargar datos de entrenamiento y prueba
	train_texts, train_labels, train_users = load_data(train_file)
	test_texts, test_labels, test_users = load_data(test_file)
	

		
	print(f'\n** Procesando con el modelo: {name}')
	
	# Procesar datos de entrenamiento
	print('** Generando representaciones de entrenamiento...')
	
	train_embeddings = get_embeddings(train_texts, tokenizer, model)
	train_output_file = os.path.join(output_dir, f'{name}_train_embeddings.csv')
	
	save_embeddings(train_embeddings, train_labels, train_users, train_output_file)
	
	print(f'** Embeddings de entrenamiento guardados en {train_output_file}')


	# Clear session
	K.clear_session()
	torch.cuda.empty_cache()

	# Volver a agregar las semillas
	random.seed(SEED)
	np.random.seed(SEED)
	tf.random.set_seed(SEED)
	set_seed(SEED)
	

	# Procesar datos de prueba
	print('** Generando representaciones de prueba...')
	
	test_embeddings = get_embeddings(test_texts, tokenizer, model)
	test_output_file = os.path.join(output_dir, f'{name}_test_embeddings.csv')
	
	save_embeddings(test_embeddings, test_labels, test_users, test_output_file)
	
	print(f'** Embeddings de prueba guardados en {test_output_file}')

	# Clear session
	K.clear_session()
	torch.cuda.empty_cache()

	# Volver a agregar las semillas
	random.seed(SEED)
	np.random.seed(SEED)
	tf.random.set_seed(SEED)
	set_seed(SEED)



if __name__ == "__main__":

	### Nombres de los modelos a usar
	# 'Bert'			'bert-base-uncased'
	# 'RoBERTa'			'roberta-base'
	# 'MentalBert'		'mental/mental-bert-base-uncased'
	# 'MentalRoBERTa'	'mental/mental-roberta-base'
	# 'DistilBert'		'distilbert/distilbert-base-uncased'
	# 'DistilRoBERTa'	'distilbert/distilroberta-base

	modelName = 'Bert'
	modelCheckpoint = 'bert-base-uncased'
	tokenizer, model = get_model_and_tokenizer(modelCheckpoint)
	


	# Archivos de entrada
	train_file = '/Experimentos/Full/train_eng.csv'
	test_file = '/Experimentos/Full/test_eng.csv'
	
	# Directorio de salida para los archivos de embeddings
	output_dir = '/Experimentos/Embeddings'
	
	main(train_file, test_file, output_dir, modelName, tokenizer, model)
