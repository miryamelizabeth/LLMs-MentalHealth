import pandas as pd
import numpy as np
import time
import os

from gensim.models import KeyedVectors



def load_data(file_path):
	"""Carga los datos desde un archivo CSV."""

	data = pd.read_csv(file_path)

	return data['text'], data['class'], data['user_id']


def load_embedding_model(model_path):
	"""Carga un modelo de incrustación de palabras desde un archivo."""

	print(f'Load embeddings...')
	start_time = time.time()

	model = KeyedVectors.load_word2vec_format(model_path, binary=False)

	print(f'{(time.time() - start_time) / 60.0}')

	return model


def get_average_embedding(text, model):
	"""Obtiene la representación promedio de un texto usando el modelo de incrustación."""

	words = text.split()
	word_vectors = [model[word] for word in words if word in model]
	
	if not word_vectors:
		# Si no hay palabras en el modelo, retornamos un vector de ceros
		return np.zeros(model.vector_size)
	
	return np.mean(word_vectors, axis=0)


def get_embeddings(texts, model):
	"""Obtiene las representaciones de los textos usando el modelo de incrustación especificado."""

	print(f'Get embeddings...')
	start_time = time.time()

	embeddings = [get_average_embedding(text, model) for text in texts]

	print(f'{(time.time() - start_time) / 60.0}')

	return embeddings




def save_embeddings(embeddings, labels, users, output_file):
	"""Guarda las representaciones de las incrustaciones en un archivo de texto."""

	print(f'Guardando...')
	start_time = time.time()

	totalColumns = len(embeddings[0])
	totalRows = len(embeddings)

	print(f'--> Shape: ({totalRows}, {totalColumns})')

	data = np.reshape(embeddings, (totalRows, totalColumns))
	cols = [f'c{i+1}' for i in range(totalColumns)]

	final = pd.DataFrame(data, columns=cols)
	final['class'] = labels
	final['user_id'] = users

	orderCols = ['class', 'user_id'] + cols
	final[orderCols].to_csv(output_file, index=False)

	print(f'{(time.time() - start_time) / 60.0}')




def main(train_file, test_file, output_dir, name, model):
	"""Proceso principal para obtener y guardar las incrustaciones de los tweets."""

	print(f'\n\n\n==========================================')
	
	# Cargar datos de entrenamiento y prueba
	train_texts, train_labels, train_users = load_data(train_file)
	test_texts, test_labels, test_users = load_data(test_file)
	

	print(f'\n** Procesando con el modelo: {name}')
	
	# Procesar datos de entrenamiento
	print('** Generando representaciones de entrenamiento...')
	
	train_embeddings = get_embeddings(train_texts, model)
	train_output_file = os.path.join(output_dir, f'{name}_train_embeddings.csv')
	
	save_embeddings(train_embeddings, train_labels, train_users, train_output_file)
	
	print(f'** Embeddings de entrenamiento guardados en {train_output_file}')

	

	# Procesar datos de prueba
	print('** Generando representaciones de prueba...')
	
	test_embeddings = get_embeddings(test_texts, model)
	test_output_file = os.path.join(output_dir, f'{name}_test_embeddings.csv')
	
	save_embeddings(test_embeddings, test_labels, test_users, test_output_file)
	
	print(f'** Embeddings de prueba guardados en {test_output_file}')



if __name__ == "__main__":

	#===========================================================
	# fastText
	# crawl-300d-2M.vec.zip: 2 million word vectors trained on Common Crawl (600B tokens)
	modelName = 'fastText'
	modelPath = r'Embeddings\fasttext-crawl-300d-2M.vec'
	model = load_embedding_model(modelPath)
	


	# Archivos de entrada
	train_file = r'dataset\Full\train_eng.csv'
	test_file = r'dataset\Full\test_eng.csv'
	
	# Directorio de salida para los archivos de embeddings
	output_dir = r'dataset\Embeddings'
	
	main(train_file, test_file, output_dir, modelName, model)
