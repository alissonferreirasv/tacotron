from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''
    Args:
      in_dir: O diretorio em que voce baixou o conjunto de dados
      out_dir: O diretorio para gravar a saida
      num_workers: Numero opcional de processos de trabalho para paralelizar
      tqdm: Opcionalmente, voce pode passar o tqdm para obter uma boa barra de progresso

    Returns:
      Uma lista de tuplas descrevendo os exemplos de treinamento. Isso deve ser escrito para train.txt
  '''

  # Usamos ProcessPoolExecutor para paralelizar processos. Esta e apenas uma otimizacao e voce
  # pode omitir e chamar _process_utterance em cada entrada, se desejar.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  with open(os.path.join(in_dir, 'texts.csv'), encoding='utf-8-sig') as f:
    for line in f:
      parts = line.strip().split('==')
      wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
      text = parts[1]
      futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
      index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
  '''Pre-processa um unico par de audio / texto de enunciado.

  Isso grava os espectrogramas mel e linear em disco e retorna uma tupla para gravar no arquivo train.txt.

  Args:
    out_dir: O diretorio para gravar os espectrogramas
    index: O indice numerico a ser usado nos nomes de arquivos do espectrograma.
    wav_path: Caminho para o arquivo de audio que contem a entrada de fala
    text: O texto falado no arquivo de audio de entrada

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Carregue o áudio em uma matriz numpy:
  wav = audio.load_wav(wav_path)

  # Calcule o espectrograma em escala linear a partir do wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Calcule um espectrograma em escala de mel a partir do wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Escreva os espectrogramas no disco:
  spectrogram_filename = 'ptbr-spec-%05d.npy' % index
  mel_filename = 'ptbr-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Retorne uma tupla descrevendo este exemplo de treinamento:
  return (spectrogram_filename, mel_filename, n_frames, text)
