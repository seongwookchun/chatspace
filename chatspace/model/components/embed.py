"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import tqdm
import torch.nn as nn
import numpy as np
from gensim.models.wrappers import FastText


class CharEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reserved_tokens = ["[PAD]", "[UNK]", "[SEP]",
                                "[SEPT]", "[SOS]", "[EOS]", "[MASK]", "#", "[CLS]"]
        self.embedding_size = config["vocab_size"]
        self.embedding_dim = config["embedding_dim"]

        self.embedding = nn.Embedding(
            num_embeddings=self.embedding_size, embedding_dim=self.embedding_dim, padding_idx=0)

        self.vocab_list = config["vocab_list"]
        self.word_vec_path = config["word_vec_path"]
        # self.load_word2vec()

    def load_word2vec(self):
        raw_word2vec = FastText.load_fasttext_format(self.word_vec_path)


        word2vec = []
        oov_cnt = 0
        for v in tqdm.tqdm(self.vocab_list):
            if v == "[PAD]":
                vec = np.zeros(self.embedding_dim)
            elif v in self.reserved_tokens:
                vec = np.random.randn(self.embedding_dim) * 0.1
            else:
                if v in raw_word2vec:
                    vec = raw_word2vec[v]
                else:
                    oov_cnt += 1
                    vec = np.random.randn(self.embedding_dim) * 0.1
            word2vec.append(vec)
        self.embedding.from_pretrained(torch.FloatTensor(word2vec), padding_idx=0)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab_list)))

    def forward(self, input_seq):
        #print(self.embedding(input_seq))
        return self.embedding(input_seq)
