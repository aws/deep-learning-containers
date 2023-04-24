# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import json
import os

import gluonnlp as nlp
import mxnet as mx


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    """
    ctx = mx.cpu()
    bert, vocab = nlp.model.get_model(
        'bert_12_768_12',
        dataset_name='book_corpus_wiki_en_uncased',
        pretrained=False,
        ctx=ctx,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)
    tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
    sentence_transform = nlp.data.BERTSentenceTransform(tokenizer,
                                                        max_seq_length=128,
                                                        vocab=vocab,
                                                        pad=True,
                                                        pair=False)
    batchify = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0,
                              pad_val=vocab[vocab.padding_token]),  # input
        nlp.data.batchify.Stack(),  # length
        nlp.data.batchify.Pad(axis=0, pad_val=0))  # segment
    # Set dropout to non-zero, to match pretrained model parameter names
    net = nlp.model.BERTClassifier(bert, dropout=0.1)
    net.load_parameters(os.path.join(model_dir, 'bert_sst.params'), mx.cpu(0))
    net.hybridize()

    return net, sentence_transform, batchify


def transform_fn(model, data, input_content_type, output_content_type):
    """
    Transform a request using the GluonNLP model. Called once per request.
    :param model: The Gluon model and the vocab
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    net, sentence_transform, batchify = model
    batch = json.loads(data)

    model_input = batchify(
        [sentence_transform(sentence) for sentence in batch])

    inputs, valid_length, token_types = [
        arr.as_in_context(mx.cpu()) for arr in model_input
    ]
    inference_output = net(inputs, token_types, valid_length.astype('float32'))
    inference_output = inference_output.as_in_context(mx.cpu())

    return mx.nd.softmax(inference_output).argmax(
        axis=1).astype('int').asnumpy().tolist()
