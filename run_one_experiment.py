"""
Run one iteration of the experiment, training on one language and testing on another.
"""
import argparse
import csv
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
from transformers import BertTokenizer, BertModel

from utils import train_classifier, eval_classifier, eval_classifier_ood
import data

# The size to cap the training data. Size is measured in cased nouns.
# We chose the number of cased nouns in Basque as our limit.
BASQUE_CASED_NOUNS = 13128
BASQUE_AO_CASED_NOUNS = 4312
BASQUE_AO_CASED_NOUNS_BALANCED = 2025

TEST_DATA_LIMIT = 2000000

COCA_DATA_LIMIT = 29*5*1000

BERT_LIMIT = 25000 # (don't run out of RAM)

def run_experiment(args):
    train_tb_name = os.path.split(args.train_lang_base_path)[1]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased',
                                     output_hidden_states=True)
    model.eval()
    num_layers = model.config.num_hidden_layers

    training_sent_roles = ["A", "O"]
    role_code = "".join(training_sent_roles).lower()

    if args.balance:
        classifier_paths = [os.path.join("classifiers", f"aso_{train_tb_name}_{args.seed}_{role_code}_balanced_{layer}_exproles") for layer in range(num_layers + 1)]
    else:
        classifier_paths = [os.path.join("classifiers", f"aso_{train_tb_name}_{args.seed}_{role_code}_{layer}_exproles") for layer in range(num_layers + 1)]
    if args.average_embs:
        classifier_paths = [path + "_average" for path in classifier_paths]

    # Set the size of our training set to the number of Basque data points, depending on the type of experiment.
    if args.balance:
        training_data_limit = BASQUE_AO_CASED_NOUNS_BALANCED 
    else:
        training_data_limit = BASQUE_AO_CASED_NOUNS

    has_trained_classifiers = all([os.path.exists(path) for path in classifier_paths])
    if has_trained_classifiers:
        print("Classifiers already trained!")

    if args.f_type == "txt":
        coca = data.CaseDataset(args.train_lang_base_path, model, tokenizer, limit=COCA_DATA_LIMIT, f_type=args.f_type, average=args.average_embs)
    else:
        coca = None
    if not has_trained_classifiers:
        train_classifiers(
            args, classifier_paths, model, tokenizer, training_data_limit, training_sent_roles, balanced=args.balance, average=args.average_embs, cd=coca)
    if args.reeval_src_test:
        print(f"Loading the source test set, with limit {TEST_DATA_LIMIT}")
        src_test = data.CaseDataset(
            args.train_lang_base_path + "-test.conllu", model, tokenizer,
            limit=TEST_DATA_LIMIT, f_type=args.f_type, average=args.average_embs)
    print(f"Loading the dest test set, with limit {TEST_DATA_LIMIT}")
    if args.f_type == "conllu":
        dest_test = data.CaseDataset(args.test_lang_fn, model, tokenizer, limit=TEST_DATA_LIMIT, f_type=args.f_type, average=args.average_embs)
    else:
        dest_test = coca

    out_df = pd.DataFrame([])
    # Layers trained in reverse so we can make sure code is working with informative layers early
    for layer in reversed(range(num_layers+1)):
        print("On layer", layer)
        classifier_path = classifier_paths[layer]
        classifier, labelset, labeldict, src_test_accuracy = pickle.load(open(classifier_path, "rb"))
        print(f"Loaded case classifier from {classifier_path}!")
        print("src_test_accuracy:", src_test_accuracy)
        if args.reeval_src_test:
            src_test_dataset = data.CaseLayerDataset(src_test, layer_num=layer, labeldict=labeldict)
            src_test_accuracy = eval_classifier(classifier, src_test_dataset)
            print("src_test_accuracy [re-eval]:", src_test_accuracy, "Saving new src test accuracy")
            with open(classifier_path, 'wb') as pkl_file:
                pickle.dump((classifier, labelset, labeldict, src_test_accuracy), pkl_file)
        dest_test_dataset = data.CaseLayerDataset(dest_test, layer_num=layer, labeldict=labeldict)
        print("There are", len(dest_test_dataset), "examples to evaluate on.")
        results = eval_classifier_ood(classifier, labelset, dest_test_dataset)
        results["layer"] = layer
        for key in src_test_accuracy.keys():
            results[f"source_test_accuracy_{key}"] = src_test_accuracy[key]
        print(results)
        out_df = pd.concat((out_df, results), ignore_index=True)

    if args.test_on_train:
        if args.reeval_src_test:
            print(f"Loading the source test set, with limit {TEST_DATA_LIMIT}")
            src_test = data.CaseDataset(
                args.train_lang_base_path + "-train.conllu", model, tokenizer,
                limit=TEST_DATA_LIMIT, f_type=args.f_type, average=args.average_embs)
        print(f"Loading the dest test set, with limit {TEST_DATA_LIMIT}")
        dest_test = data.CaseDataset(args.train_lang_fn, model, tokenizer, limit=TEST_DATA_LIMIT, f_type=args.f_type, average=args.average_embs)
        for layer in reversed(range(num_layers+1)):
            print("On layer", layer)
            classifier_path = classifier_paths[layer]
            classifier, labelset, labeldict, src_test_accuracy = pickle.load(open(classifier_path, "rb"))
            print(f"Loaded case classifier from {classifier_path}!")
            print("src_test_accuracy:", src_test_accuracy)
            if args.reeval_src_test:
                src_test_dataset = data.CaseLayerDataset(src_test, layer_num=layer, labeldict=labeldict)
                src_test_accuracy = eval_classifier(classifier, src_test_dataset)
                print("src_test_accuracy [re-eval]:", src_test_accuracy, "Saving new src test accuracy")
                with open(classifier_path, 'wb') as pkl_file:
                    pickle.dump((classifier, labelset, labeldict, src_test_accuracy), pkl_file)
            dest_test_dataset = data.CaseLayerDataset(dest_test, layer_num=layer, labeldict=labeldict)
            print("There are", len(dest_test_dataset), "examples to evaluate on.")
            results = eval_classifier_ood(classifier, labelset, dest_test_dataset)
            results["layer"] = layer
            for key in src_test_accuracy.keys():
                results[f"source_test_accuracy_{key}"] = src_test_accuracy[key]
            print(results)
            out_df = pd.concat((out_df, results), ignore_index=True)

    out_df.to_csv(os.path.join("results", args.output_fn))

def train_classifiers(args, classifier_paths, model, tokenizer, training_data_limit, training_role_set, balanced=False, average=False, cd=None):
    print("Need to train classifiers!")
    print(f"Loading the source train set, with limit {training_data_limit}")
    if args.f_type == "txt":
        fname = args.train_lang_base_path
        src_train = cd
    elif args.f_type == "conllu":
        fname = args.train_lang_base_path + "-train.conllu"
        src_train = data.CaseDataset(fname,
            model, tokenizer, limit=training_data_limit, role_set=training_role_set, f_type=args.f_type, balanced=balanced, average=average)
    print(f"Length of train set is {len(src_train)}, limit is {training_data_limit}")
    if len(src_train) < training_data_limit:
        print("Too small! Exiting")
        sys.exit()
    if args.f_type == "txt":
        src_test = src_train
    elif args.f_type == "conllu":
        src_test = data.CaseDataset(args.train_lang_base_path + "-test.conllu", model, tokenizer, limit=TEST_DATA_LIMIT, f_type=args.f_type, average=average)
    num_layers = model.config.num_hidden_layers
    for layer in reversed(range(num_layers+1)):
        classifier_path = classifier_paths[layer]
        if os.path.exists(classifier_path):
            continue
        train_dataset = data.CaseLayerDataset(src_train, layer_num=layer)
        print("train dataset labeldict", train_dataset.labeldict)
        print("Training on", len(train_dataset), "data points.")
        classifier = train_classifier(train_dataset)
        print("Trained a case classifier!")
        src_test_dataset = data.CaseLayerDataset(src_test, layer_num=layer, labeldict=train_dataset.labeldict)
        src_test_accuracy = eval_classifier(classifier, src_test_dataset)
        print(f"Accuracy on test set of training language: {src_test_accuracy}")
        print(f"Saving classifier to {classifier_path}")
        with open(classifier_path, 'wb') as pkl_file:
            pickle.dump((classifier, train_dataset.get_label_set(), train_dataset.labeldict, src_test_accuracy), pkl_file)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-lang-base-path', type=str,
        default="ud/en_ewt-ud",
        help="The path to the UD treebanks of the language we're training the classifier on. The path should be without the '-train.connlu'/'-test.conllu' part")
    parser.add_argument('--test-lang-fn', type=str,
        default="ud/en_ewt-ud-test.conllu",
        help="The path to the UD treebank file we're testing the classifier on")
    parser.add_argument('--train-lang-fn', type=str,
        default="ud/en_ewt-ud-train.conllu",
        help="The path to the UD treebank file we're training the classifier on")
    parser.add_argument('--balance', action='store_true', 
                        help="When this option is set, ")
    parser.add_argument('--average-embs', action='store_true', help='With this option, use the average embedding of the subwords of a word, rather than the first subword')
    parser.add_argument("--output-fn", type=str, default="last_run",
                        help="Where to save this run's output")
    parser.add_argument("--reeval-src-test", action="store_true",
                        help="Reevaluate the test set of the source language")
    parser.add_argument("--seed", type=int, default=-1, help="random seed")
    parser.add_argument("--test-on-train", action='store_true', help="test on train S labels")
    parser.add_argument('--f-type', type=str,
        default="conllu",
        help="conllu or txt")

    args = parser.parse_args()

    print("args:", args)

    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Just set the seed to {args.seed}")
    else:
        print("Not setting random seed")

    run_experiment(args)

if __name__ == "__main__":
    __main__()
