from collections import defaultdict, Counter
import numpy as np
import os
import pickle
import random
import torch
import torch.utils.data as data

import utils

def custom_collate_fn(batch):
    batch = [torch.tensor(l, dtype=torch.long) for l in batch]
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

class CaseDataset:
    def __init__(self, fname, model, tokenizer, limit=-1, f_type="conllu", role_set=None, balanced=False, average=False):
      self.fname = fname
      self.role_set = role_set
      self.balanced = balanced
      self.average = average
  
      role_string = "aso" if role_set is None else ''.join(role_set)
      if balanced:
          limit_type = f"{role_string}_balanced"
      else:
          limit_type = f"{role_string}_unbalanced"
      # Get the filename from the path.
      save_fn = os.path.split(fname)[1]
      # Remove the extension.
      save_fn = os.path.splitext(save_fn)[0]
      save_fn = "all_features_aso_exps_" + save_fn
      save_fn = f"{save_fn}_{limit_type}"
      if limit > 0:
          save_fn = f"{save_fn}_{str(limit)}"
      else:
          save_fn += "_nolimit"
      if self.average:
          save_fn += "_average"
  
      tokens_labels_dir = "cached_datasets"
      tokens_labels_path = os.path.join(tokens_labels_dir, save_fn + '.pkl')
      if os.path.exists(tokens_labels_path):
          print("Loading all of the tokens and non-bert stuff from", tokens_labels_path)
          self.tokens, self.role_labels, self.word_forms_list, \
          self.verb_type_labels, self.len, self.relevant_examples_index, \
          self.bert_tokens, self.bert_ids, self.orig_to_bert_map, \
          self.bert_to_orig_map = \
              pickle.load(open(tokens_labels_path, 'rb'))
      else:
          self.tokens, self.role_labels, self.word_forms_list, \
          self.verb_type_labels, self.len, self.relevant_examples_index  = \
              utils.get_tokens_and_labels(self.fname, limit=limit, f_type=f_type, role_set=role_set, balanced=balanced)
          self.bert_tokens, self.bert_ids, self.orig_to_bert_map, self.bert_to_orig_map = \
              utils.get_bert_tokens(self.tokens, tokenizer)
          print("lengths of bert ids etc", len(self.bert_tokens), len(self.bert_ids), len(self.orig_to_bert_map), len(self.bert_to_orig_map))
          print("Saving all of the tokens and non-bert stuff to", tokens_labels_path)
          pickle.dump(
              (self.tokens, self.role_labels, self.word_forms_list,
               self.verb_type_labels, self.len, self.relevant_examples_index,
               self.bert_tokens, self.bert_ids, 
               self.orig_to_bert_map, self.bert_to_orig_map),
              open(tokens_labels_path, 'wb'))
  
      # We need to check whether the length is large enough before we run through BERT. 
      # Otherwise, super unbalanced datasets will end up running whole training 
      # treebanks through BERT.
      print(f"There are {self.len} relevant tokens, and {len(self.tokens)} overall sentences")
      if self.len <  limit and balanced:
          print(f"Set is smaller than limit! Length {self.len}, limit {limit}.")
          return
  
      bert_vectors_dir = 'cached_bert_vectors'
      hdf5_path = os.path.join(bert_vectors_dir, save_fn + ".hdf5")
      self.bert_outputs = utils.get_bert_outputs(hdf5_path, self.bert_ids, model)
      print("length of bert outputs", len(self.bert_outputs))
  
    def __len__(self):
      return self.len
  
    def get_bert_id_dataloader(self, batch_size=32):
      return data.DataLoader(self.bert_ids, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

class CaseLayerDataset(data.Dataset):
    def __init__(self, case_dataset, layer_num, labeldict=None, verbose=False):
      self.layer_num = layer_num
      self.verbose = verbose
      self.case_dataset = case_dataset
      self.balanced = self.case_dataset.balanced
      self.pool_method = "average" if case_dataset.average else "first"
      self.embs, self.role_labels, self.word_forms, \
      self.verb_type_labels, self.idxs, indices_by_role = \
        self.get_labels(case_dataset.bert_outputs, case_dataset.role_labels,
                        case_dataset.word_forms_list,
                        case_dataset.verb_type_labels, case_dataset.orig_to_bert_map,
                        case_dataset.relevant_examples_index, pool_method=self.pool_method)
      if self.balanced:
          min_role_len = min([len(indices_by_role[role]) for role in case_dataset.role_set])
          print(f"Balancing cases to all have {min_role_len} elements")
          combined_indices = []
          for role in case_dataset.role_set:
              combined_indices += indices_by_role[role][:min_role_len]
          print(f"After trimming cases, have {len(combined_indices)} total indices")
          # For curriculum reasons, we probably don't want to have our training
          # examples with all roles in order.
          random.shuffle(combined_indices)
          self.embs = [self.embs[index] for index in combined_indices]
          self.role_labels = [self.role_labels[index] for index in combined_indices]
          self.word_forms = [self.word_forms[index] for index in combined_indices]
          self.verb_type_labels = [self.verb_type_labels[index] for index in combined_indices]
          self.idxs = [self.idxs[index] for index in combined_indices]

      print("Examples #", len(self.idxs))
      self.labeldict = self.get_label_dict(labeldict)
      print("labeldict", self.labeldict)

      self.processed_labels = [(self.labeldict[x] if x in self.labeldict else -1) for x in self.role_labels]

    def __getitem__(self, idx):
        verb_type_label = self.verb_type_labels[idx] if self.verb_type_labels[idx] is not None else ""
        return self.embs[idx], self.processed_labels[idx], (self.word_forms[idx], verb_type_label, self.idxs[idx])


    def __len__(self):
      return len(self.embs)

    def get_label_dict(self, old_labeldict):
      # Make a labeldict of all of the labels in this dataset, keeping the same 
      # name fo
      labelset = sorted(list(set(self.role_labels)))
      if old_labeldict is None:
          curr_label = 0
          labeldict = {}
      else:
          labeldict = old_labeldict
          curr_label = len(old_labeldict)
      for label in labelset:
          if old_labeldict is None or label not in old_labeldict:
              labeldict[label] = curr_label
              curr_label += 1
      return labeldict

    def get_label_set(self):
        return sorted(self.labeldict.keys(), key=lambda x: self.labeldict[x])

    def get_num_labels(self):
      return len(self.labeldict)

    def get_labels(self, bert_outputs, role_labels, word_forms_list, verb_type_labels, orig_to_bert_map, relevant_examples_index, pool_method="first"):
        train = []
        out_role_labels, out_word_forms, out_verb_type_labels, out_index = [], [], [], [], [], []
        indices_by_role = defaultdict(list)
        for sentence_num, word_num in relevant_examples_index:
            role_label = role_labels[sentence_num][word_num]
            out_role_labels.append(role_label)
            out_word_forms.append(word_forms_list[sentence_num][word_num])
            out_verb_type_labels.append(verb_type_labels[sentence_num][word_num])
            bert_start_index = orig_to_bert_map[sentence_num][word_num]
            if len(orig_to_bert_map[sentence_num]) > word_num+1:
              bert_end_index = orig_to_bert_map[sentence_num][word_num + 1]
            else:
              bert_end_index = -1
            bert_sentence = bert_outputs[sentence_num][self.layer_num].squeeze()
            if pool_method == "first":
                train.append(bert_sentence[bert_start_index])
            elif pool_method == "average":
                train.append(np.mean(bert_outputs[sentence_num][self.layer_num].squeeze()[bert_start_index:bert_end_index]))
            indices_by_role[role_label].append(len(out_role_labels) - 1)
            out_index.append((sentence_num, bert_start_index, bert_end_index, word_num))

        return train, out_role_labels, out_word_forms, out_verb_type_labels, out_index, indices_by_role

    def get_dataloader(self, batch_size=32, shuffle=True):
      return data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)
