import math
from collections import defaultdict

import numpy as np

def precook(s, n=4, out=False):
  words = s.split()
  counts = defaultdict(int)
  for k in xrange(1,n+1):
    for i in xrange(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      counts[ngram] += 1
  return counts


def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
  return [precook(ref, n) for ref in refs]

#######################################################
# optimized cider class
# usage for calculating cider for groundtruth sentence
# cider = OptimziedCider()
# cider.set_dataset(video2captions)
# video2cider = cider.compute_leave_one_out_score()
# tool functions:
# 1. prepare ngram tf-idf from cooked caption
# vec_ref, norm_ref, length_ref = cider._counts2vec(cooked_caption)
# 2. calculate cider score between two sentence using the prepared ngram tf-idf
# score += cider._sim(ivec, jvec, inorm, jnorm, ilength, jlength)
class OptimizedCider(object):
  def __init__(self, n=4, sigma=6.0):
    self.n = n
    self.sigma = sigma
    self.ids = []
    self.crefs = []
    self.document_frequency = defaultdict(float)
    self.ref_len = None

  def set_dataset(self, gts):
    self.ids = gts.keys()
    for id in self.ids:
      ref = gts[id]
      self._cook_append(ref)

    self._compute_doc_freq()

  def sim(self, cooked_sentence_lhs, cooked_sentence_rhs):
    vec_lhs, norm_lhs, length_lhs = self._counts2vec(cooked_sentence_lhs)
    vec_rhs, norm_rhs, length_rhs = self._counts2vec(cooked_sentence_rhs)
    score = self._sim(vec_lhs, vec_rhs, norm_lhs, norm_rhs, length_lhs, length_rhs)

  # @profile
  def compute_leave_one_out_score(self):
    out = {}
    scores = []
    for id, refs in zip(self.ids, self.crefs):
      # compute vector for ref captions
      vec_refs = []
      norm_refs = []
      length_refs = []
      for ref in refs:
        vec_ref, norm_ref, length_ref = self._counts2vec(ref)
        vec_refs.append(vec_ref)
        norm_refs.append(norm_ref)
        length_refs.append(length_ref)

      scores = [] 
      num_ref = len(refs)
      for i in range(num_ref):
        score = np.array([0.0 for _ in range(self.n)])
        for j in range(num_ref):
          if j == i : continue
          score += self._sim(vec_refs[i], vec_refs[j], norm_refs[i], norm_refs[j], length_refs[i], length_refs[j])
        score_avg = np.mean(score)
        score_avg /= len(refs)-1
        score_avg *= 10.0
        # append score of an image to the score list
        scores.append(score_avg)
      out[id] = scores

      if id >= 1000: break

    return out

  def _cook_append(self, refs):
    self.crefs.append(cook_refs(refs))

  def _compute_doc_freq(self):
    for refs in self.crefs:
      # refs, k ref captions of one image
      for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
        self.document_frequency[ngram] += 1
    self.ref_len = np.log(float(len(self.crefs)))

  def _counts2vec(self, cnts):
    vec = [defaultdict(float) for _ in range(self.n)]
    length = 0
    norm = [0.0 for _ in range(self.n)]
    for (ngram,term_freq) in cnts.iteritems():
      # give word count 1 if it doesn't appear in reference corpus
      df = np.log(max(1.0, self.document_frequency[ngram]))
      # ngram index
      n = len(ngram)-1
      # tf (term_freq) * idf (precomputed idf) for n-grams
      vec[n][ngram] = float(term_freq)*(self.ref_len - df)
      # compute norm for the vector.  the norm will be used for computing similarity
      norm[n] += pow(vec[n][ngram], 2)

      if n == 1:
        length += term_freq
    norm = [np.sqrt(n) for n in norm]
    return vec, norm, length

  # @profile
  def _sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
    delta = float(length_hyp - length_ref)
    # measure consine similarity
    val = np.array([0.0 for _ in range(self.n)])
    for n in range(self.n):
      # ngram
      # for (ngram,count) in vec_hyp[n].iteritems():
        # vrama91 : added clipping
        # val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

      # better implementation by chenjia, guard against dictionary inflation
      for ngram in vec_hyp[n]:
        if ngram in vec_ref[n]:
          val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
        else:
          val[n] = 0

      if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
        val[n] /= (norm_hyp[n]*norm_ref[n])

      assert(not math.isnan(val[n]))
      # vrama91: added a length based gaussian penalty
      val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
    return val

  def compute_score(test, refs):
    vec, norm, length = self._counts2vec(test)
    # compute vector for ref captions
    score = np.array([0.0 for _ in range(self.n)])
    for ref in refs:
      vec_ref, norm_ref, length_ref = counts2vec(ref)
      score += self._sim(vec, vec_ref, norm, norm_ref, length, length_ref)
    # change by vrama91 - mean of ngram scores, instead of sum
    score_avg = np.mean(score)
    # divide by number of references
    score_avg /= len(refs)
    # multiply score by 10
    score_avg *= 10.0
    return score_avg