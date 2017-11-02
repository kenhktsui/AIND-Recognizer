import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_components = min_n_components # newly added

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        BestBIC = float("inf")
        try:    
            beststate = self.n_constant
            for num_states in range(self.min_n_components,self.max_n_components+1):
                model = self.base_model(num_states)
                p = num_states * (num_states-1) + 2 * self.X.shape[1] * num_states  # transition probability + no of X x Mean and Sd x num_states 
                LL = model.score(self.X, self.lengths)
                BIC = -2 * LL + p * np.log(self.X.shape[0])
                if BIC < BestBIC:
                    BestBIC = BIC
                    beststate = num_states
                    
        except:
            pass

        return self.base_model(beststate)
        
         

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def DFC(self,n):
        model = self.base_model(n)
        anti = []
        for word, (X, lengths) in self.hwords.items():
            if word != self.this_word:
                anti.append(model.score(X, lengths))
        return model.score(self.X, self.lengths) - sum(anti)/(length(self.hwords.keys())-1)

    def DFC2nd(self):
        dfc2 = []
        counti = 0
        countj = 0
        for word, (X, lengths) in self.hwords.items():
            if word != self.this_word:
                countj += X.shape[1]
            else:
                counti += X.shape[1]
                dfc2.append(np.log(countj/counti))
        return sum(dfc2)
    
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        BestDIC = float("inf")
        try:    
            beststate = self.n_constant
            for num_states in range(self.min_n_components,self.max_n_components+1):
                model = self.base_model(num_states)
                p = num_states * (num_states-1) + 2 * self.X.shape[1] * num_states  # transition probability + no of X x Mean and Sd x num_states 
                DFC = self.DFC(num_states)
                DIC = self.DFC(num_states) +  p/(2 * (length(self.hwords.keys())-1))* self.DFC2nd()
                if DIC < BestDIC:
                    BestDIC = DIC
                    beststate = num_states
                    
        except:
            pass

        return self.base_model(beststate)
        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
        
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        bestLL = float("inf")
        try:
            beststate = self.n_constant 
            split_method = KFold()
            for num_states in range(self.min_n_components,self.max_n_components+1):
                LLlist = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    XTrain, LnTrain= combine_sequences(cv_train_idx,self.sequences)
                    XTest, LnTest= combine_sequences(cv_test_idx,self.sequences)
                    model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(XTrain, LnTrain)
                    LLlist.append(model.score(XTest,LnTest))
                averageLL = np.mean(LLlist)

                if averageLL < bestLL:
                    bestLL = averageLL
                    beststate = num_states                 
                    self.n_components = beststate
        
        except:
            pass

        return self.base_model(beststate)
