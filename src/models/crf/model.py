import pycrfsuite
import sklearn_crfsuite

def build_crf_model(c1=0.1, c2=0.1, max_iterations=100) -> sklearn_crfsuite.CRF:
    """
    Initializes a Conditional Random Field model.
    Using L-BFGS optimization.
    c1: L1 regularization penalty (promotes sparsity)
    c2: L2 regularization penalty
    """
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
        verbose=True
    )
    return crf
