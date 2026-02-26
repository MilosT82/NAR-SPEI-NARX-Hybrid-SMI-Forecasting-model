import pandas as pd
import numpy as np


def compute_spei(P, T, scale=12):
    """
    Simplified SPEI calculation using rolling water balance.
    """
    D = P - T  # climatic water balance

    D_roll = D.rolling(window=scale).sum()

    spei = (D_roll - D_roll.mean()) / D_roll.std()

    return spei