from .spatial_distribution import (plot_spatial_distribution,
                                   plot_prepost,
                                   plot_post,
                                   plot_samples)
from .coverage import plot_coverage
from .precision_recall_curve import plot_precision_recall

def plot_all():
    plot_spatial_distribution()
    plot_coverage()
    plot_precision_recall()
