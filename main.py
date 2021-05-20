import fire

from plot import plot_all
from streetview import (download_streetview_image
                        calculate_coverage,
                        calculate_zone,
                        calculate_road_length)


if __name__ == "__main__":
    fire.Fire()
