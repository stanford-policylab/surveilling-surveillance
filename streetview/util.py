from geographiclib.geodesic import Geodesic


def get_heading(lat1, lon1, lat2, lon2):
    return Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']

#def is_close(lat1, lon1, lat2, lon2, d=None):
    