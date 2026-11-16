"""flatprint: data infrastructure for the generator-fingerprint study."""

__version__ = "0.1.0"

from pillow_heif import register_heif_opener

register_heif_opener()  # Pillow opens HEIC/HEIF (phone photos) like any other image
