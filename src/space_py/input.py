import tifffile as tf
import numpy as np
import re
from warnings import warn


class Palette(tuple):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    @staticmethod
    def is_hex_color(string: str):
        """
        Check if a string is a valid hexadecimal color code,
        i.e. #RGB or #RRGGBB.
        """
        return bool(re.match(r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$", string))

    def validate(self):
        assert len(self) > 0, "Palette must have at least one color."
        for color in self:
            assert Palette.is_hex_color(color), (
                "{color} is not a valid hexadecimal color."
            )
        if self[0] not in {"#000", "#000000", "#FFF", "#FFFFFF"}:
            warn("The first palette color is neither black nor white for background.")


class Image(np.ndarray):
    def __init__(self, name: str, palette: Palette, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.name = name
        self.palette = palette

    def validate(self):
        assert self.ndim == 4, "Image has {self.ndim} dimensions but should have 4."


class ObjectImage(Image):
    def __init__(self, name: str, palette: Palette, *args, **kwargs):
        super().__init__(name, palette, *args, **kwargs)

    def validate(self):
        super().validate()
        assert self.dtype == np.uint8, (
            "Object image has {self.dtype} pixels but should have uint8."
        )
        assert self.shape[3] == 1, (
            "Object image has {self.shape[3]} channels but should have 1."
        )
        num_objects = len(np.unique(self))
        palette_length = len(self.palette)
        assert palette_length == num_objects, (
            "Object image has {palette_length} palette colors but should have {num_objects}."
        )


class ScalarImage(Image):
    def __init__(self, name: str, palette: Palette, *args, **kwargs):
        super().__init__(name, palette, *args, **kwargs)

    def validate(self):
        super().validate()
        assert self.dtype == np.float16, (
            "Object image has {self.dtype} pixels but should have float16."
        )
        num_channels = len(np.unique(self))
        palette_length = len(self.palette)
        assert palette_length == num_channels, (
            "Object image has {palette_length} palette colors but should have {num_channels}."
        )


# Start with classes:
# Image parent class
#   ObjectImage
#   ScalarImage
# Table parent class
#   ObjectTable
#   ProfileTable
#   LinkTable
