# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from enum import Enum


class DecoderType(Enum):
    PYAV = "pyav"
    PYAV_PARTIAL = "pyav_partial"
    TORCHVISION = "torchvision"
    DECORD = "decord"
