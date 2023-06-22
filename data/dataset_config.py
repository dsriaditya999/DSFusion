import os
from dataclasses import dataclass, field
from typing import Dict

from effdet.data.dataset_config import *



# FLIR-Aligned Dataset
@dataclass
class FlirAlignedFullCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='images_thermal_train/flir_train.json', img_dir='images_thermal_train/data/', has_labels=True),
        val=dict(ann_filename='images_thermal_train/flir_val.json', img_dir='images_thermal_train/data/', has_labels=True),
        test=dict(ann_filename='images_thermal_test/flir.json', img_dir='images_thermal_test/data/', has_labels=True),
    ))


@dataclass
class FlirAlignedDayCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='images_thermal_train_day/day_flir_train.json', img_dir='images_thermal_train_day/data/', has_labels=True),
        val=dict(ann_filename='images_thermal_train_day/day_flir_val.json', img_dir='images_thermal_train_day/data/', has_labels=True),
        test=dict(ann_filename='images_thermal_test_day/day_flir.json', img_dir='images_thermal_test_day/data/', has_labels=True),
    ))



@dataclass
class FlirAlignedNightCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='images_thermal_train_night/night_flir_train.json', img_dir='images_thermal_train_night/data/', has_labels=True),
        val=dict(ann_filename='images_thermal_train_night/night_flir_val.json', img_dir='images_thermal_train_night/data/', has_labels=True),
        test=dict(ann_filename='images_thermal_test_night/night_flir.json', img_dir='images_thermal_test_night/data/', has_labels=True),
    ))

# M3FD Dataset
@dataclass
class M3fdDayCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-daytime-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-daytime-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-daytime-test.json', img_dir='Ir', has_labels=True)
    ))


@dataclass
class M3fdNightCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-night-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-night-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-night-test.json', img_dir='Ir', has_labels=True)
    ))


@dataclass
class M3fdOvercastCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-overcast-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-overcast-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-overcast-test.json', img_dir='Ir', has_labels=True)
    ))

@dataclass
class M3fdChallengeCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-challenge-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-challenge-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-challenge-test.json', img_dir='Ir', has_labels=True)
    ))

@dataclass
class M3fdFullCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-test.json', img_dir='Ir', has_labels=True)
    ))

# Seeing Through Fog Dataset

@dataclass
class StfClearDayCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/all/stf-clear_day-train.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        val=dict(ann_filename='meta/all/stf-clear_day-val.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        test=dict(ann_filename='meta/all/stf-clear_day-test.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
    ))

@dataclass
class StfClearNightCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/all/stf-clear_night-train.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        val=dict(ann_filename='meta/all/stf-clear_night-val.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        test=dict(ann_filename='meta/all/stf-clear_night-test.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
    ))

@dataclass
class StfFogDayCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/all/stf-fog_day-train.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        val=dict(ann_filename='meta/all/stf-fog_day-val.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        test=dict(ann_filename='meta/all/stf-fog_day-test.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
    ))

@dataclass
class StfFogNightCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/all/stf-fog_night-train.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        val=dict(ann_filename='meta/all/stf-fog_night-val.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        test=dict(ann_filename='meta/all/stf-fog_night-test.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
    ))

@dataclass
class StfSnowDayCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/all/stf-snow_day-train.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        val=dict(ann_filename='meta/all/stf-snow_day-val.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        test=dict(ann_filename='meta/all/stf-snow_day-test.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
    ))



@dataclass
class StfSnowNightCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/all/stf-snow_night-train.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        val=dict(ann_filename='meta/all/stf-snow_night-val.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        test=dict(ann_filename='meta/all/stf-snow_night-test.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
    ))

@dataclass
class StfRainCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/all/stf-rain-train.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        val=dict(ann_filename='meta/all/stf-rain-val.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        test=dict(ann_filename='meta/all/stf-rain-test.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
    ))

@dataclass
class StfFullCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/STF/all/stf-full-train.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        val=dict(ann_filename='meta/STF/all/stf-full-val.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
        test=dict(ann_filename='meta/STF/all/stf-full-test.json', img_dir='gated_full_acc_rect_aligned', has_labels=True),
    ))



@dataclass
class LLVIPCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='llvip_train.json', img_dir='infrared/train/', has_labels=True),
        val=dict(ann_filename='llvip_test.json', img_dir='infrared/test/', has_labels=True),
        test=dict(ann_filename='llvip_test.json', img_dir='infrared/test/', has_labels=True),
    ))