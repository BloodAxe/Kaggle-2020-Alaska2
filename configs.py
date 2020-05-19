import json

import jsonpickle

from alaska2.config import ExperimenetConfig, StageConfig

cfg = ExperimenetConfig()
cfg.model_name = "ela_skresnext50_32x4d"
cfg.dropout = 0.2
cfg.num_workers = 16
cfg.fold = 0

main = StageConfig()
main.epochs = 75
main.optimizer = "Ranger"
main.schedule = "flat_cos"
main.fp16 = True
main.modification_flag_loss = [["ce", 0.1]]
main.modification_type = [["bce", 1]]

cfg.stages = [main]

jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
with open("configs/ela_skresnext50_32x4d_fold0.json", "w") as f:
    f.write(jsonpickle.encode(cfg, make_refs=False))
