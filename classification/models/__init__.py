#Tiny
from .Deformba_T import Deformba
# from .Deformba_SB import Deformba
def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["Deformba"]:
        model = Deformba(
            in_chans=config.MODEL.Deformba.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.Deformba.DEPTHS,
            dims=config.MODEL.Deformba.EMBED_DIM,
            mlp_ratios=config.MODEL.Deformba.MLP_RATIO,
            head_dim=config.MODEL.Deformba.HEAD_DIM,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layerscale=config.MODEL.Deformba.LAYERSCALE
        )
        return model
    return None
