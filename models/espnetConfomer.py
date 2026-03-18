import numpy as np
if not hasattr(np, "int"):
    np.int = int
    np.float = float
    np.bool = bool
import espnet_model_zoo.downloader as espnet_downloader
import espnet2.bin.asr_inference as espnet2
from utils.CheckCompatibilities import check_gpu_compatibility
from utils.logger import get_logger

logger = get_logger("espnet_conformer")

def main():

    d = espnet_downloader.ModelDownloader()
    model_name = "espnet/thai_commonvoice_blstm"

    model_config = d.download_and_unpack(model_name)
    device = "cuda" if check_gpu_compatibility() else "cpu"

    logger.info(f"Loading model '{model_name}' from ESPnet2...")
    try:
        model = espnet2.Speech2Text(
            **model_config,
            device=device,
            minlenratio=0.0,
            maxlenratio=0.0,
            ctc_weight=0.3,
            beam_size=10,
            batch_size=1
        )
        logger.info(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    main()