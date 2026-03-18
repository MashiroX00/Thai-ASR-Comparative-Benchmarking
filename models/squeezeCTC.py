import nemo.collections.asr as nemo_asr
from utils.CheckCompatibilities import check_gpu_compatibility
from utils.logger import get_logger

logger = get_logger("squeeze_ctc")

def main():
    model_name = "stt_en_squeezeformer_ctc_small_ls"
    device = "cuda" if check_gpu_compatibility() else "cpu"

    logger.info(f"Loading model '{model_name}' from NVIDIA NeMo...")
    try:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=model_name,
            map_location=device,
        )
        logger.info(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    main()