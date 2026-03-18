import io
import os
import time
import torch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio
from jiwer import wer
from ptflops import get_model_complexity_info
from utils.logger import setup_logger

logger = setup_logger("main")

# Import models from models/ directory
try:
    from models.espnetConfomer import main as load_espnet
    from models.NemoAsrNvidia import main as load_fastconformer
    from models.squeezeCTC import main as load_squeezeformer
    from models.TyphoonAsr import main as load_typhoon_asr
except ImportError as e:
    logger.error(f"Error importing model modules: {e}")
    # Define dummy loaders if imports fail (should not happen in user's environment)
    load_espnet = load_fastconformer = load_squeezeformer = load_typhoon_asr = lambda: None

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    if hasattr(model, 'asr_model'): # ESPnet
        return sum(p.numel() for p in model.asr_model.parameters() if p.requires_grad)
    elif hasattr(model, 'parameters'): # NeMo / Standard PyTorch
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0

from fvcore.nn import FlopCountAnalysis

def estimate_gmucs(model, is_nemo=True):
    """Estimates GMACs using fvcore for more robustness."""
    try:
        class NeMoEncoderWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            def forward(self, x):
                lengths = torch.tensor([x.size(-1)]).to(x.device)
                return self.encoder(audio_signal=x, length=lengths)

        class ESPnetEncoderWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            def forward(self, x):
                ilens = torch.tensor([x.size(1)]).to(x.device)
                return self.encoder(x, ilens)

        target_module = None
        dummy_input = None

        if is_nemo:
            if hasattr(model, 'encoder'):
                target_module = NeMoEncoderWrapper(model.encoder)
                dummy_input = torch.randn(1, 80, 100)
        else:
            if hasattr(model, 'asr_model') and hasattr(model.asr_model, 'encoder'):
                target_module = ESPnetEncoderWrapper(model.asr_model.encoder)
                dummy_input = torch.randn(1, 100, 80)

        if target_module and dummy_input is not None:
            target_module.cpu().eval()
            flops = FlopCountAnalysis(target_module, dummy_input)
            # FLOPs / 2
            # Giga MACs = (Total FLOPs / 2) / 1e9
            return flops.total() / 2 / 1e9
    except Exception as e:
        logger.warning(f"Could not estimate GMACs: {e}")
    return 0.0

def benchmark_model(model_loader, model_label, dataset, num_samples=5):
    """Benchmarks a single model on a subset of the dataset."""
    logger.info(f"--- Benchmarking: {model_label} ---")
    model = model_loader()
    if model is None:
        logger.error(f"Failed to load {model_label}. Skipping.")
        return None

    param_count = count_parameters(model)
    is_nemo = hasattr(model, 'transcribe')
    gmucs = estimate_gmucs(model, is_nemo=is_nemo)
    
    logger.info(f"Model Parameters: {param_count:,}")
    logger.info(f"Estimated GMACs (1s audio): {gmucs:.4f}")

    results = []
    total_latency = 0
    total_duration = 0
    
    
    # Re-sample/Cast dataset is already done
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
            
        if sample['audio'] is None:
            logger.warning(f"Sample {i+1} audio is None. Skipping.")
            continue

        try:
            audio_bytes = sample['audio']['bytes']
            if audio_bytes is None:
                logger.warning(f"Sample {i+1} audio bytes are None. Skipping.")
                continue
                
            with io.BytesIO(audio_bytes) as f:
                audio_data, sampling_rate = sf.read(f)
            
            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data.T)
            
            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000

        except Exception as e:
            logger.error(f"Error decoding audio for sample {i+1}: {e}")
            continue

        reference_text = sample['sentence']
        
        # Audio duration for RTF calculation
        audio_duration = len(audio_data) / sampling_rate
        
        start_time = time.time()
        
        try:
            if is_nemo:
                # NeMo transcribe is optimized for batch/files, save to temp wav for simplicity
                temp_filename = f"temp_{model_label.replace(' ', '_').replace('(', '').replace(')', '')}.wav"
                sf.write(temp_filename, audio_data, sampling_rate)
                
                # transcribe returns a list of results (one per file)
                # In some versions it returns (hypotheses, metadata)
                result = model.transcribe([temp_filename])
                
                if isinstance(result, tuple):
                    result = result[0]
                
                transcription = result[0]
                
                # Handle Hypothesis objects
                if hasattr(transcription, 'text'):
                    transcription = transcription.text
                elif not isinstance(transcription, str):
                    transcription = str(transcription)
                
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            else:
                # ESPnet Speech2Text call
                with torch.no_grad():
                    nbests = model(audio_data)
                    transcription = nbests[0][0] # First hypothesis, first element (text)
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Metric calculation
            error_rate = wer(reference_text, transcription)
            rtf = latency / audio_duration if audio_duration > 0 else 0
            
            results.append({
                'WER': error_rate,
                'Latency (s)': latency,
                'RTF': rtf,
                'Text': transcription,
                'Ref': reference_text
            })
            
            total_latency += latency
            total_duration += audio_duration
            
            logger.info(f"Sample {i+1}/{num_samples} | RTF: {rtf:.4f} | WER: {error_rate:.4f}")
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")

    # Memory cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not results:
        return None

    # Calculate average metrics
    avg_wer = np.mean([r['WER'] for r in results])
    avg_rtf = total_latency / total_duration if total_duration > 0 else 0
    avg_latency = np.mean([r['Latency (s)'] for r in results])

    return {
        'Model': model_label,
        'Parameters': param_count,
        'GMACs (1s)': gmucs,
        'Avg WER': avg_wer,
        'Avg Latency (s)': avg_latency,
        'Avg RTF': avg_rtf,
        'Samples/sec': 1.0 / avg_rtf if avg_rtf > 0 else 0
    }


def main():
    # 1. Dataset Loading
    # Switching to TVSpeech as it has proper audio columns for benchmarking
    dataset_name = "typhoon-ai/TVSpeech"
    logger.info(f"Loading Dataset: {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split='test', streaming=True)
        
        # 2. Disable automatic decoding to avoid torchcodec dependency
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        logger.error(f"Could not load or cast dataset correctly: {e}")
        return

    # 3. Model Configurations
    models_to_test = [
        (load_espnet, "ESPnet BLSTM (Baseline)"),
        (load_fastconformer, "Fast-Conformer"),
        (load_squeezeformer, "Squeezeformer"),
        (load_typhoon_asr, "Typhoon ASR (Real-time)")
    ]

    # 4. Evaluation Loop
    summary_results = []
    num_test_samples = 5 # Set to a small number for benchmarking

    for loader, label in models_to_test:
        res = benchmark_model(loader, label, ds, num_samples=num_test_samples)
        if res:
            summary_results.append(res)

    # 5. Metrics Collection & Output
    if summary_results:
        df = pd.DataFrame(summary_results)
        
        # Display as Table
        logger.info("\n" + "="*50)
        logger.info("BENCHMARKING RESULTS")
        logger.info("="*50)
        logger.info("\n" + df.to_string(index=False))
        
        # Save to CSV
        output_file = "asr_benchmark_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Results successfully saved to {output_file}")
    else:
        logger.warning("No results were collected.")

if __name__ == "__main__":
    main()
