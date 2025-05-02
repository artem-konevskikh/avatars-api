import os  # Added import

import torch

from src.services.csm.csm import CSM
from src.services.csm.utils import AudioFormat

# Determine device based on CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Removed @patch decorators for load_model and _save_audio
def test_csm_generate_saves_wav() -> None:  # Removed mock_save_audio argument
    """Test if CSM.generate correctly generates and saves a WAV file.

    This test requires actual model loading and generation.
    """
    # Minimal mock config sufficient for initialization and generate call
    MOCK_CONFIG = {
        'model_type': 'mock_type',  # Keep as mock for config, actual model loaded below
        'device': device,  # Use determined device
        'local_model_path': None,
        'audio_num_codebooks': 1,
        'optimize_for_streaming': False,
        'stream_buffer_size': 1024,
        'compile_model': False,
        'cache_size': 128,
        'default_speaker_id': 0,
        'temperature': 0.7,
        'topk': 5,
        'max_audio_length_ms': 5000,  # Reduced for faster testing
    }

    # Ensure the output directory exists or create it
    output_dir = 'test_outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, 'test_output.wav')

    # Clean up previous test file if it exists
    if os.path.exists(output_filename):
        os.remove(output_filename)

    csm_instance = CSM(config=MOCK_CONFIG)  # type: ignore
    # No need to mock generator or set _is_loaded, load_model will be called by generate

    test_text = 'Hello world, this is a test.'

    # Call the generate method requesting a WAV file output
    # This will now load the model and perform actual generation
    csm_result = csm_instance.generate(text=test_text, output_file=output_filename, output_format=AudioFormat.WAV)

    # Assertions
    assert csm_result is None, 'Generate should return None when output_file is specified.'
    # Check if the output file was actually created
    assert os.path.exists(output_filename), f'Output file {output_filename} was not created.'
    assert os.path.getsize(output_filename) > 0, f'Output file {output_filename} is empty.'

    # Optional: Clean up the generated file after test
    # os.remove(output_filename)
