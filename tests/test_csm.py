import pytest
import torch  # Added import

from src.services.csm.csm import CSM

# Determine device based on CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_csm_initialization() -> None:
    """Test if the CSM class can be initialized successfully with a basic
    config."""
    # Minimal mock config sufficient for initialization
    # Keys accessed by __init__ or methods called soon after (like setup_optimizations)
    # or potentially by logging within __init__ should be included.
    MOCK_CONFIG = {
        'model_type': 'mock_type',  # Needed for logging in load_model, though not called by init
        'device': device,  # Use determined device
        'local_model_path': None,
        'audio_num_codebooks': 1,  # Needed for ModelArgs in load_model
        'optimize_for_streaming': False,
        'stream_buffer_size': 1024,
        'compile_model': False,
        'cache_size': 128,
        'default_speaker_id': 0,
        'temperature': 0.7,
        'topk': 5,
        'max_audio_length_ms': 30000,
        # Add any other keys potentially accessed early
    }
    try:
        csm_instance = CSM(config=MOCK_CONFIG)  # type: ignore
    except Exception as err:
        pytest.fail(f'CSM initialization raised an unexpected exception: {err}')

    assert isinstance(csm_instance, CSM), 'Failed to create an instance of CSM.'
    assert csm_instance.config == MOCK_CONFIG, 'CSM instance did not store the config correctly.'
    assert not csm_instance._is_loaded, 'CSM instance should initially be marked as not loaded.'
