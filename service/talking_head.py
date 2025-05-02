from service.ditto.inference import StreamSDK, run


class Ditto(object):
    def __init__(self, data_root: str, cfg_pkl: str) -> None:
        self.data_root = data_root
        self.cfg_pkl = cfg_pkl
        self.sdk = StreamSDK(cfg_pkl, data_root)

    def run(self, audio_path: str, source_path: str, output_path: str):
        run(self.sdk, audio_path, source_path, output_path)


if __name__ == '__main__':
    data_root = 'data/weights/ditto/ditto_trt_Ampere_Plus'
    cfg_pkl = 'data/weights/ditto/ditto_cfg/v0.4_hubert_cfg_trt.pkl'
    ditto = Ditto(data_root, cfg_pkl)

    audio_path = 'output.wav'  # .wav
    source_path = 'image.png'  # video|image
    output_path = 'result.mp4'  # .mp4
