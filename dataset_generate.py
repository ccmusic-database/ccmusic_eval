import os
import csv
import random
import librosa
import datasets
import numpy as np
from tqdm import tqdm
from glob import glob

_NAMES = {
    "chanyin": 0,
    "dianyin": 6,
    "shanghua": 2,
    "xiahua": 3,
    "huazhi": 4,
    "guazou": 4,
    "lianmo": 4,
    "liantuo": 4,
    "yaozhi": 5,
    "boxian": 1,
}

_NAME = [
    "chanyin",  # Vibrato
    "boxian",  # Plucks
    "shanghua",  # Upward Portamento
    "xiahua",  # Downward Portamento
    "huazhi/guazou/lianmo/liantuo",  # Glissando
    "yaozhi",  # Tremolo
    "dianyin",  # Point Note
]

_HOMEPAGE = f"https://www.modelscope.cn/datasets/ccmusic-database/{os.path.basename(__file__)[:-3]}"

_DOMAIN = f"{_HOMEPAGE}/resolve/master/data"

_URLS = {
    "audio": f"{_DOMAIN}/audio.zip",
    "mel": f"{_DOMAIN}/mel.zip",
    "label": f"{_DOMAIN}/label.zip",
}

_TIME_LENGTH = 3  # seconds
_SAMPLE_RATE = 44100
_HOP_LENGTH = 512  # SAMPLE_RATE * ZHEN_LENGTH // 1000


class Guzheng_Tech99(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=(
                datasets.Features(
                    {
                        "audio": datasets.Audio(sampling_rate=44100),
                        "mel": datasets.Image(),
                        "label": datasets.Sequence(
                            feature={
                                "onset_time": datasets.Value("float32"),
                                "offset_time": datasets.Value("float32"),
                                "IPT": datasets.ClassLabel(num_classes=7, names=_NAME),
                                "note": datasets.Value("int8"),
                            }
                        ),
                    }
                )
                if self.config.name == "default"
                else datasets.Features(
                    {
                        "mel": datasets.features.Array3D(
                            dtype="float32", shape=(128, 258, 1)
                        ),
                        "cqt": datasets.features.Array3D(
                            dtype="float32", shape=(88, 258, 1)
                        ),
                        "chroma": datasets.features.Array3D(
                            dtype="float32", shape=(12, 258, 1)
                        ),
                        "label": datasets.features.Array2D(
                            dtype="float32", shape=(7, 258)
                        ),
                    }
                )
            ),
            homepage=_HOMEPAGE,
            license="CC-BY-NC-ND",
            version="1.2.0",
        )

    def _RoW_norm(self, data):
        common_sum = 0
        square_sum = 0
        tfle = 0
        for i in range(len(data)):
            tfle += (data[i].sum(-1).sum(0) != 0).astype("int").sum()
            common_sum += data[i].sum(-1).sum(-1)
            square_sum += (data[i] ** 2).sum(-1).sum(-1)

        common_avg = common_sum / tfle
        square_avg = square_sum / tfle
        std = np.sqrt(square_avg - common_avg**2)
        return common_avg, std

    def _norm(self, data):
        size = data.shape
        avg, std = self._RoW_norm(data)
        avg = np.tile(avg.reshape((1, -1, 1, 1)), (size[0], 1, size[2], size[3]))
        std = np.tile(std.reshape((1, -1, 1, 1)), (size[0], 1, size[2], size[3]))
        return (data - avg) / std

    def _load(self, wav_dir, csv_dir, groups):
        def files(wav_dir, csv_dir, group):
            flacs = sorted(glob(os.path.join(wav_dir, group, "*.flac")))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(wav_dir, group, "*.wav")))

            csvs = sorted(glob(os.path.join(csv_dir, group, "*.csv")))
            files = list(zip(flacs, csvs))
            if len(files) == 0:
                raise RuntimeError(f"Group {group} is empty")

            result = []
            for audio_path, csv_path in files:
                result.append((audio_path, csv_path))

            return result

        def logMel(y, sr=_SAMPLE_RATE):
            # 帧长为32ms (1000ms/(16000/512) = 32ms), D2的频率是73.418
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                hop_length=_HOP_LENGTH,
                fmin=27.5,
            )
            return (
                (1.0 / 80.0) * librosa.core.amplitude_to_db(np.abs(mel), ref=np.max)
            ) + 1.0

        # Returns the CQT of the input audio
        def logCQT(y, sr=_SAMPLE_RATE):
            # 帧长为32ms (1000ms/(16000/512) = 32ms), D2的频率是73.418
            cqt = librosa.cqt(
                y,
                sr=sr,
                hop_length=_HOP_LENGTH,
                fmin=27.5,
                n_bins=88,
                bins_per_octave=12,
            )
            return (
                (1.0 / 80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)
            ) + 1.0

        def logChroma(y, sr=_SAMPLE_RATE):
            # 帧长为32ms (1000ms/(16000/512) = 32ms), D2的频率是73.418
            chroma = librosa.feature.chroma_stft(
                y=y,
                sr=sr,
                hop_length=_HOP_LENGTH,
            )
            return (
                (1.0 / 80.0) * librosa.core.amplitude_to_db(np.abs(chroma), ref=np.max)
            ) + 1.0

        def chunk_data(f):
            x = []
            xdata = np.transpose(f)
            s = _SAMPLE_RATE * _TIME_LENGTH // _HOP_LENGTH
            length = int(np.ceil((int(len(xdata) / s) + 1) * s))
            app = np.zeros((length - xdata.shape[0], xdata.shape[1]))
            xdata = np.concatenate((xdata, app), 0)
            for i in range(int(length / s)):
                data = xdata[int(i * s) : int(i * s + s)]
                x.append(np.transpose(data[:s, :]))

            return np.array(x)

        def load_all(audio_path, csv_path, hop=_HOP_LENGTH, n_IPTs=7, technique=_NAMES):
            # Load audio features: The shape of cqt (88, 8520), 8520 is the number of frames on the time axis
            y, sr = librosa.load(audio_path, sr=_SAMPLE_RATE)
            mel = logMel(y, sr)
            cqt = logCQT(y, sr)
            chroma = logChroma(y, sr)
            # Load the ground truth label
            n_steps = cqt.shape[1]
            IPT_label = np.zeros([n_IPTs, n_steps], dtype=int)
            with open(csv_path, "r", encoding="utf-8") as f:  # csv file for each audio
                reader = csv.DictReader(f, delimiter=",")
                for label in reader:  # each note
                    onset = float(label["onset_time"])
                    offset = float(label["offset_time"])
                    IPT = int(technique[label["IPT"]])
                    left = int(round(onset * _SAMPLE_RATE / hop))
                    frame_right = int(round(offset * _SAMPLE_RATE / hop))
                    frame_right = min(n_steps, frame_right)
                    IPT_label[IPT, left:frame_right] = 1

            return dict(
                audio_path=audio_path,
                csv_path=csv_path,
                mel=mel,
                cqt=cqt,
                chroma=chroma,
                IPT_label=IPT_label,
            )

        data = []
        # print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} ")
        for group in groups:
            for input_files in files(wav_dir, csv_dir, group):
                data.append(load_all(*input_files))

        for i, dic in tqdm(enumerate(data), total=len(data), desc="Feature extracting"):
            x_mel = chunk_data(dic["mel"])
            x_cqt = chunk_data(dic["cqt"])
            x_chroma = chunk_data(dic["chroma"])
            y_i = dic["IPT_label"]
            y_i = chunk_data(y_i)
            if i == 0:
                Xtr_mel = x_mel
                Xtr_cqt = x_cqt
                Xtr_chroma = x_chroma
                Ytr_i = y_i

            else:
                Xtr_mel = np.concatenate([Xtr_mel, x_mel], axis=0)
                Xtr_cqt = np.concatenate([Xtr_cqt, x_cqt], axis=0)
                Xtr_chroma = np.concatenate([Xtr_chroma, x_chroma], axis=0)
                Ytr_i = np.concatenate([Ytr_i, y_i], axis=0)

        # Transform the shape of the input
        Xtr_mel = np.expand_dims(Xtr_mel, axis=3)
        Xtr_cqt = np.expand_dims(Xtr_cqt, axis=3)
        Xtr_chroma = np.expand_dims(Xtr_chroma, axis=3)
        # Normalize
        Xtr_mel = self._norm(Xtr_mel)
        Xtr_cqt = self._norm(Xtr_cqt)
        Xtr_chroma = self._norm(Xtr_chroma)
        return [list(Xtr_mel), list(Xtr_cqt), list(Xtr_chroma)], list(Ytr_i)

    def _parse_csv_label(self, csv_file):
        label = []
        with open(csv_file, mode="r", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                label.append(
                    {
                        "onset_time": float(row["onset_time"]),
                        "offset_time": float(row["offset_time"]),
                        "IPT": _NAME[_NAMES[row["IPT"]]],
                        "note": int(row["note"]),
                    }
                )

        return label

    def _split_generators(self, dl_manager):
        audio_files = dl_manager.download_and_extract(_URLS["audio"])
        csv_files = dl_manager.download_and_extract(_URLS["label"])
        trainset, validset, testset = [], [], []
        if self.config.name == "default":
            files = {}
            mel_files = dl_manager.download_and_extract(_URLS["mel"])
            for path in dl_manager.iter_files([audio_files]):
                fname: str = os.path.basename(path)
                if fname.endswith(".flac"):
                    item_id = fname.split(".")[0]
                    files[item_id] = {"audio": path}

            for path in dl_manager.iter_files([mel_files]):
                fname = os.path.basename(path)
                if fname.endswith(".jpg"):
                    item_id = fname.split(".")[0]
                    files[item_id]["mel"] = path

            for path in dl_manager.iter_files([csv_files]):
                fname = os.path.basename(path)
                if fname.endswith(".csv"):
                    item_id = fname.split(".")[0]
                    files[item_id]["label"] = self._parse_csv_label(path)

            for item in files.values():
                if "train" in item["audio"]:
                    trainset.append(item)

                elif "validation" in item["audio"]:
                    validset.append(item)

                elif "test" in item["audio"]:
                    testset.append(item)

        else:
            audio_dir = audio_files + "\\audio"
            csv_dir = csv_files + "\\label"
            X_train, Y_train = self._load(audio_dir, csv_dir, ["train"])
            X_valid, Y_valid = self._load(audio_dir, csv_dir, ["validation"])
            X_test, Y_test = self._load(audio_dir, csv_dir, ["test"])
            for i in range(len(Y_train)):
                trainset.append(
                    {
                        "mel": X_train[0][i],
                        "cqt": X_train[1][i],
                        "chroma": X_train[2][i],
                        "label": Y_train[i],
                    }
                )

            for i in range(len(Y_valid)):
                validset.append(
                    {
                        "mel": X_valid[0][i],
                        "cqt": X_valid[1][i],
                        "chroma": X_valid[2][i],
                        "label": Y_valid[i],
                    }
                )

            for i in range(len(Y_test)):
                testset.append(
                    {
                        "mel": X_test[0][i],
                        "cqt": X_test[1][i],
                        "chroma": X_test[2][i],
                        "label": Y_test[i],
                    }
                )

        random.shuffle(trainset)
        random.shuffle(validset)
        random.shuffle(testset)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"files": trainset}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"files": validset}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"files": testset}
            ),
        ]

    def _generate_examples(self, files):
        for i, path in enumerate(files):
            yield i, path