import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
import pickle

def collate_fn(batch):
    """
    Manually build a batch from a list to a tensor
    :param batch: a list of tensors
    :return: 
        list_wav: a list of wav with length B , [wav1, wav2, wav3, ...]
        list_fr: a list of sample rates with length B, [sr1, sr2, sr3, ...]
        batch_spec: a tensor with shape of (B, max_len, num_fbank, 1)
        batch_tvs: a tensor with shape of (B, max_len, num_tvs_dim)
        batch_track: a tensor with shape of (B, num_track_dim)
        batch_seq_len: a tensor with shape of (B, )
    """
    # list_wav, list_sr, list_spec, list_tvs, list_track, list_seq_len = map(list, zip(*batch))
    # batch_wav_len = torch.tensor([waveform.shape[0] for waveform in list_wav])
    # batch_wav = pad_sequence(list_wav, batch_first=True)
    # batch_sr = torch.stack(list_sr)
    # batch_spec = pad_sequence(list_spec, batch_first=True).unsqueeze(-1)
    # batch_tvs = pad_sequence(list_tvs, batch_first=True)
    # batch_track = torch.stack(list_track)
    # batch_seq_len = torch.stack(list_seq_len)

    # return batch_wav, batch_wav_len, batch_sr, batch_spec, batch_tvs, batch_track, batch_seq_len

    spec_list = map(list, zip(*batch))
    batch_spec = pad_sequence(spec_list, batch_first=True).unsqueeze(-1)
    return batch_spec

class SingDataset(DatasetFolder):
    '''
    Dataset with preprocessed audios and trm_params as input: processed_fps
    '''
    def __init__(self, root):
        '''
            since r0 of tvs always have same value, no need to train it
        '''
        super(DatasetFolder, self).__init__()

    def __getitem__(self, index):
        with open(self.data[index], 'rb') as item:
            spec = pickle.load(item)

        return torch.tensor(spec, dtype=torch.float32)

    def __len__(self):
        return len(self.data)