import msaf
import os
import librosa
import IPython.display
import core
import jams


def process_single():
    audio_dir = os.path.join(core.DATASET_DIR, "audio")
    file_list = os.listdir(audio_dir)
    for i in range(len(file_list)):

        path = os.path.join(audio_dir, file_list[i])
        path = str(path)
        print("-----processing file: ", path)
        """
        Returns
        -------
        results : list
            List containing tuples of (est_times, est_labels) of estimated
            boundary times and estimated labels.
            If labels_id is None, est_labels will be a list of -1.           
        """

        # the beat information will be read from: MSA_dataset \\ 'references'
        # save to: MSA_dataset \\ 'estimations'
        msaf.process(
            path, feature="cqt", boundaries_id="sf", labels_id="fmc2d", annot_beats=True
        )


if __name__ == "__main__":
    # track the beat for all files and save in references folder
    process_single()
