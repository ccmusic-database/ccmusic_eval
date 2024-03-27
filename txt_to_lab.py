"""
Transform .txt to .lab

    original label present in MSA_dataset\\annotations is in .txt
    but mir_eval need .lab
"""

import os
import core
import jams


def annotations():
    path = os.path.join(core.DATASET_DIR, "annotations")
    save_path = os.path.join(core.DATASET_DIR, "labAnnotations")
    file_list = os.listdir(path)

    for i in range(len(file_list)):
        file_path = os.path.join(path, file_list[i])
        transformed_data = []
        with open(file_path, "r") as f:
            for line in f:
                # Split each line by tabs or spaces
                parts = line.split()
                # Extract and convert the components to the appropriate data types
                st = str(parts[0])
                start_time = str(st[:-2] + "." + st[-2:])
                et = str(parts[1])
                end_time = str(et[:-2] + "." + et[-2:])
                label = (parts[2]).strip('"')  # Remove double quotes from the label
                st = start_time
                transformed_data.append(f"{start_time}\t{end_time}\t{label}")

        output_file_path = os.path.join(save_path, file_list[i].replace(".txt", ".lab"))
        with open(output_file_path, "w") as output_file:
            output_file.write("\n".join(transformed_data))

        print(f"file saved: {file_list[i]}")


def estimations():
    path = os.path.join(core.DATASET_DIR, "annotations")
    save_path = os.path.join(core.DATASET_DIR, "labAnnotations")
    os.makedirs(save_path, exist_ok=True)
    file_list = os.listdir(path)

    for i in range(len(file_list)):
        file = os.path.join(path, file_list[i])
        jam = jams.load(file)
        # Find the specific segmentation result with 'annot_beats': true
        desired_result = None
        save_data = []
        for annotation in jam.annotations:
            sandbox = annotation.sandbox  # Access the sandbox dictionary
            if "annot_beats" in sandbox and sandbox["annot_beats"] == True:
                desired_result = annotation
                break

        if desired_result:
            data = desired_result.data
            for obs in data:
                start_time = obs.time
                value = obs.value
                if obs.duration > 0.1:
                    end_time = start_time + obs.duration
                    save_data.append(f"{start_time:.2f}\t{end_time:.2f}\t{value}")
                # Process the segment as needed
                output_file_path = os.path.join(
                    save_path, file_list[i].replace(".jams", ".lab")
                )
        else:
            raise ValueError("Desired segmentation result not found in the JAMS file.")

        with open(output_file_path, "w") as output_file:
            output_file.write("\n".join(save_data))

        print(f"file saved: {file_list[i]}")


if __name__ == "__main__":
    # first transform .txt to .lab in \\annotations then in \\estimations
    annotations()
    estimations()
