import os
import shutil


def move_files(source_folder, destination_folder):
    try:
        # Check if the source and destination folders exist
        if not os.path.exists(source_folder):
            print("Source folder does not exist.")
            return
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Get a list of all files in the source folder
        files_to_move = [
            f
            for f in os.listdir(source_folder)
            if os.path.isfile(os.path.join(source_folder, f))
        ]

        # Move each file to the destination folder
        for file_name in files_to_move:
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(
                destination_folder, file_name
            )
            shutil.move(source_path, destination_path)
            print(f"Moved: {file_name}")

        print("All files moved successfully.")

    except Exception as e:
        print("An error occurred:", str(e))


if __name__ == "__main__":
    source_folder = ""
    destination_folder = ""
    move_files(source_folder, destination_folder)
