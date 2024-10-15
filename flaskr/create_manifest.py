import os
storage_dir = "Surveys/"

manifest_dict = {}

survey_folders = [sub_folder for sub_folder in os.listdir(storage_dir)
                  if os.path.isdir(os.path.join(storage_dir, sub_folder))]

for survey in survey_folders:
    survey_files = [sub_folder for sub_folder in os.listdir(os.path.join(storage_dir, survey))
                    if os.path.isdir(os.path.join(storage_dir,survey, sub_folder))]
    manifest_dict[survey] = survey_files

manifest_file = os.path.join(storage_dir, "_manifest.csv")
with open(manifest_file, "w") as f:
    for key in manifest_dict:
        new_line = [key]
        new_line.extend(manifest_dict[key])
        str_line = ",".join(new_line)
        f.write(str_line + "\n")
