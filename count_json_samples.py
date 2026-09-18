import argparse
import json
from pathlib import Path

import pandas as pd


CLASS_NAME_MAP = {
	"vc-open": "vocal-throat",
	"vc-closed": "vocal-throat",
	"nose-left": "nose",
	"nose-right": "nose",
	"ear-left": "ear",
	"ear-right": "ear",
	"throat": "throat",
}

CLASS_COLUMNS = ["vocal-throat", "nose", "ear", "throat"]


def build_row(record, images_dir, image_prefix):
	raw_label = record["Classification"]
	class_name = CLASS_NAME_MAP.get(raw_label)
	if class_name is None:
		raise ValueError(f"Unsupported Classification value: {raw_label}")

	image_name = record["Path"]
	if not (images_dir / image_name).exists():
		raise FileNotFoundError(f"Image file not found: {images_dir / image_name}")

	row = {
		"image_path": f"{image_prefix}/{image_name}".replace("\\", "/"),
		"vocal-throat": 0,
		"nose": 0,
		"ear": 0,
		"throat": 0,
		"text": f"a photo of {class_name}",
	}
	row[class_name] = 1
	return row


def convert_json_to_csv(json_path, output_csv_path, images_dir, image_prefix):
	with json_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError("Expected top-level JSON list of records")

	rows = [build_row(record, images_dir, image_prefix) for record in payload]
	df = pd.DataFrame(rows, columns=["image_path", *CLASS_COLUMNS, "text"])
	df.to_csv(output_csv_path, index=True)
	return len(df)


def main():
	parser = argparse.ArgumentParser(description="Convert ENTREP-style JSON annotations to entrep_data.csv format.")
	parser.add_argument(
		"--json_path",
		type=Path,
		default=Path(r"D:\Gradient_based_ES_for_Attack_MedVLM\local_data\entrep_test\data.json"),
		help="Path to the source JSON annotation file",
	)
	parser.add_argument(
		"--output_csv_path",
		type=Path,
		default=Path(r"D:\Gradient_based_ES_for_Attack_MedVLM\local_data\entrep_test\entrep_data.csv"),
		help="Path to the output CSV file",
	)
	parser.add_argument(
		"--images_dir",
		type=Path,
		default=Path(r"D:\Gradient_based_ES_for_Attack_MedVLM\local_data\entrep_test\images"),
		help="Directory containing the source images",
	)
	parser.add_argument(
		"--image_prefix",
		type=str,
		default="local_data/entrep_test/images",
		help="Relative image path prefix written into the CSV",
	)
	args = parser.parse_args()

	args.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
	count = convert_json_to_csv(
		json_path=args.json_path,
		output_csv_path=args.output_csv_path,
		images_dir=args.images_dir,
		image_prefix=args.image_prefix,
	)

	print(f"Wrote {count} samples to {args.output_csv_path}")


if __name__ == "__main__":
	main()