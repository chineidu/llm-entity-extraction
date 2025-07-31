#!/usr/bin/env python3
"""
Script to fix malformed CSV files with various data corruption issues.
Handles both extra column issues and misaligned txnId data.
"""

import csv
import sys
from pathlib import Path


def fix_csv_file(input_path: str, output_path: str) -> None:
    """
    Fix CSV files by handling various types of data corruption:
    1. Rows with extra columns (text duplicated, entities shifted)
    2. Rows with misaligned txnId data (text in txnId column)

    Args:
        input_path: Path to the input CSV file
        output_path: Path to the output fixed CSV file
    """
    fixed_rows = []
    problematic_rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_cols = len(header)
        print(f"Header: {header}")
        print(f"Expected columns: {expected_cols}")

        for i, row in enumerate(reader, 2):  # Start from line 2
            if len(row) == expected_cols:
                # Check for misaligned txnId data (batch_2 type issue)
                try:
                    int(row[1])  # Try to convert txnId to int
                    # Also check if entities column is valid JSON
                    entities_str = row[3]  # entities column
                    if entities_str and not entities_str.startswith("["):
                        # This row has text in entities column instead of JSON
                        problematic_rows.append((i, "Invalid JSON in entities", row))
                        continue
                    # Valid row
                    fixed_rows.append(row)
                except ValueError:
                    # txnId contains text instead of number - data is misaligned (batch_2 type)
                    try:
                        generated_txnid = int(row[0]) + 20000  # Offset to avoid conflicts
                        fixed_row = [
                            row[0],  # id
                            str(generated_txnid),  # generated txnId
                            row[1],  # text (originally in txnId position)
                            row[3],  # entities
                            row[4],  # analysisId
                            row[5],  # createdAt
                        ]
                        fixed_rows.append(fixed_row)
                        problematic_rows.append((i, "Fixed misaligned txnId", row))
                    except (ValueError, IndexError):
                        problematic_rows.append((i, "Skipped - couldn't fix txnId", row))
            elif len(row) == expected_cols + 1:
                # Row has extra column - entities and text are swapped (batch_3 type issue)
                # Structure: id, txnId, text, text_again, entities, analysisId, createdAt
                fixed_row = [
                    row[0],  # id
                    row[1],  # txnId
                    row[2],  # text (use first text field)
                    row[4],  # entities (moved from position 4)
                    row[5],  # analysisId
                    row[6],  # createdAt
                ]
                fixed_rows.append(fixed_row)
                problematic_rows.append((i, "Extra column fixed", row))
            else:
                # Other issues
                problematic_rows.append((i, f"Wrong column count: {len(row)}", row))

    # Write the fixed CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(fixed_rows)

    print(f"\nFixed {len(fixed_rows)} rows")
    print(f"Found {len(problematic_rows)} problematic rows")

    # Show some problematic rows for verification
    print("\nFirst 5 problematic rows:")
    for _, (line_num, issue, row) in enumerate(problematic_rows[:5]):
        print(f"Line {line_num}: {issue}")
        print(f"  Original: {row[:4]}...")


def main() -> None:
    """Main function to handle command line arguments and process files."""
    if len(sys.argv) != 3:
        print("Usage: python fix_bad_csv.py <input_file> <output_file>")
        print("Example: python fix_bad_csv.py data/batch_3.csv data/batch_3_fixed.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    print(f"Fixing CSV file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print("-" * 50)

    try:
        fix_csv_file(input_file, output_file)
        print("\n✅ Successfully fixed CSV file!")
        print(f"Fixed file saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error fixing CSV file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # If run without arguments, use default files for backward compatibility
    if len(sys.argv) == 1:
        print("No arguments provided. Available CSV files to fix:")
        data_dir = Path("data/data")
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            for i, csv_file in enumerate(csv_files, 1):
                if not csv_file.name.endswith("_fixed.csv"):
                    print(f"  {i}. {csv_file}")
        print("\nUsage: python fix_bad_csv.py <input_file> <output_file>")
        print("Example: python fix_bad_csv.py data/data/batch_3.csv data/data/batch_3_fixed.csv")
    else:
        main()
