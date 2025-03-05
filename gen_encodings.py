#!/usr/bin/env python
import argparse
import torch
import esm
from Bio import SeqIO  # For reading FASTA files

import sys

def main():
    parser = argparse.ArgumentParser(
        description="Generate protein encodings for each sequence in a FASTA file and output the protein id, sequence, and encoding."
    )
    parser.add_argument("input_fasta", help="FASTA file containing protein sequences (headers should have unique protein IDs)")
    parser.add_argument("output_file", help="Output file to store protein ID, sequence, and encoding")
    parser.add_argument("model_file", help="Path to the locally downloaded ESM model file")
    args = parser.parse_args()

    # Set device (use GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 🔹 Load the locally stored ESM model and alphabet
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.model_file)
        model = model.to(device)
        batch_converter = alphabet.get_batch_converter()
        model.eval()
    except Exception as e:
        print(f"Error loading model from {args.model_file}: {e}")
        sys.exit(1)

    # Read the FASTA file
    protein_data = []
    try:
        for record in SeqIO.parse(args.input_fasta, "fasta"):
            protein_id = record.id
            sequence = str(record.seq)
            protein_data.append((protein_id, sequence))
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        sys.exit(1)

    if not protein_data:
        print("No data to process. Exiting.")
        sys.exit(0)

    print(f"Processing {len(protein_data)} protein sequences")

    output_lines = []
    for index, (protein_id, sequence) in enumerate(protein_data):
        # Generate encoding only if the sequence length is below threshold.
        if len(sequence) < 1022:
            single_protein = [(protein_id, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(single_protein)
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33].cpu()
            # Mean-pool the token representations (excluding special tokens)
            encoding = token_representations[0, 1: batch_lens[0]-1].mean(0)
            encoding_str = " ".join(str(val.item()) for val in encoding)
        else:
            encoding_str = "NA"

        output_lines.append((protein_id, sequence, encoding_str))

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(protein_data)} sequences")

    # Write output: protein ID, sequence, and encoding (tab-separated)
    with open(args.output_file, "w") as out_f:
        for pid, seq, enc in output_lines:
            out_f.write(f"{pid}\t{seq}\t{enc}\n")

    print(f"Finished processing. Encodings saved to {args.output_file}")

if __name__ == "__main__":
    main()
