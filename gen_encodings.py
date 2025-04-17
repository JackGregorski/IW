#!/usr/bin/env python
import argparse
import torch
import esm
from Bio import SeqIO
import pandas as pd
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings for a subset of sequences defined by unique_proteins.tsv"
    )
    parser.add_argument("input_fasta", help="FASTA file with all protein sequences")
    parser.add_argument("unique_protein_file", help="TSV file with a 'protein' column listing IDs to process")
    parser.add_argument("output_file", help="TSV file to save protein embeddings")
    parser.add_argument("model_file", help="Path to local ESM model")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load ESM model
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.model_file)
        model = model.to(device)
        batch_converter = alphabet.get_batch_converter()
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load target protein IDs
    try:
        unique_df = pd.read_csv(args.unique_protein_file, sep='\t')
        target_ids = set(unique_df['protein'])
    except Exception as e:
        print(f"Error reading unique protein list: {e}")
        sys.exit(1)

    print(f"Loaded {len(target_ids)} unique protein IDs")

    # Read FASTA and filter
    protein_data = []
    try:
        for record in SeqIO.parse(args.input_fasta, "fasta"):
            if record.id in target_ids:
                protein_data.append((record.id, str(record.seq)))
    except Exception as e:
        print(f"Error reading FASTA: {e}")
        sys.exit(1)

    print(f"Found {len(protein_data)} sequences to process")
    skipped = 0
    with open(args.output_file, "w") as out_f:
        for i, (pid, seq) in enumerate(protein_data):
            if len(seq) < 1022:
                batch_labels, batch_strs, batch_tokens = batch_converter([(pid, seq)])
                batch_tokens = batch_tokens.to(device)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                reps = results["representations"][33].cpu()
                encoding = reps[0, 1: batch_lens[0] - 1].mean(0)
                embedding_str = " ".join(str(x.item()) for x in encoding)
                out_f.write(f"{pid}\t{embedding_str}\n")
            else:
                skipped+=1
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(protein_data)} proteins")

    print(f"✅ Done. Embeddings saved to {args.output_file}")
    print(f"⏩ Skipped {skipped} proteins with length ≥ 1022")

if __name__ == "__main__":
    main()
