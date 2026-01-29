import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    """
    Main function to read sensitivity analysis results and generate plots.

    Reads 'attention_TP.csv' from the specified input directory, parses the data,
    groups it by experimental configuration, and plots Latency vs Sequence Length (Prefill)
    or KV Cache Length (Decoding).
    """
    parser = argparse.ArgumentParser(
        description="Plot Attention Performance Comparisons"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory containing attention_TP.csv",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    csv_file = os.path.join(input_dir, "attention_TP.csv")
    output_dir = os.path.join(input_dir, "plots")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading data from {csv_file}...")

    # Data structure to hold grouped data
    grouped_data = defaultdict(list)
    is_prefill = False

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        # Check field names to determine if it is prefill or decoding
        if reader.fieldnames and "seq_len" in reader.fieldnames:
            is_prefill = True
            print("Detected Prefill data format (seq_len present).")
        else:
            print("Detected Decoding data format (seq_len_q/seq_len_kv expected).")

        for row in reader:
            try:
                # Parse common keys
                bs = int(row["batch_size"])
                nh_q = int(row["num_heads_q"])
                nh_kv = int(row["num_heads_kv"])
                h_dim = int(row["head_dim"])

                # Parse values
                ours = float(row["Ours"])
                roofline = float(row["Roofline"])
                gpu = float(row["GPU"])

                if is_prefill:
                    # Prefill format: group by (bs, nh_q, nh_kv, h_dim), x-axis is seq_len
                    sl = int(row["seq_len"])
                    key = (bs, nh_q, nh_kv, h_dim)
                    item = {
                        "seq_len": sl,
                        "Ours": ours,
                        "Roofline": roofline,
                        "GPU": gpu,
                    }
                else:
                    # Decoding format: group by (bs, nh_q, nh_kv, h_dim, sl_q), x-axis is seq_len_kv
                    sl_q = int(row["seq_len_q"])
                    sl_kv = int(row["seq_len_kv"])
                    key = (bs, nh_q, nh_kv, h_dim, sl_q)
                    item = {
                        "seq_len_kv": sl_kv,
                        "Ours": ours,
                        "Roofline": roofline,
                        "GPU": gpu,
                    }

                grouped_data[key].append(item)
            except ValueError as e:
                print(f"Skipping row due to error: {e}, Row: {row}")
                continue

    # Plotting
    print(f"Generating plots in {output_dir}...")

    for key, items in grouped_data.items():
        if is_prefill:
            bs, nh_q, nh_kv, h_dim = key
            # Sort items by seq_len
            items.sort(key=lambda x: x["seq_len"])
            x_vals = [item["seq_len"] for item in items]
            xlabel = "Sequence Length (seq_len)"
            title = f"Prefill Comparison (BS={bs}, HQ={nh_q}, HKV={nh_kv}, HD={h_dim})"
            filename = f"prefill_bs{bs}_nhq{nh_q}_nhkv{nh_kv}_hd{h_dim}.png"
        else:
            bs, nh_q, nh_kv, h_dim, sl_q = key
            # Sort items by seq_len_kv
            items.sort(key=lambda x: x["seq_len_kv"])
            x_vals = [item["seq_len_kv"] for item in items]
            xlabel = "KV Cache Length (seq_len_kv)"
            title = (
                f"Decoding Comparison (BS={bs}, HQ={nh_q}, HKV={nh_kv}, "
                f"HD={h_dim}, SL_Q={sl_q})"
            )
            filename = f"comp_bs{bs}_nhq{nh_q}_nhkv{nh_kv}_hd{h_dim}_slq{sl_q}.png"

        y_ours = [item["Ours"] for item in items]
        y_roofline = [item["Roofline"] for item in items]
        y_gpu = [item["GPU"] for item in items]

        plt.figure(figsize=(10, 6))

        plt.plot(x_vals, y_ours, marker="o", label="Ours", color="r")
        plt.plot(x_vals, y_roofline, linestyle="--", label="Roofline", color="g")
        plt.plot(
            x_vals,
            y_gpu,
            marker="s",
            linestyle="-",
            label="GPU (Measured)",
            color="b",
            alpha=0.7,
        )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Latency (ms)")
        plt.legend()
        plt.grid(True)

        # Construct filename
        save_path = os.path.join(output_dir, filename)

        plt.savefig(save_path)
        plt.close()
        # print(f"Saved {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
