import argparse
import zarr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr_path", type=str, required=True)
    args = ap.parse_args()

    root = zarr.open(args.zarr_path, mode="r")
    print("group keys:", list(root.group_keys()))
    for gname in root.group_keys():
        g = root[gname]
        print(f"\n[{gname}] array keys:", list(g.array_keys()))
        for k in g.array_keys():
            arr = g[k]
            print(f"  - {k}: shape={arr.shape}, dtype={arr.dtype}")

if __name__ == "__main__":
    main()
