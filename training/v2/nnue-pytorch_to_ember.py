#!/usr/bin/env python3
import argparse
import glob
import os
import struct
import sys
import gc

PINNED_COMMIT = "a7830b2a91d15f6d3214bd21b1a6cc5cf7701b82"

VERSION = 0x6A448AFA
ARCH_HASH = 0x0256ACDF
FT_HASH = 0x6165DDC9
STACK_HASH = 0x63337116

THREAT_DIMS = 60720
PSQ_DIMS = 22528
L1 = 1024
BUCKETS = 8

STACK_BYTES = (
    4
    + 32 * 4
    + 32 * 1024
    + 32 * 4
    + 32 * 64
    + 4
    + 128
)

MAGIC = b"COMPRESSED_LEB128"


def import_nnue_pytorch(repo):
    repo = os.path.abspath(repo)

    if not os.path.isdir(repo):
        sys.exit(f"Repository not found: {repo}")

    if not os.path.exists(os.path.join(repo, "model", "__init__.py")) and not os.path.exists(
        os.path.join(repo, "model.py")
    ):
        sys.exit(f"Invalid nnue-pytorch repository: {repo}")

    try:
        import subprocess

        rev = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        if rev and rev != PINNED_COMMIT:
            print(
                f"WARNING: repository commit is {rev[:12]}, expected {PINNED_COMMIT[:12]}",
                file=sys.stderr,
            )
    except Exception:
        pass

    sys.path.insert(0, repo)

    import model as M
    import torch

    return M, torch


def write_sleb128_value(f, value):
    v = int(value)

    while True:
        byte = v & 0x7F
        v >>= 7

        if (v == 0 and (byte & 0x40) == 0) or (
            v == -1 and (byte & 0x40) != 0
        ):
            f.write(bytes((byte,)))
            return

        f.write(bytes((byte | 0x80,)))


def encode_tensor_stream(f, tensor, chunk_size=65536):
    arr = tensor.detach().cpu().contiguous().view(-1).numpy()

    total = len(arr)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)

        chunk = arr[start:end]
        encoded = bytearray()

        for x in chunk:
            v = int(x)

            while True:
                byte = v & 0x7F
                v >>= 7

                if (v == 0 and (byte & 0x40) == 0) or (
                    v == -1 and (byte & 0x40) != 0
                ):
                    encoded.append(byte)
                    break

                encoded.append(byte | 0x80)

        f.write(encoded)

        del encoded

    del arr


def write_leb_block(f, tensor):
    f.write(MAGIC)

    length_pos = f.tell()
    f.write(b"\x00\x00\x00\x00")

    start = f.tell()

    encode_tensor_stream(f, tensor)

    end = f.tell()
    length = end - start

    current = f.tell()
    f.seek(length_pos)
    f.write(struct.pack("<I", length))
    f.seek(current)


def make_config(M):
    return M.NNUELightningConfig(
        features="Full_Threats+HalfKAv2_hm^",
        model_config=M.ModelConfig(
            L1=1024,
            L2=32,
            L3=32,
        ),
    )


def load_model(M, torch, ckpt):
    print("Loading checkpoint:", ckpt)

    nnue = M.NNUE.load_from_checkpoint(
        ckpt,
        config=make_config(M),
        map_location=torch.device("cpu"),
    )

    nnue.eval()

    model = nnue.model

    model.input.coalesce()
    model.layer_stacks.coalesce_layer_stacks_inplace()

    return model


def write_raw_tensor(f, tensor):
    arr = tensor.detach().cpu().contiguous().numpy()

    f.write(arr.tobytes())

    del arr


def write_container(torch, model, out, description):
    quant = model.quantization

    with open(out, "wb") as f:
        f.write(
            struct.pack(
                "<III",
                VERSION,
                ARCH_HASH,
                len(description),
            )
        )

        f.write(description)

        f.write(struct.pack("<I", FT_HASH))

        print("Writing feature transformer bias...")

        bias = model.input.bias.data[:L1].to(torch.float32).contiguous()

        bias_q = quant.quantize_feature_transformer_bias(bias)

        write_leb_block(f, bias_q)

        del bias
        del bias_q
        gc.collect()

        print("Preparing feature transformer weights...")

        export_w = model.input.get_export_weights().float()

        w = export_w[:, :L1]
        ps = export_w[:, L1:]

        print("Quantizing threat weights...")

        th_w, th_p = quant.quantize_feature_transformer_weights(
            w[:THREAT_DIMS].contiguous(),
            ps[:THREAT_DIMS].contiguous(),
            torch.int8,
        )

        print("Writing threat weights...")

        write_raw_tensor(f, th_w)

        print("Writing threat PSQT...")

        write_leb_block(f, th_p)

        del th_w
        del th_p
        gc.collect()

        print("Quantizing HalfKAv2 weights...")

        ha_w, ha_p = quant.quantize_feature_transformer_weights(
            w[THREAT_DIMS:].contiguous(),
            ps[THREAT_DIMS:].contiguous(),
            torch.int16,
        )

        print("Writing HalfKAv2 weights...")

        write_leb_block(f, ha_w)

        print("Writing HalfKAv2 PSQT...")

        write_leb_block(f, ha_p)

        del ha_w
        del ha_p
        del w
        del ps
        del export_w

        gc.collect()

        print("Writing layer stacks...")

        stacks = model.layer_stacks.get_coalesced_layer_stacks()

        for i, (l1, l2, out_layer) in enumerate(stacks):
            print(f"Writing stack {i + 1}/{BUCKETS}...")

            l1_b, l1_w = quant.quantize_fc_layer(
                l1.bias.data,
                l1.weight.data,
                "ls_l1",
            )

            l2_b, l2_w = quant.quantize_fc_layer(
                l2.bias.data,
                l2.weight.data,
                "ls_l2",
            )

            o_b, o_w = quant.quantize_fc_layer(
                out_layer.bias.data,
                out_layer.weight.data,
                "ls_output",
            )

            f.write(struct.pack("<I", STACK_HASH))

            write_raw_tensor(f, l1_b)
            write_raw_tensor(f, l1_w)

            write_raw_tensor(f, l2_b)
            write_raw_tensor(f, l2_w)

            write_raw_tensor(f, o_b)
            write_raw_tensor(f, o_w)

            del l1_b
            del l1_w
            del l2_b
            del l2_w
            del o_b
            del o_w

            gc.collect()


def verify(path):
    size = os.path.getsize(path)

    with open(path, "rb") as f:
        header = f.read(12)

        if len(header) != 12:
            raise RuntimeError("File is too small")

        version, arch, dlen = struct.unpack("<III", header)

        if version != VERSION:
            raise RuntimeError(
                f"Invalid version: {hex(version)}"
            )

        if arch != ARCH_HASH:
            raise RuntimeError(
                f"Invalid architecture hash: {hex(arch)}"
            )

        description = f.read(dlen).decode("utf-8")

        ft = struct.unpack("<I", f.read(4))[0]

        if ft != FT_HASH:
            raise RuntimeError(
                f"Invalid FT hash: {hex(ft)}"
            )

        magic = f.read(len(MAGIC))

        if magic != MAGIC:
            raise RuntimeError("Missing LEB128 bias block")

        block_length = struct.unpack("<I", f.read(4))[0]

        f.seek(block_length, os.SEEK_CUR)

        threat_bytes = THREAT_DIMS * L1

        f.seek(threat_bytes, os.SEEK_CUR)

        for _ in range(3):
            magic = f.read(len(MAGIC))

            if magic != MAGIC:
                raise RuntimeError("Invalid LEB128 block")

            block_length = struct.unpack("<I", f.read(4))[0]

            f.seek(block_length, os.SEEK_CUR)

        remaining = size - f.tell()

        expected = STACK_BYTES * BUCKETS

        if remaining != expected:
            raise RuntimeError(
                f"Invalid stack size: {remaining}, expected {expected}"
            )

        for i in range(BUCKETS):
            stack_hash = struct.unpack("<I", f.read(4))[0]

            if stack_hash != STACK_HASH:
                raise RuntimeError(
                    f"Invalid stack hash at stack {i}: {hex(stack_hash)}"
                )

            f.seek(STACK_BYTES - 4, os.SEEK_CUR)

    return {
        "bytes": size,
        "description": description,
    }


def find_last_ckpt(root):
    ckpts = sorted(
        glob.glob(
            os.path.join(
                root,
                "**",
                "checkpoints",
                "last.ckpt",
            ),
            recursive=True,
        ),
        key=os.path.getmtime,
    )

    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        default="",
    )

    parser.add_argument(
        "--out",
        default="",
    )

    parser.add_argument(
        "--root",
        default="",
    )

    parser.add_argument(
        "--description",
        default="",
    )

    args = parser.parse_args()

    if not args.ckpt and not args.root:
        parser.error(
            "Specify --ckpt or --root"
        )

    ckpt = args.ckpt

    if not ckpt:
        ckpt = find_last_ckpt(args.root)

    if not ckpt:
        sys.exit("Checkpoint not found")

    if not os.path.isfile(ckpt):
        sys.exit(
            f"Checkpoint does not exist: {ckpt}"
        )

    M, torch = import_nnue_pytorch(args.repo)

    if args.out:
        out = os.path.abspath(args.out)
    else:
        out = os.path.join(
            os.path.dirname(os.path.abspath(ckpt)),
            "ember-"
            + os.path.splitext(
                os.path.basename(ckpt)
            )[0]
            + ".nnue",
        )

    description = args.description

    if not description:
        description = (
            "Ember external net: "
            "FullThreats(60720)+"
            "HalfKAv2_hm(22528), "
            "L1=1024, stacks 32x32, "
            "8 buckets; "
            "trained with nnue-pytorch "
            "(a7830b2)"
        )

    model = load_model(
        M,
        torch,
        ckpt,
    )

    print("Writing:", out)

    write_container(
        torch,
        model,
        out,
        description.encode("utf-8"),
    )

    print("Verifying...")

    info = verify(out)

    print("Wrote:", out)
    print(
        "Size:",
        info["bytes"],
        "bytes",
        f"({info['bytes'] / 1024 / 1024:.1f} MiB)",
    )
    print("Description:", info["description"])
    print("VERIFY_OK")


if __name__ == "__main__":
    main()

# python .\nnue-pytorch_to_ember.py --repo "nnue-pytorch" --ckpt "D:\nnue-pytorch\***.ckpt" --out "net.nnue"