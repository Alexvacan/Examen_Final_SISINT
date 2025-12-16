import os
import json
import argparse


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_file(path: str) -> str:
    data = load_json(path)

    if "video" not in data or "items" not in data:
        return "Falta 'video' o 'items'"

    items = data["items"]
    if not isinstance(items, list) or len(items) == 0:
        return "items vacío o no es lista"

    x = items[0]
    # campos base
    for k in ["t", "face", "text"]:
        if k not in x:
            return f"Item no tiene '{k}'"

    # face
    if "dominant" not in x["face"]:
        return "face sin dominant"

    # text (puede tener content null)
    for k in ["segment_start", "segment_end", "content", "dominant"]:
        if k not in x["text"]:
            return f"text sin '{k}'"

    # t debe ser número
    if not isinstance(x["t"], (int, float)):
        return "t no es numérico"

    return "OK"


def main():
    ap = argparse.ArgumentParser(description="Validación rápida outputs Día 3")
    ap.add_argument("--dir", default="outputs/multimodal", help="Carpeta multimodal")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"No existe la carpeta: {args.dir}")

    files = sorted([f for f in os.listdir(args.dir) if f.endswith("_multimodal.json")])
    if not files:
        raise SystemExit("No hay archivos *_multimodal.json para validar")

    ok = 0
    fail = 0

    for f in files:
        path = os.path.join(args.dir, f)
        status = validate_file(path)
        if status == "OK":
            print(f"✅ {f}: OK")
            ok += 1
        else:
            print(f"❌ {f}: {status}")
            fail += 1

    print(f"\nResumen: OK={ok} | FAIL={fail}")


if __name__ == "__main__":
    main()
