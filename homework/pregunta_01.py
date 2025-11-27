# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Genera los CSVs de train y test a partir de `files/input.zip`.

Salida:
- `files/output/train_dataset.csv`
- `files/output/test_dataset.csv`

Cada CSV contiene las columnas `phrase` y `target`.
"""


def pregunta_01():
    """Genera los datasets de train y test en `files/output`.

    Extrae `files/input.zip` si es necesario y luego recorre las carpetas
    `train/{positive,negative,neutral}` y `test/{...}` para construir los CSVs.
    """
    import os
    import zipfile
    import pandas as pd

    base_dir = os.path.join(os.getcwd(), "files")
    zip_path = os.path.join(base_dir, "input.zip")
    extract_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")

    # Extraer si existe el zip y la carpeta no existe
    if os.path.exists(zip_path) and not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_dir)

    os.makedirs(output_dir, exist_ok=True)

    def create_dataset(dir_path):
        rows = []
        if not os.path.isdir(dir_path):
            return pd.DataFrame(rows)
        for sentiment in ["positive", "negative", "neutral"]:
            sentiment_dir = os.path.join(dir_path, sentiment)
            if not os.path.isdir(sentiment_dir):
                continue
            for fname in os.listdir(sentiment_dir):
                if not fname.lower().endswith(".txt"):
                    continue
                fpath = os.path.join(sentiment_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        text = fh.read().strip()
                except Exception:
                    with open(fpath, "r", encoding="latin-1") as fh:
                        text = fh.read().strip()
                rows.append({"phrase": text, "target": sentiment})
        return pd.DataFrame(rows)

    train_dir = os.path.join(extract_dir, "train")
    test_dir = os.path.join(extract_dir, "test")

    train_df = create_dataset(train_dir)
    test_df = create_dataset(test_dir)

    train_df.to_csv(os.path.join(output_dir, "train_dataset.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_dataset.csv"), index=False)
