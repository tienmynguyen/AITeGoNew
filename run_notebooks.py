#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script thực thi notebook huấn luyện (train_one_hand.ipynb và train_two_hands.ipynb)
tự động ghi lại kết quả outputs vào file notebook.
"""

import sys
import os
import json
import io
import contextlib
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
plt.show = lambda *args, **kwargs: plt.close('all')


def execute_notebook(nb_path):
    print("=" * 60)
    print(f"BẮT ĐẦU THỰC THI NOTEBOOK: {nb_path}")
    print("=" * 60)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Namespace dùng chung cho các cell trong cùng 1 notebook
    nb_globals = {
        "__name__": "__main__",
        "__file__": nb_path,
    }

    execution_count = 1
    total_cells = len(nb_data.get("cells", []))

    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        raw_source = cell.get("source", [])
        if isinstance(raw_source, list):
            code_text = "".join(raw_source)
        else:
            code_text = str(raw_source)

        print(f"\n[Cell {idx + 1}/{total_cells}] Đang chạy...")
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        cell_error = None
        last_err = None
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(compile(code_text, f"{os.path.basename(nb_path)}_cell_{idx}", "exec"), nb_globals)
        except Exception as e:
            last_err = e
            cell_error = traceback.format_exc()
            print(f"[ERROR ở Cell {idx + 1}]:\n{cell_error}")

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()

        if stdout_val:
            sys.stdout.write(stdout_val)
        if stderr_val:
            sys.stderr.write(stderr_val)

        # Cập nhật kết quả vào cell
        cell["execution_count"] = execution_count
        execution_count += 1
        cell["outputs"] = []

        if stdout_val:
            cell["outputs"].append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_val.splitlines()]
            })

        if stderr_val:
            cell["outputs"].append({
                "name": "stderr",
                "output_type": "stream",
                "text": [line + "\n" for line in stderr_val.splitlines()]
            })

        if cell_error:
            cell["outputs"].append({
                "ename": type(last_err).__name__,
                "evalue": str(last_err),
                "output_type": "error",
                "traceback": cell_error.splitlines()
            })
            # Ghi lại file với lỗi và dừng
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb_data, f, indent=1, ensure_ascii=False)
            raise RuntimeError(f"Lỗi khi thực thi cell {idx + 1} trong {nb_path}")

    # Ghi lại file notebook sau khi hoàn tất thành công
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)

    print(f"\n[HOÀN TẤT THÀNH CÔNG NOTEBOOK]: {nb_path}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        notebooks = sys.argv[1:]
    else:
        notebooks = [
            os.path.join("notebooks", "train_one_hand.ipynb"),
            os.path.join("notebooks", "train_two_hands.ipynb")
        ]

    for nb in notebooks:
        execute_notebook(nb)

    print("==========================================================")
    print(" CÁC NOTEBOOK ĐÃ ĐƯỢC HUẤN LUYỆN VÀ XUẤT TFLITE XONG!")
    print("==========================================================")
