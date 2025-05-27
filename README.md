For the original repo readme, refer to `README_original.md`.

# LLM4Rec: How to Run

## 1. Preprocess Data

1. The LLM-processed session data is located in the `data/` directory.
2. *(Optional)* Clean the CSV data:
    ```bash
    python3 src/clean_csv.py
    ```
    > The current CSV is already cleaned, so this step is optional.
3. Generate prompts:
    ```bash
    python3 src/gen_prompt.py
    ```

## 2. Train

Run the training script:
```bash
./run.sh
```