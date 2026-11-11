
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# --------------------------------------------------
# 1. Paths and model configuration
# --------------------------------------------------

MODEL_FILE = Path("models/punctuation/onnx/model_quantized.onnx")
TOKENIZER_FOLDER = Path("models/punctuation/tokenizer")

INPUT_FOLDER = Path("data/model_input")
OUTPUT_FOLDER = Path("data/punctuated")






MODEL_ID2LABEL = {
    0: "0",
    1: ".",
    2: ",",
    3: "?",
    4: "-",
    5: ":"
}

CHUNK_SIZE = 400
OVERLAP_SIZE = 50


# --------------------------------------------------
# 2. Punctuation restoration class
# --------------------------------------------------

class PunctuationRestorer:

    def __init__(self):

        # Check model and tokenizer files
        if not MODEL_FILE.exists():
            raise FileNotFoundError(MODEL_FILE)

        if not TOKENIZER_FOLDER.exists():
            raise FileNotFoundError(TOKENIZER_FOLDER)

        # Load tokenizer from local files
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_FOLDER,
            local_files_only=True,
            use_fast=True
        )

        # Load quantized ONNX model on CPU
        self.session = ort.InferenceSession(
            str(MODEL_FILE),
            providers=["CPUExecutionProvider"]
        )

        # Store special token IDs
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

        print("Punctuation model loaded successfully!")
        print("Inference provider:", self.session.get_providers())


    # --------------------------------------------------
    # 3. Remove existing punctuation
    # --------------------------------------------------

    @staticmethod
    def remove_existing_punctuation(text):

        punctuation_pattern = (
            r"[.,;:!?。，、；：！？؟،¿¡…।॥۔።፣፧]"
        )

        def should_preserve(match):

            mark = match.group()

            # Preserve decimal points and numeric separators
            if mark in ".,:":
                position = match.start()

                if (
                    0 < position < len(text) - 1
                    and text[position - 1].isdigit()
                    and text[position + 1].isdigit()
                ):
                    return mark

            return ""

        return re.sub(
            punctuation_pattern,
            should_preserve,
            text
        )


    # --------------------------------------------------
    # 4. Convert predicted labels into Hindi punctuation
    # --------------------------------------------------

    @staticmethod
    def convert_label(label, previous_character):

        # Full stop becomes Hindi danda
        if label == ".":
            return "।"

        # Keep other supported marks
        if label == "-":
            return " -"

        return label


    # --------------------------------------------------
    # 5. Predict labels using ONNX
    # --------------------------------------------------

    def predict_labels(self, text):

        # Tokenize without adding special tokens
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        predictions = {}
        start = 0

        while start < len(input_ids):

            end = min(
                start + CHUNK_SIZE,
                len(input_ids)
            )

            # Select current chunk
            chunk_ids = input_ids[start:end]

            # Add CLS and SEP tokens
            model_ids = (
                [self.cls_id]
                + chunk_ids
                + [self.sep_id]
            )

            # ONNX requires NumPy int64 arrays
            input_array = np.array(
                [model_ids],
                dtype=np.int64
            )

            attention_mask = np.ones_like(input_array)

            # Run model inference
            logits = self.session.run(
                ["logits"],
                {
                    "input_ids": input_array,
                    "attention_mask": attention_mask
                }
            )[0]

            # Get the highest-scoring label for each token
            predicted_ids = np.argmax(
                logits[0, 1:-1, :],
                axis=-1
            )

            # Avoid duplicate predictions in overlapping chunks
            keep_until = (
                end
                if end == len(input_ids)
                else end - OVERLAP_SIZE
            )

            for index in range(start, keep_until):

                token_position = index - start

                label_id = int(predicted_ids[token_position])
                label = MODEL_ID2LABEL[label_id]

                token_start, token_end = offsets[index]

                if token_end > token_start:
                    predictions[token_end] = label

            if end == len(input_ids):
                break

            start = keep_until

        return predictions


    # --------------------------------------------------
    # 6. Reconstruct punctuated text
    # --------------------------------------------------

    def restore(self, text):

        # Standardize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove existing punctuation before prediction
        text = self.remove_existing_punctuation(text)

        if not text:
            return text

        # Predict punctuation at token boundaries
        predictions = self.predict_labels(text)

        result = []
        previous_position = 0

        for position in sorted(predictions):

            label = predictions[position]

            # Ignore tokens with no punctuation
            if label == "0":
                continue

            # Add punctuation only at word boundaries
            if not (
                position == len(text)
                or text[position] == " "
            ):
                continue

            previous_character = text[position - 1]

            punctuation = self.convert_label(
                label,
                previous_character
            )

            result.append(
                text[previous_position:position]
                + punctuation
            )

            previous_position = position

        result.append(text[previous_position:])

        return "".join(result)


# --------------------------------------------------
# 7. Test on sample-1.txt
# --------------------------------------------------

# --------------------------------------------------
# 7. Process all input files
# --------------------------------------------------

if __name__ == "__main__":

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load the model only once
    restorer = PunctuationRestorer()

    # Find all text files
    text_files = sorted(INPUT_FOLDER.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(
            f"No text files found in {INPUT_FOLDER}"
        )

    total_files = 0

    for input_file in text_files:

        print(f"\nProcessing: {input_file.name}")

        # Read original text
        text = input_file.read_text(encoding="utf-8")

        # Restore punctuation
        punctuated_text = restorer.restore(text)

        # Save output
        output_file = OUTPUT_FOLDER / input_file.name

        output_file.write_text(
            punctuated_text,
            encoding="utf-8"
        )

        # Display statistics
        print("Characters:", len(text))
        print("Full stops:", punctuated_text.count("।"))
        print("Commas:", punctuated_text.count(","))
        print("Questions:", punctuated_text.count("?"))
        print("Saved:", output_file)

        total_files += 1

    print("\n" + "=" * 45)
    print("PUNCTUATION RESTORATION COMPLETED")
    print("Total files processed:", total_files)
    print("Output folder:", OUTPUT_FOLDER)