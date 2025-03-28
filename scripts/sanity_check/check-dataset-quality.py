import argparse
import json
import os
from datetime import datetime

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from utils.model_wrappers import OpenAIInterface
from utils.parsers import parse_checker_response
from utils.prompts import sanity_check_prompt

load_dotenv()


def main():
    debug = False

    parser = argparse.ArgumentParser(description="Check dataset quality using LLM")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Huggingface dataset name"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to check"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument(
        "--provider", type=str, default="openai", help="Provider of the model"
    )
    parser.add_argument(
        "--limit", type=int, default=1, help="Limit number of questions to check"
    )
    args = parser.parse_args()

    if args.provider == "openai":
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via OPENAI_KEY env var")

        llm = OpenAIInterface(api_key=api_key, model=args.model)

    split_2_list = [
        "Biology (High School in kaz)",
        "Biology (High School in rus)",
        "Biology (Professional & University in rus)",
        "Chemistry (High School in kaz)",
        "Chemistry (High School in rus)",
        "Culture and Art (Professional & University in rus)",
        "Economics and Entrepreneurship (Professional in rus)",
        "Education and Training (Professional & University in rus)",
        "Finance (Professional & University in rus)",
        "General Education Disciplines (Professional & University in rus)",
        "Geography (High School in kaz)",
        "Geography (High School in rus)",
        "Informatics (High School in kaz)",
        "Informatics (High School in rus)",
        "Jurisprudence (Professional & University in rus)",
        "Kazakh History (High School in kaz)",
        "Kazakh History (High School in rus)",
        "Kazakh Language (High School in kaz)",
        "Kazakh Literature (High School in kaz)",
        "Law (High School in kaz)",
        "Law (High School in rus)",
        "Management and Marketing (Professional & University in rus)",
        "Math (High School in kaz)",
        "Math (High School in rus)",
        "Math Literacy (High School in rus)",
        "Medicine (Professional & University in rus)",
        "Philosophy and Psychology (Professional & University in rus)",
        "Physics (High School in kaz)",
        "Physics (High School in rus)",
        "Reading Literacy (High School in kaz)",
        "Reading Literacy (High School in rus)",
        "Russian Language (High School in rus)",
        "Russian Literature (High School in rus)",
        "World History (High School in kaz)",
        "World History (High School in rus)",
    ]

    exclude_list = [
        "Accounting and Auditing (Professional & University in rus)",
        "Social Science (Professional & University in rus)",
    ]
    split_2_list = [s for s in split_2_list if s not in exclude_list]

    for s2 in split_2_list:
        print(f"Loading dataset {args.dataset}, split: {args.split}. Subject: {s2}")
        dataset = load_dataset(args.dataset, s2, split=args.split)
        dataset_name = "_".join(
            [e for e in [args.dataset.split("/")[1], args.split, s2] if e]
        )

        if args.limit:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

        results = []

        for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
            question = item["Question"].strip()
            options = {
                "A": item["Option A"],
                "B": item["Option B"],
                "C": item["Option C"],
                "D": item["Option D"],
                "E": item["Option E"] if "Option E" in item else None,
            }

            options = {k: v for k, v in options.items() if v is not None}
            marked_answer = item["Answer Key"]

            options = "\n".join([f"{k}: {v.strip()}" for k, v in options.items()])

            prompt = sanity_check_prompt.format(
                question=question,
                options=options,
                marked_answer=marked_answer,
            )

            if debug:
                print("prompt:\n", prompt, "\n\n")

            try:
                llm_response = llm.query(prompt)
                if debug:
                    print("llm_response:\n", llm_response, "\n\n")

                llm_answer, evaluation, explanation = parse_checker_response(
                    llm_response
                )

                is_correct = evaluation == "YES"

                results.append(
                    {
                        "dataset_name": dataset_name,
                        "question_idx": idx,
                        "question": question,
                        "marked_answer": marked_answer,
                        "llm_answer": llm_answer,
                        "is_correct": is_correct,
                        "explanation": explanation,
                    }
                )

            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                results.append(
                    {
                        "dataset_name": dataset_name,
                        "question_idx": idx,
                        "question": question,
                        "marked_answer": marked_answer,
                        "llm_answer": "ERROR",
                        "is_correct": False,
                        "explanation": str(e),
                    }
                )

        with open(f"{dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        correct_count = sum(1 for item in results if item["is_correct"])
        total_count = len(results)

        with open("sanity_check_result.txt", "a") as f:
            f.write(f"Check on {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
            f.write(f"Dataset {dataset_name} Check Summary:\n")
            f.write(f"Total questions checked: {total_count}\n")
            f.write(
                f"Correct answers: {correct_count} ({correct_count/total_count:.2%})\n"
            )
            f.write(
                f"Incorrect answers: {total_count-correct_count} ({(total_count-correct_count)/total_count:.2%})\n"
            )
            f.write(f"Detailed results saved to {dataset_name}.json\n\n")
            f.write("\n")


if __name__ == "__main__":
    main()
