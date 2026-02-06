import json
import os
import time
from datetime import datetime

from openai import OpenAI

from problems import problems

models = [
    "gpt-4o",
    "o3",
    "deepseek-chat",
    "deepseek-reasoner"
]


def call_model(model_name, problem, attempt):
    print(f"  尝试 {attempt + 1}: 调用 {model_name}...")

    client = OpenAI(
        api_key="sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt",
        base_url="https://svip.xty.app/v1"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": problem}
        ],
        temperature=0.5,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "attempt": attempt + 1
    }


def main():
    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    trials = 1

    print(f"开始处理 {len(problems)} 个问题...")
    print(f"使用模型: {', '.join(models)}")
    print(f"每个模型回答 {trials} 次\n")

    for idx, problem in enumerate(problems, 1):
        print(f"\n{'=' * 60}")
        print(f"问题 {idx}/{len(problems)}: {problem}")
        print(f"{'=' * 60}")

        problem_results = {
            "problem": problem,
            "models": {}
        }

        # 遍历每个模型
        for model_name in models:
            print(f"\n模型: {model_name}")

            model_results = []

            for attempt in range(trials):
                result = call_model(model_name, problem, attempt)
                model_results.append(result)

                if attempt < trials - 1:  # 最后一次不需要延迟
                    time.sleep(1)

            problem_results["models"][model_name] = model_results

        all_results.append(problem_results)

        # 每个问题处理完后保存一次（防止中途出错丢失数据）
        temp_output_file = os.path.join(output_dir, f"results_temp_{timestamp}.json")
        with open(temp_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    final_output_file = os.path.join(output_dir, f"results_final_{timestamp}.json")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"所有问题处理完成！")
    print(f"结果已保存到: {final_output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
