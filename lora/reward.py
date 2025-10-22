import re
from fractions import Fraction

def extract_boxed_answer(s: str) -> str | None:
    if not s:
        return None
    s = s.replace('\\!', '')
    m = re.findall(r'\\?boxed\{([^}]*)\}', s)
    if not m:
        return None
    ans = m[-1].strip()
    ans = re.sub(r'[^0-9\-/\.]', '', ans)
    if ans != '' and ans[-1] == '/':
        ans = ans[:-1]
    return ans or None

def parse_math_value(expr: str):
    if not expr:
        return None
    expr = expr.strip().replace(',', '').replace(' ', '')
    if re.match(r'^-?\d+/\d+$', expr):
        try:
            return Fraction(expr)
        except ZeroDivisionError:
            return None
    try:
        return float(expr)
    except ValueError:
        return None

def math_equiv(a, b, tol=1e-6) -> bool:
    va, vb = parse_math_value(a), parse_math_value(b)
    if va is None or vb is None:
        return False
    try:
        if isinstance(va, Fraction) and isinstance(vb, Fraction):
            return va == vb
        return abs(float(va) - float(vb)) < tol
    except Exception:
        return False

def format_reward_func(completions) -> list[float]:
    pattern = r'\\?boxed\{[^}]*\}'
    responses = [completion[0]['content'] for completion in completions]
    rewards = [1.0 if re.search(pattern, r) else 0.0 for r in responses]
    return rewards

def strict_reward_func(completions, answers) -> list[float]:
    if isinstance(completions,str):
        completions = [[{"content":completions}]]
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_boxed_answer(r) for r in responses]
    rewards = [2.0 if math_equiv(r_pred, r_gt) else 0.0 for r_pred, r_gt in zip(extracted_responses, answers)]
    return rewards

def math_rewards(completions, answers) -> dict[str, list[float]]:
    """
    返回总奖励和分解奖励：
      - total_rewards
      - format_rewards
      - strict_rewards
    """
    format_rewards = format_reward_func(completions)
    strict_rewards = strict_reward_func(completions, answers)
    total_rewards = [f + s for f, s in zip(format_rewards, strict_rewards)]
    return {
        'total_rewards': total_rewards,
        'format_rewards': format_rewards,
        'strict_rewards': strict_rewards
    }

REWARD_FUNCS = {
    "math_rewards_func":math_rewards,
}

if __name__ == "__main__":
    # 测试
    completions = [
        [{'role': 'assistant', 'content': 'The answer is \\boxed{1/2 m/s}.'}],
        [{'role': 'assistant', 'content': 'Answer: 0.5'}],
        [{'role': 'assistant', 'content': 'The solution is \\boxed{2/3}.'}]
    ]
    answers = ['0.5', '0.5', '0.6666666667']

    rewards = math_rewards(completions, answers)
    print(rewards)
