import json
import re

def parse_pomdp_policy(file_path: str) -> dict:
    """
    Parses a policy text (which uses => instead of :) into a Python dictionary.

    The parser:
      1. Strips out lines starting with '#'.
      2. Replaces => with :.
      3. Wraps bare words (unquoted keys) in quotes to produce valid JSON.
      4. Uses the json library to parse into a dictionary.

    Returns:
      A dict with fields like:
        {
          "policyType": "...",
          "numPlanes": ...,
          "planes": [
            {
              "action": ...,
              "numEntries": ...,
              "entries": [...]
            },
            ...
          ]
        }
    """
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    text_no_comments = "\n".join(lines)

    # 2. Replace "=>" with ":"
    text_colon = text_no_comments.replace("=>", ":")

    # 3. We need to ensure that unquoted keys become valid JSON strings.
    #    This approach uses a simple regex to capture bare keys at start of
    #    object lines, e.g.  policyType: "MaxPlanesLowerBound"
    #
    #    The pattern:
    #      - Start of line or punctuation that might precede keys ([{,])
    #      - A sequence of word chars or underscores ([a-zA-Z0-9_]+)
    #      - Followed by colon
    #
    #    Then we put quotes around that key group. For example:
    #        policyType: "MaxPlanesLowerBound"
    #      becomes
    #        "policyType": "MaxPlanesLowerBound"
    #
    #    Note: This is a very simplistic approach. In a more complex scenario,
    #    you might want a proper parser or more sophisticated regex logic.
    def quote_unquoted_keys(match):
        return f'"{match.group(1)}":'

    # Regex: look for something like   planes:   or   planes :
    pattern = r'([,{]\s*)([a-zA-Z0-9_]+)\s*:'
    # Insert a placeholder character at the start so we can match keys at the very beginning
    text_colon = " " + text_colon
    text_quoted = re.sub(pattern, lambda m: f'{m.group(1)}"{m.group(2)}":', text_colon)

    # Trim off that extra space we prepended
    text_quoted = text_quoted.strip()

    # 4. Now parse with json
    parsed_policy = json.loads(text_quoted)
    return parsed_policy


def evaluate_pomdp_policy(policy: dict, belief: list[float]) -> (float, int):
    """
    Given a policy (parsed dictionary) and a belief vector b,
    1. Filters out planes not 'applicable' to b.
       A plane is applicable if the *non-zero* indices of b
       are a subset of the indices present in the plane's alpha vector.
    2. Among all applicable planes, compute the inner product of the plane's
       alpha vector with the belief.
    3. Return the maximum inner product (value) and the corresponding action.

    Args:
      policy: A dict with the same structure as returned by parse_pomdp_policy().
      belief: A list of floats (the belief vector).

    Returns:
      (best_value, best_action)
    """
    best_value = float('-inf')
    best_action = None

    # Each plane has "action", "numEntries", and "entries".
    # "entries" is a flat list [index0, value0, index1, value1, ...].
    for plane in policy["planes"]:
        # Build a dictionary for alpha-vector: { index -> value }
        alpha_dict = {}
        entries = plane["entries"]
        for i in range(0, len(entries), 2):
            idx = entries[i]
            val = entries[i + 1]
            alpha_dict[idx] = val

        # Check applicability:
        #   The plane is applicable if for every i where b[i] != 0,
        #   i is in alpha_dict.
        is_applicable = True
        for i, prob in enumerate(belief):
            if prob != 0.0 and i not in alpha_dict:
                is_applicable = False
                break

        if not is_applicable:
            continue

        # Compute inner product of alpha vector and b
        # Because alpha_dict typically only has a few non-zero entries, we can sum just those:
        dot_product = 0.0
        for i, val in alpha_dict.items():
            dot_product += val * belief[i]

        # Track the maximum
        if dot_product > best_value:
            best_value = dot_product
            best_action = plane["action"]

    return best_value, best_action


if __name__ == "__main__":
    policy_dict = parse_pomdp_policy("tiger.hsvi")
    belief_example = [0.5, 0.5]
    value, action = evaluate_pomdp_policy(policy_dict, belief_example)
    print(f"\nBelief: {belief_example}")
    print(f"Best value:  {value}")
    print(f"Best action: {action}")