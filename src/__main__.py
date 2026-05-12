

from pydantic import BaseModel
from src.parser import Parser
import sys
from typing import Any, Callable
import json

from llm_sdk import Small_LLM_Model


def get_json(iterable: list[Any]) -> str:
    return json.dumps([elem.model_dump() for elem in iterable], indent=2)


def create_prompt(obj: Parser) -> list[dict[str, Any]]:

    data = get_json(obj.function_definition)
    output: str = (
            "You have access to the following functions:\n\n"
            f"{data}\n\n"
            'Based on the functions above, for each prompt below select the most '
            'relevant function and respond ONLY with a JSON object where:\n'
            '- "prompt" is the original question as-is\n'
            '- "name" is the function name to call\n'
            '- "parameters" are the values parsed from the prompt,'
            " matching the function's parameter names and types\n"
            "\nrespond ONLY with:\n"
            "{\"prompt\": \"...\",  \"name\": \"...\", \"parameters\": {...}}\n"
            "No explanation, no extra text.\n")

    return output


def main(args: list[str]) -> None:
    data = Parser(args=sys.argv)
#    print(data.function_definition)
#    print(data.inputs)
#    print(data.output)

    base_prompt = create_prompt(data)

    obj = Small_LLM_Model()
    string = (
            base_prompt +
            "\nPrompts:\n" +
            "".join(["\"" + elem.prompt + "\"\n" for elem in data.inputs]) +
            "\nAnswer:\n"
            )
    tensi = obj.encode(string)
    keep = tensi.tolist()[0]
    answer: list[str] = []
   
    def p0(logits: Any) -> int:
        i: int = 0
        case: str
        while i < len(logits):
            case = obj.decode([[i]])[0]
            if not case.startswith("[\n"):
                logits[i] = float("-inf")
            i += 1

        return 1

    def p1(logits: Any) -> int:
        return 1

    phases: list[Callable] = [
                p1,
                p1,
            ]

    i: int = 0
    debug = 0
    while debug < 60:
        logits: Any = obj.get_logits_from_input_ids(keep)

        # CONSTRAINED DECODING
        i = phases[i](logits)
        index = logits.index(max(logits))
        
        keep.append(index)
        letters = obj.decode([[index]])[0]
        answer.append(letters)
        if letters.endswith("]"):
            break
        debug += 1

    print("\n\nOUTPUT\n\n", string + "".join(answer))
        

if __name__ == "__main__":
    main(sys.argv)
