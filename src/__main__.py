

from pydantic import BaseModel
from src.parser import Parser
import sys
from typing import Any, Callable
import json

from llm_sdk import Small_LLM_Model


def get_json(iterable: list[Any]) -> str:
    for i in range(len(iterable)):
        iterable[i] = iterable[i].model_dump()
    out = json.dumps([elem for elem in iterable], indent=2)
    for elem in iterable:
        elem["name"] = '"' + elem["name"] + '"'
        inside_data = [[key, value] for key, value in elem["parameters"].items()]
        for key, value in inside_data:
            elem["parameters"]['"' + key + '"'] = elem["parameters"].pop(key)

    return out


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

def contrain_basique(logits: list[Any], whitelist: str, dict_to_asc: dict[int, str]) -> None:
    
    for i in range(len(logits)):
        case: str | None = dict_to_asc.get(i)
        if not case or (not whitelist.startswith(case)):
            logits[i] = float("-inf")

def take_one(logits: list[Any], whitelist: str, dict_to_asc: dict[int, str]) -> None:
    
    pass

def test_int(data: str) -> bool:
    try:
        float(data)
    except Exception:
        return False
    return True

def choose_param(
        logits: list[Any], sub_sequence: list [str | int],
        dict_to_asc: dict[int, str]) -> int:

    last_i = len(sub_sequence) - 1

    index: int = logits.index(max(logits))
    if sub_sequence[last_i]:
        while 1:
            case = dict_to_asc[index]
            if case.startswith(',') or case.startswith('}'):
                if sub_sequence[last_i][0] == '"':
                    if sub_sequence[last_i][len(sub_sequence[last_i]) - 1] == '"':
                        sub_sequence[last_i] = ""
                        return -1
                else:
                    sub_sequence[last_i] = ""
                    return -1
            if sub_sequence[last_i][0] == '"':
                if '"' in case and (not case.endswith('"')):
                    logits[index] = float("-inf")
                    index: int = logits.index(max(logits))
                else:
                    break
            else:
                if '.' in sub_sequence[last_i]:
                    if '.' in case:
                        logits[index] = float("-inf")
                        index: int = logits.index(max(logits))
                    if not case.isdigit():
                        logits[index] = float("-inf")
                        index: int = logits.index(max(logits))
                elif case.count('.') > 1 or (not case.lstrip('.').isdigit()):
                    logits[index] = float("-inf")
                    index: int = logits.index(max(logits))
                else:
                    break
    else:
        while 1:
            case = dict_to_asc[index]
            if '"' in case and (not case.startswith('"')):
                logits[index] = float("-inf")
                index: int = logits.index(max(logits))
            elif not test_int(case):
                logits[index] = float("-inf")
                index: int = logits.index(max(logits))
            else:
                break

    sub_sequence[last_i] += case 
    return index


def contrain_param(
        logits: list[Any], sub_sequence: list [str | int],
        dict_to_asc: dict[int, str]) -> int:

    i = sub_sequence[0]
    match sub_sequence[i]:
        case -1:
            index: int = choose_param(logits, sub_sequence, dict_to_asc)
            if index == -1:
                sub_sequence[0] += 1
                return None
        case -2:
            index: int = choose_param(logits, sub_sequence, dict_to_asc)
            if index == -1:
                sub_sequence[0] = -1
                return None
        case _:
            contrain_basique(logits, sub_sequence[i], dict_to_asc)
            index: int = logits.index(max(logits))
            sub_sequence[i] = sub_sequence[i][len(dict_to_asc[index]):]
            if sub_sequence[i] == "":
                sub_sequence[0] += 1
    return index


def contrain_func(
        logits: list[Any], previous: str,
        whitelist: list[dict[str, Any]], dict_to_asc: dict[int, str]) -> None:

    for i in range(len(logits)):
        case: str | None = dict_to_asc.get(i)
        if not case or (not any(
            elem["name"].startswith(previous + case) for elem in whitelist)):
            logits[i] = float("-inf")


def create_sub_sequence(iterable: list[dict[str, Any]], func_name: str) -> list[str | int]:
    
    func_dict: dict[str, Any]
    sequence: list[str | int] = [1]

    for elem in iterable:
        if elem["name"] == func_name:
            func_dict = elem
            break

    for elem in func_dict["parameters"].keys():
        sequence.append(f" {elem}: ")
        sequence.append(-1)
    sequence[1] = sequence[1][1:]
    sequence[len(sequence) - 1] = -2
    sequence.append("")

    return sequence


def constrained_decoding(
        data: Parser, logits: list[Any],
        sequence: list[str | int], dict_to_asc: dict[int, str]) -> int:
    i = sequence[0]
   # print("\nBEFOR\n", sequence)
    index = None # TEMPORAIRE

    if sequence[i] == -3:
        return -1
    match sequence[i]:
        case -1:
            contrain_func(logits, sequence[7], data.function_definition, dict_to_asc)
            index: int = logits.index(max(logits))
            sequence[7] += dict_to_asc[index]
            if any(elem for elem in data.function_definition if sequence[7] == elem["name"]):
                sequence.append(
                        create_sub_sequence(data.function_definition, sequence[7]))
                sequence[0] += 1
        case -2:
            index = contrain_param(logits, sequence[8], dict_to_asc)
            if sequence[8][0] == -1:
                sequence[0] += 1
        case _:
            contrain_basique(logits, sequence[i], dict_to_asc)
            index: int = logits.index(max(logits))
            sequence[i] = sequence[i][len(dict_to_asc[index]):]
          #  print(f"'{dict_to_asc[index]}'")
            if sequence[i] == "":
                sequence[0] += 1
    print("\nAFTER\n", sequence)
    return index



def create_sequence(prompt: str, dict_to_log: dict[str, int], state: int = 1) -> list[int]:

    brac1 = "{"
    sequence: list[str | int] = [1]

    if state == 0:
        sequence_1: str = f'[\n  {brac1} \n    "prompt": "{prompt}",\n    "name": '
    else:
        sequence_1: str = f'  {brac1} \n    "prompt": "{prompt}",\n    "name": '

    # -1
    sequence_2: str = ',\n    "parameters": {'
    # -2
    if state == 2:
        sequence_3: str = '}\n  }\n]\n'
    else:
        sequence_3: str = '}\n  },\n'

    sequence.append(sequence_1)
    sequence.append(-1)
    sequence.append(sequence_2)
    sequence.append(-2)
    sequence.append(sequence_3)
    sequence.append(-3)
    sequence.append("")


    return sequence


def main(args: list[str]) -> None:

    obj = Small_LLM_Model()
    data = Parser(args=sys.argv)
#    print(data.function_definition[0].name)
#    print(data.inputs)
#    print(data.output)
    base_prompt = create_prompt(data)
    string = (
            base_prompt +
            "\nPrompts:\n" +
            "".join(["\"" + elem.prompt + "\"\n" for elem in data.inputs]) +
            "\nAnswer:\n"
            )

    tensi = obj.encode(string)
    keep = tensi.tolist()[0]

    logits: Any = obj.get_logits_from_input_ids(keep)
    dict_to_log = {elem: i for i, elem in enumerate(obj.decode([[i] for i in range(len(logits))]))}
    dict_to_asc = {value: key for key, value in dict_to_log.items()}

    answer: list[str] = []
    
    i: int = 0
   
    last_i = len(data.inputs) - 1
    while i < len(data.inputs):
        if i == 0:
            sequence = create_sequence(data.inputs[i].prompt, dict_to_log, 0)
        elif i == last_i:
            sequence = create_sequence(data.inputs[i].prompt, dict_to_log, 2)
        else:
            sequence = create_sequence(data.inputs[i].prompt, dict_to_log)
        while 1:
            logits: Any = obj.get_logits_from_input_ids(keep)

            # CONSTRAINED DECODING
            index: int = constrained_decoding(data, logits, sequence, dict_to_asc)
            
            if index == -1:
                print(f"ANSWER {i} / {last_i} DONE");
                break

            if index:
                keep.append(index)
        i += 1


#    print("\n\nOUTPUT\n\n", string + "".join(answer))
    oute = obj.decode([keep])[0]
    print("\n\nOUTPUT\n\n", oute)       

if __name__ == "__main__":
    main(sys.argv)
