

from pydantic import BaseModel
from src.parser import Parser
import sys
from typing import Any, Callable
import json
import numpy as np

from llm_sdk import Small_LLM_Model


def get_json(iterable: list[Any]) -> str:
    for i in range(len(iterable)):
        iterable[i] = iterable[i].model_dump()
    out = json.dumps([elem for elem in iterable])
    for elem in iterable:
        elem["name"] = '"' + elem["name"] + '"'
        inside_data = [[key, value] for key, value in elem["parameters"].items()]
        for key, value in inside_data:
            elem["parameters"]['"' + key + '"'] = elem["parameters"].pop(key)

    return out


def create_prompt(obj: Parser) -> list[dict[str, Any]]:

    data = get_json(obj.function_definition)
    output: str = (
        f"Functions:\n{data}\n\n"
        "For the prompt below, pick the most relevant function.\n"
        "Respond ONLY with a JSON array:\n"
        "[{\"prompt\":\"...\",\"name\":\"...\",\"parameters\":{...}} ...]\n"
        "- prompt: original question\n"
        "- name: function name\n"
        "- parameters: values from prompt matching parameter names and types\n"
        "No explanation, no extra text."
    )
    return output

def contrain_basique(logits: list[Any], whitelist: str,
                     dict_to_asc: dict[int, str], dict_to_log: dict[int, str]) -> None:

    masque = np.zeros(len(logits), dtype=bool)
    w_len = len(whitelist)
    for i in range(w_len):
        ind = dict_to_log.get(whitelist[:w_len - i])
        if ind:
            masque[ind] = True

    logits[~masque] = float("-inf")
    return np.argmax(logits)

def validate_case(data: str, memory: str) -> int:
    authorize = [" null", " true", " false"]
    if data.startswith(' "'):
        match data.count('"'):
            case 1:
                return 1
            case 2:
                if data.endswith('"'):
                    return 2
                else:
                    return 4
            case _:
                return 0

    if data == " -" or data == " +":
        return 1

    if data.startswith(' '):
        try:
            float(data[1:])
            return 1
        except Exception:
            try:
                float(memory[1:])
                return 3
            except Exception:
                pass

    for elem in authorize:
        if elem.startswith(data):
            if elem == data:
                return 2
            else:
                return 1
    return 0

def choose_param(
        logits: list[Any], sub_sequence: list [str | int],
        dict_to_asc: dict[int, str], dict_to_log: dict[int, str]) -> tuple[int, int]:

    last_i: int = len(sub_sequence) - 1
    memory: str = sub_sequence[last_i]



    while 1:
        index: int = np.argmax(logits)
        case = dict_to_asc[index]
        test_case = memory + case
        match validate_case(test_case, memory):
            case 0:
                logits[index] = float("-inf")
            case 1:
                break
            case 2:
                sub_sequence[last_i] = ""
                return (index, -1)
            case 3:
                sub_sequence[last_i] = ""
                return (index, -2)
            case 4:
                sub_data = case.split('"')
                sub_data.pop()
                case = '"'.join(sub_data) + '"'
                index = dict_to_log.get(case)
                if index is not None:
                    sub_sequence[last_i] = ""
                    return (index, -1)
                logits[index] = float("-inf")



    sub_sequence[last_i] += case 
    return (index, 0)


def contrain_param(
        logits: list[Any], sub_sequence: list [str | int],
        dict_to_asc: dict[int, str], dict_to_log: dict[str, int]) -> int | None:

    i = sub_sequence[0]
    match sub_sequence[i]:
        case -1:
            index:int
            check: int
            index, check = choose_param(logits, sub_sequence, dict_to_asc, dict_to_log)
            if check < 0:
                sub_sequence[0] += 1
                if check == -2:
                    return None
        case -2:
            index, check = choose_param(logits, sub_sequence, dict_to_asc, dict_to_log)
            if check < 0:
                sub_sequence[0] = -1
                if check == -2:
                    return None
        case _:
            index = contrain_basique(logits, sub_sequence[i], dict_to_asc, dict_to_log)
            sub_sequence[i] = sub_sequence[i][len(dict_to_asc[index]):]
            if sub_sequence[i] == "":
                sub_sequence[0] += 1
    return index


def contrain_func(
        logits: list[Any], previous: str,
        whitelist: list[dict[str, Any]],
        dict_to_asc: dict[int, str], dict_to_log: dict[str, int]) -> int:

    masque = np.zeros(len(logits), dtype=bool)
    for funcs in whitelist:
        elem = funcs["name"]
        if elem.startswith(previous):
            elem = elem[len(previous):]
            w_len = len(elem)
            for i in range(w_len):
                letters = elem[:w_len - i]
                ind = dict_to_log.get(letters)
                if ind:
                    masque[ind] = True

    logits[~masque] = float("-inf")
    return np.argmax(logits)

def create_sub_sequence(iterable: list[dict[str, Any]], func_name: str) -> list[str | int]:
    
    func_dict: dict[str, Any]
    sequence: list[str | int] = [1]

    for elem in iterable:
        if elem["name"] == func_name:
            func_dict = elem
            break

    for elem in func_dict["parameters"].keys():
        sequence.append(f", {elem}:")
        sequence.append(-1)
    sequence[1] = sequence[1][2:]
    sequence[len(sequence) - 1] = -2
    sequence.append("")

    return sequence


def constrained_decoding(
        data: Parser, logits: list[Any],
        sequence: list[str | int],
        dict_to_asc: dict[int, str], dict_to_log: dict[str, int]) -> int | None:

    i = sequence[0]
    index = None
    if sequence[i] == -3:
        return -1
    match sequence[i]:
        case -1:
            contrain_func(logits, sequence[7], data.function_definition,
                          dict_to_asc, dict_to_log)
            index: int = np.argmax(logits)
            sequence[7] += dict_to_asc[index]
            if any(elem for elem in data.function_definition if sequence[7] == elem["name"]):
                sequence.append(
                        create_sub_sequence(data.function_definition, sequence[7]))
                sequence[0] += 1
        case -2:
            index = contrain_param(logits, sequence[8], dict_to_asc, dict_to_log)
            if sequence[8][0] == -1:
                sequence[0] += 1
        case _:
            index = contrain_basique(logits, sequence[i], dict_to_asc, dict_to_log)
            sequence[i] = sequence[i][len(dict_to_asc[index]):]
            if sequence[i] == "":
                sequence[0] += 1
    print("STATUS:\n", sequence)
    return index



def create_sequence(prompt: str, dict_to_log: dict[str, int], start: bool = False, end: bool = False) -> list[int]:

    brac1 = "{"
    sequence: list[str | int] = [1]

    if start:
        sequence_1: str = f'[\n{brac1}"prompt": "{prompt}", "name": '
    else:
        sequence_1: str = f'{brac1}"prompt": "{prompt}", "name": '

    # -1
    sequence_2: str = ', "parameters": {'
    # -2
    if end:
        sequence_3: str = '}}\n]\n'
    else:
        sequence_3: str = '}},\n'

    sequence.append(sequence_1)
    sequence.append(-1)
    sequence.append(sequence_2)
    sequence.append(-2)
    sequence.append(sequence_3)
    sequence.append(-3)
    sequence.append("")


    return sequence


import time # DEBUG TODO
def main(args: list[str]) -> None:

    obj = Small_LLM_Model()
    data = Parser(args=sys.argv)
#    print(data.output)
    base_prompt = create_prompt(data)

    prompts = [(
            base_prompt +
            "\nPrompt:\n" +
            '"' + elem.prompt + '"\n' +
            "\nAnswer:\n"
            ) for elem in data.inputs]
    encoded = []
    for elem in prompts:
        encoded.append(obj.encode(elem))
    keeps = [elem.tolist()[0] for elem in encoded]

    logits = obj.get_logits_from_input_ids(keeps[0])

    dict_to_log = {elem: i for i, elem in enumerate(obj.decode([[i] for i in range(len(logits))]))}
    dict_to_asc = {value: key for key, value in dict_to_log.items()}

    with open(data.output, 'a') as file:
        file.write('\n')
   
    i: int = 8
    last_i = len(data.inputs) - 1
    while i <= last_i:
        sequence = create_sequence(data.inputs[i].prompt, dict_to_log, (i == 0), (i == last_i))
        answer: list[Any] = []
        while 1:
            t_one = time.time()
            logits: Any = np.array(obj.get_logits_from_input_ids(keeps[i]))

            print(time.time() - t_one, "TIME TO GET LOGITS ---------------------------------------------------")
            t_one = time.time()
            index: int | None = constrained_decoding(data, logits, sequence,
                                                     dict_to_asc, dict_to_log)
            print(time.time() - t_one, "TIME TO CONS DECOD ---------------------------------------------------")
            if index == -1:
                print(f"ANSWER {i} / {last_i} DONE");
                break

            if not index is None:
                keeps[i].append(index)
                answer.append(index)
        with open(data.output, 'a') as file:
            t_one = time.time()
            file.write(obj.decode(answer))
            print(time.time() - t_one, "TIME TO WRITE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        i += 1

#    print("\n\nOUTPUT\n\n", string + "".join(answer))
    oute = obj.decode(answer)
    print("\n\nOUTPUT\n\n", oute)

if __name__ == "__main__":
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        print(f"execution killed from keyboard")
    except BaseException as e:
        print(f"an error occured : {e}")
