

from pydantic import BaseModel
from llm_sdk import Small_LLM_Model

def main() -> None:
    obj = Small_LLM_Model()
    string = "j'ai mal a"
    
    tensi = obj.encode(string)
    keep = tensi.tolist()[0]
    print("keep", keep)

    for _ in range(20):
        logits = obj.get_logits_from_input_ids(keep)
        index = logits.index(max(logits))
        keep.append(index)
        oute = obj.decode([keep])


    print(oute)


if __name__ == "__main__":
    main()
