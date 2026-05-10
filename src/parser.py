
import json
from pydantic import BaseModel, model_validator, ConfigDict
from typing import Any, TypedDict
from os.path import exists
import sys


class InputsDict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


class FuncArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class ReturnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class FuncDefsDict(BaseModel):
    
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, FuncArgs]
    returns: ReturnArgs


class ValidDefsJson(BaseModel):
    data: list[FuncDefsDict]


class ValidInputJson(BaseModel):
    data: list[InputsDict]


class ParserJson(BaseModel):

    path: str
    validation: type[ValidDefsJson] | type[ValidInputJson]
    filtered: list[dict[str, Any] | FuncDefsDict | InputsDict]= []

    @model_validator(mode="after")
    def load_file(self) -> "ParserJson":
        try:
            with open(self.path, 'r') as file:
                data: str = file.read()
                self.filtered = json.loads(data)
            self.filtered = self.validation(data=self.filtered).data
        except Exception as e:
            raise ValueError(f"JSON can't be extract : {e}")
        return self

class ParserOutput(BaseModel):

    path:str

    @model_validator(mode="after")
    def choose_file_name(self) -> "ParserOutput":
        i: int = 0

        if not self.path.endswith(".json"):
            raise ValueError("output must finish with .json")
        while i < 101:
            if exists(self.path):
                self.path = self.path.removesuffix(".json") + "_1.json"
            else:
                if i != 0:
                    print(f"this file name already used, name set to : {self.path}", file=sys.strerr)
                break;
            i += 1
        if i == 101:
            raise ValueError("too many files already created, "
                             "please make your folder clean")
        return self

class Parser(BaseModel):

    args: list[str]
    function_definition: list[FuncDefsDict] = []
    inputs: list[InputsDict] = []
    output: str = ""
    _end: bool = False

    @model_validator(mode="after")
    def check_args(self) -> "ArgsCheck":

        if self.output is None:
            return
        check_fdf: bool = False
        check_inp: bool = False
        check_out: bool = False
        self.args.append("")
        for i in range(len(self.args)):
            match self.args[i]:
                case "--functions_definition":
                    check_fdf = True
                    self.function_definition.extend(ParserJson(
                            path=self.args[i + 1],
                            validation=ValidDefsJson
                            ).filtered)
                case "--input":
                    check_inp = True
                    self.inputs.extend(ParserJson(
                            path=self.args[i + 1],
                            validation=ValidInputJson).filtered)
                case "--output":
                    if check_out:
                        raise ValueError("only one output name can be set")
                    check_out = True
                    self.output = ParserOutput(path=self.args[i + 1]).path

        if self._end:
            return
        self.args = []
        if not check_fdf:
            self.args.extend(["--functions_definition", "data/input/functions_definition.json"])
            self._end = True
        if not check_inp:
            self.args.extend(["--input", "data/input/function_calling_tests.json"])
            self._end = True
        if not check_out:
            self.args.extend(["--output", "data/output/output.json"])
        if self.args:
            self.check_args()

        return self
